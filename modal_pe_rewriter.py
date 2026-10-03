"""Qwen-Image-2.1 prompt rewriter (PE-T2I) on Modal — direct transformers pattern.

Runs the official Qwen/Qwen-Image-2.1-PE-T2I checkpoint (fine-tuned
Qwen3.5-VL 9B, ~18.8 GB bf16): turns a brief image request in any language
into a detailed English prompt plus a recommended aspect ratio. Text in,
JSON out — it draws nothing itself.

GPU: L4 (24 GB) — hardcoded, not MODAL_GPU. The 9B bf16 weights (~18.8 GB)
do not fit a T4 (16 GB).

Run:
  modal setup
  modal secret create huggingface-secret HF_TOKEN=hf_...   # optional, raises rate limits
  modal run modal_pe_rewriter.py --prompt "astronaut cat"
  modal run modal_pe_rewriter.py --prompt "barish me chai ki dukaan" --max-tokens 2048
"""
from __future__ import annotations

import json
import os

import modal

APP_NAME = "qwen21-pe-rewrite"
VOL_NAME = "qwen21-pe-cache"

HF_REPO_PE_T2I = "Qwen/Qwen-Image-2.1-PE-T2I"


def _download_all():
    from huggingface_hub import snapshot_download

    token = os.environ.get("HF_TOKEN")
    local = snapshot_download(repo_id=HF_REPO_PE_T2I, cache_dir="/cache", token=token)
    print(f"pe-t2i snapshot: {local}", flush=True)


def _hf_secrets() -> list:
    try:
        s = modal.Secret.from_name("huggingface-secret")
        s.hydrate()
        return [s]
    except Exception:
        print("No 'huggingface-secret'; using env HF_TOKEN (may be empty).", flush=True)
        return [modal.Secret.from_dict({"HF_TOKEN": os.environ.get("HF_TOKEN", "")})]


vol = modal.Volume.from_name(VOL_NAME, create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch", "torchvision", index_url="https://download.pytorch.org/whl/cu130")
    .pip_install("transformers>=5.4.0", "accelerate", "pillow",
                 "huggingface_hub[hf_transfer]")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "HF_XET_HIGH_PERFORMANCE": "1"})
    .run_function(_download_all, volumes={"/cache": vol}, secrets=_hf_secrets())
)

app = modal.App(APP_NAME, image=image)


def _snapshot_dir() -> str:
    import glob

    hits = glob.glob("/cache/models--Qwen--Qwen-Image-2.1-PE-T2I/snapshots/*")
    if not hits:
        raise RuntimeError("PE-T2I snapshot missing from /cache (image build download failed?)")
    return sorted(hits)[-1]


@app.cls(
    gpu="L4",  # fixed: 9B bf16 (~18.8 GB) needs 24 GB; T4 cannot fit it
    volumes={"/cache": vol},
    scaledown_window=1800,  # 30 min warm: Enhance is latency-sensitive, cold load is ~1 min.
    # NOTE: warm container bills L4 ($0.80/hr) while idle. Lower to 120 if
    # Enhance is used rarely.
    max_containers=1,
    timeout=900,
    # NO memory snapshot: snapshotting a 19 GB model causes restore
    # slowness/failure (observed: 900s timeout). Cold load is only ~1 min.
    enable_memory_snapshot=False,
)
@modal.concurrent(max_inputs=1)
class PERewrite:
    @modal.enter()
    def load(self):
        import torch
        from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration

        print(f"torch {torch.__version__}, cuda={torch.cuda.is_available()}", flush=True)
        if torch.cuda.is_available():
            print(f"gpu: {torch.cuda.get_device_name(0)}", flush=True)
        d = _snapshot_dir()
        self.processor = AutoProcessor.from_pretrained(d, trust_remote_code=True)
        self.model = Qwen3_5ForConditionalGeneration.from_pretrained(
            d, dtype=torch.bfloat16, device_map="auto", trust_remote_code=True,
        ).eval()
        print("PE-T2I loaded", flush=True)

    @modal.method()
    def rewrite(self, prompt: str, max_tokens: int = 4096) -> dict:
        """Brief request -> {"rewritten_prompt": str, "wh_ratio": str}.

        Thinking mode always ON (no-think emits prose, not the JSON contract).
        4096 cap: observed thinks run 700-1100 tokens before the answer;
        2048 truncated one mid-think into an empty result.
        """
        import time
        import torch

        t0 = time.time()
        nudged = (prompt.strip() + "\n\nThink briefly (a short paragraph), "
                  "then output only the JSON answer.")
        messages = [{"role": "user", "content": [{"type": "text", "text": nudged}]}]
        inputs = self.processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt",
        ).to(self.model.device)
        with torch.inference_mode():
            out = self.model.generate(
                **inputs, max_new_tokens=max_tokens,
                do_sample=True, temperature=0.7, top_p=0.95, top_k=20,
            )
        n_new = len(out[0]) - inputs["input_ids"].shape[-1]
        gen = self.processor.tokenizer.decode(
            out[0, inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
        _, _, answer = gen.partition("</think>")
        try:
            result = json.loads(answer.strip())
        except json.JSONDecodeError:
            result = {"rewritten_prompt": "", "wh_ratio": ""}
        if not isinstance(result, dict):
            result = {"rewritten_prompt": "", "wh_ratio": ""}
        out_d = {"rewritten_prompt": str(result.get("rewritten_prompt", "")),
                 "wh_ratio": str(result.get("wh_ratio", ""))}
        print(f"rewrite: {n_new} tokens, head: {gen[:150]!r}, "
              f"empty={not out_d['rewritten_prompt'].strip()}", flush=True)
        if not out_d["rewritten_prompt"].strip():
            raise RuntimeError(f"PE-T2I empty output; raw tail: {gen[:300]!r}")
        print(f"rewrite took {time.time()-t0:.1f}s", flush=True)
        return out_d


@app.local_entrypoint()
def main(prompt: str = "astronaut cat riding a horse in the rain",
         max_tokens: int = 4096):
    if not prompt.strip():
        raise ValueError("--prompt is required")
    out: dict = PERewrite().rewrite.remote(prompt, max_tokens)
    print(f"ratio: {out['wh_ratio'] or '(none)'}")
    print(f"rewritten: {out['rewritten_prompt']}")

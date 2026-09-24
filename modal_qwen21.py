"""Qwen-Image 2.1 (4-bit quantized) on Modal + ComfyUI.

Two supported 4-bit paths (see README section below):
  A) OFFICIAL / recommended, any CUDA GPU:
     DiT  = qwen_image_2.1_int8_convrot.safetensors (7.26 GB)
     TE   = qwen3vl_8b_w4a8.safetensors             (6.31 GB, W4A8 = 4-bit weights)
     VAE  = qwen_image_2.1_vae_bf16.safetensors     (0.68 GB)
     Total disk ~14.2 GB. Runs on A100-40GB / L40S / H100 / B200.
  B) SMALLEST / Blackwell-only NVFP4 DiT (community):
     DiT  = qwen_image_2.1_nvfp4.safetensors (3.91 GB, pottokao/...-DiT-NVFP4-ComfyUI)
     TE   = qwen3vl_8b_w4a8.safetensors (same as above, most compatible)
     VAE  = same. Total ~10.9 GB. Needs B200/B300 for native FP4 kernels,
     else ComfyUI emulates (memory saving kept, no speedup).

Usage:
  modal setup
  modal volume create qwen21-comfy-cache   # optional, auto-created
  modal run modal_qwen21.py --prompt "a cat in a spacesuit" --use-nvfp4-dit
  modal serve modal_qwen21.py              # temporary UI URL
  modal deploy modal_qwen21.py             # persistent UI + web endpoint
"""
from __future__ import annotations

import json
import subprocess
import time
import urllib.parse
from pathlib import Path

import modal

APP_NAME = "qwen21-4bit-comfyui"
VOL_NAME = "qwen21-comfy-cache"
COMFY_PORT = 8188

HF_REPO_OFFICIAL = "Comfy-Org/Qwen-Image-2.1"
HF_REPO_NVFP4_DIT = "pottokao/Qwen-Image-2.1-DiT-NVFP4-ComfyUI"

# Official 4-bit-capable set (DiT int8 + TE W4A8 + VAE bf16).
OFFICIAL_FILES = [
    ("diffusion_models", "qwen_image_2.1_int8_convrot.safetensors"),
    ("text_encoders", "qwen3vl_8b_w4a8.safetensors"),
    ("text_encoders", "qwen3vl_8b_int8_convrot.safetensors"),  # fallback / template default
    ("vae", "qwen_image_2.1_vae_bf16.safetensors"),
]
# Community NVFP4 DiT (true 4-bit DiT). File must be renamed to match workflow.
NVFP4_FILES = [
    ("diffusion_models", "qwen_image_2.1_nvfp4.safetensors"),
]


def _download_official():
    import os

    from huggingface_hub import hf_hub_download

    cache = Path("/cache")
    for subdir, fname in OFFICIAL_FILES:
        local = hf_hub_download(
            repo_id=HF_REPO_OFFICIAL,
            filename=f"{subdir}/{fname}",
            cache_dir=str(cache),
            token=os.environ.get("HF_TOKEN"),
        )
        target = Path(f"/root/comfy/ComfyUI/models/{subdir}/{fname}")
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() or target.is_symlink():
            target.unlink()
        target.symlink_to(local)
        print(f"linked {fname} -> {target}", flush=True)


def _download_nvfp4_dit():
    import os

    from huggingface_hub import hf_hub_download

    fname = "qwen_image_2.1_nvfp4.safetensors"
    local = hf_hub_download(
        repo_id=HF_REPO_NVFP4_DIT,
        filename=fname,
        cache_dir="/cache",
        token=os.environ.get("HF_TOKEN"),
    )
    target = Path(f"/root/comfy/ComfyUI/models/diffusion_models/{fname}")
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.is_symlink():
        target.unlink()
    target.symlink_to(local)
    print(f"linked NVFP4 DiT {fname} -> {target}", flush=True)


def _download_all():
    _download_official()
    try:
        _download_nvfp4_dit()
    except Exception as e:  # NVFP4 is optional; official set must still work
        print(f"NVFP4 DiT download skipped/failed (official set still usable): {e}", flush=True)


def _hf_secrets() -> list:
    try:
        s = modal.Secret.from_name("huggingface-secret")
        s.hydrate()
        return [s]
    except Exception:
        import os

        print("No Modal secret 'huggingface-secret'; using env HF_TOKEN (may be empty).", flush=True)
        return [modal.Secret.from_dict({"HF_TOKEN": os.environ.get("HF_TOKEN", "")})]


vol = modal.Volume.from_name(VOL_NAME, create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "git-lfs", "libgl1-mesa-dev", "libglib2.0-0", "ffmpeg")
    .pip_install("comfy-cli", "huggingface_hub[hf_transfer]", "requests", "websocket-client")
    .run_commands("comfy --skip-prompt install --nvidia")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "HF_XET_HIGH_PERFORMANCE": "1"})
    .run_function(_download_all, volumes={"/cache": vol}, secrets=_hf_secrets())
)

# Ship the API-format workflow into the image so the remote method can load it.
image = image.add_local_file(
    Path(__file__).parent / "qwen21_workflow_api.json",
    "/root/qwen21_workflow_api.json",
    copy=True,
)

app = modal.App(APP_NAME, image=image)


def _wait_for_port(port: int, timeout: int = 300) -> None:
    import socket

    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=2):
                return
        except OSError:
            time.sleep(1)
    raise TimeoutError(f"ComfyUI did not open port {port} in {timeout}s")


def _load_workflow_template() -> dict:
    with open("/root/qwen21_workflow_api.json") as f:
        return json.load(f)


def _submit_and_wait(base_url: str, workflow: dict, timeout: int = 1200) -> list[bytes]:
    """Classic ComfyUI REST API: POST /prompt, poll /history, GET /view.

    This is the dependency-free 'comfyui python library' path: plain HTTP,
    works against any ComfyUI server. Returns raw PNG bytes.
    """
    import uuid

    import requests

    client_id = str(uuid.uuid4())
    r = requests.post(f"{base_url}/prompt", json={"prompt": workflow, "client_id": client_id}, timeout=60)
    r.raise_for_status()
    prompt_id = r.json()["prompt_id"]

    history_url = f"{base_url}/history/{prompt_id}"
    deadline = time.time() + timeout
    outputs_meta = None
    while time.time() < deadline:
        h = requests.get(history_url, timeout=30).json()
        if prompt_id in h and h[prompt_id].get("status", {}).get("completed", False):
            outputs_meta = h[prompt_id]["outputs"]
            break
        time.sleep(2)
    if outputs_meta is None:
        raise TimeoutError(f"ComfyUI job {prompt_id} did not finish in {timeout}s")

    blobs: list[bytes] = []
    for node_out in outputs_meta.values():
        for img in node_out.get("images", []):
            qs = urllib.parse.urlencode(
                {
                    "filename": img["filename"],
                    "subfolder": img.get("subfolder", ""),
                    "type": img.get("type", "output"),
                }
            )
            vr = requests.get(f"{base_url}/view?{qs}", timeout=120)
            vr.raise_for_status()
            blobs.append(vr.content)
    if not blobs:
        raise RuntimeError(f"No images in ComfyUI outputs: {list(outputs_meta.keys())}")
    return blobs


@app.cls(
    gpu="T4",  # cheapest Modal GPU ($0.59/hr). See note below: L4 is cheapest viable for 2K.
    volumes={"/cache": vol},
    scaledown_window=60,
    timeout=3600,
    enable_memory_snapshot=True,
)
@modal.concurrent(max_inputs=4)
class ComfyQwen21:
    @modal.enter(snap=True)
    def launch(self):
        self.proc = subprocess.Popen(
            "python /root/comfy/ComfyUI/main.py --listen 0.0.0.0 "
            f"--port {COMFY_PORT} --disable-auto-launch",
            shell=True,
        )
        _wait_for_port(COMFY_PORT, timeout=300)
        print("ComfyUI ready", flush=True)

    @modal.exit()
    def stop(self):
        proc = getattr(self, "proc", None)
        if proc is not None:
            try:
                proc.terminate()
            except (ProcessLookupError, OSError):
                pass

    @modal.web_server(COMFY_PORT, startup_timeout=300)
    def ui(self):
        pass  # Modal forwards to the ComfyUI process above

    @modal.method()
    def generate(
        self,
        prompt: str,
        negative_prompt: str = "",
        width: int = 1024,
        height: int = 1024,
        seed: int = 42,
        steps: int = 25,
        use_nvfp4_dit: bool = False,
        text_encoder: str = "qwen3vl_8b_w4a8.safetensors",
    ) -> bytes:
        """Run the Qwen-Image 2.1 4-bit workflow and return one PNG."""
        wf = _load_workflow_template()
        dit = "qwen_image_2.1_nvfp4.safetensors" if use_nvfp4_dit else "qwen_image_2.1_int8_convrot.safetensors"
        wf["3"]["inputs"]["unet_name"] = dit
        wf["4"]["inputs"]["clip_name"] = text_encoder
        wf["4"]["inputs"]["type"] = "qwen_image"
        wf["6"]["inputs"]["prompt"] = prompt
        wf["6"]["inputs"]["negative_prompt"] = negative_prompt
        wf["6"]["inputs"]["resolution"] = max(width, height)
        wf["7"]["inputs"]["width"] = width
        wf["7"]["inputs"]["height"] = height
        wf["8"]["inputs"]["seed"] = seed
        wf["8"]["inputs"]["steps"] = steps
        wf["8"]["inputs"]["cfg"] = 1
        wf["8"]["inputs"]["sampler_name"] = "euler"
        wf["8"]["inputs"]["scheduler"] = "simple"
        blobs = _submit_and_wait(f"http://127.0.0.1:{COMFY_PORT}", wf)
        return blobs[0]


@app.local_entrypoint()
def main(
    prompt: str = "cinematic portrait of an astronaut in a neon Tokyo alley, rain reflections, ultra detailed",
    width: int = 1024,
    height: int = 1024,
    steps: int = 25,
    seed: int = 42,
    use_nvfp4_dit: bool = False,
    out: str = "qwen21_modal_out.png",
):
    png: bytes = ComfyQwen21().generate.remote(
        prompt, "", width, height, seed, steps, use_nvfp4_dit
    )
    Path(out).write_bytes(png)
    print(f"saved {out} ({len(png)/1e6:.2f} MB)")

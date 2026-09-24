"""Qwen-Image 2.1 4-bit on Modal — direct ComfyUI Python-library pattern.

Mirrors z_image_turbo_jupyter (1).py: no server, no /prompt API, no comfy-sdk.
Uses ComfyUI nodes in-process via NODE_CLASS_MAPPINGS:

  UNETLoader / CLIPLoader / VAELoader / TextEncodeQwenImage21 /
  EmptyLatentImage / KSampler / VAEDecode

Text-to-image and edit: pass one or more reference images with --ref-images
(comma-separated local paths, up to 16). TextEncodeQwenImage21 splices them
into the sequence as VAE latents; sampling uses the latent it returns for the
first reference, so the output follows that reference's size and composition.

4-bit file set (Comfy-Org/Qwen-Image-2.1, public):
  diffusion_models/qwen_image_2.1_int8_convrot.safetensors (7.26 GB)
  text_encoders/qwen3vl_8b_w4a8.safetensors                (6.31 GB, W4A8 4-bit)
  vae/qwen_image_2.1_vae_bf16.safetensors                  (0.68 GB)
Optional true-4-bit DiT (community):
  diffusion_models/qwen_image_2.1_nvfp4.safetensors (3.91 GB,
    pottokao/Qwen-Image-2.1-DiT-NVFP4-ComfyUI)

GPU: L4 (24 GB, $0.80/hr) by default — fastest and cheapest per image
(native int8 paths, no decode OOM). Override with MODAL_GPU=T4 ($0.59/hr,
slower, 16 GB) or MODAL_GPU=B200 (native NVFP4).

Run:
  modal setup
  modal secret create huggingface-secret HF_TOKEN=hf_...   # optional, raises rate limits
  modal run modal_qwen21_direct.py --prompt "astronaut in neon Tokyo alley"
  modal run modal_qwen21_direct.py --use-nvfp4-dit --width 1024 --height 1024 --steps 25
  modal run modal_qwen21_direct.py --prompt "make the jacket bright red" --ref-images photo.png
  modal run modal_qwen21_direct.py --prompt "merge these two into one scene" --ref-images "a.png,b.png"
  modal run modal_qwen21_direct.py --scaledown-window 300 --prompt "..."   # keep container warm 5 min
"""
from __future__ import annotations

import io as pyio
import os
import sys
from pathlib import Path

import modal

APP_NAME = "qwen21-4bit-direct"
VOL_NAME = "qwen21-comfy-cache"
COMFY_DIR = "/root/comfy/ComfyUI"
# Idle seconds before a container scales down. Default 60; override per run
# with `modal run modal_qwen21_direct.py --scaledown-window 300 ...`.
DEFAULT_SCALEDOWN_WINDOW = 60

HF_REPO_OFFICIAL = "Comfy-Org/Qwen-Image-2.1"
HF_REPO_NVFP4_DIT = "pottokao/Qwen-Image-2.1-DiT-NVFP4-ComfyUI"

UNET_INT8 = "qwen_image_2.1_int8_convrot.safetensors"
UNET_NVFP4 = "qwen_image_2.1_nvfp4.safetensors"
CLIP_W4A8 = "qwen3vl_8b_w4a8.safetensors"  # 4-bit weights / 8-bit acts, any CUDA GPU
CLIP_INT8 = "qwen3vl_8b_int8_convrot.safetensors"  # template default, larger
VAE = "qwen_image_2.1_vae_bf16.safetensors"


def _download_all():
    from huggingface_hub import hf_hub_download

    token = os.environ.get("HF_TOKEN")
    jobs = [
        (HF_REPO_OFFICIAL, f"diffusion_models/{UNET_INT8}", f"diffusion_models/{UNET_INT8}"),
        (HF_REPO_OFFICIAL, f"text_encoders/{CLIP_W4A8}", f"text_encoders/{CLIP_W4A8}"),
        (HF_REPO_OFFICIAL, f"vae/{VAE}", f"vae/{VAE}"),
    ]
    for repo, remote, rel in jobs:
        local = hf_hub_download(repo_id=repo, filename=remote, cache_dir="/cache", token=token)
        target = Path(f"{COMFY_DIR}/models/{rel}")
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() or target.is_symlink():
            target.unlink()
        target.symlink_to(local)
        print(f"linked {rel}", flush=True)
    try:
        local = hf_hub_download(repo_id=HF_REPO_NVFP4_DIT, filename=UNET_NVFP4, cache_dir="/cache", token=token)
        target = Path(f"{COMFY_DIR}/models/diffusion_models/{UNET_NVFP4}")
        if target.exists() or target.is_symlink():
            target.unlink()
        target.symlink_to(local)
        print(f"linked diffusion_models/{UNET_NVFP4}", flush=True)
    except Exception as e:
        print(f"NVFP4 DiT optional download skipped: {e}", flush=True)


def _hf_secrets() -> list:
    try:
        s = modal.Secret.from_name("huggingface-secret")
        s.hydrate()
        return [s]
    except Exception:
        print("No 'huggingface-secret'; using env HF_TOKEN (may be empty).", flush=True)
        return [modal.Secret.from_dict({"HF_TOKEN": os.environ.get("HF_TOKEN", "")})]


def _fetch_modal_metered_cost():
    """Month-to-date Modal metered cost (USD, pre-credit) via the billing API.

    Returns a Decimal, or None if the lookup fails. Never raises: cost logging
    must not break generation (e.g. unauthenticated env, API hiccup).
    """
    try:
        summary = modal.Workspace.from_context().billing.summary()
        print(
            f"Modal billing: month-to-date metered ${float(summary.metered_cost):.8f}"
            f" (billed ${float(summary.billed_cost):.8f})",
            flush=True,
        )
        return summary.metered_cost
    except Exception as e:
        print(f"Modal billing lookup failed, cost logging skipped: {e}", flush=True)
        return None


vol = modal.Volume.from_name(VOL_NAME, create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "git-lfs", "libgl1-mesa-dev", "libglib2.0-0", "ffmpeg")
    .pip_install("comfy-cli", "huggingface_hub[hf_transfer]", "pillow", "numpy")
    .run_commands("comfy --skip-prompt install --nvidia")
    # Qwen-Image 2.1 nodes (TextEncodeQwenImage21) merged Sep 2026 (ComfyUI >=0.37):
    # force latest master in case comfy-cli pinned an older stable.
    .run_commands("cd /root/comfy/ComfyUI && git fetch origin && git reset --hard origin/master")
    # ComfyUI requires torch cu130+ for Blackwell-era optimized ops (T4 sm75 still supported).
    .run_commands("pip install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "HF_XET_HIGH_PERFORMANCE": "1"})
    .run_function(_download_all, volumes={"/cache": vol}, secrets=_hf_secrets())
)

app = modal.App(APP_NAME, image=image)


@app.cls(
    gpu=os.environ.get("MODAL_GPU", "L4"),  # default L4; override: MODAL_GPU=T4 modal run ...
    volumes={"/cache": vol},
    scaledown_window=DEFAULT_SCALEDOWN_WINDOW,
    timeout=900,  # max container lifetime: 15 min
    enable_memory_snapshot=True,
)
@modal.concurrent(max_inputs=2)
class Qwen21Direct:
    @modal.enter(snap=True)
    def load(self):
        import asyncio
        import time as _time

        import torch

        t0 = _time.time()
        sys.path.insert(0, COMFY_DIR)
        # comfy_extras nodes (incl. TextEncodeQwenImage21) only register through
        # init_extra_nodes(), which main.py normally calls. Importing `nodes`
        # alone gives core nodes only -> KeyError on the Qwen node.
        import nodes as comfy_nodes

        asyncio.run(comfy_nodes.init_extra_nodes(init_custom_nodes=False, init_api_nodes=False))
        from nodes import NODE_CLASS_MAPPINGS

        print(f"node registry loaded in {_time.time()-t0:.1f}s: {len(NODE_CLASS_MAPPINGS)} nodes", flush=True)

        try:
            import comfyui_version

            print(f"ComfyUI version: {comfyui_version.__version__}", flush=True)
        except Exception:
            pass
        print(f"torch {torch.__version__}, cuda={torch.cuda.is_available()}", flush=True)
        if torch.cuda.is_available():
            print(f"gpu: {torch.cuda.get_device_name(0)}, sm={torch.cuda.get_device_capability(0)}", flush=True)
        missing = [n for n in ("UNETLoader", "CLIPLoader", "VAELoader", "TextEncodeQwenImage21", "KSampler", "VAEDecode", "EmptyLatentImage") if n not in NODE_CLASS_MAPPINGS]
        if missing:
            qwen_nodes = sorted(n for n in NODE_CLASS_MAPPINGS if "Qwen" in n)
            raise RuntimeError(f"ComfyUI too old, missing nodes: {missing}. Qwen nodes present: {qwen_nodes}")

        self.torch = torch
        self.UNETLoader = NODE_CLASS_MAPPINGS["UNETLoader"]()
        self.CLIPLoader = NODE_CLASS_MAPPINGS["CLIPLoader"]()
        self.VAELoader = NODE_CLASS_MAPPINGS["VAELoader"]()
        self.TextEncode = NODE_CLASS_MAPPINGS["TextEncodeQwenImage21"]()
        self.KSampler = NODE_CLASS_MAPPINGS["KSampler"]()
        self.VAEDecode = NODE_CLASS_MAPPINGS["VAEDecode"]()
        self.EmptyLatent = NODE_CLASS_MAPPINGS["EmptyLatentImage"]()
        self._models = {}
        print("ComfyUI nodes loaded (direct library mode)", flush=True)

    def _get_models(self, use_nvfp4_dit: bool, clip_name: str):
        key = (use_nvfp4_dit, clip_name)
        if key not in self._models:
            unet_name = UNET_NVFP4 if use_nvfp4_dit else UNET_INT8
            with self.torch.inference_mode():
                unet = self.UNETLoader.load_unet(unet_name, "default")[0]
                clip = self.CLIPLoader.load_clip(clip_name, type="qwen_image")[0]
                vae = self.VAELoader.load_vae(VAE)[0]
            self._models[key] = (unet, clip, vae)
            print(f"models cached: {unet_name} + {clip_name} + {VAE}", flush=True)
        return self._models[key]

    def _bytes_to_image(self, raw: bytes):
        """Local PNG/JPEG bytes -> ComfyUI IMAGE tensor [1, H, W, 3] float32 0-1."""
        import numpy as np
        from PIL import Image

        img = Image.open(pyio.BytesIO(raw)).convert("RGB")
        arr = np.array(img).astype(np.float32) / 255.0
        return self.torch.from_numpy(arr).unsqueeze(0)

    def _encode(self, clip, vae, prompt: str, negative_prompt: str, resolution: int, ref_images=None):
        """Encode prompt (+ optional reference images). Returns (positive, negative, latent|None).

        In edit mode the current node also returns an empty latent sized to the
        first reference — sampling must use it, any other size shifts the edit.
        """
        enc = self.TextEncode
        if hasattr(enc, "encode"):  # older/custom-node shape: text-to-image only
            if ref_images:
                raise RuntimeError("reference images need the current TextEncodeQwenImage21 (execute()) node")
            out = enc.encode(clip, prompt, negative_prompt, resolution)
            return out[0], out[1], None
        # current ComfyUI io.ComfyNode shape: execute(clip, prompt, negative_prompt, vae, resolution, images)
        images = {f"image_{i}": t for i, t in enumerate(ref_images or [], 1)}
        out = enc.execute(clip, prompt, negative_prompt, vae if images else None, resolution, images)
        return out[0], out[1], (out[2] if images else None)

    @modal.method()
    def generate(
        self,
        prompt: str,
        negative_prompt: str = "",
        width: int = 1024,
        height: int = 1024,
        seed: int = 42,
        steps: int = 25,
        cfg: float = 1.0,
        sampler_name: str = "euler",
        scheduler: str = "simple",
        use_nvfp4_dit: bool = False,
        clip_name: str = CLIP_W4A8,
        ref_images: list[bytes] | None = None,
    ) -> bytes:
        import random
        import time

        from PIL import Image

        if seed == 0:
            random.seed(int(time.time()))
            seed = random.randint(0, 18446744073709551615)

        unet, clip, vae = self._get_models(use_nvfp4_dit, clip_name)
        ref_tensors = [self._bytes_to_image(b) for b in (ref_images or [])]
        t_inf = time.time()
        with self.torch.inference_mode():
            positive, negative, edit_latent = self._encode(
                clip, vae, prompt, negative_prompt, max(width, height), ref_tensors
            )
            if edit_latent is not None:
                latent = edit_latent
            else:
                latent = self.EmptyLatent.generate(width, height, batch_size=1)[0]
            samples = self.KSampler.sample(
                unet, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent, denoise=1.0
            )[0]
            decoded = self.VAEDecode.decode(vae, samples)[0].detach()
        if ref_tensors:
            print(
                f"edit mode: {len(ref_tensors)} reference image(s); sampling latent {tuple(latent['samples'].shape)}",
                flush=True,
            )
        print(f"inference (encode+sample+decode) took {time.time()-t_inf:.1f}s at {steps} steps", flush=True)
        arr = (decoded * 255).to(self.torch.uint8).cpu().numpy()[0]
        buf = pyio.BytesIO()
        Image.fromarray(arr).save(buf, format="PNG")
        return buf.getvalue()


@app.local_entrypoint()
def main(
    prompt: str = "cinematic portrait of an astronaut in a neon Tokyo alley, rain reflections, ultra detailed",
    negative: str = "",
    ref_images: str = "",  # comma-separated local paths -> edit mode (up to 16)
    width: int = 1024,
    height: int = 1024,
    # Official Qwen-Image 2.1 sampling settings (ComfyUI template defaults):
    # 25 steps, cfg 1.0, euler + simple. All overridable via CLI flags.
    steps: int = 25,
    cfg: float = 1.0,
    sampler: str = "euler",
    scheduler: str = "simple",
    seed: int = 42,
    use_nvfp4_dit: bool = False,
    scaledown_window: int = DEFAULT_SCALEDOWN_WINDOW,  # idle seconds before scale-down (default 60)
    out: str = "qwen21_direct_out.png",
):
    import time as _billing_time

    # Exact-cost logging: snapshot month-to-date metered cost BEFORE any spend.
    cost_before = _fetch_modal_metered_cost()
    if scaledown_window < 1:
        raise ValueError("--scaledown-window must be at least 1 second")
    refs = [Path(p.strip()).read_bytes() for p in ref_images.split(",") if p.strip()]
    if refs:
        print(f"edit mode: {len(refs)} reference image(s) -> output follows the first reference's size")
    cls = Qwen21Direct
    if scaledown_window != DEFAULT_SCALEDOWN_WINDOW:
        # Dynamic config: bind a variant with this idle window (it autoscales
        # independently of the static default; base config is untouched).
        cls = cls.with_options(scaledown_window=scaledown_window)
        print(f"scaledown_window: {scaledown_window}s idle before scale-down")
    png: bytes = cls().generate.remote(
        prompt, negative, width, height, seed, steps, cfg, sampler, scheduler, use_nvfp4_dit, CLIP_W4A8,
        ref_images=refs or None,
    )
    Path(out).write_bytes(png)
    print(f"saved {out} ({len(png)/1e6:.2f} MB)")

    # Exact-cost logging: snapshot metered cost AFTER the run and report delta.
    # Billing ingestion lags the run by ~1-3 min, so poll until the meter
    # moves (up to ~3 min), then subtract: cost = after - before.
    cost_after = _fetch_modal_metered_cost()
    if cost_before is not None and cost_after is not None:
        deadline = _billing_time.time() + 180
        while cost_after <= cost_before and _billing_time.time() < deadline:
            print("Modal billing has not ingested this run yet; re-checking in 15s ...", flush=True)
            _billing_time.sleep(15)
            cost_after = _fetch_modal_metered_cost()
            if cost_after is None:
                break
        if cost_after is not None:
            run_cost = cost_after - cost_before
            print(
                f"Modal cost for this generation: ${float(run_cost):.8f}"
                f" (metered ${float(cost_before):.8f} -> ${float(cost_after):.8f})",
                flush=True,
            )
            if run_cost <= 0:
                print(
                    "Note: billing still shows no increase; it lags a few minutes —"
                    " re-run `modal billing summary --json` shortly for the final figure."
                    " Delta also includes any concurrent workspace usage.",
                    flush=True,
                )

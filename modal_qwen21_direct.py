"""Qwen-Image 2.1 4-bit on Modal — direct ComfyUI Python-library pattern.

Mirrors z_image_turbo_jupyter (1).py: no server, no /prompt API, no comfy-sdk.
Uses ComfyUI nodes in-process via NODE_CLASS_MAPPINGS:

  UNETLoader / CLIPLoader / VAELoader / TextEncodeQwenImage21 /
  EmptyLatentImage / KSampler / VAEDecode

4-bit file set (Comfy-Org/Qwen-Image-2.1, public):
  diffusion_models/qwen_image_2.1_int8_convrot.safetensors (7.26 GB)
  text_encoders/qwen3vl_8b_w4a8.safetensors                (6.31 GB, W4A8 4-bit)
  vae/qwen_image_2.1_vae_bf16.safetensors                  (0.68 GB)
Optional true-4-bit DiT (community):
  diffusion_models/qwen_image_2.1_nvfp4.safetensors (3.91 GB,
    pottokao/Qwen-Image-2.1-DiT-NVFP4-ComfyUI)

GPU: T4 — cheapest Modal GPU ($0.59/hr). L4 ($0.80/hr, 24 GB) is the
cheapest *viable* pick for 1024px+ with headroom; T4 works at 1024px with
w4a8 TE + offload but is slow (Turing, NVFP4 emulated, no native FP4).

Run:
  modal setup
  modal secret create huggingface-secret HF_TOKEN=hf_...   # optional, raises rate limits
  modal run modal_qwen21_direct.py --prompt "astronaut in neon Tokyo alley"
  modal run modal_qwen21_direct.py --use-nvfp4-dit --width 1024 --height 1024 --steps 25
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
    gpu=os.environ.get("MODAL_GPU", "L4"),  # override: MODAL_GPU=L4 modal run ...
    volumes={"/cache": vol},
    scaledown_window=60,
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

    def _encode(self, clip, prompt: str, negative_prompt: str, resolution: int):
        enc = self.TextEncode
        if hasattr(enc, "encode"):  # older/custom-node shape
            out = enc.encode(clip, prompt, negative_prompt, resolution)
        else:  # current ComfyUI io.ComfyNode shape: execute()
            out = enc.execute(clip, prompt, negative_prompt, None, resolution, {})
        positive, negative = out[0], out[1]
        return positive, negative

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
    ) -> bytes:
        import random
        import time

        import numpy as np
        from PIL import Image

        if seed == 0:
            random.seed(int(time.time()))
            seed = random.randint(0, 18446744073709551615)

        unet, clip, vae = self._get_models(use_nvfp4_dit, clip_name)
        t_inf = time.time()
        with self.torch.inference_mode():
            positive, negative = self._encode(clip, prompt, negative_prompt, max(width, height))
            latent = self.EmptyLatent.generate(width, height, batch_size=1)[0]
            samples = self.KSampler.sample(
                unet, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent, denoise=1.0
            )[0]
            decoded = self.VAEDecode.decode(vae, samples)[0].detach()
        print(f"inference (encode+sample+decode) took {time.time()-t_inf:.1f}s for {width}x{height}@{steps} steps", flush=True)
        arr = np.array(decoded * 255, dtype=np.uint8)[0]
        buf = pyio.BytesIO()
        Image.fromarray(arr).save(buf, format="PNG")
        return buf.getvalue()


@app.local_entrypoint()
def main(
    prompt: str = "cinematic portrait of an astronaut in a neon Tokyo alley, rain reflections, ultra detailed",
    negative: str = "",
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
    out: str = "qwen21_direct_out.png",
):
    png: bytes = Qwen21Direct().generate.remote(
        prompt, negative, width, height, seed, steps, cfg, sampler, scheduler, use_nvfp4_dit, CLIP_W4A8
    )
    Path(out).write_bytes(png)
    print(f"saved {out} ({len(png)/1e6:.2f} MB)")

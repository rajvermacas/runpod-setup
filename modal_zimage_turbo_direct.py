"""Z-Image-Turbo on Modal T4 — direct ComfyUI Python-library pattern.

Mirrors modal_qwen21_direct.py: no server, no /prompt API, no comfy-sdk.
Uses ComfyUI nodes in-process via NODE_CLASS_MAPPINGS:

  UNETLoader / CLIPLoader / VAELoader / CLIPTextEncode /
  ConditioningZeroOut / EmptySD3LatentImage / ModelSamplingAuraFlow /
  KSampler / VAEDecode

Text-to-image only (Z-Image-Edit is unreleased; Turbo has no edit mode).

Model source: Tongyi-MAI/Z-Image-Turbo is diffusers format (~20 GB:
12.3 GB transformer + 8 GB text encoder), not directly loadable by
ComfyUI UNETLoader and too big for a 16 GB T4. This script uses its
official ComfyUI repack, Comfy-Org/z_image_turbo (converted from that
exact repo), with the T4-sized official int8 template file set:

  diffusion_models/z_image_turbo_int8_convrot.safetensors (6.20 GB)
  text_encoders/qwen_3_4b_fp8_mixed.safetensors            (5.63 GB, template default)
  vae/ae.safetensors                                      (0.34 GB)
Optional VRAM-saver (same repo):
  text_encoders/qwen_3_4b_fp4_mixed.safetensors            (3.48 GB)
  diffusion_models/z_image_turbo_bf16.safetensors          (12.3 GB, needs bigger GPU)

Sampling mirrors the official ComfyUI int8 template
(image_z_image_turbo_int8.json): EmptySD3LatentImage ->
ModelSamplingAuraFlow (shift 3) -> KSampler, negative = ConditioningZeroOut
of positive (Turbo uses guidance 0 / cfg 1.0, so no negative prompt).

GPU: T4 (16 GB, $0.59/hr) by default — the whole point of this script.
Override with MODAL_GPU=L4 for headroom.

Run:
  modal setup
  modal secret create huggingface-secret HF_TOKEN=hf_...   # optional, raises rate limits
  modal run modal_zimage_turbo_direct.py --prompt "astronaut in neon Tokyo alley"
  modal run modal_zimage_turbo_direct.py --clip-name qwen_3_4b_fp4_mixed.safetensors --prompt "..."
  modal run modal_zimage_turbo_direct.py --scaledown-window 300 --prompt "..."   # keep container warm 5 min
"""
from __future__ import annotations

import io as pyio
import os
import sys
from pathlib import Path

import modal

APP_NAME = "zimage-turbo-direct"
VOL_NAME = "zimage-turbo-comfy-cache"
COMFY_DIR = "/root/comfy/ComfyUI"
# Idle seconds before a container scales down. Default 200; override per run
# with `modal run modal_zimage_turbo_direct.py --scaledown-window 300 ...`.
DEFAULT_SCALEDOWN_WINDOW = 2

# Comfy-Org/z_image_turbo is the ComfyUI repack of Tongyi-MAI/Z-Image-Turbo.
HF_REPO_COMFY = "Comfy-Org/z_image_turbo"

UNET_INT8 = "z_image_turbo_int8_convrot.safetensors"  # 6.20 GB, T4 default
UNET_BF16 = "z_image_turbo_bf16.safetensors"  # 12.3 GB, needs bigger GPU
CLIP_FP8 = "qwen_3_4b_fp8_mixed.safetensors"  # 5.63 GB, official int8-template default
CLIP_FP4 = "qwen_3_4b_fp4_mixed.safetensors"  # 3.48 GB, extra VRAM headroom on T4
CLIP_BF16 = "qwen_3_4b.safetensors"  # 8.04 GB, needs bigger GPU
VAE = "ae.safetensors"


def _download_all():
    from huggingface_hub import hf_hub_download

    token = os.environ.get("HF_TOKEN")
    jobs = [
        (HF_REPO_COMFY, f"split_files/diffusion_models/{UNET_INT8}", f"diffusion_models/{UNET_INT8}"),
        (HF_REPO_COMFY, f"split_files/text_encoders/{CLIP_FP8}", f"text_encoders/{CLIP_FP8}"),
        (HF_REPO_COMFY, f"split_files/text_encoders/{CLIP_FP4}", f"text_encoders/{CLIP_FP4}"),
        (HF_REPO_COMFY, f"split_files/vae/{VAE}", f"vae/{VAE}"),
    ]
    for repo, remote, rel in jobs:
        local = hf_hub_download(repo_id=repo, filename=remote, cache_dir="/cache", token=token)
        target = Path(f"{COMFY_DIR}/models/{rel}")
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() or target.is_symlink():
            target.unlink()
        target.symlink_to(local)
        print(f"linked {rel}", flush=True)


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
    # Z-Image nodes are core ComfyUI (CLIPLoader type lumina2, ModelSamplingAuraFlow):
    # force latest master in case comfy-cli pinned an older stable.
    .run_commands("cd /root/comfy/ComfyUI && git fetch origin && git reset --hard origin/master")
    # ComfyUI requires torch cu130+ for Blackwell-era optimized ops (T4 sm75 still supported).
    .run_commands("pip install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "HF_XET_HIGH_PERFORMANCE": "1"})
    .run_function(_download_all, volumes={"/cache": vol}, secrets=_hf_secrets())
)

app = modal.App(APP_NAME, image=image)


@app.cls(
    gpu=os.environ.get("MODAL_GPU", "T4"),  # default T4; override: MODAL_GPU=L4 modal run ...
    volumes={"/cache": vol},
    scaledown_window=DEFAULT_SCALEDOWN_WINDOW,
    max_containers=1,  # cap parallel spend while testing; concurrent calls queue
    timeout=600,  # max container lifetime: 10 min
    enable_memory_snapshot=True,
)
@modal.concurrent(max_inputs=1)  # one job at a time; avoids VRAM contention/OOM while testing
class ZImageTurboDirect:
    @modal.enter(snap=True)
    def load(self):
        import asyncio
        import time as _time

        import torch

        t0 = _time.time()
        sys.path.insert(0, COMFY_DIR)
        # comfy_extras nodes (incl. ModelSamplingAuraFlow) only register through
        # init_extra_nodes(), which main.py normally calls. Importing `nodes`
        # alone gives core nodes only -> KeyError on the sampling node.
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
        missing = [n for n in ("UNETLoader", "CLIPLoader", "VAELoader", "CLIPTextEncode", "ConditioningZeroOut", "EmptySD3LatentImage", "ModelSamplingAuraFlow", "KSampler", "VAEDecode") if n not in NODE_CLASS_MAPPINGS]
        if missing:
            raise RuntimeError(f"ComfyUI too old, missing nodes: {missing}")

        self.torch = torch
        self.UNETLoader = NODE_CLASS_MAPPINGS["UNETLoader"]()
        self.CLIPLoader = NODE_CLASS_MAPPINGS["CLIPLoader"]()
        self.VAELoader = NODE_CLASS_MAPPINGS["VAELoader"]()
        self.CLIPTextEncode = NODE_CLASS_MAPPINGS["CLIPTextEncode"]()
        self.ZeroOut = NODE_CLASS_MAPPINGS["ConditioningZeroOut"]()
        self.EmptyLatent = NODE_CLASS_MAPPINGS["EmptySD3LatentImage"]()
        self.AuraFlow = NODE_CLASS_MAPPINGS["ModelSamplingAuraFlow"]()
        self.KSampler = NODE_CLASS_MAPPINGS["KSampler"]()
        self.VAEDecode = NODE_CLASS_MAPPINGS["VAEDecode"]()
        self._models = {}
        print("ComfyUI nodes loaded (direct library mode)", flush=True)

    def _get_models(self, unet_name: str, clip_name: str):
        key = (unet_name, clip_name)
        if key not in self._models:
            with self.torch.inference_mode():
                unet = self.UNETLoader.load_unet(unet_name, "default")[0]
                clip = self.CLIPLoader.load_clip(clip_name, type="lumina2", device="default")[0]
                vae = self.VAELoader.load_vae(VAE)[0]
            self._models[key] = (unet, clip, vae)
            print(f"models cached: {unet_name} + {clip_name} + {VAE}", flush=True)
        return self._models[key]

    @modal.method()
    def generate(
        self,
        prompt: str,
        width: int = 1024,
        height: int = 1024,
        seed: int = 42,
        steps: int = 8,
        cfg: float = 1.0,
        sampler_name: str = "res_multistep",
        scheduler: str = "simple",
        shift: float = 3.0,
        unet_name: str = UNET_INT8,
        clip_name: str = CLIP_FP8,
    ) -> bytes:
        import random
        import time

        from PIL import Image

        if seed == 0:
            random.seed(int(time.time()))
            seed = random.randint(0, 18446744073709551615)

        unet, clip, vae = self._get_models(unet_name, clip_name)
        t_inf = time.time()
        with self.torch.inference_mode():
            # Official int8 template: positive encoded from prompt, negative =
            # zeroed positive (Turbo is guidance-distilled, cfg 1.0).
            positive = self.CLIPTextEncode.encode(clip, prompt)[0]
            negative = self.ZeroOut.zero_out(positive)[0]
            latent = self.EmptyLatent.generate(width, height, batch_size=1)[0]
            # NOTE: call patch_aura (the node's FUNCTION entrypoint), NOT the
            # inherited patch(): patch() defaults multiplier=1000 (SD3 timestep
            # scale) while patch_aura uses multiplier=1.0 (Aura timestep scale).
            # patch() with 1000 silently produces posterized neon garbage.
            unet_shifted = self.AuraFlow.patch_aura(unet, shift)[0]
            samples = self.KSampler.sample(
                unet_shifted, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent, denoise=1.0
            )[0]
            decoded = self.VAEDecode.decode(vae, samples)[0].detach()
        print(f"inference (encode+sample+decode) took {time.time()-t_inf:.1f}s at {steps} steps", flush=True)
        arr = (decoded * 255).to(self.torch.uint8).cpu().numpy()[0]
        buf = pyio.BytesIO()
        Image.fromarray(arr).save(buf, format="PNG")
        return buf.getvalue()


@app.local_entrypoint()
def main(
    prompt: str = "cinematic portrait of an astronaut in a neon Tokyo alley, rain reflections, ultra detailed",
    width: int = 1024,
    height: int = 1024,
    # Official Z-Image-Turbo sampling settings (ComfyUI int8 template defaults):
    # 8 steps, cfg 1.0, res_multistep + simple, ModelSamplingAuraFlow shift 3.
    # (9 num_inference_steps in diffusers == 8 DiT forwards.) All overridable.
    steps: int = 8,
    cfg: float = 1.0,
    sampler: str = "res_multistep",
    scheduler: str = "simple",
    shift: float = 3.0,
    seed: int = 42,
    unet_name: str = UNET_INT8,
    clip_name: str = CLIP_FP8,
    scaledown_window: int = DEFAULT_SCALEDOWN_WINDOW,  # idle seconds before scale-down
    out: str = "zimage_turbo_direct_out.png",
):
    if scaledown_window < 1:
        raise ValueError("--scaledown-window must be at least 1 second")
    cls = ZImageTurboDirect
    if scaledown_window != DEFAULT_SCALEDOWN_WINDOW:
        # Dynamic config: bind a variant with this idle window (it autoscales
        # independently of the static default; base config is untouched).
        cls = cls.with_options(scaledown_window=scaledown_window)
        print(f"scaledown_window: {scaledown_window}s idle before scale-down")
    png: bytes = cls().generate.remote(
        prompt, width, height, seed, steps, cfg, sampler, scheduler, shift, unet_name, clip_name,
    )
    Path(out).write_bytes(png)
    print(f"saved {out} ({len(png)/1e6:.2f} MB)")

"""BFS head-swap on Modal — direct ComfyUI Python-library pattern.

Mirrors modal_qwen21_direct.py: no server, no /prompt API, no comfy-sdk.
Uses ComfyUI nodes in-process via NODE_CLASS_MAPPINGS:

  UNETLoader / CLIPLoader / VAELoader / LoraLoaderModelOnly /
  TextEncodeQwenImage21 / EmptyLatentImage / KSampler / VAEDecode

Runs the Alissonerdx/BFS-Best-Face-Swap head-swap LoRA for Qwen-Image 2.1
on top of the same 4-bit base weights as modal_qwen21_direct.py:

  Image 1 / <image1> (--body-image): target/base image. Its body, pose,
      composition, lighting and environment form the result.
  Image 2 / <image2> (--head-image): reference head. Identity, hair, eye
      color, nose structure are transferred onto image 1.
  Keep this order. Reversing the images swaps who is the target.

LoRA weights (Alissonerdx/BFS-Best-Face-Swap, all drop-in compatible):
  bfs_head_v1.1_qwen_2.1.safetensors             (default, recommended)
  bfs_head_v1.1_alternative_qwen_2.1.safetensors (stronger expression copy)
  bfs_head_v1_qwen_2.1.safetensors               (original release)

Base weights are shared with modal_qwen21_direct.py via the same Modal
volume (qwen21-comfy-cache), so no re-download of the ~14 GB base set.

GPU: T4 ($0.59/hr, cheapest hourly) by default, same as modal_qwen21_direct.py.
Override with MODAL_GPU=L4 ($0.80/hr, faster) or MODAL_GPU=B200.

Run:
  modal setup
  modal secret create huggingface-secret HF_TOKEN=hf_...   # optional, raises rate limits
  modal run modal_bfs_headswap_direct.py --body-image body.png --head-image head.png
  modal run modal_bfs_headswap_direct.py --body-image body.png --head-image head.png --bfs-lora bfs_head_v1.1_alternative_qwen_2.1.safetensors
  modal run modal_bfs_headswap_direct.py --body-image body.png --head-image head.png --use-accel-lora --steps 8
  modal run modal_bfs_headswap_direct.py --body-image body.png --head-image head.png --custom-size --width 1024 --height 1024
"""
from __future__ import annotations

import io as pyio
import os
import sys
from pathlib import Path

import modal

APP_NAME = "bfs-headswap-direct"
VOL_NAME = "qwen21-comfy-cache"  # shared with modal_qwen21_direct.py (base weights)
COMFY_DIR = "/root/comfy/ComfyUI"
# Idle seconds before a container scales down. Default 2; override per run
# with `modal run modal_bfs_headswap_direct.py --scaledown-window 300 ...`.
DEFAULT_SCALEDOWN_WINDOW = 2

HF_REPO_OFFICIAL = "Comfy-Org/Qwen-Image-2.1"
HF_REPO_BFS = "Alissonerdx/BFS-Best-Face-Swap"
HF_REPO_PRUNA = "PrunaAI/Pruna-Qwen-Image-2.1"

UNET_INT8 = "qwen_image_2.1_int8_convrot.safetensors"
CLIP_W4A8 = "qwen3vl_8b_w4a8.safetensors"  # 4-bit weights / 8-bit acts, any CUDA GPU
VAE = "qwen_image_2.1_vae_bf16.safetensors"

BFS_LORA_DEFAULT = "bfs_head_v1.1_qwen_2.1.safetensors"
BFS_LORA_KNOWN = (
    "bfs_head_v1.1_qwen_2.1.safetensors",
    "bfs_head_v1.1_alternative_qwen_2.1.safetensors",
    "bfs_head_v1_qwen_2.1.safetensors",
)
# Optional 8-step acceleration LoRA from the BFS reference workflow
# (not part of BFS; skip with --no-accel, which is the default).
PRUNA_ACCEL_LORA = "p_qwen_image_2.1_8step_v0.1.safetensors"

DEFAULT_PROMPT = (
    "head_swap: start with <image1> as the base image, keeping its lighting, "
    "environment, and background. remove the head from <image1> completely and "
    "replace it with the head from <image2>, strictly preserving the hair, eye "
    "color, nose structure from <image2>. copy the direction of the eye, head "
    "rotation, micro expressions from <image1>, high quality, sharp details, 4k"
)


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
    for fname in BFS_LORA_KNOWN:
        local = hf_hub_download(repo_id=HF_REPO_BFS, filename=fname, cache_dir="/cache", token=token)
        target = Path(f"{COMFY_DIR}/models/loras/{fname}")
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() or target.is_symlink():
            target.unlink()
        target.symlink_to(local)
        print(f"linked loras/{fname}", flush=True)
    try:
        local = hf_hub_download(repo_id=HF_REPO_PRUNA, filename=PRUNA_ACCEL_LORA, cache_dir="/cache", token=token)
        target = Path(f"{COMFY_DIR}/models/loras/{PRUNA_ACCEL_LORA}")
        if target.exists() or target.is_symlink():
            target.unlink()
        target.symlink_to(local)
        print(f"linked loras/{PRUNA_ACCEL_LORA}", flush=True)
    except Exception as e:
        print(f"Pruna accel LoRA optional download skipped: {e}", flush=True)


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
    gpu=os.environ.get("MODAL_GPU", "T4"),  # default T4; override: MODAL_GPU=L4 modal run ...
    volumes={"/cache": vol},
    scaledown_window=DEFAULT_SCALEDOWN_WINDOW,
    max_containers=1,  # cap parallel spend while testing; concurrent calls queue
    timeout=600,  # max container lifetime: 10 min
    enable_memory_snapshot=True,
)
@modal.concurrent(max_inputs=1)  # one job at a time; avoids VRAM contention/OOM while testing
class BFSHeadSwapDirect:
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
        missing = [n for n in ("UNETLoader", "CLIPLoader", "VAELoader", "LoraLoaderModelOnly", "TextEncodeQwenImage21", "KSampler", "VAEDecode", "EmptyLatentImage") if n not in NODE_CLASS_MAPPINGS]
        if missing:
            qwen_nodes = sorted(n for n in NODE_CLASS_MAPPINGS if "Qwen" in n)
            raise RuntimeError(f"ComfyUI too old, missing nodes: {missing}. Qwen nodes present: {qwen_nodes}")

        self.torch = torch
        self.UNETLoader = NODE_CLASS_MAPPINGS["UNETLoader"]()
        self.CLIPLoader = NODE_CLASS_MAPPINGS["CLIPLoader"]()
        self.VAELoader = NODE_CLASS_MAPPINGS["VAELoader"]()
        self.LoraLoader = NODE_CLASS_MAPPINGS["LoraLoaderModelOnly"]()
        self.TextEncode = NODE_CLASS_MAPPINGS["TextEncodeQwenImage21"]()
        self.KSampler = NODE_CLASS_MAPPINGS["KSampler"]()
        self.VAEDecode = NODE_CLASS_MAPPINGS["VAEDecode"]()
        self.EmptyLatent = NODE_CLASS_MAPPINGS["EmptyLatentImage"]()
        self._models = {}
        print("ComfyUI nodes loaded (direct library mode)", flush=True)

    def _get_models(self, clip_name: str, lora_name: str, lora_strength: float, use_accel_lora: bool):
        key = (clip_name, lora_name, lora_strength, use_accel_lora)
        if key not in self._models:
            with self.torch.inference_mode():
                unet = self.UNETLoader.load_unet(UNET_INT8, "default")[0]
                clip = self.CLIPLoader.load_clip(clip_name, type="qwen_image")[0]
                vae = self.VAELoader.load_vae(VAE)[0]
                # Same order as the BFS reference workflow: accel LoRA first, then BFS.
                if use_accel_lora:
                    unet = self.LoraLoader.load_lora_model_only(unet, PRUNA_ACCEL_LORA, 1.0)[0]
                unet = self.LoraLoader.load_lora_model_only(unet, lora_name, lora_strength)[0]
            self._models[key] = (unet, clip, vae)
            print(f"models cached: {UNET_INT8} + {clip_name} + {VAE} + {lora_name}@{lora_strength}" + (" + accel" if use_accel_lora else ""), flush=True)
        return self._models[key]

    def _bytes_to_image(self, raw: bytes):
        """Local PNG/JPEG bytes -> ComfyUI IMAGE tensor [1, H, W, 3|4] float32 0-1.

        Keeps alpha when present: TextEncodeQwenImage21 composites RGBA over
        white for the vision tower and encodes all four channels with the VAE,
        so converting everything to RGB would silently drop transparency edits.
        """
        import numpy as np
        from PIL import Image

        img = Image.open(pyio.BytesIO(raw))
        if img.mode in ("RGBA", "LA") or "transparency" in img.info:
            img = img.convert("RGBA")
        else:
            img = img.convert("RGB")
        arr = np.array(img).astype(np.float32) / 255.0
        return self.torch.from_numpy(arr).unsqueeze(0)

    def _encode(self, clip, vae, prompt: str, negative_prompt: str, resolution: int, ref_images=None):
        """Encode prompt + the two head-swap references. Returns (positive, negative, latent|None).

        ref_images must be [body, head]: the node returns an empty latent sized
        to the first reference (the body), which sampling must use unless a
        custom canvas size is requested.
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
        body_image: bytes,
        head_image: bytes,
        prompt: str = DEFAULT_PROMPT,
        negative_prompt: str = "",
        width: int = 1024,
        height: int = 1024,
        custom_size: bool = False,
        seed: int = 42,
        steps: int = 0,  # 0 = auto: 8 with --use-accel-lora, else 25
        cfg: float = 1.0,
        sampler_name: str = "euler",
        scheduler: str = "simple",
        lora_name: str = BFS_LORA_DEFAULT,
        lora_strength: float = 1.0,
        use_accel_lora: bool = False,
        clip_name: str = CLIP_W4A8,
    ) -> bytes:
        import random
        import time

        from PIL import Image

        if seed == 0:
            random.seed(int(time.time()))
            seed = random.randint(0, 18446744073709551615)
        if steps <= 0:
            steps = 8 if use_accel_lora else 25

        unet, clip, vae = self._get_models(clip_name, lora_name, lora_strength, use_accel_lora)
        ref_tensors = [self._bytes_to_image(body_image), self._bytes_to_image(head_image)]
        t_inf = time.time()
        with self.torch.inference_mode():
            positive, negative, edit_latent = self._encode(
                clip, vae, prompt, negative_prompt, max(width, height), ref_tensors
            )
            if custom_size:
                latent = self.EmptyLatent.generate(width, height, batch_size=1)[0]
            elif edit_latent is not None:
                latent = edit_latent
            else:
                latent = self.EmptyLatent.generate(width, height, batch_size=1)[0]
            samples = self.KSampler.sample(
                unet, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent, denoise=1.0
            )[0]
            decoded = self.VAEDecode.decode(vae, samples)[0].detach()
        print(
            f"head-swap: canvas {tuple(latent['samples'].shape)} "
            f"({'custom-size' if custom_size else 'follows body image'}); "
            f"lora {lora_name}@{lora_strength}",
            flush=True,
        )
        print(f"inference (encode+sample+decode) took {time.time()-t_inf:.1f}s at {steps} steps", flush=True)
        arr = (decoded * 255).to(self.torch.uint8).cpu().numpy()[0]
        buf = pyio.BytesIO()
        Image.fromarray(arr).save(buf, format="PNG")
        return buf.getvalue()


@app.local_entrypoint()
def main(
    body_image: str = "",  # target/base image -> <image1> (required)
    head_image: str = "",  # reference head -> <image2> (required)
    prompt: str = DEFAULT_PROMPT,
    negative: str = "",
    width: int = 1024,  # custom-size canvas width; else resolution budget with height
    height: int = 1024,  # custom-size canvas height; else resolution budget with width
    custom_size: bool = False,  # use width x height canvas (workflow ResolutionSelector path) instead of body-image size
    # Qwen-Image 2.1 official sampling: cfg 1.0, euler + simple.
    # steps=0 auto-selects: 8 with --use-accel-lora, else 25.
    steps: int = 0,
    cfg: float = 1.0,
    sampler: str = "euler",
    scheduler: str = "simple",
    seed: int = 42,
    bfs_lora: str = BFS_LORA_DEFAULT,  # one of the three BFS Qwen-2.1 head weights
    lora_strength: float = 1.0,  # BFS guide: start at 1.0
    use_accel_lora: bool = False,  # add the optional Pruna 8-step accel LoRA (pair with --steps 8)
    scaledown_window: int = DEFAULT_SCALEDOWN_WINDOW,  # idle seconds before scale-down (default 2)
    out: str = "bfs_headswap_out.png",
):
    if not body_image or not head_image:
        raise ValueError("--body-image (target, <image1>) and --head-image (reference head, <image2>) are both required")
    if bfs_lora not in BFS_LORA_KNOWN:
        raise ValueError(f"--bfs-lora must be one of {list(BFS_LORA_KNOWN)}")
    if scaledown_window < 1:
        raise ValueError("--scaledown-window must be at least 1 second")
    body = Path(body_image).read_bytes()
    head = Path(head_image).read_bytes()
    print(f"head-swap: body(<image1>)={body_image}, head(<image2>)={head_image}")
    cls = BFSHeadSwapDirect
    if scaledown_window != DEFAULT_SCALEDOWN_WINDOW:
        # Dynamic config: bind a variant with this idle window (it autoscales
        # independently of the static default; base config is untouched).
        cls = cls.with_options(scaledown_window=scaledown_window)
        print(f"scaledown_window: {scaledown_window}s idle before scale-down")
    png: bytes = cls().generate.remote(
        body, head, prompt, negative, width, height, custom_size, seed, steps, cfg,
        sampler, scheduler, bfs_lora, lora_strength, use_accel_lora, CLIP_W4A8,
    )
    Path(out).write_bytes(png)
    print(f"saved {out} ({len(png)/1e6:.2f} MB)")

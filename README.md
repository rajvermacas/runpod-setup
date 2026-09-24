# Qwen-Image 2.1 (4-bit) on Modal

Run **Qwen-Image 2.1** — the 4-bit quantized variants — on [Modal](https://modal.com) GPUs using ComfyUI's Python node library directly (no web UI, no server). Generates 1024×1024 images from text prompts and saves them locally as PNG.

## Model files (4-bit set)

All public, downloaded automatically at image build into the Modal Volume `qwen21-comfy-cache`:

| Component | File | Size | Source |
|---|---|---|---|
| Diffusion model (INT8 ConvRot) | `qwen_image_2.1_int8_convrot.safetensors` | 7.26 GB | `Comfy-Org/Qwen-Image-2.1` |
| Text encoder (W4A8 = 4-bit weights) | `qwen3vl_8b_w4a8.safetensors` | 6.31 GB | `Comfy-Org/Qwen-Image-2.1` |
| VAE | `qwen_image_2.1_vae_bf16.safetensors` | 0.68 GB | `Comfy-Org/Qwen-Image-2.1` |
| *(optional)* NVFP4 DiT (smallest, Blackwell) | `qwen_image_2.1_nvfp4.safetensors` | 3.91 GB | `pottokao/Qwen-Image-2.1-DiT-NVFP4-ComfyUI` |

Sampling defaults: 25 steps, CFG 1.0, euler/simple, 1024×1024 (1 MP). Qwen-Image 2.1 supports native 2K (set `--width 2048 --height 2048`, needs more VRAM).

## Prerequisites

- Python 3.10+ and `pip install modal`
- A Modal account + API token (`modal token new`, or tokens from modal.com → Settings → API tokens)
- Optional: a HuggingFace token for higher download bandwidth (models are public)

## Setup (one time)

```bash
pip install modal
modal token new                        # or: modal token set --token-id ... --token-secret ...
modal profile activate <your-profile>  # e.g. mrinalrajprom1

# Optional but recommended — used by the image build for model downloads:
modal secret create huggingface-secret HF_TOKEN=hf_...
```

## Run an image generation

```bash
modal run modal_qwen21_direct.py \
  --prompt "cinematic photorealistic portrait of Shah Rukh Khan sitting on a wooden bench in a lush green park, soft golden evening light, ultra detailed" \
  --width 1024 --height 1024 --steps 25 --seed 42 \
  --out srk_park.png
```

Flags (defined in `modal_qwen21_direct.py` → `main()`):

| Flag | Default | Notes |
|---|---|---|
| `--prompt` | astronaut example | text prompt |
| `--width/--height` | 1024/1024 | multiples of 32; 2048 = native 2K |
| `--steps` | 25 | official range ~25–50 at CFG 1 |
| `--seed` | 42 | `0` = random seed |
| `--use-nvfp4-dit` | off | swap to NVFP4 DiT (smaller; full speed needs B200) |
| `--out` | `qwen21_direct_out.png` | local output path |

Examples:

```bash
# random seed
modal run modal_qwen21_direct.py --prompt "a cat in a spacesuit" --seed 0 --out cat.png

# NVFP4 DiT (edit gpu="T4" -> "B200" in modal_qwen21_direct.py for native FP4 speed)
modal run modal_qwen21_direct.py --prompt "..." --use-nvfp4-dit
```

## Measured timings (T4, 1024×1024, 25 steps)

| Stage | First run (cold) | Second run (warm) |
|---|---|---|
| Wall clock, launch → PNG | 5 min 17 s | 3 min 27 s |
| Sampling | 2 min 35 s (~6.2 s/it) | 2 min 07 s (~5.1 s/it) |
| Encode + VAE decode | ~36 s | ~27 s |
| **Pure inference** | **3 min 11 s** | **2 min 34 s** |
| Cost (T4 @ $0.59/hr) | ~$0.05 | ~$0.04 |

First run includes one-time image build + ~14 GB model download (10–20 min once). Later runs reuse the cached image and Modal Volume, so only container start + inference.

## GPU choice

| GPU | $/hr | Notes |
|---|---|---|
| **T4** | 0.59 | cheapest; 16 GB — works at 1024px with a decode OOM warning that ComfyUI recovers from |
| L4 | 0.80 | 24 GB — recommended for clean 1024px+ decode headroom |
| B200 | 6.25 | needed for native NVFP4 kernels (`--use-nvfp4-dit`) |

Change `gpu="T4"` in `modal_qwen21_direct.py` (~line 103) to switch.

## Files

| File | Purpose |
|---|---|
| `modal_qwen21_direct.py` | **main entry** — direct ComfyUI nodes (`NODE_CLASS_MAPPINGS` + `torch.inference_mode()`), no server |
| `modal_qwen21.py` | alternative: runs ComfyUI as an HTTP server on Modal (`/prompt` REST API) |
| `qwen21_workflow_api.json` | API-format workflow used by `modal_qwen21.py` |
| `client_qwen21.py` | local client for the server approach (submit + poll + download) |
| `z_image_turbo_jupyter (1).py` | reference notebook the direct pattern was adapted from |
| `AGENTS.md` | lessons learned (read before editing) |

## Troubleshooting

- **`KeyError: 'TextEncodeQwenImage21'`** — importing ComfyUI `nodes` directly registers core nodes only. The code calls `asyncio.run(nodes.init_extra_nodes(init_custom_nodes=False, init_api_nodes=False))` before reading `NODE_CLASS_MAPPINGS`; keep that call if you modify `load()`.
- **`memory allocation failed with OOM` during decode on T4** — expected at 1024px; ComfyUI falls back and still saves the image. Use `gpu="L4"` for headroom.
- **First run slow / model download** — models live in the Modal Volume `qwen21-comfy-cache`; the first build downloads ~14 GB. Subsequent runs are fast.
- **License** — Qwen weights are under the Qwen Research License (research/evaluation; non-commercial).

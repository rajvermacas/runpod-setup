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

### Flags

Defaults are the **official Qwen-Image 2.1 settings** (25 steps, CFG 1.0, euler + simple). Every value can be overridden.

| Flag | Default | Notes |
|---|---|---|
| `--prompt` | astronaut example | text prompt |
| `--negative` | `""` | negative prompt (relevant when CFG > 1) |
| `--width/--height` | 1024/1024 | multiples of 32; 2048 = native 2K |
| `--steps` | 25 | official range ~25–50 at CFG 1 |
| `--cfg` | 1.0 | official path keeps CFG 1; raise only with a negative prompt |
| `--sampler` | `euler` | any ComfyUI sampler (`euler_ancestral`, `dpmpp_2m`, `res_multistep`, …) |
| `--scheduler` | `simple` | `normal`, `karras`, … |
| `--seed` | 42 | `0` = random seed |
| `--use-nvfp4-dit` | off | swap to NVFP4 DiT (full speed needs B200) |
| `--out` | `qwen21_direct_out.png` | local output path |

GPU is selected with the `MODAL_GPU` env var (default `L4`).

### Execution examples

```bash
# 1) default run (L4, official sampling: 25 steps / CFG 1 / euler + simple)
modal run modal_qwen21_direct.py --prompt "a cat in a spacesuit" --out cat.png

# 2) cheapest hourly rate (T4, slower with a recoverable decode OOM warning)
MODAL_GPU=T4 modal run modal_qwen21_direct.py \
  --prompt "cinematic photorealistic full body shot of Sylvester Stallone standing on a sunny beach, muscular build, beach shorts, ocean waves behind him, golden sunlight, white sand, ultra detailed" \
  --seed 0 --out stallone_beach_t4.png

# 3) random seed (every run gives a new image)
modal run modal_qwen21_direct.py --prompt "a red fox in a snowy forest" --seed 0 --out fox.png

# 4) override sampling (e.g. the sd.cpp-style values from Unsloth's GGUF example)
modal run modal_qwen21_direct.py \
  --prompt "a cartoon sloth mascot waving, flat vector illustration, bright colours" \
  --steps 20 --cfg 6.0 --sampler euler --scheduler simple \
  --negative "blurry, low quality" --out sloth.png

# 5) native 2K output (default L4; slower)
modal run modal_qwen21_direct.py \
  --prompt "aerial view of a coral reef, turquoise water, ultra detailed" \
  --width 2048 --height 2048 --out reef_2k.png

# 6) NVFP4 DiT (smallest; native speed only on B200/B300)
MODAL_GPU=B200 modal run modal_qwen21_direct.py --prompt "..." --use-nvfp4-dit --out out_nvfp4.png
```

## Full ComfyUI UI on Modal (optional)

```bash
modal deploy modal_qwen21.py          # persistent app, prints a stable URL
modal app stop qwen21-4bit-comfyui    # stop it when done (redeploy to bring back)
```

Open the printed URL → **Templates → Qwen-Image 2.1** → generate. Billed only while a container is warm (L4 $0.80/hr); scales to zero ~5 min after the last activity. An open browser tab keeps it warm — close it when finished.

## Measured timings (1024×1024, 25 steps)

| Metric | T4 ($0.59/hr) | L4 ($0.80/hr) |
|---|---|---|
| Wall clock, launch → PNG | 3 min 34 s | **1 min 45 s** |
| Sampling | 2 min 14 s (~5.5 s/it) | **20 s (~0.75 s/it)** |
| Encode + VAE decode | ~48 s | ~13 s |
| **Pure inference** | 162.1 s | **33.0 s** |
| Decode OOM warning | yes (recovers) | none |
| **Cost per image** | $0.038 | **~$0.025** |

L4 is both faster and cheaper per image despite the higher hourly rate — ~5× faster sampling (native int8 paths, no VRAM thrashing). The first-ever run includes a one-time image build + ~14 GB model download (10–20 min); after that, only container start + inference.

## GPU choice

| GPU | $/hr | Notes |
|---|---|---|
| T4 | 0.59 | cheapest hourly; 16 GB — works at 1024px with a recoverable decode OOM warning; slow (Turing has no native int8/FP8 paths) |
| **L4** | 0.80 | 24 GB — **default**; **best value per image** (~2× faster wall, ~35% cheaper per image than T4) |
| B200 | 6.25 | native NVFP4 kernels for `--use-nvfp4-dit` |

Select with `MODAL_GPU=T4 modal run modal_qwen21_direct.py ...` (default: L4).

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
- **`memory allocation failed with OOM` during decode** — only happens when overriding to `MODAL_GPU=T4` at 1024px; ComfyUI falls back and still saves the image. The default L4 (24 GB) has headroom and shows no warning.
- **First run slow / model download** — models live in the Modal Volume `qwen21-comfy-cache`; the first build downloads ~14 GB. Subsequent runs are fast.
- **License** — Qwen weights are under the Qwen Research License (research/evaluation; non-commercial).

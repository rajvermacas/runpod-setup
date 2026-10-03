# Qwen-Image 2.1 (4-bit) on Modal

Run **Qwen-Image 2.1** — the 4-bit quantized variants — on [Modal](https://modal.com) GPUs using ComfyUI's Python node library directly (no web UI, no server). Generates 1024×1024 images from text prompts, or edits your own images from one or more reference photos, and saves the result locally as PNG.

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
| `--ref-images` | `""` | comma-separated local paths → **edit mode** (up to 16 refs); output follows the first reference |
| `--width/--height` | 1024/1024 | multiples of 32; 2048 = native 2K |
| `--steps` | 25 | official range ~25–50 at CFG 1 |
| `--cfg` | 1.0 | official path keeps CFG 1; raise only with a negative prompt |
| `--sampler` | `euler` | any ComfyUI sampler (`euler_ancestral`, `dpmpp_2m`, `res_multistep`, …) |
| `--scheduler` | `simple` | `normal`, `karras`, … |
| `--seed` | 0 | random seed (fixed N = reproducible) |
| `--use-nvfp4-dit` | off | swap to NVFP4 DiT (full speed needs B200) |
| `--scaledown-window` | 60 | idle seconds before the container scales down; raise it (e.g. 300) to keep a warm container between runs |
| `--out` | `qwen21_direct_out.png` | local output path |

GPU is selected with the `MODAL_GPU` env var (default `T4`).

### Execution examples

```bash
# 1) default run (T4, official sampling: 25 steps / CFG 1 / euler + simple)
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

# 5) native 2K output (T4 default; slower — prefer L4 for 2K)
modal run modal_qwen21_direct.py \
  --prompt "aerial view of a coral reef, turquoise water, ultra detailed" \
  --width 2048 --height 2048 --out reef_2k.png

# 6) NVFP4 DiT (smallest; native speed only on B200/B300)
MODAL_GPU=B200 modal run modal_qwen21_direct.py --prompt "..." --use-nvfp4-dit --out out_nvfp4.png

# 7) keep the container warm 5 min after the run — a follow-up run with the same
#    --scaledown-window reuses it and skips the cold start
modal run modal_qwen21_direct.py --prompt "another take, same character" --seed 0 --scaledown-window 300 --out take2.png
```

### Edit with reference images

Pass one or more local images with `--ref-images` (comma-separated, up to 16). The text encoder sees them as reference latents, and the output follows the **first reference's size and composition** — in edit mode `--width/--height` set the working resolution (`resolution = max(width, height)`, ~1 MP by default), not the output size.

```bash
# 1) single-reference edit — verified on L4: 41.5 s inference
modal run modal_qwen21_direct.py \
  --prompt "make him wear a bright red jacket and black sunglasses; keep his face, pose and the park background unchanged" \
  --ref-images srk_park.png --seed 0 --out srk_edit_red.png

# 2) multiple references (comma-separated)
modal run modal_qwen21_direct.py \
  --prompt "place the woman from the second image next to the man from the first, same lighting" \
  --ref-images "a.png,b.png" --out combined.png
```

Notes:
- Output dimensions come from the first reference (resized to the `--width`×`--height` box, aspect preserved, multiples of 32). Sampling uses the latent the node returns — any other size shifts the edit.
- Edit runs are slightly slower than text-to-image (41.5 s vs 33.0 s inference on L4 at 1024²): reference latents extend the sequence.
- The full ComfyUI UI path supports the same thing: `LoadImage` → `TextEncodeQwenImage21` (images input) → sample with the node's latent output.

## Web UI (FastAPI + Jinja, own backend → Modal)

Custom UI in `web/` — mode dropdown (**Generate** / **Edit** / **Head-swap** / **Turbo**) + prompt box + reference-image upload. The backend spawns a Modal GPU job and the result page polls until the PNG is ready:

| Mode | Backend (deployed Modal app) | Input |
|---|---|---|
| Generate | `Cls.from_name("qwen21-4bit-direct", "Qwen21Direct")` | prompt only |
| Edit | same Qwen app | prompt + up to 4 refs (output follows the first) |
| Head-swap | `Cls.from_name("bfs-headswap-direct", "BFSHeadSwapDirect")` | exactly 2 images: body/target first, reference head second |
| Turbo | `Cls.from_name("zimage-turbo-direct", "ZImageTurboDirect")` | prompt only, 8 steps (refs rejected, negative ignored) |

Character slots round it out: save a portrait once (`Save a character slot` form → `web/characters/`), pick it from the Character dropdown. The portrait auto-attaches as an identity reference and its anchor text injects into the prompt. In Head-swap mode the slot portrait becomes the head, so one body upload suffices. `No character` = fresh generation, nothing attached.

### Prerequisites

```bash
modal deploy modal_qwen21_direct.py        # one time each — backend lookups need deployed apps
modal deploy modal_bfs_headswap_direct.py
modal deploy modal_zimage_turbo_direct.py
pip install -r web/requirements.txt
```

### Run

```bash
uv venv web/.venv  # restart-proof: lives in the repo (gitignored), not /tmp
uv pip install --python web/.venv/bin/python -r web/requirements.txt
web/.venv/bin/python -m uvicorn web.app:app --host 0.0.0.0 --port 8001
# open http://<this-machine-ip>:8001  (e.g. http://192.168.29.75:8001)

# open directly on a mode:
# http://<ip>:8001/?mode=headswap   (or ?mode=generate / ?mode=edit)

# preselect the form mode from the environment:
DEFAULT_MODE=headswap python3 -m uvicorn web.app:app --host 0.0.0.0 --port 8001
```

UI-only check with zero GPU spend (placeholder image in ~5 s):

```bash
MOCK_MODAL=1 python3 -m uvicorn web.app:app --port 8000
```

### How a generation flows

1. Fill **Prompt** (seed defaults to `0` = random), optionally attach up to **4 reference images** (edit mode follows the first), set size/steps.
2. **Generate on Modal** → backend spawns the job → redirects to `/result/{call_id}`.
3. The page auto-refreshes every 4 s while the GPU works (~1–3 min with cold start), then shows the image + **Download PNG**.

| Endpoint | Purpose |
|---|---|
| `GET /` | form (prompt + upload) |
| `POST /generate` | spawn Modal job → 303 to `/result/{call_id}` |
| `GET /result/{call_id}` | poll (`FunctionCall.from_id().get(timeout=0)`); 202-style wait, then image |
| `GET /image/{call_id}` | final PNG bytes |
| `GET /health` | `{"status":"ok"}` for monitors |

### Cheap testing tips

- Size **512×512**, steps **10–15** — biggest cost savers.
- Prefer L4 for lowest **per image** cost (~$0.025) vs T4 (~$0.038) despite the higher hourly rate (`MODAL_GPU=L4`).
- Infra is already minimal: 0 warm containers, `scaledown_window=2s`, `max_containers=1`, `max_inputs=1`, 10-min timeout.

Env overrides: `MODAL_APP_NAME`, `MODAL_CLS_NAME` (Qwen path), `BFS_MODAL_APP_NAME`, `BFS_MODAL_CLS_NAME` (head-swap path), `TURBO_MODAL_APP_NAME`, `TURBO_MODAL_CLS_NAME` (Turbo path), `DEFAULT_MODE` (form preselect: `generate`/`edit`/`headswap`/`turbo`), `MOCK_MODAL=1`.

### Server logs

Structured logs (`qwen21-web` logger, timestamped) cover: startup config, every `POST /generate` (client IP, prompt, size, steps, seed, ref filenames + KB), Modal spawn time + `call_id`, poll completion (elapsed seconds, PNG bytes), and warnings for bad input/unknown jobs with full tracebacks on failures. Pending polls log at DEBUG only, so the 4-s auto-refresh doesn't spam.

```bash
LOG_LEVEL=DEBUG python3 -m uvicorn web.app:app --port 8000   # verbose polling
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

Edit mode (one 1024² reference) adds ~25% to inference: **41.5 s** on L4 (vs 33.0 s text-to-image).

## GPU choice

| GPU | $/hr | Notes |
|---|---|---|
| T4 | 0.59 | cheapest hourly; 16 GB — works at 1024px with a recoverable decode OOM warning; slow (Turing has no native int8/FP8 paths) |
| **L4** | 0.80 | 24 GB — **best value per image** (~2× faster wall, ~35% cheaper per image than T4); select with `MODAL_GPU=L4` |
| B200 | 6.25 | native NVFP4 kernels for `--use-nvfp4-dit` |

Select with `MODAL_GPU=L4 modal run modal_qwen21_direct.py ...` (default: T4).

## Z-Image-Turbo on T4 (direct nodes)

Branch `zimage-turbo-t4-direct`, entry point `modal_zimage_turbo_direct.py` — same direct-`NODE_CLASS_MAPPINGS` pattern as the Qwen script, but for [Tongyi-MAI/Z-Image-Turbo](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo). GPU default is **T4** (16 GB, $0.59/hr). Text-to-image only (Z-Image-Edit is unreleased, so no `--ref-images` path).

The official repo is diffusers format (~20 GB, not loadable by ComfyUI `UNETLoader` and OOM on T4), so the script uses its official ComfyUI repack `Comfy-Org/z_image_turbo`, auto-downloaded at image build into the Modal Volume `zimage-turbo-comfy-cache`:

| Component | File | Size |
|---|---|---|
| Diffusion model (INT8) | `diffusion_models/z_image_turbo_int8_convrot.safetensors` | 6.20 GB |
| Text encoder (default, template) | `text_encoders/qwen_3_4b_fp8_mixed.safetensors` | 5.63 GB |
| Text encoder (VRAM-saver) | `text_encoders/qwen_3_4b_fp4_mixed.safetensors` | 3.48 GB |
| VAE | `vae/ae.safetensors` | 0.34 GB |

Sampling mirrors the official ComfyUI int8 template (`image_z_image_turbo_int8.json`): `EmptySD3LatentImage` → `ModelSamplingAuraFlow` (shift 3) → `KSampler`, negative = `ConditioningZeroOut(positive)`, defaults 8 steps / CFG 1.0 / `res_multistep` + `simple`.

### Execution command

```bash
# default run (T4, 8 steps / CFG 1 / res_multistep + simple / shift 3) — verified 2026-09-25: 33.8 s inference, 1024×1024 PNG
modal run modal_zimage_turbo_direct.py \
  --prompt "cinematic portrait of an astronaut in a neon Tokyo alley, rain reflections, ultra detailed" \
  --width 1024 --height 1024 --steps 8 --seed 42 \
  --out zimage_turbo_t4_fixed.png

# extra VRAM headroom on T4 (3.48 GB clip instead of 5.63 GB)
modal run modal_zimage_turbo_direct.py --prompt "a cat in a spacesuit" \
  --clip-name qwen_3_4b_fp4_mixed.safetensors --out cat_z.png

# keep the container warm 5 min between runs
modal run modal_zimage_turbo_direct.py --prompt "..." --scaledown-window 300 --out take2.png
```

| Flag | Default | Notes |
|---|---|---|
| `--prompt` | astronaut example | text prompt (negative is zeroed conditioning — Turbo is guidance-distilled) |
| `--width/--height` | 1024/1024 | output size exactly |
| `--steps` | 8 | official template value (9 diffusers steps == 8 DiT forwards) |
| `--cfg` | 1.0 | keep at 1.0 for Turbo |
| `--sampler/--scheduler` | `res_multistep`/`simple` | template defaults |
| `--shift` | 3.0 | `ModelSamplingAuraFlow` shift |
| `--seed` | 0 | random seed (fixed N = reproducible) |
| `--unet-name` | int8 DiT | `z_image_turbo_bf16.safetensors` needs a bigger GPU |
| `--clip-name` | `qwen_3_4b_fp8_mixed.safetensors` | `qwen_3_4b_fp4_mixed.safetensors` for extra headroom |
| `--scaledown-window` | 200 | idle seconds before scale-down |
| `--out` | `zimage_turbo_direct_out.png` | local output path |

GPU override: `MODAL_GPU=L4 modal run modal_zimage_turbo_direct.py ...`

## BFS head-swap on Qwen-Image 2.1 (direct nodes)

Branch `bfs-headswap-modal`, entry point `modal_bfs_headswap_direct.py` — same direct-`NODE_CLASS_MAPPINGS` pattern as the Qwen script, but for the [BFS head-swap LoRA](https://huggingface.co/Alissonerdx/BFS-Best-Face-Swap) ([Qwen 2.1 guide](https://huggingface.co/Alissonerdx/BFS-Best-Face-Swap/blob/main/docs/qwen-image-2.1.md)). Runs on the **same 4-bit base weights** (shared Modal Volume `qwen21-comfy-cache`, no re-download) with a `LoraLoaderModelOnly` step applied after `UNETLoader`.

Input order matters: `--body-image` → `<image1>` (target: body, pose, background), `--head-image` → `<image2>` (reference head: identity, hair, eyes, nose). Keep this order — reversing the images changes who is the target. Do not use with public figures or people who haven't consented (model terms).

LoRA weights, auto-downloaded at image build into `ComfyUI/models/loras/`:

| File | Notes |
|---|---|
| `bfs_head_v1.1_qwen_2.1.safetensors` | default, recommended |
| `bfs_head_v1.1_alternative_qwen_2.1.safetensors` | stronger expression copy, slightly larger pose error |
| `bfs_head_v1_qwen_2.1.safetensors` | original release (softest skin) |
| `p_qwen_image_2.1_8step_v0.1.safetensors` | optional Pruna accel LoRA from the BFS reference workflow (not part of BFS) |

### Execution commands

```bash
# basic swap (T4 default, v1.1 LoRA @ 1.0, 25 steps / CFG 1 / euler + simple)
# verified 2026-10-03 on L4: ~53 s inference, output follows the body image size
modal run modal_bfs_headswap_direct.py --body-image woman.png --head-image man.png --out headswap_out.png

# alternative LoRA (stronger expression transfer)
modal run modal_bfs_headswap_direct.py --body-image woman.png --head-image man.png \
  --bfs-lora bfs_head_v1.1_alternative_qwen_2.1.safetensors --out headswap_alt.png

# 8-step accel path (Pruna LoRA + fewer steps)
modal run modal_bfs_headswap_direct.py --body-image woman.png --head-image man.png \
  --use-accel-lora --steps 8 --out headswap_fast.png

# explicit canvas instead of body-image size (workflow ResolutionSelector path)
modal run modal_bfs_headswap_direct.py --body-image woman.png --head-image man.png \
  --custom-size --width 1024 --height 1024 --seed 0 --out headswap_custom.png
```

| Flag | Default | Notes |
|---|---|---|
| `--body-image` / `--head-image` | required | local paths → `<image1>` target / `<image2>` reference head |
| `--prompt` | head_swap template | BFS guide prompt; keep the `<image1>`/`<image2>` tokens |
| `--negative` | `""` | relevant when CFG > 1 |
| `--custom-size` | off | off = canvas follows the body image; on = `--width`×`--height` canvas |
| `--width/--height` | 1024/1024 | custom-size canvas, or resolution budget (`max(w,h)`) otherwise |
| `--steps` | 0 (= auto) | 8 with `--use-accel-lora`, else 25 |
| `--cfg` / `--sampler` / `--scheduler` | 1.0 / `euler` / `simple` | official Qwen-Image 2.1 path |
| `--bfs-lora` | v1.1 | one of the three BFS Qwen-2.1 head weights above |
| `--lora-strength` | 1.0 | guide says start at 1.0 |
| `--use-accel-lora` | off | adds the Pruna 8-step LoRA (pair with `--steps 8`) |
| `--seed` | 0 | random seed (fixed N = reproducible) |
| `--scaledown-window` | 2 | idle seconds before scale-down |
| `--out` | `bfs_headswap_out.png` | local output path |

GPU default is T4 (`MODAL_GPU` override, same as the Qwen script). The BFS reference workflow's `deis_2m` sampler needs the res4lyf custom-node pack, so the script defaults to `euler`/`simple` to stay dependency-identical to `modal_qwen21_direct.py`.

## Generic workflow runner (any workflow JSON)

`modal_workflow_direct.py` — runs any ComfyUI workflow file on a Modal GPU
through ComfyUI's **native execution**: the container boots the ComfyUI server
and the workflow is run via the official `/prompt` API (history polled,
images fetched via `/view`) — the same path the UI uses. No re-implemented
executor.

```bash
# UI format, incl. subgraph-wrapped official templates (translated to API first)
modal run modal_workflow_direct.py --workflow image_z_image_turbo_int8.json --prompt "astronaut in neon Tokyo alley" --out turbo.png
# API format ("Export (API)") — posted as-is
modal run modal_workflow_direct.py --workflow qwen21_workflow_api.json --prompt "a red fox in a snowy forest" --random-seed --out fox.png
```

`--prompt` is injected into the positive text encoder (auto-traced via the sampler; `--prompt-node <id|title>` to pick explicitly). Seed priority: explicit `--seed N` > `--seed 0` / `--random-seed` (random) > workflow's own seed (literal, or its randomize-control) > template fallback (random, logged). Models lazy-download on first use from a registry (Qwen-Image 2.1 + Z-Image-Turbo sets) into volume `workflow-comfy-cache`. Overrides: `--steps/--cfg/--sampler/--scheduler/--width/--height/--negative/--unet/--clip/--vae/--ckpt`. T4 default, `MODAL_GPU` override. Verified 2026-09-25 on L4: official Z-Image int8 template (UI+subgraph, template seed) in 38.5 s, Qwen API workflow (`--seed 42`) in 48.4 s — both 1024², correct output.

## Files

| File | Purpose |
|---|---|
| `modal_qwen21_direct.py` | **main entry** — direct ComfyUI nodes (`NODE_CLASS_MAPPINGS` + `torch.inference_mode()`), no server |
| `modal_zimage_turbo_direct.py` | Z-Image-Turbo entry (branch `zimage-turbo-t4-direct`) — same direct pattern, T4 default, 8-step Turbo sampling |
| `modal_bfs_headswap_direct.py` | BFS head-swap entry (branch `bfs-headswap-modal`) — same direct pattern, Qwen-Image 2.1 4-bit base + BFS head LoRA (`--body-image` → `<image1>`, `--head-image` → `<image2>`) |
| `modal_workflow_direct.py` | generic runner: `--workflow file.json --prompt "..."` executes any API/UI-format workflow on Modal GPU |
| `modal_qwen21.py` | alternative: runs ComfyUI as an HTTP server on Modal (`/prompt` REST API) |
| `qwen21_workflow_api.json` | API-format workflow used by `modal_qwen21.py` |
| `client_qwen21.py` | local client for the server approach (submit + poll + download) |
| `web/` | FastAPI + Jinja UI: prompt box + reference upload → backend spawns Modal job (see Web UI section) |
| `z_image_turbo_jupyter (1).py` | reference notebook the direct pattern was adapted from |
| `AGENTS.md` | lessons learned (read before editing) |

## Troubleshooting

- **`KeyError: 'TextEncodeQwenImage21'`** — importing ComfyUI `nodes` directly registers core nodes only. The code calls `asyncio.run(nodes.init_extra_nodes(init_custom_nodes=False, init_api_nodes=False))` before reading `NODE_CLASS_MAPPINGS`; keep that call if you modify `load()`.
- **`memory allocation failed with OOM` during decode** — can happen on the T4 default (16 GB) at 1024px; ComfyUI falls back and still saves the image. Override to `MODAL_GPU=L4` (24 GB) for headroom and no warning.
- **Edit output size ignores `--width/--height`** — expected in edit mode: the output follows the first reference's aspect ratio, resized to the working resolution (`max(width, height)`). Text-to-image uses `--width/--height` exactly.
- **First run slow / model download** — models live in the Modal Volume `qwen21-comfy-cache`; the first build downloads ~14 GB. Subsequent runs are fast.
- **License** — Qwen weights are under the Qwen Research License (research/evaluation; non-commercial).

"""Generic ComfyUI workflow runner on Modal — NATIVE execution.

The workflow is executed by ComfyUI itself: this script launches the ComfyUI
server on the Modal GPU container and runs the workflow through the official
`/prompt` API (the same path the UI uses), then fetches the output images via
`/view`. No re-implemented executor — scheduling, lazy eval, typing and node
plumbing are 100 % stock ComfyUI behavior.

  modal run modal_workflow_direct.py --workflow workflow.json --prompt "..." --out out.png

Accepted workflow formats (auto-detected):
  - API format  ("Export (API)" in ComfyUI: {id: {class_type, inputs}}) — posted as-is.
  - UI format   (canvas save: {nodes, links}), including subgraph-wrapped
    templates (definitions.subgraphs). Translated to API format first
    (subgraphs flattened, widget values mapped via live INPUT_TYPES) —
    the same translation the frontend performs on export.

--prompt is injected into the positive text encoder (auto-traced through the
sampler's `positive` input; override with --prompt-node). Seed priority:
explicit --seed > --seed 0 / --random-seed (random) > workflow's own seed
(literal, or its randomize-control) > template fallback (random, logged).
Everything else comes from the workflow file; best-effort overrides exist
(--steps, --cfg, --sampler, --scheduler, --width, --height, --unet, --clip,
--vae, --ckpt, --negative).

Model files are lazy-downloaded on first use from a small public-file
registry (Qwen-Image 2.1 + Z-Image-Turbo sets) into the Modal Volume
`workflow-comfy-cache`. Unknown files fail fast with the exact filename so
you can extend MODEL_FILES.

GPU: T4 (16 GB, $0.59/hr) by default — cheapest hourly rate.
Override with MODAL_GPU=L4 (24 GB headroom) / MODAL_GPU=B200.

Run:
  modal setup
  modal secret create huggingface-secret HF_TOKEN=hf_...   # optional, raises rate limits
  modal run modal_workflow_direct.py --workflow image_z_image_turbo_int8.json --prompt "astronaut in neon Tokyo alley"
  modal run modal_workflow_direct.py --workflow qwen21_workflow_api.json --prompt "a red fox in a snowy forest" --random-seed
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import urllib.parse
from pathlib import Path

import modal

APP_NAME = "workflow-direct"
VOL_NAME = "workflow-comfy-cache"
COMFY_DIR = "/root/comfy/ComfyUI"
COMFY_PORT = 8188
DEFAULT_SCALEDOWN_WINDOW = 200

# filename -> (hf repo, path inside repo, path under ComfyUI/models).
# Covers the Qwen-Image 2.1 and Z-Image-Turbo sets validated in this repo.
MODEL_FILES = {
    "qwen_image_2.1_int8_convrot.safetensors": (
        "Comfy-Org/Qwen-Image-2.1",
        "diffusion_models/qwen_image_2.1_int8_convrot.safetensors",
        "diffusion_models/qwen_image_2.1_int8_convrot.safetensors"),
    "qwen_image_2.1_nvfp4.safetensors": (
        "pottokao/Qwen-Image-2.1-DiT-NVFP4-ComfyUI",
        "qwen_image_2.1_nvfp4.safetensors",
        "diffusion_models/qwen_image_2.1_nvfp4.safetensors"),
    "qwen3vl_8b_w4a8.safetensors": (
        "Comfy-Org/Qwen-Image-2.1",
        "text_encoders/qwen3vl_8b_w4a8.safetensors",
        "text_encoders/qwen3vl_8b_w4a8.safetensors"),
    "qwen_image_2.1_vae_bf16.safetensors": (
        "Comfy-Org/Qwen-Image-2.1",
        "vae/qwen_image_2.1_vae_bf16.safetensors",
        "vae/qwen_image_2.1_vae_bf16.safetensors"),
    "z_image_turbo_int8_convrot.safetensors": (
        "Comfy-Org/z_image_turbo",
        "split_files/diffusion_models/z_image_turbo_int8_convrot.safetensors",
        "diffusion_models/z_image_turbo_int8_convrot.safetensors"),
    "qwen_3_4b_fp8_mixed.safetensors": (
        "Comfy-Org/z_image_turbo",
        "split_files/text_encoders/qwen_3_4b_fp8_mixed.safetensors",
        "text_encoders/qwen_3_4b_fp8_mixed.safetensors"),
    "qwen_3_4b_fp4_mixed.safetensors": (
        "Comfy-Org/z_image_turbo",
        "split_files/text_encoders/qwen_3_4b_fp4_mixed.safetensors",
        "text_encoders/qwen_3_4b_fp4_mixed.safetensors"),
    "ae.safetensors": (
        "Comfy-Org/z_image_turbo",
        "split_files/vae/ae.safetensors",
        "vae/ae.safetensors"),
}

MODEL_EXTS = (".safetensors", ".ckpt", ".pt", ".pth", ".bin")
UI_ONLY_INPUTS = {"control_after_generate", "control_before_generate"}
SEED_CONTROLS = ("fixed", "increment", "decrement", "randomize")
LINK_TYPES = {  # input types that are always graph links, never widgets
    "MODEL", "CLIP", "VAE", "LATENT", "CONDITIONING", "IMAGE", "MASK",
    "CONTROL_NET", "STYLE_MODEL", "CLIP_VISION", "CLIP_VISION_OUTPUT",
    "NOISE", "GUIDER", "SAMPLER", "SIGMAS", "LATENT_OPERATION",
}
# subgraph-exposed input name -> CLI flag that fills it (for error hints).
FILL_FLAGS = {"text": "--prompt", "prompt": "--prompt", "width": "--width",
              "height": "--height", "seed": "--seed", "steps": "--steps",
              "cfg": "--cfg", "sampler": "--sampler", "sampler_name": "--sampler",
              "scheduler": "--scheduler", "unet_name": "--unet", "clip_name": "--clip",
              "vae_name": "--vae", "ckpt_name": "--ckpt", "denoise": "--denoise",
              "batch_size": "--batch-size", "shift": "--shift"}


def _is_api_format(doc) -> bool:
    return isinstance(doc, dict) and "nodes" not in doc and all(
        isinstance(v, dict) and "class_type" in v for v in doc.values()
    )


def _widget_input_names(cls) -> list[str]:
    """Widget-type input names in declaration order (required, then optional).

    Positional widgets_values maps onto ALL widget inputs (linked or not) —
    this is what the frontend saves.
    """
    names: list[str] = []
    try:
        spec = cls.INPUT_TYPES()
    except Exception:
        return names
    for section in ("required", "optional"):
        for name, decl in (spec.get(section) or {}).items():
            if name in UI_ONLY_INPUTS:
                continue
            t = decl[0] if isinstance(decl, (list, tuple)) and decl else decl
            if isinstance(t, list) or t in ("INT", "FLOAT", "STRING", "BOOLEAN"):
                names.append(name)
            elif t not in LINK_TYPES and not isinstance(t, str):
                names.append(name)
    return names


def _ui_node_input_name(node: dict, slot: int) -> str | None:
    inputs = node.get("inputs") or []
    if 0 <= slot < len(inputs):
        entry = inputs[slot]
        return entry.get("name") or entry.get("localized_name")
    return None


def _flatten_subgraphs(ui: dict, fills: dict) -> dict:
    """Inline single-level subgraph nodes. Exposed inputs with no outer
    connection are recorded as {api_node_id: {input_name: value-or-None}} in
    `fills` (None = fill from template stale value / CLI later)."""
    defs = {s["id"]: s for s in (ui.get("definitions") or {}).get("subgraphs", [])}
    sg_node_ids = {n["id"] for n in ui["nodes"] if n.get("type") in defs}
    if not sg_node_ids:
        return ui
    max_node = max([n["id"] for n in ui["nodes"]] + [0])
    max_link = 0
    for l in ui["links"]:
        max_link = max(max_link, l[0])

    nodes = [n for n in ui["nodes"] if n["id"] not in sg_node_ids]
    # outer links are [id, origin_id, origin_slot, target_id, target_slot, type];
    # drop any touching a subgraph node (retargeted copies are added below).
    links = [l for l in ui["links"] if l[1] not in sg_node_ids and l[3] not in sg_node_ids]

    def _inner_links(sg: dict) -> dict:
        return {l["id"]: l for l in sg.get("links", [])}

    for node in ui["nodes"]:
        if node["id"] not in sg_node_ids:
            continue
        sg = defs[node["type"]]
        ilinks = _inner_links(sg)
        id_map: dict[int, int] = {}
        for inner in sg.get("nodes", []):
            if inner["id"] in (-10, -20):
                continue
            max_node += 1
            id_map[inner["id"]] = max_node
            nodes.append({**inner, "id": max_node})
        by_id = {n["id"]: n for n in nodes}

        for i, inp_def in enumerate(sg.get("inputs", [])):
            incoming = [l for l in ui["links"] if l[3] == node["id"] and l[4] == i]
            inner_targets = []
            for lid in inp_def.get("linkIds", []):
                if lid in ilinks and ilinks[lid]["origin_id"] == -10:
                    l = ilinks[lid]
                    inner_targets.append((id_map[l["target_id"]], l["target_slot"]))
            if incoming:
                for l in incoming:
                    for (t_id, t_slot) in inner_targets:
                        max_link += 1
                        links.append([max_link, l[1], l[2], t_id, t_slot, l[5]])
            else:
                name = inp_def.get("name", "")
                for (t_id, t_slot) in inner_targets:
                    iname = _ui_node_input_name(by_id[t_id], t_slot)
                    if iname:
                        fills.setdefault(str(t_id), {})[iname] = fills.get(name)
        for o, out_def in enumerate(sg.get("outputs", [])):
            outgoing = [l for l in ui["links"] if l[1] == node["id"] and l[2] == o]
            for l in outgoing:
                for lid in out_def.get("linkIds", []):
                    if lid in ilinks and ilinks[lid]["target_id"] == -20:
                        src = ilinks[lid]
                        max_link += 1
                        links.append([max_link, id_map[src["origin_id"]], src["origin_slot"],
                                      l[3], l[4], l[5]])
        for l in sg.get("links", []):
            if l["origin_id"] in (-10, -20) or l["target_id"] in (-10, -20):
                continue
            max_link += 1
            links.append([max_link, id_map[l["origin_id"]], l["origin_slot"],
                          id_map[l["target_id"]], l["target_slot"], l["type"]])
    return {"nodes": nodes, "links": links}


def _ui_to_api(ui: dict, node_classes: dict) -> dict:
    """Classic UI graph -> API graph using live INPUT_TYPES for widget order."""
    if (ui.get("definitions") or {}).get("subgraphs"):
        raise RuntimeError(
            "nested/remaining subgraph definitions after flattening are not supported; "
            "export the workflow with ComfyUI's 'Export (API)' instead")
    by_id = {n["id"]: n for n in ui.get("nodes", [])}
    api: dict[str, dict] = {}
    for node in ui.get("nodes", []):
        ctype = node.get("type")
        if ctype == "Reroute":
            continue
        if ctype not in node_classes:
            touched = [l for l in ui.get("links", []) if l[1] == node["id"] or l[3] == node["id"]]
            if touched:
                raise RuntimeError(
                    f"node type '{ctype}' (id {node.get('id')}) is wired into the graph "
                    f"but not installed in this image")
            print(f"skipping UI-only node {node.get('id')} ({ctype})", flush=True)
            continue
        cls = node_classes[ctype]
        inputs: dict = {}
        names = _widget_input_names(cls)
        vals = node.get("widgets_values") or []
        vi = 0
        for ni, name in enumerate(names):
            if vi >= len(vals):
                break
            inputs[name] = vals[vi]
            vi += 1
            # The frontend injects a seed-control widget right after each seed
            # widget; it is not part of INPUT_TYPES. Skip it only on surplus.
            if name in ("seed", "noise_seed") and (len(vals) - vi) > (len(names) - ni - 1) \
                    and isinstance(vals[vi], str) and vals[vi] in SEED_CONTROLS:
                vi += 1
        api[str(node["id"])] = {"class_type": ctype, "inputs": inputs,
                                "_title": node.get("title", "")}
        for v in vals:
            if isinstance(v, str) and v in SEED_CONTROLS:
                api[str(node["id"])]["_seed_control"] = v
                break
    # reroute bypass map: reroute_id -> (origin_id, origin_slot)
    reroutes: dict[int, tuple] = {}
    for node in ui.get("nodes", []):
        if node.get("type") == "Reroute":
            outs = [l for l in ui.get("links", []) if l[1] == node["id"]]
            ins = [l for l in ui.get("links", []) if l[3] == node["id"]]
            if outs and ins:
                reroutes[node["id"]] = (outs[0][1], outs[0][2])
    for l in ui.get("links", []):
        _lid, o_id, o_slot, t_id, t_slot, _lt = l
        while o_id in reroutes:
            o_id, o_slot = reroutes[o_id]
        if str(t_id) not in api or str(o_id) not in api:
            continue
        iname = _ui_node_input_name(by_id[t_id], t_slot)
        if iname and iname not in UI_ONLY_INPUTS:
            prev = api[str(t_id)]["inputs"].get(iname, None)
            if prev is not None and not isinstance(prev, list):
                # linked widget's last UI value = template default fallback.
                api[str(t_id)].setdefault("_stale", {})[iname] = prev
            api[str(t_id)]["inputs"][iname] = [str(o_id), o_slot]
    return api


def _ensure_files(api: dict):
    from huggingface_hub import hf_hub_download

    token = os.environ.get("HF_TOKEN")
    wanted: dict[str, None] = {}
    for node in api.values():
        for key, val in node.get("inputs", {}).items():
            if isinstance(val, str) and val.lower().endswith(MODEL_EXTS) and Path(val).name == val:
                wanted[Path(val).name] = None
    for fname in wanted:
        if fname not in MODEL_FILES:
            raise RuntimeError(
                f"model file '{fname}' is referenced by the workflow but not in MODEL_FILES; "
                f"add 'repo_id / repo_path / models_rel' mapping and re-run")
        repo, remote, rel = MODEL_FILES[fname]
        target = Path(f"{COMFY_DIR}/models/{rel}")
        if not target.exists():
            local = hf_hub_download(repo_id=repo, filename=remote, cache_dir="/cache", token=token)
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists() or target.is_symlink():
                target.unlink()
            target.symlink_to(local)
            print(f"downloaded+linked {rel}", flush=True)
        else:
            print(f"model present: {rel}", flush=True)


def _find_sampler(api: dict) -> str:
    for nid in sorted(api, key=int):
        n = api[nid]
        if n["class_type"] == "KSampler" or ("positive" in n["inputs"] and "negative" in n["inputs"]):
            return nid
    raise RuntimeError("no sampler node (with positive/negative inputs) found in workflow")


def _trace_encoder(api: dict, sampler_id: str, key: str) -> str | None:
    ref = api[sampler_id]["inputs"].get(key)
    if isinstance(ref, list):
        return str(ref[0])
    return None


def _resolve_seed(api: dict, sampler_id: str, args: dict) -> int:
    """Decide the sampler seed and store it on the API graph.

    Priority: explicit --seed (nonzero) > --seed 0 / --random-seed (random) >
    workflow seed literal > stale template value honoring its control
    (fixed => literal, randomize => random) > random fallback (logged).
    """
    import random
    import time

    def _random(reason: str) -> int:
        random.seed(int(time.time()))
        val = random.randint(0, 18446744073709551615)
        print(f"seed: randomized ({reason}): {val}", flush=True)
        return val

    s_inputs = api[sampler_id]["inputs"]
    if args.get("seed") is not None and args["seed"] != 0:
        s_inputs["seed"] = args["seed"]
        print(f"seed: explicit CLI --seed {args['seed']}", flush=True)
        return args["seed"]
    if args.get("seed") == 0 or args.get("random_seed"):
        s_inputs["seed"] = _random("CLI requested random")
        return s_inputs["seed"]
    lit = s_inputs.get("seed")
    if isinstance(lit, bool) or not isinstance(lit, int):
        lit = api[sampler_id].get("_stale", {}).get("seed")
        if isinstance(lit, bool) or not isinstance(lit, int):
            lit = None
    control = api[sampler_id].get("_seed_control")
    if lit is not None and control in (None, "fixed"):
        s_inputs["seed"] = lit
        print(f"seed: workflow literal {lit}", flush=True)
        return lit
    reason = f"workflow control is '{control}'" if control else "workflow has no seed value"
    s_inputs["seed"] = _random(reason)
    return s_inputs["seed"]


def _apply_overrides(api: dict, args: dict):
    """Prompt + best-effort CLI overrides as plain API-dict edits."""
    sampler_id = _find_sampler(api)
    s_inputs = api[sampler_id]["inputs"]

    pos_id = args.get("prompt_node") or _trace_encoder(api, sampler_id, "positive")
    if pos_id is not None and args.get("prompt"):
        if pos_id not in api:
            matches = [nid for nid, n in api.items()
                       if args["prompt_node"].lower() in (n.get("_title") or "").lower()]
            if not matches:
                raise RuntimeError(f"--prompt-node '{args['prompt_node']}' matches no node id/title")
            pos_id = matches[0]
        enc = api[pos_id]["inputs"]
        for key in ("text", "prompt", "positive_prompt", "positive"):
            if key in enc and not isinstance(enc[key], list):
                enc[key] = args["prompt"]
                print(f"prompt injected into node {pos_id} ({api[pos_id]['class_type']}.{key})", flush=True)
                break
        else:
            raise RuntimeError(
                f"node {pos_id} ({api[pos_id]['class_type']}) has no text input; "
                f"inputs: {sorted(enc)}")
    if args.get("negative") is not None:
        neg_id = _trace_encoder(api, sampler_id, "negative")
        if neg_id is not None and neg_id in api:
            enc = api[neg_id]["inputs"]
            for key in ("text", "prompt", "negative_prompt", "negative"):
                if key in enc and not isinstance(enc[key], list):
                    enc[key] = args["negative"]
                    print(f"negative injected into node {neg_id}", flush=True)
                    break
            else:
                print(f"--negative ignored: node {neg_id} ({api[neg_id]['class_type']}) "
                      f"takes no text (e.g. ConditioningZeroOut)", flush=True)
    for flag in ("steps", "cfg", "sampler_name", "scheduler", "denoise"):
        if args.get(flag) is not None and flag in s_inputs and not isinstance(s_inputs[flag], list):
            s_inputs[flag] = args[flag]
    if args.get("sampler") is not None and "sampler_name" in s_inputs \
            and not isinstance(s_inputs["sampler_name"], list):
        s_inputs["sampler_name"] = args["sampler"]
    if args.get("width") is not None or args.get("height") is not None:
        lat_ref = s_inputs.get("latent_image")
        if isinstance(lat_ref, list) and str(lat_ref[0]) in api:
            lat = api[str(lat_ref[0])]
            if lat["class_type"] in ("EmptyLatentImage", "EmptySD3LatentImage") \
                    and "width" in lat["inputs"] and "height" in lat["inputs"]:
                if args.get("width") is not None:
                    lat["inputs"]["width"] = args["width"]
                if args.get("height") is not None:
                    lat["inputs"]["height"] = args["height"]
                print(f"size override on node {lat_ref[0]}: "
                      f"{lat['inputs']['width']}x{lat['inputs']['height']}", flush=True)
            else:
                print("--width/--height ignored: sampler latent is not an empty-latent node", flush=True)
    for flag, keys in (("unet", ("unet_name",)), ("clip", ("clip_name",)),
                       ("vae", ("vae_name",)), ("ckpt", ("ckpt_name",))):
        if args.get(flag) is None:
            continue
        for nid, n in api.items():
            for key in keys:
                if key in n["inputs"] and not isinstance(n["inputs"][key], list):
                    print(f"model override node {nid}.{key}: {n['inputs'][key]} -> {args[flag]}", flush=True)
                    n["inputs"][key] = args[flag]


def _fill_template_defaults(api: dict, fills: dict):
    """Subgraph-exposed inputs with no outer connection and no CLI value fall
    back to the template's own stale widget value; else fail with the flag."""
    for nid, kv in fills.items():
        if not isinstance(kv, dict) or nid not in api:
            continue
        stale = api[nid].pop("_stale", {})
        for key, val in kv.items():
            if val is not None:
                api[nid]["inputs"][key] = val
            elif key in stale:
                api[nid]["inputs"][key] = stale[key]
                print(f"template default for node {nid}.{key}: {stale[key]}", flush=True)
            elif key not in api[nid]["inputs"]:
                hint = FILL_FLAGS.get(key, "")
                raise RuntimeError(
                    f"workflow needs '{key}' on node {nid} ({api[nid]['class_type']}) "
                    f"and the template supplies no value; pass {hint} to provide it")


def _strip_helpers(api: dict):
    for n in api.values():
        for key in ("_title", "_seed_control", "_stale"):
            n.pop(key, None)


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


def _submit_and_wait(base_url: str, workflow: dict, timeout: int = 600) -> list[bytes]:
    """Official ComfyUI REST API: POST /prompt, poll /history, GET /view."""
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
            if h[prompt_id]["status"].get("status_str") == "error":
                raise RuntimeError(f"ComfyUI job {prompt_id} errored: {h[prompt_id]['status']}")
            outputs_meta = h[prompt_id]["outputs"]
            break
        time.sleep(2)
    if outputs_meta is None:
        raise TimeoutError(f"ComfyUI job {prompt_id} did not finish in {timeout}s")

    blobs: list[bytes] = []
    for node_out in outputs_meta.values():
        for img in node_out.get("images", []):
            qs = urllib.parse.urlencode(
                {"filename": img["filename"], "subfolder": img.get("subfolder", ""),
                 "type": img.get("type", "output")})
            vr = requests.get(f"{base_url}/view?{qs}", timeout=120)
            vr.raise_for_status()
            blobs.append(vr.content)
    if not blobs:
        raise RuntimeError(f"No images in ComfyUI outputs: {list(outputs_meta.keys())}")
    return blobs


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
    .pip_install("comfy-cli", "huggingface_hub[hf_transfer]", "pillow", "numpy", "requests")
    .run_commands("comfy --skip-prompt install --nvidia")
    # Force latest master in case comfy-cli pinned an older stable.
    .run_commands("cd /root/comfy/ComfyUI && git fetch origin && git reset --hard origin/master")
    .run_commands("pip install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "HF_XET_HIGH_PERFORMANCE": "1"})
)

app = modal.App(APP_NAME, image=image)


@app.cls(
    gpu=os.environ.get("MODAL_GPU", "T4"),  # default T4; override: MODAL_GPU=L4 modal run ...
    volumes={"/cache": vol},
    scaledown_window=DEFAULT_SCALEDOWN_WINDOW,
    max_containers=1,
    timeout=900,
)
@modal.concurrent(max_inputs=1)
class WorkflowDirect:
    @modal.enter()
    def launch(self):
        self.proc = subprocess.Popen(
            f"python {COMFY_DIR}/main.py --listen 127.0.0.1 "
            f"--port {COMFY_PORT} --disable-auto-launch",
            shell=True,
        )
        _wait_for_port(COMFY_PORT, timeout=300)
        print("ComfyUI server ready", flush=True)

    @modal.exit()
    def stop(self):
        proc = getattr(self, "proc", None)
        if proc is not None:
            try:
                proc.terminate()
            except (ProcessLookupError, OSError):
                pass

    @modal.method()
    def generate(self, workflow_text: str, args: dict) -> list[bytes]:
        import asyncio

        import torch

        sys.path.insert(0, COMFY_DIR)
        import nodes as comfy_nodes

        asyncio.run(comfy_nodes.init_extra_nodes(init_custom_nodes=False, init_api_nodes=False))
        from nodes import NODE_CLASS_MAPPINGS

        node_classes = dict(NODE_CLASS_MAPPINGS)
        print(f"node registry: {len(node_classes)} nodes", flush=True)
        print(f"torch {torch.__version__}, cuda={torch.cuda.is_available()}", flush=True)
        if torch.cuda.is_available():
            print(f"gpu: {torch.cuda.get_device_name(0)}, sm={torch.cuda.get_device_capability(0)}", flush=True)

        doc = json.loads(workflow_text)
        if _is_api_format(doc):
            api = {str(k): {"class_type": v["class_type"], "inputs": dict(v.get("inputs", {}))}
                   for k, v in doc.items()}
            print(f"workflow format: API ({len(api)} nodes)", flush=True)
        else:
            if "nodes" not in doc or "links" not in doc:
                raise RuntimeError("unrecognized workflow JSON: need API format or UI format (nodes+links)")
            print(f"workflow format: UI ({len(doc['nodes'])} nodes) — translating to API", flush=True)
            fills = {"text": args.get("prompt"), "prompt": args.get("prompt")}
            for key in ("width", "height", "seed", "steps", "cfg", "sampler", "sampler_name",
                        "scheduler", "denoise", "batch_size", "unet_name", "clip_name",
                        "vae_name", "ckpt_name"):
                if args.get(key) is not None and args.get(key) != 0:
                    fills[key] = args[key]
            ui = _flatten_subgraphs(doc, fills)
            api = _ui_to_api(ui, node_classes)
            _fill_template_defaults(api, fills)
            print(f"translated to API: {len(api)} nodes", flush=True)

        _resolve_seed(api, _find_sampler(api), args)
        _apply_overrides(api, args)
        _ensure_files(api)
        _strip_helpers(api)
        t0 = time.time()
        blobs = _submit_and_wait(f"http://127.0.0.1:{COMFY_PORT}", api)
        print(f"native execution (queue+inference+fetch) took {time.time()-t0:.1f}s", flush=True)
        return blobs


@app.local_entrypoint()
def main(
    workflow: str = "",
    prompt: str = "",
    prompt_node: str = "",
    negative: str | None = None,
    seed: int | None = None,  # None = use workflow's seed; 0 = random
    random_seed: bool = False,  # randomize regardless of workflow
    steps: int | None = None,
    cfg: float | None = None,
    sampler: str | None = None,
    scheduler: str | None = None,
    width: int | None = None,
    height: int | None = None,
    unet: str | None = None,
    clip: str | None = None,
    vae: str | None = None,
    ckpt: str | None = None,
    scaledown_window: int = DEFAULT_SCALEDOWN_WINDOW,
    out: str = "workflow_direct_out.png",
):
    if not workflow:
        raise ValueError("--workflow workflow.json is required")
    if not prompt:
        raise ValueError("--prompt \"...\" is required")
    if scaledown_window < 1:
        raise ValueError("--scaledown-window must be at least 1 second")
    workflow_text = Path(workflow).read_text()
    args = {"prompt": prompt, "prompt_node": prompt_node or None, "negative": negative,
            "seed": seed, "random_seed": random_seed, "steps": steps, "cfg": cfg,
            "sampler": sampler, "scheduler": scheduler, "width": width, "height": height,
            "unet": unet, "clip": clip, "vae": vae, "ckpt": ckpt}
    cls = WorkflowDirect
    if scaledown_window != DEFAULT_SCALEDOWN_WINDOW:
        cls = cls.with_options(scaledown_window=scaledown_window)
        print(f"scaledown_window: {scaledown_window}s idle before scale-down")
    pngs: list[bytes] = cls().generate.remote(workflow_text, args)
    base = Path(out)
    for i, png in enumerate(pngs):
        p = base if len(pngs) == 1 else base.with_name(f"{base.stem}_{i}{base.suffix}")
        p.write_bytes(png)
        print(f"saved {p} ({len(png)/1e6:.2f} MB)")

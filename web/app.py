"""Qwen-Image 2.1 web UI — FastAPI + Jinja.

Own UI -> this backend -> Modal GPU (no `modal serve` UI needed).

Flow (async, avoids HTTP timeouts on 1-3 min GPU jobs):
  GET  /                -> form (prompt textbox + reference image upload)
  POST /generate        -> spawn Modal job, redirect to /result/{call_id}
  GET  /result/{call_id}-> poll Modal; 202-style pending page auto-refreshes
  GET  /image/{call_id} -> final PNG bytes

Backend -> Modal uses `modal.Cls.from_name(...)` with `.spawn()` /
`FunctionCall.from_id().get(timeout=0)` against three deployed apps:

  generate / edit -> `Qwen21Direct` on `qwen21-4bit-direct`
  headswap       -> `BFSHeadSwapDirect` on `bfs-headswap-direct`
  turbo          -> `ZImageTurboDirect` on `zimage-turbo-direct` (text-to-image only)

(Deploy once each — `Cls.from_name` needs a deployed app. Overrides:
MODAL_APP_NAME / MODAL_CLS_NAME for the Qwen path,
BFS_MODAL_APP_NAME / BFS_MODAL_CLS_NAME for the head-swap path,
TURBO_MODAL_APP_NAME / TURBO_MODAL_CLS_NAME for the Turbo path.)

Modes (dropdown on the form, `mode` form field):
  generate: text-to-image (Qwen-Image 2.1 4-bit), no references needed.
  edit:     Qwen edit with reference photos (up to 4, output follows the first).
  headswap: BFS LoRA — exactly 2 images, body/target first, reference head second.
  turbo:    Z-Image-Turbo, 8-step text-to-image (no refs, negative ignored).

Character slots: portraits saved under web/characters/ appear in the
Character dropdown; the slot portrait auto-attaches as an identity
reference and its anchor text injects into the prompt. In headswap mode
the slot portrait becomes the head, so one body upload suffices.

Run:
  pip install -r web/requirements.txt
  uvicorn web.app:app --reload --port 8000
  # mock mode (no Modal spend, placeholder image):
  MOCK_MODAL=1 uvicorn web.app:app --reload --port 8000
"""
from __future__ import annotations

import base64
import io
import json
import logging
import os
import re
import time
import uuid
from pathlib import Path

from fastapi import FastAPI, Form, Request, UploadFile
from fastapi.responses import HTMLResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.concurrency import run_in_threadpool
from starlette.datastructures import UploadFile as StarletteUploadFile

logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
)
log = logging.getLogger("qwen21-web")

BASE_DIR = Path(__file__).resolve().parent
TEMPLATES_DIR = BASE_DIR / "templates"

MODAL_APP_NAME = os.environ.get("MODAL_APP_NAME", "qwen21-4bit-direct")
MODAL_CLS_NAME = os.environ.get("MODAL_CLS_NAME", "Qwen21Direct")
BFS_MODAL_APP_NAME = os.environ.get("BFS_MODAL_APP_NAME", "bfs-headswap-direct")
BFS_MODAL_CLS_NAME = os.environ.get("BFS_MODAL_CLS_NAME", "BFSHeadSwapDirect")
TURBO_MODAL_APP_NAME = os.environ.get("TURBO_MODAL_APP_NAME", "zimage-turbo-direct")
TURBO_MODAL_CLS_NAME = os.environ.get("TURBO_MODAL_CLS_NAME", "ZImageTurboDirect")
# Official Z-Image-Turbo int8 template defaults (see modal_zimage_turbo_direct.py).
TURBO_DEFAULTS = dict(steps=8, cfg=1.0, sampler="res_multistep", scheduler="simple",
                      shift=3.0, unet="z_image_turbo_int8_convrot.safetensors",
                      clip="qwen_3_4b_fp8_mixed.safetensors")
MODAL_CLIP = "qwen3vl_8b_w4a8.safetensors"
MOCK_MODAL = os.environ.get("MOCK_MODAL", "") == "1"

MODES = ("generate", "edit", "headswap", "turbo")
# Form default; BFS_HEADSWAP=1 keeps the earlier single-purpose toggle working
# by preselecting headswap.
DEFAULT_MODE = os.environ.get("DEFAULT_MODE", "headswap" if os.environ.get("BFS_HEADSWAP", "") == "1" else "generate")
if DEFAULT_MODE not in MODES:
    DEFAULT_MODE = "generate"

HEADSWAP_DEFAULT_PROMPT = (
    "head_swap: start with <image1> as the base image, keeping its lighting, "
    "environment, and background. remove the head from <image1> completely and "
    "replace it with the head from <image2>, strictly preserving the hair, eye "
    "color, nose structure from <image2>. copy the direction of the eye, head "
    "rotation, micro expressions from <image1>, high quality, sharp details, 4k"
)

# Per-mode prompt defaults (form prefill + JS swap on mode change).
MODE_DEFAULT_PROMPTS = {
    "generate": "cinematic portrait of an astronaut in a neon Tokyo alley, rain reflections, ultra detailed",
    "edit": "change the jacket to bright red; keep the face, pose and background unchanged",
    "headswap": HEADSWAP_DEFAULT_PROMPT,
    "turbo": "cinematic portrait of an astronaut in a neon Tokyo alley, rain reflections, ultra detailed",
}

MAX_REFS = 4
MAX_FILE_MB = 10

# call_id -> job dict (in-memory; single-process dev server)
JOBS: dict[str, dict] = {}
MAX_JOBS_KEPT = 50  # bound memory: drop oldest finished jobs beyond this


def _prune_jobs() -> None:
    done = [(cid, j) for cid, j in JOBS.items() if j["status"] != "pending"]
    for cid, _ in sorted(done, key=lambda kv: kv[1]["created"])[:max(0, len(done) - MAX_JOBS_KEPT)]:
        del JOBS[cid]


def _load_dotenv() -> None:
    """Load KEY=VALUE pairs from .env (project root) without overriding real env."""
    env_path = Path(__file__).resolve().parent.parent / ".env"
    try:
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip("'\""))
    except FileNotFoundError:
        pass


_load_dotenv()
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
OPENROUTER_MODEL = os.environ.get("OPENROUTER_MODEL", "inclusionai/ling-3.1-flash")

ENHANCE_SYSTEM_FALLBACK = (
    "You are a prompt engineer for the Qwen-Image-2.1 text-to-image model. "
    "Expand the user's short idea into one detailed English image prompt: "
    "concrete subject, action/pose, environment texture, light direction and color, "
    "camera/lens, medium and grain, mood. No people who could be real, no text overlays. "
    'Reply with JSON only: {"rewritten_prompt": "<single detailed prompt>", '
    '"wh_ratio": "<W:H suggestion like 16:9, or empty>"}.'
)


def _enhance_system() -> str:
    """Official Qwen-Image-2.1 T2I rewriter system prompt (vendored from
    QwenLM/Qwen-Image-2.1 prompt_rewrite/prompts/system_prompt_t2i.txt),
    so Enhance follows Qwen's own rewriting contract verbatim."""
    try:
        return (BASE_DIR / "prompts" / "enhance_t2i.txt").read_text()
    except FileNotFoundError:
        log.warning("official enhance prompt missing, using fallback")
        return ENHANCE_SYSTEM_FALLBACK


ENHANCE_SYSTEM = _enhance_system() + """

ADDITIONAL CONSTRAINT (overrides the size guidance above — follow everything
else as written): keep rewritten_prompt SURGICAL, 80-120 words, one paragraph.
Include only what decides the image: subject + key action/pose, the 2-3 most
important environment details, light direction and color, camera/lens, one style
word. Drop the full frame-walk inventory, secondary objects, and closing summary.
No quality boosters. wh_ratio rule unchanged.
"""

# Saved character slots: web/characters/<name>.png|jpg + <name>.txt (identity anchor).
CHARACTERS_DIR = BASE_DIR / "characters"
CHARACTERS_DIR.mkdir(exist_ok=True)


def _valid_character_name(name: str) -> bool:
    return bool(name) and len(name) <= 32 and all(c.isalnum() or c in "-_" for c in name)


def list_characters() -> list[dict]:
    """Saved slots: [{"name":..., "identity":..., "has_image":...}], sorted."""
    out: list[dict] = []
    if not CHARACTERS_DIR.is_dir():
        return out
    for img in sorted(CHARACTERS_DIR.glob("*")):
        if img.suffix.lower() not in (".png", ".jpg", ".jpeg", ".webp") or not img.is_file():
            continue
        txt = img.with_suffix(".txt")
        out.append({
            "name": img.stem,
            "identity": txt.read_text().strip() if txt.is_file() else "",
            "has_image": True,
        })
    return out


def load_character(name: str) -> tuple[bytes, str]:
    """Return (image bytes, identity text) for a saved slot; raises ValueError."""
    for c in list_characters():
        if c["name"] == name:
            for ext in (".png", ".jpg", ".jpeg", ".webp"):
                p = CHARACTERS_DIR / f"{name}{ext}"
                if p.is_file():
                    return p.read_bytes(), c["identity"]
    raise ValueError(f"unknown character: {name!r}")

app = FastAPI(title="Qwen-Image 2.1 web UI")
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

log.info(
    "startup modal_app=%s cls=%s bfs_app=%s bfs_cls=%s turbo_app=%s turbo_cls=%s mock=%s default_mode=%s max_refs=%d max_file_mb=%d",
    MODAL_APP_NAME, MODAL_CLS_NAME, BFS_MODAL_APP_NAME, BFS_MODAL_CLS_NAME,
    TURBO_MODAL_APP_NAME, TURBO_MODAL_CLS_NAME,
    MOCK_MODAL, DEFAULT_MODE, MAX_REFS, MAX_FILE_MB,
)


def _mock_png(prompt: str, width: int = 512, height: int = 512) -> bytes:
    """Placeholder PNG so the UI can be tested without Modal/GPU spend."""
    from PIL import Image, ImageDraw

    w, h = max(256, min(width, 1024)), max(256, min(height, 1024))
    img = Image.new("RGB", (w, h), (24, 24, 32))
    d = ImageDraw.Draw(img)
    d.rectangle([8, 8, w - 8, h - 8], outline=(120, 120, 160))
    d.text((20, 20), "MOCK_MODAL=1", fill=(255, 200, 80))
    txt = (prompt or "(empty prompt)")[:120]
    y = 50
    while txt:
        d.text((20, y), txt[:60], fill=(230, 230, 230))
        txt = txt[60:]
        y += 18
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _spawn_modal(prompt: str, negative: str, width: int, height: int,
                 seed: int, steps: int, refs: list[bytes], mode: str,
                 char_image: bytes | None = None, identity: str = "") -> str:
    """Spawn a Modal job, return the FunctionCall id.

    char_image/identity come from a saved character slot: the portrait is
    appended as an identity reference and the anchor text is injected.
    In headswap mode the slot portrait serves as the head (<image2>), so a
    single body upload suffices.
    """
    t0 = time.time()
    refs = list(refs)
    if identity:
        prompt = f"{prompt.strip()}\nSame identity: {identity.strip()}"
    if mode == "headswap" and char_image is not None:
        if not refs:
            raise ValueError("head-swap with a character needs 1 body upload (the head comes from the slot)")
        refs = [refs[0], char_image]  # extra uploads ignored, logged below
    elif char_image is not None and len(refs) < MAX_REFS:
        refs.append(char_image)
    if MOCK_MODAL:
        call_id = f"mock-{uuid.uuid4().hex[:12]}"
        JOBS[call_id] = {
            "status": "pending", "png": None, "error": None,
            "prompt": prompt, "negative": negative,
            "width": width, "height": height, "seed": seed, "steps": steps,
            "mode": mode,
            "ref_count": len(refs), "created": time.time(), "mock": True,
            "ref_previews": _previews(refs),
        }
        log.info("spawn mock call_id=%s mode=%s prompt=%.60r %dx%d steps=%d seed=%d refs=%d (%.2fs)",
                 call_id, mode, prompt, width, height, steps, seed, len(refs), time.time() - t0)
        return call_id
    import modal

    if mode == "turbo":
        if refs:
            raise ValueError("turbo is text-to-image only — remove reference images or switch to Edit")
        cls = modal.Cls.from_name(TURBO_MODAL_APP_NAME, TURBO_MODAL_CLS_NAME)
        inst = cls()
        t = TURBO_DEFAULTS
        call = inst.generate.spawn(
            prompt, width, height, seed, steps,
            t["cfg"], t["sampler"], t["scheduler"], t["shift"], t["unet"], t["clip"],
        )
    elif mode == "headswap":
        if len(refs) < 2:
            raise ValueError("head-swap needs 2 images: body/target first, reference head second")
        cls = modal.Cls.from_name(BFS_MODAL_APP_NAME, BFS_MODAL_CLS_NAME)
        inst = cls()
        call = inst.generate.spawn(
            refs[0], refs[1], prompt or HEADSWAP_DEFAULT_PROMPT, negative,
            width, height, False, seed, steps,
            1.0, "euler", "simple",
            "bfs_head_v1.1_qwen_2.1.safetensors", 1.0, False, MODAL_CLIP,
        )
    else:
        cls = modal.Cls.from_name(MODAL_APP_NAME, MODAL_CLS_NAME)
        inst = cls()
        call = inst.generate.spawn(
            prompt, negative, width, height, seed, steps,
            1.0, "euler", "simple", False, MODAL_CLIP,
            ref_images=refs or None,
        )
    call_id: str = call.object_id
    _prune_jobs()
    JOBS[call_id] = {
        "status": "pending", "png": None, "error": None,
        "prompt": prompt, "negative": negative,
        "width": width, "height": height, "seed": seed, "steps": steps,
        "mode": mode,
        "ref_count": len(refs), "created": time.time(), "mock": False,
        "ref_previews": _previews(refs),
    }
    log.info("spawn modal call_id=%s mode=%s prompt=%.60r %dx%d steps=%d seed=%d refs=%d (%.2fs)",
             call_id, mode, prompt, width, height, steps, seed, len(refs), time.time() - t0)
    return call_id


def _is_image(data: bytes) -> bool:
    """True if bytes decode as an image (early 400 > late GPU failure)."""
    from PIL import Image

    try:
        with Image.open(io.BytesIO(data)) as img:
            img.load()
        return True
    except Exception:
        return False


def _openrouter_enhance(text: str, timeout: int = 90) -> dict:
    """Rewrite via OpenRouter LLM. Returns {"rewritten_prompt", "wh_ratio"}."""
    text = (text or "").strip()
    if not text:
        raise ValueError("nothing to enhance: prompt is empty")
    if MOCK_MODAL:
        return {"rewritten_prompt": f"[mock enhanced] {text} — cinematic light, "
                                    "shallow depth of field, ultra detailed",
                "wh_ratio": "16:9"}
    if not OPENROUTER_API_KEY:
        raise RuntimeError("OPENROUTER_API_KEY missing (.env)")
    from openai import OpenAI

    client = OpenAI(base_url="https://openrouter.ai/api/v1",
                    api_key=OPENROUTER_API_KEY, timeout=timeout)
    # NOTE: no response_format — the serving provider rejects structured
    # outputs. The system prompt already demands JSON-only; parse leniently.
    # max_tokens 2048: surgical rewrites are short; headroom for reasoning.
    resp = client.chat.completions.create(
        model=OPENROUTER_MODEL,
        messages=[{"role": "system", "content": ENHANCE_SYSTEM},
                  {"role": "user", "content": text}],
        temperature=0.7, max_tokens=2048,
    )
    content = (resp.choices[0].message.content or "").strip()
    if content.startswith("```"):
        content = re.sub(r"^```(?:json)?\s*", "", content)
        content = re.sub(r"\s*```$", "", content)
    try:
        result = json.loads(content)
    except json.JSONDecodeError:
        result = {"rewritten_prompt": content, "wh_ratio": ""}
    rewritten = str(result.get("rewritten_prompt", "")).strip()
    if not rewritten:
        raise ValueError("enhancer returned an empty rewrite")
    return {"rewritten_prompt": rewritten, "wh_ratio": str(result.get("wh_ratio", ""))}


def _previews(refs: list[bytes], limit: int = 4) -> list[str]:
    out: list[str] = []
    for b in refs[:limit]:
        try:
            out.append("data:image/png;base64," + base64.b64encode(b).decode())
        except Exception:
            pass
    return out


def _poll_modal(call_id: str) -> None:
    """Poll once; on success store PNG and mark done, on Timeout leave pending."""
    job = JOBS.get(call_id)
    if job is None or job["status"] != "pending":
        return
    elapsed = time.time() - job["created"]
    if job.get("mock"):
        # simulate ~5 s GPU delay so polling UI can be exercised
        if time.time() - job["created"] < 5:
            log.debug("poll %s still pending (mock, %.0fs)", call_id, elapsed)
            return
        try:
            job["png"] = _mock_png(job["prompt"], job["width"], job["height"])
            job["status"] = "done"
        except Exception as e:
            job["status"] = "error"
            job["error"] = str(e)
            log.exception("poll %s mock render failed", call_id)
            return
        log.info("poll %s done (mock, %.0fs, %d bytes)", call_id, elapsed, len(job["png"]))
        return
    try:
        import modal

        fc = modal.FunctionCall.from_id(call_id)
        png = fc.get(timeout=0)  # raises TimeoutError while running
    except TimeoutError:
        log.debug("poll %s still pending (%.0fs)", call_id, elapsed)
        return
    except Exception as e:
        job["status"] = "error"
        job["error"] = f"{type(e).__name__}: {e}"
        log.exception("poll %s failed after %.0fs", call_id, elapsed)
        return
    job["png"] = bytes(png)
    job["status"] = "done"
    log.info("poll %s done (%.0fs, %d bytes)", call_id, elapsed, len(job["png"]))


@app.get("/", response_class=HTMLResponse)
def index(request: Request, mode: str = "", char_error: str = ""):
    mode = mode if mode in MODES else DEFAULT_MODE
    return templates.TemplateResponse(
        request, "index.html",
        {"mode": mode, "modes": MODES, "mode_prompts": MODE_DEFAULT_PROMPTS,
         "default_prompt": MODE_DEFAULT_PROMPTS[mode],
         "characters": list_characters(), "character": "",
         "error": char_error or None},
    )


@app.get("/health")
def health():
    return {"status": "ok", "modal_app": MODAL_APP_NAME,
            "bfs_modal_app": BFS_MODAL_APP_NAME,
            "turbo_modal_app": TURBO_MODAL_APP_NAME, "mock": MOCK_MODAL,
            "openrouter": bool(OPENROUTER_API_KEY), "openrouter_model": OPENROUTER_MODEL,
            "modes": list(MODES), "default_mode": DEFAULT_MODE,
            "characters": len(list_characters())}


@app.post("/characters")
async def save_character(
    request: Request,
    name: str = Form(""),
    identity: str = Form(""),
):
    """Save a portrait + identity anchor as a reusable character slot."""
    client = request.client.host if request.client else "?"
    name = (name or "").strip().lower().replace(" ", "-")
    name = "".join(c for c in name if c.isalnum() or c in "-_")
    identity = (identity or "").strip()

    def _fail(msg: str):
        from urllib.parse import quote_plus
        log.warning("POST /characters from %s rejected: %s (%r)", client, msg, name)
        return RedirectResponse(url=f"/?mode=generate&char_error={quote_plus(msg)}",
                                status_code=303)

    if not _valid_character_name(name):
        return _fail("slot name needs 1-32 letters/numbers/dashes")
    form = await request.form()
    upload = next((v for k, v in form.multi_items()
                   if k == "portrait" and isinstance(v, StarletteUploadFile) and v.filename), None)
    if upload is None:
        return _fail("no portrait uploaded")
    data = await upload.read()
    if not data:
        return _fail("portrait file is empty")
    if len(data) > MAX_FILE_MB * 1024 * 1024:
        return _fail(f"portrait exceeds {MAX_FILE_MB} MB")
    if not _is_image(data):
        return _fail("portrait is not a valid image file")
    from PIL import Image

    img = Image.open(io.BytesIO(data))
    ext = ".png" if (img.mode in ("RGBA", "LA") or "transparency" in img.info) else ".jpg"
    for old in CHARACTERS_DIR.glob(f"{name}.*"):
        old.unlink()
    (CHARACTERS_DIR / f"{name}{ext}").write_bytes(data)
    (CHARACTERS_DIR / f"{name}.txt").write_text(identity + "\n" if identity else "")
    log.info("POST /characters from %s saved slot %r (%d KB)", client, name, len(data) // 1024)
    return RedirectResponse(url=f"/?mode=generate", status_code=303)


@app.post("/enhance")
async def enhance(request: Request):
    """OpenRouter rewrite: {text} -> {rewritten_prompt, wh_ratio} (no GPU)."""
    try:
        body = await request.json()
        text = body.get("text") or ""
    except Exception:
        return Response('{"error": "invalid JSON"}', status_code=400, media_type="application/json")
    try:
        return await run_in_threadpool(_openrouter_enhance, text)
    except ValueError as e:
        return Response(f'{{"error": "{e}"}}', status_code=400, media_type="application/json")
    except Exception as e:
        log.exception("POST /enhance failed")
        return Response(f'{{"error": "enhance failed: {type(e).__name__}: {e}"}}',
                        status_code=502, media_type="application/json")


@app.post("/characters/delete")
async def delete_character(request: Request, name: str = Form("")):
    """Delete a saved character slot (portrait + anchor)."""
    client = request.client.host if request.client else "?"
    name = (name or "").strip().lower()
    if not _valid_character_name(name):
        return RedirectResponse(url="/", status_code=303)
    removed = [p.name for p in CHARACTERS_DIR.glob(f"{name}.*") if p.is_file()]
    for p in CHARACTERS_DIR.glob(f"{name}.*"):
        if p.is_file():
            p.unlink()
    log.info("POST /characters/delete from %s removed slot %r (%s)",
             client, name, ", ".join(removed) or "was already gone")
    return RedirectResponse(url="/", status_code=303)


@app.post("/generate")
async def generate(
    request: Request,
    mode: str = Form("generate"),
    flow: str = Form("generate"),
    character: str = Form(""),
    prompt: str = Form(""),
    negative_prompt: str = Form(""),
    width: int = Form(1024),
    height: int = Form(1024),
    steps: int = Form(25),
    seed: int = Form(0),
    batch: int = Form(1),
):
    def _form_ctx(error: str, status: int):
        return templates.TemplateResponse(
            request, "index.html",
            {"error": error, "mode": mode if mode in MODES else DEFAULT_MODE,
             "modes": MODES, "mode_prompts": MODE_DEFAULT_PROMPTS,
             "default_prompt": prompt or MODE_DEFAULT_PROMPTS.get(mode, ""),
             "characters": list_characters(), "character": character},
            status_code=status,
        )

    if mode not in MODES:
        log.warning("POST /generate rejected: unknown mode %r", mode)
        mode = DEFAULT_MODE
    prompt = (prompt or "").strip()
    client = request.client.host if request.client else "?"
    if mode == "headswap" and not prompt:
        prompt = HEADSWAP_DEFAULT_PROMPT
        log.info("POST /generate from %s: empty prompt, using head-swap default", client)
    if not prompt:
        log.warning("POST /generate from %s rejected: empty prompt", client)
        return _form_ctx("Prompt is required.", 400)
    char_image: bytes | None = None
    identity = ""
    if character:
        try:
            char_image, identity = load_character(character)
        except ValueError as e:
            log.warning("POST /generate from %s rejected: %s", client, e)
            return _form_ctx(str(e), 400)
    width = max(256, min(width, 2048))
    height = max(256, min(height, 2048))
    steps = max(1, min(steps, 100))
    seed = max(0, seed)  # Modal treats 0 as random; negative seeds crash the sampler

    # NOTE: parse the multipart form manually instead of
    # `ref_images: list[UploadFile] = File(...)`. Starlette parses a part with
    # an empty filename (filename="") as a plain str field, which fails
    # list[UploadFile] validation and 422s the WHOLE request — even when valid
    # files accompany it. Filtering here keeps one stray part from nuking the
    # submission. NB: form values are starlette UploadFiles; fastapi's
    # UploadFile is a *subclass*, so isinstance must target the starlette base.
    form = await request.form()
    uploads: list[UploadFile] = [
        v for k, v in form.multi_items()
        if k == "ref_images" and isinstance(v, StarletteUploadFile) and v.filename
    ]

    refs: list[bytes] = []
    ref_names: list[str] = []
    for f in uploads:
        if len(refs) >= MAX_REFS:
            log.warning("POST /generate from %s: too many refs, keeping first %d", client, MAX_REFS)
            break
        if f.size is not None and f.size > MAX_FILE_MB * 1024 * 1024:
            log.warning("POST /generate from %s rejected: %s %.1f MB over limit",
                        client, f.filename, f.size / 1048576)
            return _form_ctx(f"{f.filename}: exceeds {MAX_FILE_MB} MB limit.", 400)
        data = await f.read()
        if not data:
            log.warning("POST /generate from %s: skipping empty file %r", client, f.filename)
            continue
        if len(data) > MAX_FILE_MB * 1024 * 1024:
            log.warning("POST /generate from %s rejected: %s %.1f MB over limit",
                        client, f.filename, len(data) / 1048576)
            return _form_ctx(f"{f.filename}: exceeds {MAX_FILE_MB} MB limit.", 400)
        if not _is_image(data):
            log.warning("POST /generate from %s rejected: %r is not a decodable image",
                        client, f.filename)
            return _form_ctx(f"{f.filename}: not a valid image file.", 400)
        refs.append(data)
        ref_names.append(f"{f.filename} ({len(data) // 1024} KB)")
    log.info("POST /generate from %s mode=%s char=%s prompt=%.80r %dx%d steps=%d seed=%d refs=[%s]",
             client, mode, character or "none", prompt, width, height, steps, seed,
             ", ".join(ref_names) or "none")

    if mode == "turbo" and (refs or char_image is not None):
        log.warning("POST /generate from %s rejected: turbo takes no reference images", client)
        return _form_ctx(
            "Turbo is text-to-image only — remove reference images and the character, or switch mode.", 400)

    if mode == "headswap" and (len(refs) + (1 if char_image is not None else 0)) < 2:
        log.warning("POST /generate from %s rejected: head-swap needs 2 images, got %d uploads + %s slot",
                    client, len(refs), "1" if char_image is not None else "0")
        return _form_ctx(
            "Head-swap needs a body upload plus a head: either upload 2 images, or upload 1 body and pick a character slot as the head.", 400)

    original_prompt = ""
    ratio = ""
    if flow == "enhance-generate":
        if mode not in ("generate", "turbo"):
            return _form_ctx("Enhance & Generate is only offered for Generate/Turbo modes.", 400)
        log.info("POST /generate from %s: openrouter enhance-first...", client)
        original_prompt = prompt
        try:
            enhanced = await run_in_threadpool(_openrouter_enhance, prompt)
            prompt, ratio = enhanced["rewritten_prompt"], enhanced["wh_ratio"]
        except Exception as e:
            log.exception("POST /generate from %s: enhance failed", client)
            return _form_ctx(f"Enhance failed: {type(e).__name__}: {e}", 502)
        log.info("POST /generate from %s: enhanced (ratio=%s), chaining to %s",
                 client, ratio or "?", mode)

    batch = max(1, min(batch, 8))

    try:
        # Batch = N Modal calls spawned back-to-back. With max_containers=1
        # Modal queues them on ONE warm container instead of cold-starting
        # each — no scaledown change needed. Seeds vary per item.
        call_ids: list[str] = []
        for i in range(batch):
            s = seed + i if seed else 0
            # blocking Modal network call -> threadpool, taaki event loop block na ho
            call_ids.append(await run_in_threadpool(
                _spawn_modal, prompt, negative_prompt.strip(), width, height, s, steps,
                refs, mode, char_image, identity,
            ))
        call_id = call_ids[0]
    except Exception as e:
        log.exception("POST /generate spawn failed for %s", client)
        return _form_ctx(f"Modal spawn failed: {type(e).__name__}: {e}", 502)
    if original_prompt:
        for cid in call_ids:
            JOBS[cid].update({"original_prompt": original_prompt, "enhanced": True,
                              "ratio": ratio, "mode": f"{mode} (enhanced)"})
    if len(call_ids) > 1:
        log.info("POST /generate from %s -> batch of %d, redirect /queue", client, len(call_ids))
        return RedirectResponse(url=f"/queue?batch={','.join(call_ids)}", status_code=303)
    log.info("POST /generate from %s -> redirect /result/%s", client, call_id)
    return RedirectResponse(url=f"/result/{call_id}", status_code=303)


@app.get("/queue", response_class=HTMLResponse)
def queue(request: Request, batch: str = ""):
    wanted = [c for c in batch.split(",") if c in JOBS]
    jobs = [(cid, JOBS[cid]) for cid in wanted] or sorted(
        JOBS.items(), key=lambda kv: kv[1]["created"], reverse=True)[:20]
    for cid, _ in jobs:
        _poll_modal(cid)
    pending = any(j["status"] == "pending" for _, j in jobs)
    return templates.TemplateResponse(
        request, "queue.html", {"jobs": jobs, "pending": pending}
    )


@app.get("/result/{call_id}", response_class=HTMLResponse)
def result(request: Request, call_id: str):
    job = JOBS.get(call_id)
    if job is None:
        log.warning("GET /result/%s: unknown job, tracking as fresh Modal lookup", call_id)
        # backend restarted or unknown id — still try Modal directly once
        JOBS[call_id] = {
            "status": "pending", "png": None, "error": None, "prompt": "",
            "negative": "", "width": 1024, "height": 1024, "seed": 0,
            "steps": 25, "mode": "?", "ref_count": 0, "created": time.time(),
            "mock": MOCK_MODAL, "ref_previews": [],
        }
        job = JOBS[call_id]
    _poll_modal(call_id)
    elapsed = int(time.time() - job["created"])
    log.debug("GET /result/%s status=%s elapsed=%ds", call_id, job["status"], elapsed)
    return templates.TemplateResponse(
        request, "result.html", {"call_id": call_id, "job": job, "elapsed": elapsed}
    )


@app.get("/image/{call_id}")
def image(call_id: str):
    job = JOBS.get(call_id)
    if job is None:
        log.warning("GET /image/%s: unknown job", call_id)
        return Response("unknown job", status_code=404)
    _poll_modal(call_id)
    if job["status"] != "done" or not job["png"]:
        log.debug("GET /image/%s: still processing (status=%s)", call_id, job["status"])
        return Response("still processing", status_code=202)
    log.info("GET /image/%s: serving %d bytes", call_id, len(job["png"]))
    return Response(content=job["png"], media_type="image/png")

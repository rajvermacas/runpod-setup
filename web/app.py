"""Qwen-Image 2.1 web UI — FastAPI + Jinja.

Own UI -> this backend -> Modal GPU (no `modal serve` UI needed).

Flow (async, avoids HTTP timeouts on 1-3 min GPU jobs):
  GET  /                -> form (prompt textbox + reference image upload)
  POST /generate        -> spawn Modal job, redirect to /result/{call_id}
  GET  /result/{call_id}-> poll Modal; 202-style pending page auto-refreshes
  GET  /image/{call_id} -> final PNG bytes

Backend -> Modal uses `modal.Cls.from_name("qwen21-4bit-direct", "Qwen21Direct")`
with `.spawn()` / `FunctionCall.from_id().get(timeout=0)`.

Run:
  pip install -r web/requirements.txt
  uvicorn web.app:app --reload --port 8000
  # mock mode (no Modal spend, placeholder image):
  MOCK_MODAL=1 uvicorn web.app:app --reload --port 8000
"""
from __future__ import annotations

import base64
import io
import logging
import os
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
MODAL_CLIP = "qwen3vl_8b_w4a8.safetensors"
MOCK_MODAL = os.environ.get("MOCK_MODAL", "") == "1"

MAX_REFS = 4
MAX_FILE_MB = 10

# call_id -> job dict (in-memory; single-process dev server)
JOBS: dict[str, dict] = {}

app = FastAPI(title="Qwen-Image 2.1 web UI")
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

log.info(
    "startup modal_app=%s cls=%s mock=%s max_refs=%d max_file_mb=%d",
    MODAL_APP_NAME, MODAL_CLS_NAME, MOCK_MODAL, MAX_REFS, MAX_FILE_MB,
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
                 seed: int, steps: int, refs: list[bytes]) -> str:
    """Spawn a Modal job, return the FunctionCall id."""
    t0 = time.time()
    if MOCK_MODAL:
        call_id = f"mock-{uuid.uuid4().hex[:12]}"
        JOBS[call_id] = {
            "status": "pending", "png": None, "error": None,
            "prompt": prompt, "negative": negative,
            "width": width, "height": height, "seed": seed, "steps": steps,
            "ref_count": len(refs), "created": time.time(), "mock": True,
            "ref_previews": _previews(refs),
        }
        log.info("spawn mock call_id=%s prompt=%.60r %dx%d steps=%d seed=%d refs=%d (%.2fs)",
                 call_id, prompt, width, height, steps, seed, len(refs), time.time() - t0)
        return call_id
    import modal

    cls = modal.Cls.from_name(MODAL_APP_NAME, MODAL_CLS_NAME)
    inst = cls()
    call = inst.generate.spawn(
        prompt, negative, width, height, seed, steps,
        1.0, "euler", "simple", False, MODAL_CLIP,
        ref_images=refs or None,
    )
    call_id: str = call.object_id
    JOBS[call_id] = {
        "status": "pending", "png": None, "error": None,
        "prompt": prompt, "negative": negative,
        "width": width, "height": height, "seed": seed, "steps": steps,
        "ref_count": len(refs), "created": time.time(), "mock": False,
        "ref_previews": _previews(refs),
    }
    log.info("spawn modal call_id=%s prompt=%.60r %dx%d steps=%d seed=%d refs=%d (%.2fs)",
             call_id, prompt, width, height, steps, seed, len(refs), time.time() - t0)
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
def index(request: Request):
    return templates.TemplateResponse(request, "index.html")


@app.get("/health")
def health():
    return {"status": "ok", "modal_app": MODAL_APP_NAME, "mock": MOCK_MODAL}


@app.post("/generate")
async def generate(
    request: Request,
    prompt: str = Form(...),
    negative_prompt: str = Form(""),
    width: int = Form(1024),
    height: int = Form(1024),
    steps: int = Form(25),
    seed: int = Form(42),
):
    prompt = (prompt or "").strip()
    client = request.client.host if request.client else "?"
    if not prompt:
        log.warning("POST /generate from %s rejected: empty prompt", client)
        return templates.TemplateResponse(
            request, "index.html", {"error": "Prompt is required."}, status_code=400
        )
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
            return templates.TemplateResponse(
                request,
                "index.html",
                {"error": f"{f.filename}: exceeds {MAX_FILE_MB} MB limit."},
                status_code=400,
            )
        data = await f.read()
        if not data:
            log.warning("POST /generate from %s: skipping empty file %r", client, f.filename)
            continue
        if len(data) > MAX_FILE_MB * 1024 * 1024:
            log.warning("POST /generate from %s rejected: %s %.1f MB over limit",
                        client, f.filename, len(data) / 1048576)
            return templates.TemplateResponse(
                request,
                "index.html",
                {"error": f"{f.filename}: exceeds {MAX_FILE_MB} MB limit."},
                status_code=400,
            )
        if not _is_image(data):
            log.warning("POST /generate from %s rejected: %r is not a decodable image",
                        client, f.filename)
            return templates.TemplateResponse(
                request,
                "index.html",
                {"error": f"{f.filename}: not a valid image file."},
                status_code=400,
            )
        refs.append(data)
        ref_names.append(f"{f.filename} ({len(data) // 1024} KB)")
    log.info("POST /generate from %s prompt=%.80r %dx%d steps=%d seed=%d refs=[%s]",
             client, prompt, width, height, steps, seed, ", ".join(ref_names) or "none")

    try:
        # blocking Modal network call -> threadpool, taaki event loop block na ho
        call_id = await run_in_threadpool(
            _spawn_modal, prompt, negative_prompt.strip(), width, height, seed, steps, refs
        )
    except Exception as e:
        log.exception("POST /generate spawn failed for %s", client)
        return templates.TemplateResponse(
            request,
            "index.html",
            {"error": f"Modal spawn failed: {type(e).__name__}: {e}"},
            status_code=502,
        )
    log.info("POST /generate from %s -> redirect /result/%s", client, call_id)
    return RedirectResponse(url=f"/result/{call_id}", status_code=303)


@app.get("/result/{call_id}", response_class=HTMLResponse)
def result(request: Request, call_id: str):
    job = JOBS.get(call_id)
    if job is None:
        log.warning("GET /result/%s: unknown job, tracking as fresh Modal lookup", call_id)
        # backend restarted or unknown id — still try Modal directly once
        JOBS[call_id] = {
            "status": "pending", "png": None, "error": None, "prompt": "",
            "negative": "", "width": 1024, "height": 1024, "seed": 42,
            "steps": 25, "ref_count": 0, "created": time.time(),
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

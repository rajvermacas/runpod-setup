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
import os
import time
import uuid
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.responses import HTMLResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

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
    if MOCK_MODAL:
        call_id = f"mock-{uuid.uuid4().hex[:12]}"
        JOBS[call_id] = {
            "status": "pending", "png": None, "error": None,
            "prompt": prompt, "negative": negative,
            "width": width, "height": height, "seed": seed, "steps": steps,
            "ref_count": len(refs), "created": time.time(), "mock": True,
            "ref_previews": _previews(refs),
        }
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
    return call_id


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
    if job.get("mock"):
        # simulate ~5 s GPU delay so polling UI can be exercised
        if time.time() - job["created"] < 5:
            return
        try:
            job["png"] = _mock_png(job["prompt"], job["width"], job["height"])
            job["status"] = "done"
        except Exception as e:
            job["status"] = "error"
            job["error"] = str(e)
        return
    try:
        import modal

        fc = modal.FunctionCall.from_id(call_id)
        png = fc.get(timeout=0)  # raises TimeoutError while running
    except TimeoutError:
        return
    except Exception as e:
        job["status"] = "error"
        job["error"] = f"{type(e).__name__}: {e}"
        return
    job["png"] = bytes(png)
    job["status"] = "done"


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse(request, "index.html")


@app.post("/generate")
async def generate(
    request: Request,
    prompt: str = Form(...),
    negative_prompt: str = Form(""),
    width: int = Form(1024),
    height: int = Form(1024),
    steps: int = Form(25),
    seed: int = Form(42),
    ref_images: Optional[list[UploadFile]] = File(default=None),
):
    prompt = (prompt or "").strip()
    if not prompt:
        return templates.TemplateResponse(
            request, "index.html", {"error": "Prompt is required."}, status_code=400
        )
    width = max(256, min(width, 2048))
    height = max(256, min(height, 2048))
    steps = max(1, min(steps, 100))

    refs: list[bytes] = []
    for f in ref_images or []:
        if not f.filename:
            continue
        if len(refs) >= MAX_REFS:
            break
        data = await f.read()
        if not data:
            continue
        if len(data) > MAX_FILE_MB * 1024 * 1024:
            return templates.TemplateResponse(
                request,
                "index.html",
                {"error": f"{f.filename}: exceeds {MAX_FILE_MB} MB limit."},
                status_code=400,
            )
        refs.append(data)

    try:
        call_id = _spawn_modal(prompt, negative_prompt.strip(), width, height, seed, steps, refs)
    except Exception as e:
        return templates.TemplateResponse(
            request,
            "index.html",
            {"error": f"Modal spawn failed: {type(e).__name__}: {e}"},
            status_code=502,
        )
    return RedirectResponse(url=f"/result/{call_id}", status_code=303)


@app.get("/result/{call_id}", response_class=HTMLResponse)
def result(request: Request, call_id: str):
    job = JOBS.get(call_id)
    if job is None:
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
    return templates.TemplateResponse(
        request, "result.html", {"call_id": call_id, "job": job, "elapsed": elapsed}
    )


@app.get("/image/{call_id}")
def image(call_id: str):
    job = JOBS.get(call_id)
    if job is None:
        return Response("unknown job", status_code=404)
    _poll_modal(call_id)
    if job["status"] != "done" or not job["png"]:
        return Response("still processing", status_code=202)
    return Response(content=job["png"], media_type="image/png")

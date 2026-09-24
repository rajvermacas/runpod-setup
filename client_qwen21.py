"""Local client for the Modal Qwen-Image 2.1 endpoint.

Option 1 (used here): call the Modal .remote() method via `modal run`
  modal run modal_qwen21.py --prompt "your prompt" --use-nvfp4-dit

Option 2: talk to the deployed ComfyUI directly with the official
`comfy-sdk` Python library (requires a deployed/serve URL):

  pip install comfy-sdk
  export COMFY_BASE_URL="https://<your-modal-app>--comfqwen21-ui.modal.run"
  python client_qwen21.py --via-sdk --workflow qwen21_workflow_api.json

Option 2 needs the comfy-api-proxy in front of ComfyUI (COMFY_BASE_URL
pointing at port 8189). The Modal app in modal_qwen21.py exposes the raw
ComfyUI server (port 8188, classic /prompt API), so the simplest reliable
client is plain requests against /prompt (see generate_classic()).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def generate_classic(base_url: str, workflow_path: str, prompt: str, timeout: int = 1200) -> bytes:
    import uuid

    import requests

    wf = json.loads(Path(workflow_path).read_text())
    wf["6"]["inputs"]["prompt"] = prompt
    client_id = str(uuid.uuid4())
    r = requests.post(f"{base_url}/prompt", json={"prompt": wf, "client_id": client_id}, timeout=60)
    r.raise_for_status()
    prompt_id = r.json()["prompt_id"]
    import time

    deadline = time.time() + timeout
    while time.time() < deadline:
        h = requests.get(f"{base_url}/history/{prompt_id}", timeout=30).json()
        if prompt_id in h and h[prompt_id].get("status", {}).get("completed"):
            outs = h[prompt_id]["outputs"]
            break
        time.sleep(2)
    else:
        raise TimeoutError(prompt_id)
    for node_out in outs.values():
        for img in node_out.get("images", []):
            vr = requests.get(
                f"{base_url}/view",
                params={"filename": img["filename"], "subfolder": img.get("subfolder", ""), "type": img.get("type", "output")},
                timeout=120,
            )
            vr.raise_for_status()
            return vr.content
    raise RuntimeError("no images returned")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8188")
    ap.add_argument("--workflow", default="qwen21_workflow_api.json")
    ap.add_argument("--prompt", default="cinematic portrait of an astronaut in a neon Tokyo alley, ultra detailed")
    ap.add_argument("--out", default="qwen21_out.png")
    args = ap.parse_args()
    png = generate_classic(args.base_url, args.workflow, args.prompt)
    Path(args.out).write_bytes(png)
    print(f"saved {args.out} ({len(png)/1e6:.2f} MB)")


if __name__ == "__main__":
    main()

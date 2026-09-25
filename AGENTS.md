Whenever I point out or ask you to remember or you catch yourself making the same mistake twice, before continuing write a rule in the #LESSONS below to avoid the same mistake in future.

# LESSONS

## ComfyUI direct node imports (Modal / notebooks)
- Importing `nodes` directly (without `main.py`) registers core nodes only; `NODE_CLASS_MAPPINGS["TextEncodeQwenImage21"]` and other comfy_extras nodes raise KeyError. Always call `asyncio.run(nodes.init_extra_nodes(init_custom_nodes=False, init_api_nodes=False))` before reading `NODE_CLASS_MAPPINGS`.
- Do not diagnose a missing node as "ComfyUI too old" until `init_extra_nodes()` has been ruled out — this exact misdiagnosis cost two failed runs (15:25 and 15:35 IST on 2026-09-24).

## ComfyUI direct node calls (Modal / notebooks)
- Call the node's FUNCTION entrypoint (e.g. `ModelSamplingAuraFlow.patch_aura`), never an inherited same-name method: `ModelSamplingSD3.patch()` defaults `multiplier=1000` (SD3 timestep scale) while `patch_aura()` uses `multiplier=1.0` — calling `patch()` silently yields posterized neon garbage with no error (Z-Image-Turbo T4 run, 2026-09-25).
- Check `FUNCTION = "..."` in the node class source before wiring any `NODE_CLASS_MAPPINGS` call in-process; the UI path always uses FUNCTION, inherited helpers may carry wrong defaults.
- New-API (io.ComfyNode) nodes return `io.NodeOutput`; unwrap `.args` to legacy values at every node boundary (`_normalize_outputs`). The server executor does this implicitly — direct callers bypass it, so legacy downstream nodes get the wrapper: `latent["samples"]` on a NodeOutput -> TypeError, slot links index a 1-tuple -> IndexError (generic workflow runner, 2026-09-25).

## GPU runs (Modal)
- Never start any GPU-billed run (`modal run`, `modal deploy`, `modal serve`, background GPU jobs) without the user's explicit permission for that specific run. Announce the exact command plus expected GPU/time/cost first and wait for approval — no implied consent from debug context (rule set 2026-09-25).
- Default GPU is always T4 (cheapest hourly) unless the user asks for another GPU for that specific run/script (rule set 2026-09-25).

## Modal app logs
- When polling a run, target the current run's app ID explicitly; `modal app list | awk '/qwen21/'` can match an older app of the same name and show stale logs (this caused a false "still failing" report).

## Time reporting
- Always express times in IST (UTC+5:30), never UTC, for this user.


Whenever I point out or ask you to remember or you catch yourself making the same mistake twice, before continuing write a rule in the #LESSONS below to avoid the same mistake in future.

# LESSONS

## ComfyUI direct node imports (Modal / notebooks)
- Importing `nodes` directly (without `main.py`) registers core nodes only; `NODE_CLASS_MAPPINGS["TextEncodeQwenImage21"]` and other comfy_extras nodes raise KeyError. Always call `asyncio.run(nodes.init_extra_nodes(init_custom_nodes=False, init_api_nodes=False))` before reading `NODE_CLASS_MAPPINGS`.
- Do not diagnose a missing node as "ComfyUI too old" until `init_extra_nodes()` has been ruled out — this exact misdiagnosis cost two failed runs (15:25 and 15:35 IST on 2026-09-24).

## Modal app logs
- When polling a run, target the current run's app ID explicitly; `modal app list | awk '/qwen21/'` can match an older app of the same name and show stale logs (this caused a false "still failing" report).

## Time reporting
- Always express times in IST (UTC+5:30), never UTC, for this user.


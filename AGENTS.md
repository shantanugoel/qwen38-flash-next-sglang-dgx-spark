# Agent notes — this recipe repo

You are iterating on a **single DGX Spark / GB10** SGLang (or vLLM) recipe for
`RadixArk/Qwen3.8-Flash-Next-NVFP4`. Plan: `EXPLORATION.md`. Results:
`RESEARCH_LOG.md`. Do not edit `~/ai`.

## DO NOT RUN THINGS IN THE FOREGROUND

Long work on this box (image pull, `prepare.sh`, first model load 8–20 min,
decode benches, 8k/32k prefills, GSM8K, docker logs -f) will **block the
agent with no way out** if you wait on them in the foreground.

**Always:**

1. Launch the job **in the background** (`docker run -d`, `nohup … &`, or the
   harness `timeout`/`block_until_ms` equivalent of “return immediately”).
2. Redirect stdout/stderr to a file under `results/` (gitignored).
3. **Poll** that log, `docker logs --tail`, `/health`, or `nvidia-smi` with
   short waits. Do not `docker logs -f`, do not `sleep 900`, do not set a
   20-minute blocking timeout on the tool call that started the job.
4. If a poll shows the job died, read the tail of the log, fix, restart in
   the background again.
5. Benches that take more than ~30s go the same way: `nohup python3 bench/….py
   > results/….log 2>&1 & echo $! > results/….pid` then poll the log until
   `RESULT` / `OK` / traceback appears.

Serve is already detached (`docker run -d`). First load is still 8–20 min of
silence while the PLE mmap fills — that is **not** hung. Poll `/health` every
30–60s for up to ~25 min, then read `docker logs --tail 80`.

## Other rules

- One GPU occupant. Unload llama-swap first:
  `curl -sS -X POST http://127.0.0.1:8080/api/models/unload`
- No sudo. Do not uncap GPU clocks.
- Do not commit weights, PLE backing files, `results/`, logs, or `.env`.
- Default recipe: thinking **on**; caller may set
  `chat_template_kwargs.enable_thinking=false`.
- Keep **vision**. Do not pass `--language-only` or `--language-model-only`.
  Those skip or disaggregate the multimodal encoder. Hashd1ve used
  `--language-only` for text-only KV headroom; we do not.
- 262k is the default context; 512k is optional and must not replace it
  unless it actually works.
- Runtime-only env (not committed): reuse the existing mmap with
  `PLE_DIR=/home/shantanu/ai/cache/sglang/flash-next-ple-mmap` and
  `HF_CACHE=$HF_HOME`.
- The PLE mmap is **reused** across boots (`patches/ple_reuse.py`). A boot is
  ~10 min, not ~60. If a boot suddenly takes an hour again, check the log for
  `PLE table: N/N shards already on disk` — if it is missing, the patch did not
  apply. The table is written by `copy_ple_rows_to_tp_embedding` in
  `load_weights`, **not** by `param.weight_loader`; do not "fix" it there.
- **Terminal-Bench cannot run on this box**: TB 2.x task images are amd64-only
  and there is no qemu binfmt handler. `scripts/bench_tb.sh` is correct and runs
  from an x86 host against this server. Never report a TB score measured here.
- One experiment = one `nohup TAG=… ./scripts/run_config.sh &`. It restarts,
  waits for `/health` with a timeout, and runs the fast bench suite.
- After each plan step: append `RESEARCH_LOG.md`, commit scripts/docs only.

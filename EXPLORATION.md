# Exploration plan — one-GB10 Flash-Next recipe

Goal: keep a **single DGX Spark / GB10** recipe, stay on **Radix NVFP4**, and raise
quality then speed for long-horizon agentic work (search/scrape/facts/tools/code/prose,
100–150 turns, native **262k** context, **512k optional** if it actually works).

Constraints (locked 2026-08-28):

- Balanced: no quality regression vs the current recipe, then take free speed.
- Thinking **on by default**; caller can disable via `chat_template_kwargs.enable_thinking`.
- Do **not** edit `~/ai`. Unload llama-swap before occupying the GPU. No sudo.
- Treat **vLLM as a real contender** if it beats this SGLang path on the same box.
- Keep **vision**. Never `--language-only` / `--language-model-only`.
- Look at newer NVFP4 checkpoints / images, but do not download a second 135 GB
  pack unless research shows a quality or fit win.

Each step ends with a commit of scripts/docs only (no weights, PLE files, logs, or
raw dumps). Results live in `RESEARCH_LOG.md`.

**Foreground is forbidden** (also in `AGENTS.md`): never block on image pulls,
`prepare.sh`, first load, `docker logs -f`, or benches. Start them in the
background, log under `results/`, poll `/health` or the log file. Repeat this
rule at every step so a later agent cannot “just wait” on an 8–20 min boot.

## What we already know (do not re-learn)

Current public recipe (`scripts/serve.sh`): SGLang `lmsysorg/sglang:qwen38flashnext`,
PLE `torch.from_file` mmap, QSA SM120 gate, MTP NEXTN 3/1/4 + `unquant`, decode
`trtllm_mha`, prefill `triton`, CUDA graphs on, `--mem-fraction-static 0.95`,
`--context-length 262144`, `--chunked-prefill-size 4096`, `--max-running-requests 4`.

Published on this box 2026-08-27 (clock-capped GB10, thinking **off**): **40.2 tok/s
code** (median 38.8) / **~20 tok/s** thinking-on chat. vLLM mmap+MTP=2 was ~27 tok/s.

Community map (full notes in `RESEARCH_LOG.md`):

| Path | Speed | Quality risk | Decision |
| --- | ---: | --- | --- |
| hashd1ve mmap + QSA pin + MTP 3/1/4 | 41.5 code / 22.8 prose | GSM8K 96% vs ckpt 97.27 | **Baseline family** |
| QSA ring-width 8, MTP 7/1/8 | +18% code, **−36% prose** | GSM8K ok | Skip as default (agentic is mixed) |
| Death-By-Tokens HashK PLE (4× compress) | 27–46, more KV headroom | cosine ~0.50; not GSM8K-audited | Skip (quality) |
| blazux vLLM mmap + MTP=2 | ~25–28 decode, 2–2.6k prefill | needle at 414k w/ YaRN | **Bake off later** |
| llama.cpp UD-Q4_K_XL | 16–18 tok/s, no QSA/MTP | 93.5% same-top-1 | Skip as primary |
| mixed NVFP4+FP8 community packs | maybe denser | no RadixArk-class eval | Skip download this round |
| Kernel no-ops (hashd1ve n=10) | tf32 / ssm bf16 / cudnn fp4 / continuous-decode-2 | — | Do not retest |

## Benchmarks we will run

**Fast, every restart** (must finish in <15 min after warmup):

1. Health + `"12*17"` → 204.
2. Decode suite (`bench/decode.py`): code EN + prose, thinking **off** and **on**, n=3 after 1 warmup.
3. Tool-call smoke: one weather-style function, parser must emit `tool_calls`.
4. Code smoke: write a tiny Python function; we execute it.
5. Factual + multi-turn: plant a fact in turn 1, retrieve in turn 4.
6. Vision smoke: one image + short question (tower must stay loaded).

**Once per promising config** (longer):

6. Prefix-cache TTFT: 8k then 32k haystack, second send must be ≪ first (agentic resend).
7. Needle at ~8k and ~32k (invented access-code in real-ish prose). 128k only on the
   finalist — a 240k cold prefill has taken this box down at mem-frac 0.85.
8. GSM8K n=20, thinking off, t=0.6 — sanity vs 97.27, not a paper number.
9. Reasoning-effort sweep on one loaded server: default / medium / low / thinking-off
   (time-to-answer + correctness on one multi-hop question).

Not in this window: full GSM8K 1319, AIME26, 150-turn live agent, 240k cold prefill
sweep, HumanEval+. Those need overnight and a watchdog.

## Step-by-step

### Step 0 — Research + inventory (this file)

Background + poll (`AGENTS.md`). Image pull was `nohup docker pull … &`.

- [x] Unload llama-swap (`POST /api/models/unload`).
- [x] Confirm GPU free, 118 GiB host available, NVFP4 snapshot
      `7b719225242aacd3dbd3f9407468c2ee9a9d2594` already in `$HF_HOME`.
- [x] Confirm official image `lmsysorg/sglang:qwen38flashnext` is **not** local
      (only Felliks/vLLM side images). Pull it.
- [x] Write this plan + `RESEARCH_LOG.md`.

Commit: `docs: exploration plan and research log`.

### Step 1 — Harness (no GPU yet)

Background + poll (`AGENTS.md`). This step is local files only.

- [x] `bench/decode.py` — comparable to hashd1ve’s protocol.
- [x] `bench/quality.py` — math / tools / code-exec / multi-turn / effort.
- [x] `bench/longctx.py` — 8k/32k needle + prefix-cache TTFT.
- [x] `scripts/wait_ready.sh` — poll `/health` without sudo.
- [x] `.gitignore` — `results/`, logs, PLE, caches.
- [x] `AGENTS.md` — never run long jobs in the foreground; poll instead.

Do **not** point `PLE_DIR` at `~/ai` in committed scripts. Runtime may reuse
`/home/shantanu/ai/cache/sglang/flash-next-ple-mmap` via env to skip the 2.5 min
first-write. Default stays `./data/ple`.

Commit: `Add agentic speed/quality bench harness`.

### Step 2 — Official image + patches

Background + poll (`AGENTS.md`): `nohup ./scripts/prepare.sh > results/prepare.log 2>&1 &`.

- [x] `lmsysorg/sglang:qwen38flashnext` @ `sha256:12d3392bdc8be8d35e9a95f191df6aef99c5114bdbefd41bfdc7e760e6d25ec1`
- [x] PLE mmap patch applied; stock image has **no** SM121 SDPA intercept; SM120 QSA gate applied.
- [x] Checkpoint already present; no re-download.

If pull/patch fails: stop and log; do not silently switch to the Felliks image.

No script fix needed.

### Step 3 — Baseline (current `serve.sh` flags)

Background + poll (`AGENTS.md`): `serve.sh` is already `-d`. Poll `/health`
every 30–60s (up to ~25 min). Run benches with `nohup python3 bench/….py … &`.

Boot current flags, `PLE_DIR` reused if present.

Record: boot time, `cuda graph`, decode backend, MTP accept, RSS/`free`.

Run fast benches (thinking on **and** off). This is the gate every later config
must beat or match.

Commit: `RESEARCH_LOG.md` baseline numbers.

### Step 4 — Client knobs on the live baseline (no restart)

Background + poll (`AGENTS.md`). Same occupant; background each HTTP bench.

Same occupant:

- thinking on vs off vs `reasoning_effort` low/medium/xhigh.
- Confirm `enable_thinking: false` still works (chat template supports it even
  though the cookbook says “always reasons”).
- Tool-call parser behaviour.
- Sampling: checkpoint `generation_config.json` is already t=1.0 / top_p=0.95 /
  top_k=20, so `--preferred-sampling-params` is redundant unless a client omits
  them. Measure one greedy vs default decode-rate sample.

Commit: log only, unless we add a documented client snippet to README later.

### Step 4b — Boot cost (done out of order, 2026-08-28)

Background + poll (`AGENTS.md`).

- [x] Found the 47.7 GiB PLE refill that runs on **every** boot; added
      `patches/ple_reuse.py` (verified byte-sample fast path) and wired it into
      `prepare.sh`.
- [x] `serve.sh` parameterized (`MEMFRAC`, `PREFILL`, `MAX_RUNNING`, `CONTEXT`,
      `MAX_TOTAL`, `SPEC_*`, `CUDA_GRAPH_MAX_BS`, `EXTRA_ARGS`) so an A/B is an
      env prefix, not an edit.
- [x] Terminal-Bench moved to Harbor + `terminal-bench@2.0`; `scripts/host_proxy.py`
      exposes the loopback server on the docker bridge for the in-container agent.

### Step 5 — Serve-flag pack A (UMA / PLE residency)

Background + poll (`AGENTS.md`): stop + `serve.sh` detached, then poll `/health`.
**Never** `docker logs -f`, never a blocking timeout on the boot.

One restart. `nvidia-smi --query-compute-apps` during the baseline boot showed the
scheduler holding **82.8 GiB** of the 121 GiB UMA pool at `--mem-fraction-static
0.95`, leaving ~2–7 GiB free. The 47.7 GiB PLE table is random-access and lives in
**page cache**, so mem-fraction is a PLE-residency knob and therefore a decode-speed
knob, not only a stability knob. **Vision stays on.**

```
MEMFRAC=0.85 PREFILL=2048 MAX_RUNNING=2 CUDA_GRAPH_MAX_BS=8 \
EXTRA_ARGS="--weight-loader-drop-cache-after-load" ./scripts/serve.sh
```

- `--mem-fraction-static 0.85` — hand ~12 GiB back to the page cache.
- `--chunked-prefill-size 2048` — activation vs throughput.
- `--max-running-requests 2` — agentic is one stream plus maybe a retry.
- `--cuda-graph-max-bs-decode 8` — stock captures decode graphs to bs **256**;
  we never exceed `--max-running-requests`. Saves capture time and memory.
- `--weight-loader-drop-cache-after-load` — `posix_fadvise(DONTNEED)` per shard,
  which frees exactly the page cache the PLE table wants.
- keep MTP 3/1/4, graphs, `trtllm_mha` decode, multimodal tower.

Gate: quality no worse than baseline, 32k longctx stable, decode ≥ baseline.

### Step 6 — Agentic cache/session knobs (one restart)

Background + poll (`AGENTS.md`).

On top of the Step 5 winner, all client-invisible:

- `--strip-thinking-cache` — drop reasoning from the cached prefix so 100–150
  turn sessions keep hitting the radix cache.
- `--radix-eviction-policy lfu` (or `slru`) — keep the system prompt and tool
  definitions across a long session instead of evicting by recency.
- `--enable-gdn-replayssm-spec` — upstream handling of GDN state under
  speculative rewind; this is the corruption Death-By-Tokens patched by hand.

Measure prefix-cache hit % and 32k resend TTFT, not just decode.

### Step 7 — MTP A/B (one restart each, only if Steps 5–6 are healthy)

Background + poll (`AGENTS.md`): one detached boot per MTP id.

QSA draft cap is 4 tokens without a ring-width patch we will **not** ship.

| id | steps / topk / draft | Why |
| --- | --- | --- |
| M0 | `SPEC=off` | floor (~18 tok/s historically) |
| M1 | `SPEC_STEPS=2 SPEC_DRAFT=3` | cheaper drafts, maybe better prose |
| M2 | `SPEC_STEPS=3 SPEC_DRAFT=4` | current; Qwen paper mean accept ~4.07 |

Keep M2 unless M1 is clearly better on prose + thinking-on without losing code.

### Step 8 — Benchmarks on the finalist

Background + poll (`AGENTS.md`). These are the long ones; run them once, on the
config we intend to ship, not on every candidate.

- `bench/bfcl.py` — fixed 100-case BFCL subset; accuracy by split + invalid
  tool calls + wall clock / tokens / cache-hit / TTFT / decode tok/s.
- `scripts/bench_tb.sh` — Terminal-Bench 8-task subset, k=1, PASS/FAIL/TIMEOUT.
- GSM8K n=20 sanity, thinking off.
- Vision on vs off as the last comparison, both numbers reported.

### Step 9 — 512k optional

Background + poll (`AGENTS.md`): boot can OOM; watch `docker logs --tail` and
host `free -h` on a timer, never in the foreground.

```
CONTEXT=524288 MAX_TOTAL=524288 MAX_RUNNING=1 MEMFRAC=0.82 PREFILL=1024
```

with Qwen static YaRN (`factor=4.0`, `original_max_position_embeddings=262144`).
Pass: boots + 8k needle still works + a ~40k needle. Fail: OOM, rope error, or an
8k quality break → keep 262k default and document 512k as experimental.
Do **not** make 512k the default even if it boots.

### Step 10 — vLLM bake-off (only if the clock allows)

Background + poll (`AGENTS.md`). Lowest priority: it costs a full image + boot
cycle and the SGLang path is already measured. If it is skipped, say so in
`RESEARCH_LOG.md` rather than implying it was tested.

### Step 11 — Final recipe

Background + poll (`AGENTS.md`): confirmation benches still go via `nohup`.

Fold winners into `scripts/serve.sh` defaults + README tables (thinking on **and**
off, prefix-cache TTFT, needle, tools, BFCL, Terminal-Bench, vision on/off).
Optional `CONTEXT=524288` documented, not default.

## Skip list (intentional)

- HashK / packed-NVFP4 PLE — quality unknown vs RadixArk GSM8K.
- QSA ring-width 8 — prose/agentic regression.
- Alternative HF NVFP4 repos this round (Inferact / lovedheart / primitive-ai).
- GPU clock uncap (sudo, hard-reboot history).
- llama-swap wrap / Caddy (`~/ai` is off limits).
- Dual-Spark TP2.
- 1M YaRN.
- `--load-format dummy` (PLE OOM).
- `--attention-backend trtllm_mha` (prefill SM100 gate).

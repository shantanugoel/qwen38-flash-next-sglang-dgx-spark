# Exploration plan — one-GB10 Flash-Next recipe

Goal: keep a **single DGX Spark / GB10** recipe, stay on **Radix NVFP4**, and raise
quality then speed for long-horizon agentic work (search/scrape/facts/tools/code/prose,
100–150 turns, native **262k** context, **512k optional** if it actually works).

Constraints (locked 2026-08-28):

- Balanced: no quality regression vs the current recipe, then take free speed.
- Thinking **on by default**; caller can disable via `chat_template_kwargs.enable_thinking`.
- Do **not** edit `~/ai`. Unload llama-swap before occupying the GPU. No sudo.
- Treat **vLLM as a real contender** if it beats this SGLang path on the same box.
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

### Step 5 — Serve-flag pack A (stability / agentic memory)

Background + poll (`AGENTS.md`): stop + `serve.sh` detached, then poll `/health`.

One restart. Combined because each is a 10–15 min boot and they target the same
long-horizon failure mode (UMA starvation + vision weights we do not need):

- `--language-only`
- `--mem-fraction-static 0.85` (0.95 left ~6–8 GiB; hashd1ve hung the box on
  sequential long prefills at 0.85 even)
- `--chunked-prefill-size 2048` (activation vs throughput)
- `--max-running-requests 2` (agentic is 1 stream + maybe a retry)
- keep MTP 3/1/4, graphs, trtllm_mha decode

If quality holds and longctx 32k is stable, this becomes the new default skeleton.

Commit: `serve.sh` change only after the numbers land.

### Step 6 — MTP A/B (one restart each, only if Step 5 is healthy)

Background + poll (`AGENTS.md`): one detached boot per MTP id, never wait on logs -f.

QSA draft cap is 4 tokens without a ring-width patch we will **not** ship.

| id | steps / topk / draft | Why |
| --- | --- | --- |
| M0 | off | floor (~18 tok/s historically) |
| M1 | 2/1/3 | cheaper drafts, maybe better prose |
| M2 | 3/1/4 | current; Qwen paper mean accept ~4.07 at 4-step |
| M3 | skip 4/1/5 | needs ring >4; skip |

Keep M2 unless M1 is clearly better on **prose + thinking-on** without losing code.

Commit: serve flags if M1 wins; else log.

### Step 7 — Leftover server knobs (only if still unknown)

Background + poll (`AGENTS.md`). At most two detached restarts.

Try at most two more restarts, highest expected value first:

1. `--chunked-prefill-size 1024` vs 2048 on 32k TTFT (long-horizon prefills).
2. Drop `--disable-prefill-cuda-graph` **only if** logs show prefill graphs would
   capture on this mmap path (Felliks disk-cache path cannot; hashd1ve mmap can
   capture **decode** graphs).
3. Do **not** retune: `--fp4-gemm-backend`, `--mamba-ssm-dtype`, `--enable-tf32-matmul`,
   `--num-continuous-decode-steps`, `--moe-runner-backend` (sm_121 dead ends).

### Step 8 — 512k optional

Background + poll (`AGENTS.md`): boot can OOM; watch `docker logs --tail` and
host `free -h` on a timer, do not sit in the foreground.

Native rope is 262144. Try:

```
CONTEXT=524288 YARN=1 MAX_TOTAL=524288 MAX_RUNNING=1 MEMFRAC=0.82 PREFILL=1024
```

with Qwen static YaRN (`factor=4.0`, `original_max_position_embeddings=262144`).

Pass: boots + 8k needle still works + a ~40k needle (not a full 400k haystack).
Fail: OOM, rope error, or 8k quality break → keep 262k default, document the env
knob as experimental.

Do **not** make 512k the default even if it boots.

### Step 9 — vLLM bake-off

Background + poll (`AGENTS.md`): stop SGLang first, start vLLM detached, poll its
`/health` (port likely 18300). Image pulls in the background only.

Use the already-local `qwen38-flash-dgx:latest` (blazux mmap image) if it is that
recipe; otherwise pull `vllm/vllm-openai:qwen38-flash-next` only if disk/time
allow.

Same checkpoint, MTP=2, ctx 262k, `--no-enable-prefix-caching` (GDN sm_121 bug).

Compare: thinking-on decode, 8k/32k prefill+needle, tool-call. vLLM wins only if
it is faster **and** not worse on tools/needle/code-exec, **or** dramatically
faster on prefill with equal quality (agentic search dumps care about TTFT).

If vLLM wins, the final recipe switches engine. If not, record why SGLang stays.

### Step 10 — Final recipe

Background + poll (`AGENTS.md`): last confirmation benches still go via `nohup`.

Fold winners into `scripts/serve.sh` + README tables (thinking on **and** off,
prefix-cache TTFT, needle, tools, smoke). Optional `CTX=512k` documented, not default.

Last commit: `Ship measured one-GB10 Flash-Next recipe`.

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

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

### Step 5 — Pack A (mem-frac 0.85 pack) — **DONE, REJECTED**

`MEMFRAC=0.85 PREFILL=2048 MAX_RUNNING=2 CUDA_GRAPH_MAX_BS=8 --weight-loader-drop-cache-after-load`

Result: no measurable speed change vs a clean baseline, **−39% KV budget**
(524288 → 318464) and MTP accept down 3.80 → 3.50. **Keep the committed
defaults.** The PLE-residency hypothesis that motivated it is dead: 0.85 bought
only ~4 GiB of page cache against a 47.7 GiB table and bought nothing in speed.
Individual components were never attributed — that is what Step 13 fixes.

### Step 6 — Baseline re-measure (clean) — **DONE**

The first baseline was contaminated by a concurrent second bench client.
`bench_fast.sh` now takes an `flock`. Clean numbers are the gate in
`RESEARCH_LOG.md`; `results/baseline-contaminated/` is kept as evidence.

### Step 7 — MTP A/B — **DROPPED, with evidence**

`sglang:spec_accept_length` is **3.80 out of a 4-token draft**. M1 (2/1/3) can
only lose accepted tokens; deeper drafts need QSA ring-width > 4, already
skipped for a −36% prose regression. Recorded as reasoned-out, **not tested**.

### Step 8 — Finalist benchmarks on the shipped recipe — **DONE (thinking off)**

BFCL 100-case, GSM8K n=20, agentic 120 turns. Numbers in `RESEARCH_LOG.md`.
**Terminal-Bench is impossible on this box** (amd64-only task images, no qemu
binfmt — Step 3b). Never report a TB number measured here.

Open follow-up: the BFCL `multi_turn_base` split scores 0.0% because of **our
scorer**, not the model — it issues one tool call per turn (correct agentic
behaviour) while the ground truth lists the whole multi-call sequence for that
turn. Fix in Step 9.

### Step 9 — BFCL multi-turn scorer fix + re-score — **TODO**

Let the model loop *within* a turn (feed each tool result back until it stops
calling), then compare the union of called function names against the turn's
ground truth. Re-run the 20 multi-turn cases only. Report the corrected split
accuracy and reissue the headline number; the current 58% is not reportable.

### Step 10 — Agentic session with **thinking on** — **IN PROGRESS**

The finalist scenario, and the sharper probe for MTP-rewind / GDN-state
corruption (the reported failure mode involves reasoning traces). 120 turns,
tools, radix cache on.

### Step 11 — Kernel and memory sweep — **QUEUED** (`scripts/sweep.sh`)

`QUICK=1` (decode + quality only) per config; a boot failure is a **result**,
not a stop condition — several of these may have no sm_121 cubins.

| tag | config | question |
| --- | --- | --- |
| `lin-flashinfer` | `--linear-attn-backend=flashinfer` | GDN kernel; auto-selected on SM100, untested on SM121 |
| `lin-cutedsl` | `--linear-attn-backend=cutedsl` | same |
| `lin-nvidia-kda` | `--linear-attn-backend=nvidia_kda` | same |
| `mamba-ratio-03` | `--mamba-full-memory-ratio=0.3` | recover SSM memory for KV / concurrency |
| `prefill8192-graph8` | `PREFILL=8192 CUDA_GRAPH_MAX_BS=8 --disable-cuda-graph-padding` | TTFT bundle |
| `gdn-replayssm` | `--enable-gdn-replayssm-spec` | does the upstream fix replace our `extra_buffer` + `track_interval 64` workaround? |

### Step 12 — Radix cache A/B — **TODO**

`--disable-radix-cache` vs on, **with thinking on**, 120 turns. Speed *and*
reliability. Current evidence with the cache **on** and the Mamba guard in
place: 0 invalid tool calls across ~200 tool turns, correct 32k cached-resend
needle, cache hit 95–99%. This step is about what we give up by turning it off,
and whether the guard is load-bearing.

### Step 13 — Unbundle whatever won — **TODO**

Pack A taught this: never ship a bundle. Any winning multi-flag config from
Step 11 gets its components measured one at a time before it reaches
`serve.sh` defaults.

### Step 14 — 512k optional — **TODO**

```
CONTEXT=524288 MAX_TOTAL=524288 MAX_RUNNING=1 MEMFRAC=0.82 PREFILL=1024
```

plus Qwen static YaRN (`factor=4.0`, `original_max_position_embeddings=262144`),
layered on whatever `--mamba-full-memory-ratio` frees. Pass: boots + 8k needle
still works + a ~40k needle. Do **not** make it the default even if it boots.

### Step 15 — Vision off — **TODO**

One boot with `--language-only` purely to report both numbers, then **revert**.
Vision stays on in the shipped recipe.

### Step 16 — vLLM bake-off — **NOT PLANNED, with a reason**

vLLM on sm_121 needs `--no-enable-prefix-caching` (GDN CUBLAS bug). Prefix
caching is what carries this use case: 99.0% hit and 20.6× warm-prefill
speedup over 120 turns, against a vLLM decode number reported at ~25–28 tok/s
versus our 38.6. Trading the cache away to chase a slower decode loses on the
metric that matters. If it is never run, `RESEARCH_LOG.md` says **untested** —
it does not imply a comparison was made.

### Step 17 — Final recipe

Fold winners into `scripts/serve.sh` defaults + README tables: decode thinking
on **and** off, agentic 120-turn bands, prefix-cache TTFT, needles, BFCL
(corrected), GSM8K, vision on/off. Optional `CONTEXT=524288` documented, not
default. Last commit: `Ship measured one-GB10 Flash-Next recipe`.

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

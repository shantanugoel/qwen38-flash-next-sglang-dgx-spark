# Exploration plan — one-GB10 Flash-Next recipe

**Current campaign:** [September upstream validation plan](UPSTREAM_PLAN.md).
The August plan below is historical; stale TODO/status and skip-list entries do
not authorize rerunning completed experiments. See RESEARCH_LOG.md for outcomes.
U0 and U1 are accepted; later campaign items have not started.

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
PLE `torch.from_file` mmap, QSA SM121 Triton (#36845), MTP NEXTN 3/1/4 + `unquant`, decode
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

### Step 9 — BFCL multi-turn scorer — **DONE, split still not measurable**

Scorer fixed (model loops within a turn): 0.0% → 15.0%. Still invalid as a BFCL
number because our tool backend is a mock. Report the 80 single-turn cases
(**72.5%**) and mark multi-turn N/A.

### Step 10 — Agentic with thinking on — **DONE**

120 turns, 0 invalid tool calls, late recall PASS, cache hit 99.4%, TTFT falling
to 0.29 s. Behavioural finding: **60/120 turns emit a tool call vs 119/120 with
thinking off.**

### Step 11 — Kernel and memory sweep — **DONE**

One adoption (`--enable-gdn-replayssm-spec`), one deferred
(`--mamba-full-memory-ratio`, inert at 262k because `--max-total-tokens` binds
first), one held back (the prefill/graph bundle: worst prose-off, one borderline
quality failure, and unattributed), and one structural negative (alternative GDN
kernels are unreachable — compressed QSA pins `page_size=64`).

### Step 12 — Radix cache A/B — **IN PROGRESS**

`--disable-radix-cache` with the **full** bench, on top of the adopted
`--enable-gdn-replayssm-spec`. The QUICK sweep could not answer this: short
unique prompts make the prefix cache irrelevant by construction, which is why
`radix-off-triton` looked equal to baseline there. The 120-turn session is where
the cost shows.

### Step 13 — `--enable-gdn-replayssm-spec` — **DONE, ADOPTED**

+7.6% code decode, agentic decode 51.7 vs 49.6, `spec_accept_length`
3.80 → 3.95, 12/12 quality, same KV, 2.1 GiB less GPU. In `serve.sh` defaults.

### Step 14 — 512k optional — **TODO**

```
CONTEXT=524288 MAX_TOTAL=524288 MAX_RUNNING=1 MEMFRAC=0.82 PREFILL=1024
```

**The published Qwen YaRN recipe does not apply to this checkpoint.** There is no
`rope_scaling` field; `text_config.rope_parameters` is **mrope**:

```json
{"mrope_interleaved": true, "mrope_section": [11, 11, 10],
 "partial_rotary_factor": 0.25, "rope_theta": 10000000, "rope_type": "default"}
```

So the override must target `text_config.rope_parameters` and preserve the mrope
fields, and `SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1` must be set or
`_derive_context_length` refuses 524288 outright. Whether YaRN composes with
sectioned mrope at all is unknown — a clean failure here is an acceptable
outcome for an optional feature, but it must be documented as *why*, not as
"not tried". **Retest `--mamba-full-memory-ratio 0.3` in this step** — it is
inert at 262k because `--max-total-tokens` binds first, so this is the only
regime where it can pay off. Do not make 512k default even if it boots.

### Step 15 — Vision off — **TODO**

One boot with `--language-only`, report both numbers, then revert. Vision stays.

### Step 16 — vLLM — **NOT PLANNED**

Needs `--no-enable-prefix-caching` on sm_121. Prefix caching is what carries this
use case (99.0% hit, 21× warm prefill). Recorded as **untested**, not compared.

### Step 17 — Final recipe — **TODO**

`serve.sh` defaults are already updated with `--enable-gdn-replayssm-spec`.
Remaining: rewrite the README Measured table with the new numbers (decode
thinking on/off, agentic 120-turn bands both modes, needles, prefix-cache TTFT,
BFCL single-turn 72.5%, GSM8K 19/20, vision on/off) and the dead-end list.

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

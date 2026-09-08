# Qwen3.8-Flash-Next on one DGX Spark (SGLang)

**September 2026 upstream campaign:** **U0–U4a and U6 are accepted; U4b/U4c/U5a were rejected; U5b skipped.** Serving image is
SGLang `4ccff141` (`lmsysorg/sglang@sha256:9d2a843c706c74bc259c0d9abf360551eb2734e1e7d255ab012a6965f10480b6`).
Sparse decode on GB10 uses the 2026-08-28 Triton kernel from [SGLang #36845](https://github.com/sgl-project/sglang/pull/36845),
overlaid on this image's bundled KDA QSA (rejected in U1). ReplaySSM verify commits PLE n-gram/short-conv
state ([#37794](https://github.com/sgl-project/sglang/pull/37794) `spec_utils` hunk
only; that PR's NGRAM feature is not ported). Native PLE file backend
([#37068](https://github.com/sgl-project/sglang/pull/37068)) with recipe filename reuse.
The later KDA overlay from #36845 passed isolated tensor replay here and then emitted token-id 0 on a 32k needle;
it is not the serving default. 8k/32k needles pass. 120k–210k needles are still
unmeasured on this box. See the [staged plan](UPSTREAM_PLAN.md).

**This repo is how you run Qwen3.8-Flash-Next performantly on a single NVIDIA DGX Spark — or any other GB10 machine (ASUS Ascent GX10, MSI Atom, …).**

Serve [`RadixArk/Qwen3.8-Flash-Next-NVFP4`](https://huggingface.co/RadixArk/Qwen3.8-Flash-Next-NVFP4) with **SGLang** (128 GB unified memory). Native 262k context, MTP speculative decode, CUDA graphs.

- **What:** 125B MoE + 51B n-gram PLE, 6B active, NVFP4 routed experts. The checkpoint is ~135 GB; a Spark has 128 GB. Stock `--ple-offload-embedding` pins the 48 GB PLE table in the *same* UMA pool, so it does not fit.
- **Why mmap:** a token reads 16 PLE rows (~2.5 KB). File-backed mmap
  (`--ple-offload-backend file`) keeps the table on NVMe. CUDA graphs still work because the GPU walks the host page tables. Native pinned-host allocation OOMs the 48 GiB table on GB10.
- **QSA on sm_121:** stock SGLang gates TRT-LLM sparse decode to SM100. GB10 is SM121, so that call would fall into FA4 (does not compile) or, if the gate is widened, FlashInfer XQA (silent token-id-0 garbage). We leave the stock SM100 gate closed and route packed varlen decode through the #36845 2026-08-28 Triton kernel.
- **MTP:** the 31 draft tensors are still BF16 inside this NVFP4 pack. `--speculative-draft-model-quantization unquant` stops the draft inheriting `modelopt_fp4`. Draft logits use a 65536-id `--speculative-token-map` (`bench/draft_vocab/hot_tokens_64k.pt`); the target sampler is unchanged. `SPECULATIVE_TOKEN_MAP=off` restores the full draft head.
- **Non-root:** the container runs as your uid. Hugging Face cache and the PLE backing file stay host-owned.

Prefill uses the real QSA kernels (llama.cpp GGUF cannot). We did **not** benchmark vLLM on this box — see *What was tried and rejected* for why the comparison was not run.

## Requirements

- DGX Spark or other GB10 (aarch64, SM 12.1), Docker with NVIDIA runtime, ~230 GB free NVMe.
- Hugging Face token with access to the checkpoint (`HF_TOKEN`, or a `.env` in this directory).
- One GPU occupant. Unload anything else first.

## Run

```bash
# optional: point at an existing Hub cache
# export HF_CACHE=$HOME/.cache/huggingface

./scripts/prepare.sh          # pull image, patch qwen4_exp / QSA / spec_utils, download weights
./scripts/serve.sh            # :30000, ~10 min first load, ~10 min after
./scripts/smoke.sh            # health + "12*17" → 204
```

```bash
curl -sS http://127.0.0.1:30000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen38-flash-next-nvfp4-mtp","messages":[{"role":"user","content":"12*17"}],"max_tokens":2048}'
```

Stop: `./scripts/stop.sh`. Default bind is `127.0.0.1`. LAN: `BIND_ADDR=0.0.0.0 ./scripts/serve.sh`.

| Env | Default |
| --- | --- |
| `PORT` | `30000` |
| `HF_CACHE` | `$HF_HOME` or `~/.cache/huggingface` |
| `PLE_DIR` | `./data/ple` (sparse ~48 GB backing file) |
| `IMAGE` | `lmsysorg/sglang@sha256:9d2a843c706c74bc259c0d9abf360551eb2734e1e7d255ab012a6965f10480b6` (SGLang `4ccff141`; local alias `lmsysorg/sglang:dev-qwen38-next-local-4ccff14`) |

## Controlled experiments

Use `scripts/run_config.sh` in a detached session with a unique TAG and redirect
its output under `results/`. It now owns the whole lifecycle: inventory, original
image/source preservation, sampled checkpoint/PLE identity validation, llama-swap
unload, GPU-idle check, guarded startup, bounded benchmark suites, and graceful
shutdown. **The experiment server is stopped at the end**, including a failed
run; `scripts/serve.sh` remains the standalone serving entry point.

The shared `results/.bench.lock` covers restart and all suites. Benchmark-only
`scripts/bench_fast.sh` uses that same lock but does not own/restart/stop its
existing server. Tags cannot overwrite earlier evidence. Inspect
`results/<TAG>/suite_status.json` and `experiment_status.json`: process exit codes,
missing/malformed reports, failed quality/needle checks and invalid tool calls
all prevent a passing verdict. Suites have individual logs and JSON artifacts.

Default limits: startup 1500 seconds (`BOOT_WAIT`), each suite 900 seconds
(`SUITE_TIMEOUT`), each API request 300 seconds (`REQUEST_TIMEOUT`, total wall
clock including streaming). Deadlines kill the benchmark's process group.
`PROFILE=u0` restricts the harness to 8k/32k tests and at most 40 tool turns,
with a 32768-token text/chat preflight ceiling. The fixed small positional image
smoke remains enabled. This does not limit arbitrary callers to the server.

The non-root watchdog samples every five seconds. It stops the experiment after
15 seconds below 6 GiB MemAvailable, or below 0.5 GiB MemFree while MemAvailable
is below 10 GiB; more than 0.5 GiB new swap use also trips it. New NVIDIA Xids
trigger immediate shutdown. `NV_ERR_NO_MEMORY` in *new* kernel messages uses the
same 15-second sustain window: this host logs one-shot allocation failures during
CUDA graph capture even when the server becomes healthy. Matching journal lines
are stored on a trip. Free-memory gating avoids treating a large reclaimable file
cache as exhaustion. These U0 thresholds preserve margin above exhaustion while
allowing the historical ~1 GiB free baseline; they are not a validation of large
prefills. `MEMWATCH_AVAILABLE`, `MEMWATCH_FREE`, `MEMWATCH_FREE_GATE`, and
`MEMWATCH_SUSTAIN` configure the controller's policy. Watchdog process loss aborts
the active experiment. Memory samples and shutdown evidence stay in the tag's
results directory. `python3 tests/test_harness.py` covers the detector, lock,
deadlines, PLE identity, and suite verdicts.

PLE identity records live in `results/ple-identities`, not under the external
cache. The current guard supports the existing FP8 TP1 layout and requires its
backing file to exist. It checks every shard's first/middle/last byte windows,
records model/revision/config/index identity, and refuses reuse across registered
checkpoint identities. Sampling is not a full-file integrity checksum. A new
backend/checkpoint will need an explicit identity migration in its plan item.

## Boot time

The 48 GiB PLE table lives in a file-backed mmap that survives restarts
(`--ple-offload-backend file` on the U3 image). SGLang still re-copies it out of
the checkpoint on every boot, in shards, through `copy_ple_rows_to_tp_embedding`
unless `patches/ple_reuse.py` byte-samples each shard and skips the ones already
on disk. Native #37068 names the file differently; `patches/ple_file_compat.py`
keeps using `ple_table_{numel}_{nbytes}.bin` when that file is already present.
Only the **first** fill pays the 45–60 min copy:

```
PLE table: 128/128 shards already on disk (320001536 rows)
```

Second and later boots are **~10 min** end to end (U3 reuse boot 624 s). Set
`SGLANG_QWEN4_PLE_REUSE=0` to force the full copy. Empty `PLE_OFFLOAD_BACKEND`
must default to `file`; native pinned RAM OOMs.

## Tuning

`scripts/serve.sh` reads these; the defaults are the recipe.

| Env | Default | Why you would change it |
| --- | --- | --- |
| `MEMFRAC` | `0.95` | Lower hands UMA back to the page cache the PLE table is read from |
| `PREFILL` | `4096` | Chunked prefill size |
| `MAX_RUNNING` | `4` | Concurrent requests |
| `CONTEXT` | `262144` | Native rope limit |
| `MAX_TOTAL` | `524288` | KV token budget |
| `SPEC_STEPS` / `SPEC_TOPK` / `SPEC_DRAFT` | `3` / `1` / `4` | MTP depth; `SPEC=off` disables |
| `CUDA_GRAPH_MAX_BS` | unset | Cap decode graph captures. With `MAX_RUNNING=4`, stock already captures `bs=[1,2,3,4]` (pool size), not 256 |
| `CUDA_GRAPH_BS` | unset | Explicit `--cuda-graph-bs-decode` list (`1,2,3,4` or `1 2 3 4`). Mutually exclusive with `CUDA_GRAPH_MAX_BS` |
| `MAMBA_STRATEGY` | `extra_buffer` | Required for `page_size > 1`; guards MTP rewind vs GDN state |
| `PAGE_SIZE` | `64` | Ignored — compressed QSA pins it to 64 |
| `EXTRA_ARGS` | empty | Raw extra `sglang serve` flags |
| `SPECULATIVE_TOKEN_MAP` | `bench/draft_vocab/hot_tokens_64k.pt` | U6. 65536 draft token IDs. `off` / `0` restores the full draft head. |
| `PLE_OFFLOAD_BACKEND` | `file` when the native table module exists | Native #37068 defaults to pinned host RAM; that OOMs the 48 GiB table on GB10 |

## Client notes

- **Thinking is on by default and can be turned off**, despite what the cookbook
  says: `{"chat_template_kwargs": {"enable_thinking": false}}` works and is much
  shorter (3 tokens vs ~60 on a one-line arithmetic question).
- **`reasoning_effort` does nothing on this build.** low / medium / xhigh were
  measured at 59 / 71 / 64 completion tokens as a template kwarg and 59 / 66 as a
  top-level OpenAI field — no ordering and no trend, and the chat template
  reports no effort kwarg. Use `enable_thinking`, not effort.
- `usage.completion_tokens_details.reasoning_tokens` is always 0; reasoning is
  billed as content.
- Tool calls use the `qwen3_coder` parser and come back as OpenAI `tool_calls`.
- **Thinking on reduces tool-call frequency, by a variable amount.** Over
  identical 120-turn sessions, thinking off emitted a tool call on **119/120**
  turns every time, while thinking on gave **60/120** in one run and **101/120**
  in another. The spread between those two runs is larger than most differences
  in this README, so treat it as "fewer and unpredictable", not as a fixed ratio.
  Nothing is corrupted either way. An agent loop that treats "no tool call" as a
  stall should drive tool-heavy phases with `enable_thinking: false`.
- **The model can refuse to repeat something it was told to keep quiet.** In 120-turn
  runs the model is asked at turn 120 for a code planted at turn 3 alongside
  "Do not repeat unless asked" in `ops/secrets.md`. U2 thinking-off reprinted it;
  U2 thinking-on and both U3 thinking modes recalled the source and declined to
  print the value. Recall was intact; the harness fail is a refusal. Worth knowing
  if an agent stores credentials in its own transcript and later needs them back.

## Benchmarks

`bench/` holds the harness. All of it talks to the running server over the
OpenAI-compatible API.

| | |
| --- | --- |
| `bench/decode.py` | code + prose decode rate, thinking on and off |
| `bench/streams.py` | concurrent 1/2/4-stream decode (`STREAMS=1`) |
| `bench/mixedload.py` | 64k prefill arriving during two decodes; chunk-gap p50/p95/p99 (`MIXEDLOAD=1`) |
| `bench/quality.py` | math, tool call, executed code, multi-turn fact, positional vision, effort sweep |
| `bench/longctx.py` | 8k/32k needle and prefix-cache resend TTFT |
| `bench/agentic.py` | long-horizon tool-calling session; TTFT / decode / cache-hit banded by turn |
| `bench/bfcl.py` | fixed 100-case BFCL subset (AST, irrelevance, multi-turn) |
| `bench/gsm8k.py` | GSM8K sanity slice |
| `scripts/bench_tb.sh` | Terminal-Bench subset via Harbor |

**What this repo does NOT establish:** these numbers show the *serving stack* is
stable over long sessions — 120 turns, ~600 tool turns, no corruption, flat TTFT.
They do **not** establish that the model is good at complex multi-step coding.
The coding evidence here is one trivial executed function plus GSM8K; the
benchmark that would answer it (Terminal-Bench) cannot run on this hardware.
To find out, run `scripts/bench_tb.sh` from an x86 host against this server, or
point a real coding agent at the endpoint.

**Terminal-Bench does not run on a GB10.** The TB 2.x task images are
`linux/amd64` only and there is no qemu binfmt handler on this class of box, so
the task container exits 255 on platform mismatch. `scripts/bench_tb.sh` is
correct and works from an x86 host pointed at this server over the network. No
Terminal-Bench number in this repo was measured on a Spark.

## Measured (one GB10)

Single stream, `RadixArk/Qwen3.8-Flash-Next-NVFP4`, MTP 3/1/4, `trtllm_mha`
dense decode, QSA sparse decode via #36845 Triton, `triton` prefill, CUDA graphs,
radix cache on, **vision on**, 262k context, clock-capped GB10. U6 numbers
(`u6-draft-vocab-64k-20260908`, confirm `u6-draft-vocab-64k-confirm-20260908`)
are the current default (U3 pin plus 64k NEXTN token map). U3 (`u3-dev-4ccff14-20260908b`)
is the previous full-vocab draft head. U2 PLE commit
(`u2-ple-commit-20260907`) is the previous serving image. U1 Triton (`u1-triton-20260907`)
is the previous QSA-corrected baseline. August 28 tables below those blocks are
the historical widened-gate recipe.

### U6 current default (2026-09-08, 64k NEXTN token map)

Same U3 engine pin. `--speculative-token-map` slices the **draft** lm_head to
65536 rows (`bench/draft_vocab/hot_tokens_64k.pt`: 33 specials, 44301 corpus-ranked
IDs, 21202 id-order fill). Target vocabulary and sampling are unchanged; tokens
outside the subset still win through target verify. Held-out builder coverage
99.35%. This is a decode-bandwidth lever, not a KV or resident-weight saving
(MTP load `mem usage` 0.43–0.90 GiB vs U4a 0.60; KV still 524288).

Two boots: `u6-draft-vocab-64k-20260908` (full suite, 564 s) and confirm
`u6-draft-vocab-64k-confirm-20260908` (QUICK, 556 s). PLE `128/128` / 0 copied
both times. Quality **12/12** including `effort_thinking_off=24`. 8k/32k needles
PASS. 120-turn 0 invalid; late-recall refusals as in U3. Confirm soak was the
PROFILE=u3 3600 s default and was stopped after decode; not a U6 measurement.

| Decode median tok/s | thinking off | thinking on |
| --- | ---: | ---: |
| code EN (run 1) | **47.59** (45.86–48.70) | 35.09 (34.97–42.54) |
| code EN (confirm) | **47.57** (44.32–48.74) | 42.02 (38.00–42.95) |
| prose ES (run 1) | 21.12 (19.88–23.89) | 25.17 (23.99–26.95) |
| prose ES (confirm) | 22.13 (20.38–23.19) | 28.03 (27.90–29.50) |

Versus last accepted U4a thinking-off code **40.29** (38.06–40.94): **+18%** on
both launches, ranges do not overlap. 32k first-send 10.47 s (U4a 10.52 s).

| 120-turn (run 1) | wall | tools | invalid | late recall | decode bands |
| --- | ---: | ---: | ---: | --- | --- |
| thinking off | 122 s | 119/120 | 0 | refusal | 61.42 / 61.76 / 60.72 tok/s |
| thinking on | 346 s | 118/120 | 0 | refusal | 33.69 / 41.01 / 39.56 tok/s |

`SPECULATIVE_TOKEN_MAP=off` rolls this back. Smaller maps were not measured.

### U3 previous default (2026-09-08, SGLang `4ccff141`, full draft vocab)

Image digest `lmsysorg/sglang@sha256:9d2a843c706c74bc259c0d9abf360551eb2734e1e7d255ab012a6965f10480b6`
(commit `4ccff141`, MTP token-0 router [#38290](https://github.com/sgl-project/sglang/pull/38290),
native PLE file backend [#37068](https://github.com/sgl-project/sglang/pull/37068)).
U1 Triton overlays bundled KDA QSA. U2 ReplaySSM PLE commit stays. Prefetch stays
off (U4b: 32k cold TTFT 10.24 s on vs 10.23 s off). RSS trimming stays off (U4c:
mapping RSS ~0.4 GiB vs 8 GiB budget; 0 trim events in one hour). Filename compat
reuses `ple_table_51200245760_51200245760.bin`.

Boot 624 s, PLE `128/128 shards already on disk`, Triton SM121 QSA, ReplaySSM PLE
commit on the serving path. U4a second reuse boot (`u4a-reuse-boot-20260908`)
618 s, same `128/128` / 0 copied; native #37068 still rewrites the table without
`patches/ple_reuse.py`, so that overlay stays. Vision, tools, executed code, 8k/32k
needles all pass. 0 invalid tool calls. One-hour soak **2214/2214**. Isolated QSA
check: Triton/wrapper rel-L2 ~0 vs FP32, KDA ~0.00225, CUDA-graph replay 0.000.

Harness `experiment_status` is fail: quality 11/12 (`effort_thinking_off` answered
`28` not `24` at temperature 0) and both 120-turn late-recall checks were
disclosure-style refusals of `ops/secrets.md`, not forgotten codes. GSM8K was not
part of the U3 gate (last measured: U2 n=200 **193/200**).

| Decode median tok/s | thinking off | thinking on |
| --- | ---: | ---: |
| code EN | 38.99 (37.49–40.52) | 32.83 (31.66–34.17) |
| prose ES | 21.88 (21.27–22.69) | 24.17 (23.21–27.95) |

U2 was 38.56 / 35.74 code and 22.92 / 26.91 prose. Thinking-off overlaps U2;
thinking-on code is ~8% slower. This is a new engine image plus #38290, not a
speed claim.

| Longctx | first s | resend s | speedup | needle |
| --- | ---: | ---: | ---: | --- |
| 8k (5885 tok) | 4.34 | 0.59 | 7.3× | PASS |
| 32k (23555 tok) | 10.44 | 0.50 | 21.1× | PASS |

| 120-turn | wall | tools | invalid | late recall | decode bands |
| --- | ---: | ---: | ---: | --- | --- |
| thinking off | 172 s | 119/120 | 0 | refusal | 49.45 / 49.45 / 49.28 tok/s |
| thinking on | 497 s | 106/120 | 0 | refusal | 33.31 / 30.97 / 29.95 tok/s |

**Not a 120k+ test.** End-of-run `spec_accept_length` 3.3 after soak.

### U2 previous image (2026-09-07, PLE commit after ReplaySSM verify)

`--enable-gdn-replayssm-spec` is a deprecated alias of
`--enable-linear-replayssm-spec`. On this image that sets `replayssm_spec_fold`,
so `commit_mamba_states_after_verify` used to return before rolling PLE n-gram
and short-conv state. `patches/replayssm_ple_commit.py` applies the #37794
`spec_utils` hunk to both ReplaySSM early-return branches. NGRAM stays refused.

Boot 574 s, PLE `128/128 shards already on disk`, Triton SM121 QSA, log line
`ReplaySSM verify: committing PLE n-gram/short-conv state`. Dedicated
accept/reject + recall suite 4/4. GSM8K n=200 thinking-off **193/200 (96.5%)**.
Vision, tools, executed code, 8k/32k needles all pass. 0 invalid tool calls.

Harness `experiment_status` is fail: quality 11/12 (`effort_thinking_off` answered
`25` not `24` at temperature 0) and thinking-on 120-turn late recall was a
disclosure-style refusal, not a forgotten code. Expanded GSM8K and the dedicated
PLE recall are the gates that matter for this bug; rolling the patch back would
restore the freeze. Thinking-on tool frequency this run was 76/120 (U1 was
119/120; already recorded as unpredictable).

| Decode median tok/s | thinking off | thinking on |
| --- | ---: | ---: |
| code EN | 38.56 (36.9–40.65) | 35.74 (32.48–37.56) |
| prose ES | 22.92 (21.24–23.19) | 26.91 (25.26–27.56) |

U1 was 40.63 / 31.91 code and 23.42 / 25.66 prose. Thinking-off code overlaps
U1's range; this is a correctness change, not a speed claim.

| Longctx | first s | resend s | speedup | needle |
| --- | ---: | ---: | ---: | --- |
| 8k (5885 tok) | 3.66 | 0.57 | 6.4× | PASS |
| 32k (23555 tok) | 10.40 | 0.50 | 20.7× | PASS |

| 120-turn | wall | tools | invalid | late recall | decode bands |
| --- | ---: | ---: | ---: | --- | --- |
| thinking off | 127 s | 119/120 | 0 | PASS | 57.4 / 57.5 / 57.4 tok/s |
| thinking on | 677 s | 76/120 | 0 | refusal | 31.6 / 29.7 / 25.6 tok/s |

**Not a 120k+ test.** End-of-run `spec_accept_length` 3.725 after GSM8K.

### U1 corrected baseline (2026-09-07, Triton SM121)

Quality 12/12, 8k and 32k needles exact, 120-turn late recall PASS both thinking
modes, 0 invalid tool calls. Isolated kernel check: TRT-LLM gated off, wrapper
`_qsa_sm121_triton_varlen`, Triton rel-L2 ~0 vs FP32, CUDA-graph replay 0.000.
Boot 580 s with `128/128 shards already on disk`. Decode medians: code 40.63
(39.37–41.31) off / 31.91 (31.07–34.22) on; prose 23.42 (19.64–23.64) off /
25.66 (24.31–27.51) on — within the usual ±5% of U0.

| 120-turn | TTFT bands | decode bands | tools | wall | ctx |
| --- | ---: | ---: | ---: | ---: | ---: |
| thinking off | 0.57 / 0.57 / 0.57 s | 49.2 / 49.0 / 48.8 tok/s | 119/120 | 166 s | 14605 |
| thinking on | 0.53 / 0.49 / 0.44 s | 34.1 / 30.5 / 33.1 tok/s | 119/120 | 539 s | 20217 |

32k needle 10.40 s first / 0.50 s resend (20.9×). **Not a 120k+ test.** The KDA
overlay on this same image failed that 32k needle with 64× token id 0.

### Long-horizon agentic session — what this recipe is for

120 turns, one tool call per turn, growing context, `bench/agentic.py`.

Measured on the U3 shipped config (`u3-dev-4ccff14-20260908b`), thinking **off**:

| turns | TTFT | decode | cache hit | context |
| --- | ---: | ---: | ---: | ---: |
| 1–40 | 0.57 s | 49.45 tok/s | 96.2% | 2.7k |
| 41–80 | 0.57 s | 49.45 tok/s | 98.5% | 7.5k |
| 81–120 | 0.57 s | 49.28 tok/s | **99.1%** | 12.2k |

119/120 turns emitted a tool call, **0 invalid tool calls**, 172 s wall clock,
final context 14577 tokens. **TTFT is flat as context grows** — that is the radix
cache absorbing the resend pattern. Decode here is tool-JSON-heavy and is not the
code/prose table. Late recall refused to reprint the planted `ops/secrets.md`
code; the answer still cited that file (disclosure, not forgotten state). U2
thinking-off late recall passed.

Same session, thinking **on**:

| turns | TTFT | decode | cache hit | context |
| --- | ---: | ---: | ---: | ---: |
| 1–40 | 0.39 s | 33.31 tok/s | 94.5% | 2.8k |
| 41–80 | 0.46 s | 30.97 tok/s | 96.4% | 7.7k |
| 81–120 | 0.42 s | 29.95 tok/s | **98.7%** | 12.5k |

0 invalid tool calls, 497 s wall clock. Tool frequency 106/120 this run (U2 was
76/120, U1 119/120). Late recall was again a disclosure-style refusal.

Repeated 40-turn runs land at 48–52 tok/s decode, 0.56–0.60 s TTFT and
97.3–97.5% cache hit, with **0 invalid tool calls** every time.

### Decode

| | thinking off | thinking on |
| --- | ---: | ---: |
| code (EN) | **~40 tok/s** (39.3 / 40.4 / 41.5 over three runs) | ~32 tok/s |
| prose (ES) | ~22 tok/s | ~26 tok/s |

Run-to-run spread on this box is roughly ±5%, so single decode figures are not
meaningful to one decimal place. The baseline (without
`--enable-gdn-replayssm-spec`) measured 38.3 / 38.6 / 39.0 over three runs, so the
flag is worth **~4–5%** on code decode, not the 7.6% a best-vs-best comparison
would suggest.

`spec_accept_length` is the more stable evidence: **3.80 without the flag,
3.93–3.95 with it**, out of a 4-token draft, consistent across every run.

### Long context

| | |
| --- | ---: |
| needle 8k / 32k / 40k | **PASS / PASS / PASS** |
| TTFT 8k / 32k | 3.98 s / 10.37 s (~2.3k tok/s prefill) |
| prefix-cache resend 8k | 3.98 s → 0.57 s (**7.0×**) |
| prefix-cache resend 32k | 10.37 s → 0.49 s (**21.0×**) |

### Quality

| | |
| --- | ---: |
| smoke suite (math, tools, executed code, multi-turn, **vision**, effort) | **11/12** (U3 and U2) / 12/12 (U1) |
| MTP `spec_accept_length` | **3.3** end-of-U3 soak (U2 was 3.73 after GSM8K) |
| GSM8K, n=200 first official test, thinking off | **193/200 (96.5%)** |
| GSM8K, n=20, thinking off (August) | **19/20 (95%)** |
| BFCL fixed subset, 80 single-turn cases | **72.5%** |
| — simple / multiple / parallel | 86.7% / 73.3% / 66.7% |
| — irrelevance / live_irrelevance | 80.0% / **30.0%** |
| invalid tool calls, ~600 tool turns total | **0** |

U3 and U2 quality 11/12 is `effort_thinking_off` at temperature 0 (U3 answered
`28`, U2 answered `25`, expected `24`). Math `12×17`, tools, vision and multi-turn fact passed. GSM8K
n=200 is the expanded gate for that miss and was last run on U2.

`live_irrelevance` at 30% is the one weak spot: the model reaches for a tool when
the right move is to decline. Curated `irrelevance` is 80%, so it is the harder
live split specifically. An agent loop that trusts every tool call it receives
will waste steps on it.

BFCL `multi_turn_base` is **not reported** — scoring it needs BFCL's stateful
sandbox, and a mock tool backend measures the harness, not the model.

### What was tried and rejected

| Change | Result |
| --- | --- |
| `--mem-fraction-static 0.85` pack | No speed gain, **−39% KV budget**, accept 3.80 → 3.50 |
| `--disable-radix-cache` | TTFT 0.57 s → **2.32 s and climbing** at turn 40, 32k resend 21× → 1.0×, +73% wall clock, **no reliability benefit** |
| `--linear-attn-backend` (flashinfer / cutedsl / nvidia_kda) | **Unreachable.** Compressed QSA pins `page_size=64`, which forces `extra_buffer`, which rejects these backends. Only `--disable-radix-cache` unlocks them, and that costs far more than they give |
| `--mamba-full-memory-ratio 0.3` | Inert at 262k — `--max-total-tokens` binds before memory does |
| `PREFILL=8192` + graph bundle | Worst prose decode in the sweep, one quality failure, unattributed 3-flag bundle |
| MTP depth changes | Accept length is already 3.93–3.95 / 4.0; no headroom |
| **512k context** | Boots, but yields **201984 KV tokens — below the 262k default**. The published Qwen static-YaRN recipe targets a `rope_scaling` field this checkpoint does not have (it is sectioned **mrope** under `text_config.rope_parameters`), and `rope_type` stays `default` after the override. **Not achieved; 262k is the only supported context** |
| vLLM | **Untested.** It needs `--no-enable-prefix-caching` on sm_121, and prefix caching is worth 21× warm prefill here |
| `MAX_RUNNING=1` | No gain, **+21% wall clock** on a 40-turn session |
| `PREFILL=1024` | 32k TTFT **12.31 s vs 10.37 s** at 4096 — smaller chunks cost TTFT |
| `--language-only` (vision off) | **Does not disable vision.** It is for encoder *disaggregation*; with no encoder service configured the local tower still runs and still answers image questions correctly. 89 MiB and 0 KV difference — there was never a headroom argument |
| Widened TRT-LLM QSA gate (`is_sm120_supported()`) | **Rejected.** On SM121 that call is XQA, not trtllm-gen. Silent token-id-0 loops from ~120k. Retired by #36806/#36845 |
| #36845 KDA SM121 overlay on this image | **Rejected for serving.** Isolated rel-L2 ≤ 0.0024 and CUDA-graph replay passed; a 32k needle then returned 64× `!`. Triton-only on the same stack passed |
| #37794 NGRAM on Qwen4-Exp | **Not ported.** U2 took only the ReplaySSM PLE-commit hunk. `_prepare_ple_batch` still refuses NGRAM |

### Vision

Vision is **on** and verified by a test that cannot be passed blind: a
four-quadrant image (red / yellow / green / blue) with the model asked for one
named quadrant. All four positions answered correctly.

An earlier version of this test showed a solid red square and asked for the
dominant colour — a text-only server passes that by guessing "Red". If you write
a vision smoke test, make it positional.

A true vision-off comparison is **not possible** with this checkpoint and image:
`--language-only` does not remove the tower (see the table above). It would need
a text-only checkpoint or a disaggregated encoder deployment.

### An unexplained observation, recorded as such

One config — 512k context with four other changes — posted agentic decode of
**56–60 tok/s** against ~50–52 everywhere else. Each of its five variables was
then isolated (`MEMFRAC`, `--mamba-full-memory-ratio`, `MAX_RUNNING`, `PREFILL`,
page-cache residency) and **none reproduces it**. It is a single observation with
no surviving explanation, it is not a tuning lead, and it is not an argument for
512k. Details in `RESEARCH_LOG.md`.


## Credits

- Model: Qwen / Alibaba. NVFP4: [RadixArk](https://huggingface.co/RadixArk/Qwen3.8-Flash-Next-NVFP4).
- Engine: [SGLang](https://github.com/sgl-project/sglang) (`qwen4_exp`).
- PLE mmap and SM121 QSA Triton backport: adapted from [hashd1ve/qwen38-flash-next-one-dgx-spark](https://github.com/hashd1ve/qwen38-flash-next-one-dgx-spark) (MIT) and [sglang#36845](https://github.com/sgl-project/sglang/pull/36845) (Apache-2.0).
- Native PLE file backend: [sglang#37068](https://github.com/sgl-project/sglang/pull/37068) with [sglang#38123](https://github.com/sgl-project/sglang/pull/38123). Recipe filename reuse: `patches/ple_file_compat.py`.

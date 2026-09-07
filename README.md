# Qwen3.8-Flash-Next on one DGX Spark (SGLang)

**September 7 upstream campaign:** **U0–U2 are accepted.** Sparse decode on GB10
uses the 2026-08-28 Triton kernel from [SGLang #36845](https://github.com/sgl-project/sglang/pull/36845),
not a widened TRT-LLM/XQA gate. ReplaySSM verify commits PLE n-gram/short-conv
state ([#37794](https://github.com/sgl-project/sglang/pull/37794) `spec_utils` hunk
only; that PR's NGRAM feature is not ported). The later KDA overlay from #36845
passed isolated tensor replay here and then emitted token-id 0 on a 32k needle;
it is not the serving default. 8k/32k needles pass. 120k–210k needles are still
unmeasured on this box. See the [staged plan](UPSTREAM_PLAN.md).

**This repo is how you run Qwen3.8-Flash-Next performantly on a single NVIDIA DGX Spark — or any other GB10 machine (ASUS Ascent GX10, MSI Atom, …).**

Serve [`RadixArk/Qwen3.8-Flash-Next-NVFP4`](https://huggingface.co/RadixArk/Qwen3.8-Flash-Next-NVFP4) with **SGLang** (128 GB unified memory). Native 262k context, MTP speculative decode, CUDA graphs.

- **What:** 125B MoE + 51B n-gram PLE, 6B active, NVFP4 routed experts. The checkpoint is ~135 GB; a Spark has 128 GB. Stock `--ple-offload-embedding` pins the 48 GB PLE table in the *same* UMA pool, so it does not fit.
- **Why mmap:** a token reads 16 PLE rows (~2.5 KB). File-backed `torch.from_file` keeps the table on NVMe. CUDA graphs still work because the GPU walks the host page tables.
- **QSA on sm_121:** stock SGLang gates TRT-LLM sparse decode to SM100. GB10 is SM121, so that call would fall into FA4 (does not compile) or, if the gate is widened, FlashInfer XQA (silent token-id-0 garbage). We leave the stock SM100 gate closed and route packed varlen decode through the #36845 2026-08-28 Triton kernel.
- **MTP:** the 31 draft tensors are still BF16 inside this NVFP4 pack. `--speculative-draft-model-quantization unquant` stops the draft inheriting `modelopt_fp4`.
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
| `IMAGE` | `lmsysorg/sglang:qwen38flashnext` |

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

The 48 GiB PLE table lives in a file-backed mmap that survives restarts, but
SGLang re-copies it out of the checkpoint on every boot, in shards, through
`copy_ple_rows_to_tp_embedding`. On this box that copy alone runs 45–60 min.
`patches/ple_reuse.py` byte-samples each shard against the checkpoint and skips
the ones already on disk, so only the **first** boot pays the fill:

```
PLE table: 128/128 shards already on disk (320001536 rows)
```

Second and later boots are **~10 min** end to end. Set
`SGLANG_QWEN4_PLE_REUSE=0` to force the full copy.

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
| `CUDA_GRAPH_MAX_BS` | unset | Stock captures decode graphs up to bs 256 |
| `MAMBA_STRATEGY` | `extra_buffer` | Required for `page_size > 1`; guards MTP rewind vs GDN state |
| `PAGE_SIZE` | `64` | Ignored — compressed QSA pins it to 64 |
| `EXTRA_ARGS` | empty | Raw extra `sglang serve` flags |

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
- **Thinking on can refuse to repeat something it was told to keep quiet.** In one
  120-turn run the model was asked at turn 120 for a code it had been given at
  turn 3 alongside the note "Do not repeat unless asked". With thinking off it
  answered. With thinking on it recalled the source correctly but declined to
  repeat the value, reasoning about disclosure first. Recall was intact; the
  behaviour was a refusal. Worth knowing if an agent stores credentials in its own
  transcript and later needs them back.

## Benchmarks

`bench/` holds the harness. All of it talks to the running server over the
OpenAI-compatible API.

| | |
| --- | --- |
| `bench/decode.py` | code + prose decode rate, thinking on and off |
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
radix cache on, **vision on**, 262k context, clock-capped GB10. U2 numbers
(`u2-ple-commit-20260907`) are the current default. U1 Triton (`u1-triton-20260907`)
is the previous QSA-corrected baseline. August 28 tables below those blocks are
the historical widened-gate recipe.

### U2 current default (2026-09-07, PLE commit after ReplaySSM verify)

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

Measured on the U2 shipped config (`u2-ple-commit-20260907`), thinking **off**:

| turns | TTFT | decode | cache hit | context |
| --- | ---: | ---: | ---: | ---: |
| 1–40 | 0.56 s | 57.4 tok/s | 95.6% | 2.5k |
| 41–80 | 0.56 s | 57.5 tok/s | 98.4% | 6.7k |
| 81–120 | 0.57 s | 57.4 tok/s | **99.0%** | 10.9k |

119/120 turns emitted a tool call, **0 invalid tool calls**, a fact planted at
turn 3 was recalled correctly at turn 120, 127 s wall clock, final context 12985
tokens. **TTFT is flat as context grows** — that is the radix cache absorbing the
resend pattern. Decode here is tool-JSON-heavy and is not the code/prose table.

Same session, thinking **on**:

| turns | TTFT | decode | cache hit | context |
| --- | ---: | ---: | ---: | ---: |
| 1–40 | 0.39 s | 31.6 tok/s | 94.7% | 2.8k |
| 41–80 | 0.38 s | 29.7 tok/s | 98.0% | 7.9k |
| 81–120 | 0.33 s | 25.6 tok/s | **99.3%** | 13.7k |

0 invalid tool calls, 677 s wall clock. Tool frequency 76/120 this run (U1 was
119/120). Late recall was a disclosure-style refusal of the planted code, not a
forgotten value; the dedicated PLE-spec recall test passed.

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
| smoke suite (math, tools, executed code, multi-turn, **vision**, effort) | **11/12** (U2) / 12/12 (U1) |
| MTP `spec_accept_length` | **3.73** end-of-U2 mixed load (was 3.93–3.95) |
| GSM8K, n=200 first official test, thinking off | **193/200 (96.5%)** |
| GSM8K, n=20, thinking off (August) | **19/20 (95%)** |
| BFCL fixed subset, 80 single-turn cases | **72.5%** |
| — simple / multiple / parallel | 86.7% / 73.3% / 66.7% |
| — irrelevance / live_irrelevance | 80.0% / **30.0%** |
| invalid tool calls, ~600 tool turns total | **0** |

U2 quality 11/12 is `effort_thinking_off` answering `25` instead of `24` at
temperature 0. Math `12×17`, tools, vision and multi-turn fact passed. GSM8K
n=200 is the expanded gate for that miss.

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

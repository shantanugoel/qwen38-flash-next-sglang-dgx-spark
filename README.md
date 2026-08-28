# Qwen3.8-Flash-Next on one DGX Spark (SGLang)

**This repo is how you run Qwen3.8-Flash-Next performantly on a single NVIDIA DGX Spark — or any other GB10 machine (ASUS Ascent GX10, MSI Atom, …).**

Serve [`RadixArk/Qwen3.8-Flash-Next-NVFP4`](https://huggingface.co/RadixArk/Qwen3.8-Flash-Next-NVFP4) with **SGLang** (128 GB unified memory). Native 262k context, MTP speculative decode, CUDA graphs.

- **What:** 125B MoE + 51B n-gram PLE, 6B active, NVFP4 routed experts. The checkpoint is ~135 GB; a Spark has 128 GB. Stock `--ple-offload-embedding` pins the 48 GB PLE table in the *same* UMA pool, so it does not fit.
- **Why mmap:** a token reads 16 PLE rows (~2.5 KB). File-backed `torch.from_file` keeps the table on NVMe. CUDA graphs still work because the GPU walks the host page tables.
- **QSA on sm_121:** stock SGLang gates the TRT-LLM sparse decode kernel to SM100. GB10 is SM121, so QSA falls into an FA4 path that does not compile. We widen that gate and (if present) drop a Python-SDPA intercept.
- **MTP:** the 31 draft tensors are still BF16 inside this NVFP4 pack. `--speculative-draft-model-quantization unquant` stops the draft inheriting `modelopt_fp4`.
- **Non-root:** the container runs as your uid. Hugging Face cache and the PLE backing file stay host-owned.

Prefill uses the real QSA kernels (llama.cpp GGUF cannot). Decode with MTP is faster than the vLLM mmap recipe on the same checkpoint (~27 tok/s).

## Requirements

- DGX Spark or other GB10 (aarch64, SM 12.1), Docker with NVIDIA runtime, ~230 GB free NVMe.
- Hugging Face token with access to the checkpoint (`HF_TOKEN`, or a `.env` in this directory).
- One GPU occupant. Unload anything else first.

## Run

```bash
# optional: point at an existing Hub cache
# export HF_CACHE=$HOME/.cache/huggingface

./scripts/prepare.sh          # pull image, patch two files, download weights
./scripts/serve.sh            # :30000, ~8–20 min first load
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
- **Thinking on halves tool-call frequency.** Over an identical 120-turn session,
  119/120 turns emitted a tool call with thinking off, but only **60/120** with it
  on — the model answers in prose instead. Nothing is corrupted and recall still
  works, but an agent loop that treats "no tool call" as a stall will feel steps
  being dropped. Drive tool-heavy phases with `enable_thinking: false` and leave
  thinking on for reasoning turns.

## Benchmarks

`bench/` holds the harness. All of it talks to the running server over the
OpenAI-compatible API.

| | |
| --- | --- |
| `bench/decode.py` | code + prose decode rate, thinking on and off |
| `bench/quality.py` | math, tool call, executed code, multi-turn fact, vision, effort sweep |
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

## Measured (one GB10, 2026-08-28)

Single stream, `RadixArk/Qwen3.8-Flash-Next-NVFP4`, MTP 3/1/4, `trtllm_mha`
decode, `triton` prefill, CUDA graphs, radix cache on, **vision on**, 262k
context, clock-capped GB10. Every number below is from `bench/` in this repo.

### Long-horizon agentic session — what this recipe is for

120 turns, one tool call per turn, growing context, `bench/agentic.py`.

| turns | TTFT | decode | cache hit | context |
| --- | ---: | ---: | ---: | ---: |
| 1–40 | 0.55 s | 47.8 tok/s | 95.5% | 2.5k |
| 41–80 | 0.58 s | 41.7 tok/s | 98.4% | 7.1k |
| 81–120 | 0.57 s | 45.1 tok/s | **99.0%** | 11.8k |

Repeated 40-turn runs of the shipped config land at 48–52 tok/s decode,
0.56–0.60 s TTFT and 97.3–97.5% cache hit, with **0 invalid tool calls** every
time.

**TTFT is flat as context grows.** 119 tool turns, **0 invalid tool calls**, and a
fact planted at turn 3 recalled correctly at turn 120.

With **thinking on**, same session: TTFT 0.39 → 0.29 s, decode 32 → 25 tok/s,
cache hit 99.4%, 0 invalid tool calls, late recall correct — but only **60 of 120
turns emit a tool call** versus 119 with thinking off. See *Client notes*.

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
| smoke suite (math, tools, executed code, multi-turn, **vision**, effort) | **12/12** |
| MTP `spec_accept_length` | **3.95 / 4.0** |
| GSM8K, n=20, thinking off | **19/20 (95%)** |
| BFCL fixed subset, 80 single-turn cases | **72.5%** |
| — simple / multiple / parallel | 86.7% / 73.3% / 66.7% |
| — irrelevance / live_irrelevance | 80.0% / **30.0%** |
| invalid tool calls, ~600 tool turns total | **0** |

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
| MTP depth changes | Accept length is already 3.95/4.0; no headroom |
| **512k context** | Boots, but yields **201984 KV tokens — below the 262k default**. The published Qwen static-YaRN recipe targets a `rope_scaling` field this checkpoint does not have (it is sectioned **mrope** under `text_config.rope_parameters`), and `rope_type` stays `default` after the override. **Not achieved; 262k is the only supported context** |
| vLLM | **Untested.** It needs `--no-enable-prefix-caching` on sm_121, and prefix caching is worth 21× warm prefill here |
| `MAX_RUNNING=1` | No gain, **+21% wall clock** on a 40-turn session |
| `PREFILL=1024` | 32k TTFT **12.31 s vs 10.37 s** at 4096 — smaller chunks cost TTFT |
| `--language-only` (vision off) | **Does not disable vision.** It is for encoder *disaggregation*; with no encoder service configured the local tower still runs and still answers image questions correctly. 89 MiB and 0 KV difference — there was never a headroom argument |

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
- PLE mmap + SM120 QSA gate: adapted from [hashd1ve/qwen38-flash-next-one-dgx-spark](https://github.com/hashd1ve/qwen38-flash-next-one-dgx-spark) (MIT).

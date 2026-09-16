# Qwen3.8-Flash-Next on one DGX Spark (SGLang)

Run [`RadixArk/Qwen3.8-Flash-Next-NVFP4`](https://huggingface.co/RadixArk/Qwen3.8-Flash-Next-NVFP4)
on a single NVIDIA DGX Spark, or any other GB10 machine (ASUS Ascent GX10, MSI Atom, …),
with SGLang. You get the full native 262k context, MTP speculative decoding, CUDA graphs,
vision, tool calling, and an OpenAI-compatible API.

## Performance at a glance

Single stream on one GB10, default settings, thinking off unless noted.

| | |
| --- | ---: |
| Decode, code | **~47 tok/s** (~35–42 with thinking on) |
| Decode, prose | ~22 tok/s |
| Decode, agentic tool-calling session | ~68 tok/s |
| Time to first token, 32k prompt (cold / cached prefix) | 10.0 s / 0.4 s |
| Time to first token, 128k prompt (cold / cached prefix) | 38–41 s / 0.4 s |
| Time to first token, fresh 32k / 128k **document** (cold) | 19–27 s / 2.5–5 min |
| 4 concurrent streams, aggregate | ~130 tok/s |
| Cold start (embedding table already built) | ~8.5–9 min |
| GSM8K (n=200) | 97.5% |
| BFCL subset (80 single-turn cases) | 72.5% |
| Invalid tool calls across ~600 tool turns | 0 |

Long agent sessions stay fast: over a 120-turn tool-calling session, TTFT and decode
speed stay flat as the context grows, because the prefix cache hits ~99%. A 24-hour
mixed-load soak ran 8,879 requests with no errors and no slowdown.

### Two optional modes

Both are off by default. Each is one setting plus, for the first, a one-off local build.

| | Default | FP8 hybrid weights | 512k context |
| --- | ---: | ---: | ---: |
| Decode, code | ~47 tok/s | **~55 tok/s** | ~45 tok/s |
| Decode, prose | ~22 tok/s | **~25 tok/s** | ~21 tok/s |
| Decode, agentic session | ~68 tok/s | **~76 tok/s** | — |
| Single stream / 4 streams | 48 / 130 tok/s | **56 / 137 tok/s** | — |
| Cold start | ~8.5–9 min | **~7.8 min** | ~8.6 min |
| Longest prompt with verified recall | 262k limit | 262k limit | **294k tokens** |
| GSM8K (n=200) | 97.5% | 98.0% | not run (12/12 on the short suite) |
| Cost | — | ~3% slower prefill; 13 GB disk + a copy of the embedding table | 4–10% slower decode |

`scripts/build_fp8_hybrid_snapshot.py` builds the FP8 hybrid checkpoint (~30 min, one
off); 512k is `CONTEXT=524288` with an FP8 KV cache. Details for both are under
[Configuration](#configuration).

Full results, methodology and everything that was tried: [docs/BENCHMARKS.md](docs/BENCHMARKS.md).

## Requirements

- DGX Spark or other GB10 machine (aarch64), Docker with the NVIDIA runtime.
- ~230 GB free NVMe.
- A Hugging Face token with access to the checkpoint, as `HF_TOKEN` or in a `.env`
  file in this directory.
- Nothing else using the GPU.

## Quick start

```bash
./scripts/prepare.sh   # pull the SGLang image, apply patches, download weights
./scripts/serve.sh     # start the server on 127.0.0.1:30000
./scripts/smoke.sh     # health check + a quick "12*17" test
```

The **first** start takes about an hour: it writes a ~48 GB embedding table to disk
(`./data/ple`). Every later start reuses that file and takes about 10 minutes.

If you already have the weights in a Hugging Face cache, point at it first:
`export HF_CACHE=$HOME/.cache/huggingface`.

Stop the server with `./scripts/stop.sh`.

## Using the server

```bash
curl -sS http://127.0.0.1:30000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen38-flash-next-nvfp4-mtp","messages":[{"role":"user","content":"12*17"}],"max_tokens":2048}'
```

Any OpenAI-compatible client works. The server listens on localhost only; to expose it
on your LAN, start it with `BIND_ADDR=0.0.0.0 ./scripts/serve.sh`.

Things worth knowing:

- **Thinking is on by default.** Turn it off per request with
  `"chat_template_kwargs": {"enable_thinking": false}`. Answers get much shorter and
  decode is faster.
- **Use thinking off for tool-heavy agent loops.** With thinking off the model calls a
  tool on nearly every turn; with thinking on it calls tools less often and less
  predictably. Tool calls are well-formed either way.
- **`reasoning_effort` has no effect.** Use `enable_thinking` instead.
- `usage.completion_tokens_details.reasoning_tokens` is always 0; reasoning tokens are
  counted as regular completion tokens.
- Tool calls come back as standard OpenAI `tool_calls`.
- Vision is enabled.
- **Set a sane `max_tokens` in agent clients.** A request whose client disconnects keeps
  decoding for a while before the server reaps it (up to ~100 output steps in a 24 h
  soak, sglang#36876). It always cleared, but a bounded `max_tokens` (e.g. 2–4k) caps
  how long an abandoned request can hold one of the 4 slots.

## Configuration

`scripts/serve.sh` reads these environment variables. The defaults are the tuned recipe;
most people only need the first few.

| Variable | Default | Notes |
| --- | --- | --- |
| `PORT` | `30000` | |
| `BIND_ADDR` | `127.0.0.1` | `0.0.0.0` to serve on the network |
| `HF_CACHE` | `$HF_HOME` or `~/.cache/huggingface` | Where weights are downloaded |
| `PLE_DIR` | `./data/ple` | Location of the ~48 GB on-disk embedding table |
| `MAX_RUNNING` | `4` | Concurrent requests |
| `CONTEXT` | `262144` | Maximum context length (262k is the maximum supported) |
| `MAX_TOTAL` | `524288` | KV cache token budget |
| `MEMFRAC` | `0.95` | Fraction of memory SGLang reserves |
| `PREFILL` | `4096` | Chunked prefill size |
| `SPEC` | on | `SPEC=off` disables speculative decoding |
| `SPEC_STEPS` / `SPEC_TOPK` / `SPEC_DRAFT` | `3` / `1` / `4` | Speculative decoding depth |
| `EXTRA_ARGS` | empty | Extra flags passed straight to `sglang serve` |

<details>
<summary>Advanced options</summary>

| Variable | Default | Notes |
| --- | --- | --- |
| `SPECULATIVE_TOKEN_MAP` | `bench/draft_vocab/hot_tokens_64k.pt` | Reduced draft vocabulary; `off` uses the full draft head |
| `CUDA_GRAPH_MAX_BS` | unset | Cap decode CUDA graph batch size |
| `CUDA_GRAPH_BS` | unset | Explicit decode batch-size list (`1,2,3,4`); don't combine with `CUDA_GRAPH_MAX_BS` |
| `QUANTIZATION` | `modelopt_fp4` | Use `modelopt_mixed` for NVIDIA's NVFP4 checkpoint |
| `SPEC_DRAFT_QUANT` | `unquant` | Use `auto` for checkpoints with a quantized MTP head (e.g. NVIDIA's) |
| `MOE_RUNNER_BACKEND` | unset (auto) | Set `flashinfer_cutlass` for NVIDIA's checkpoint |
| `MAMBA_SSM_DTYPE` | `float32` | Keep at fp32; other dtypes degrade speculative-decode accuracy |
| `MAMBA_STRATEGY` | `extra_buffer` | Required by this model's attention layout |
| `PLE_OFFLOAD_BACKEND` | `file` | Keep as `file`; in-RAM storage runs out of memory |
| `SGLANG_QWEN4_PLE_REUSE` | `1` | `0` forces the on-disk embedding table to be rewritten |
| `SGLANG_QWEN4_PLE_FILE_PREFETCH` | `1` | Page-prefetch for the embedding table. Keep on: it makes a fresh long document 2–5x faster to prefill |
| `SGLANG_CHECKPOINT_KEY_FILTER` | `1` | `0` makes the speculative draft re-read all 206 checkpoint files (~55 s slower boot) |
| `IMAGE` | pinned SGLang `8874c51a` digest | Container image |

**Longer context (optional).** `CONTEXT=524288 MAX_TOTAL=524288
SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1 EXTRA_ARGS="--kv-cache-dtype fp8_e4m3"` boots
and recalls a planted fact at 294k tokens, with roughly 6 GB more memory headroom. It
costs about 4–10% decode speed, and a single prompt still cannot go much past ~340k
tokens, so 262k stays the default.

**Faster decode (optional, ~30 min to build).** `scripts/build_fp8_hybrid_snapshot.py`
writes a sibling checkpoint whose attention and GDN projections are 128x128 block FP8,
leaving the experts and `lm_head` untouched. Served with `--quantization modelopt_mixed
--fp8-gemm-backend triton`, it decodes 13–21% faster (code 54.6 tok/s, agentic 76 tok/s),
boots faster, and scored 196/200 on GSM8K against 195/200 for the stock weights. It costs
~3% on prefill and needs ~13 GB of disk plus its own copy of the embedding table. See
[docs/BENCHMARKS.md](docs/BENCHMARKS.md).

`QSA_PREFILL_SELECTION=1 ./scripts/prepare.sh` enables an experimental prefill kernel
([sglang#38209](https://github.com/sgl-project/sglang/pull/38209)). It is correct but
gave no measurable speedup, so it is off by default.

</details>

## Limitations

- **262k is the maximum context.** 512k was attempted and does not work with this
  checkpoint.
- **Long prompts are slow to prefill.** Prefill speed drops as context grows, and new
  documents are slower than repeated text because the PLE table has to be read from
  disk: with the default page prefetch, a fresh ~128k-token document takes ~2.5–5 min
  before the first token (~20–30 s at 32k). Follow-up requests that share a prefix are
  fast thanks to the prefix cache.
- **Very long fresh prompts can run out of memory.** The KV cache is committed as it
  fills, and on this unified-memory box a fresh prompt past ~150–190k tokens exhausts
  the remaining headroom. Keep single prompts well under the 262k limit.
- **A long prefill stalls other streams.** While one request is prefilling, decode on
  other concurrent requests pauses until it finishes.
- **Short arithmetic with thinking off is unreliable.** The model sometimes asserts a
  wrong answer without working it out. Turn thinking on for math. This is a property
  of the model, not of this setup.
- **It over-uses tools on off-topic requests.** On BFCL's live irrelevance split the
  model calls a tool when it should decline about 70% of the time.
- **Coding ability is not established here.** The benchmarks show the serving stack is
  fast and stable over long sessions; they don't measure how good the model is at
  multi-step coding. Terminal-Bench can't run on GB10 (its task images are x86-only),
  but `scripts/bench_tb.sh` works from an x86 host pointed at this server.

## How it works

The checkpoint is ~135 GB and a Spark has 128 GB of unified memory, so stock SGLang
can't load it. This repo applies a small set of patches on top of a pinned SGLang image:

- **Embedding table on disk.** The model's 48 GB per-layer embedding (PLE) table is
  memory-mapped from a file on NVMe instead of being held in RAM. Each token only reads
  a few KB from it, so this costs very little speed and CUDA graphs still work. The
  table file is reused across restarts, which is why only the first boot is slow.
- **Sparse attention kernel for GB10.** Stock SGLang's fast sparse-decode path doesn't
  support GB10's GPU architecture (and silently produces garbage if forced). This repo
  uses a Triton kernel from [sglang#36845](https://github.com/sgl-project/sglang/pull/36845)
  instead.
- **Speculative decoding fixes.** The MTP draft head is kept unquantized, uses a reduced
  64k-token vocabulary for speed (~18% faster decode), and a patch from
  [sglang#37794](https://github.com/sgl-project/sglang/pull/37794) keeps model state
  correct after speculative verification.
- The container runs as your user, so downloaded files stay owned by you.

## Benchmarking and development

The `bench/` directory contains the benchmark suite (decode speed, long context, quality,
tool calling, agentic sessions, GSM8K, BFCL). All of it talks to a running server over
the API. `scripts/run_config.sh` runs a full, guarded experiment (start, benchmark,
stop) and writes results under `results/`.

- [docs/BENCHMARKS.md](docs/BENCHMARKS.md): detailed results, the experiment harness,
  and the configurations that were tried and rejected.
- [RESEARCH_LOG.md](RESEARCH_LOG.md): the full research log.
- [UPSTREAM_PLAN.md](UPSTREAM_PLAN.md): the staged upgrade plan and its outcomes.

## Credits

- Model: Qwen / Alibaba. NVFP4 quantization: [RadixArk](https://huggingface.co/RadixArk/Qwen3.8-Flash-Next-NVFP4).
- Engine: [SGLang](https://github.com/sgl-project/sglang).
- On-disk PLE table and GB10 sparse attention backport adapted from
  [hashd1ve/qwen38-flash-next-one-dgx-spark](https://github.com/hashd1ve/qwen38-flash-next-one-dgx-spark) (MIT)
  and [sglang#36845](https://github.com/sgl-project/sglang/pull/36845) (Apache-2.0).
- Native PLE file backend: [sglang#37068](https://github.com/sgl-project/sglang/pull/37068)
  and [sglang#38123](https://github.com/sgl-project/sglang/pull/38123).

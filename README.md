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
| `MEMFRAC` | see below | Lower hands UMA back to the page cache the PLE table is read from |
| `PREFILL` | see below | Chunked prefill size |
| `MAX_RUNNING` | see below | Concurrent requests |
| `CONTEXT` | `262144` | Native rope limit |
| `MAX_TOTAL` | `524288` | KV token budget |
| `SPEC_STEPS` / `SPEC_TOPK` / `SPEC_DRAFT` | `3` / `1` / `4` | MTP depth; `SPEC=off` disables |
| `CUDA_GRAPH_MAX_BS` | see below | Stock captures decode graphs up to bs 256 |
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

**Terminal-Bench does not run on a GB10.** The TB 2.x task images are
`linux/amd64` only and there is no qemu binfmt handler on this class of box, so
the task container exits 255 on platform mismatch. `scripts/bench_tb.sh` is
correct and works from an x86 host pointed at this server over the network. No
Terminal-Bench number in this repo was measured on a Spark.

## Measured (one GB10, 2026-08-27)

Single stream, MTP 3/1/4, `trtllm_mha` decode, CUDA graphs on, 262k context.

| | |
| --- | ---: |
| Decode, code, thinking off | **40.2 tok/s** (median 38.8) |
| `"12*17"` → `204` | 1.9 s warm |
| llama.cpp GGUF (same class, no MTP / no QSA) | ~16–18 tok/s |
| vLLM + MTP=2 on this NVFP4 | ~27 tok/s |

## Credits

- Model: Qwen / Alibaba. NVFP4: [RadixArk](https://huggingface.co/RadixArk/Qwen3.8-Flash-Next-NVFP4).
- Engine: [SGLang](https://github.com/sgl-project/sglang) (`qwen4_exp`).
- PLE mmap + SM120 QSA gate: adapted from [hashd1ve/qwen38-flash-next-one-dgx-spark](https://github.com/hashd1ve/qwen38-flash-next-one-dgx-spark) (MIT).

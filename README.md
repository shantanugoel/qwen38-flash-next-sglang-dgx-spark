# Qwen3.8-Flash-Next on one DGX Spark (SGLang)

Serve [`RadixArk/Qwen3.8-Flash-Next-NVFP4`](https://huggingface.co/RadixArk/Qwen3.8-Flash-Next-NVFP4) with **SGLang** on a single **NVIDIA DGX Spark / GB10** (128 GB unified memory). Native 262k context, MTP speculative decode, CUDA graphs.

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

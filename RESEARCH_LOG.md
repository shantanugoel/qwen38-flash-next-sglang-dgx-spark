# Research log

Empirical log for the one-GB10 Flash-Next recipe. Newest entries at the bottom
of each step. Conclusions are one paragraph, not a dump of docker logs.

Host: ganglion, GB10 sm_121, 121 GiB unified, driver 580.173.02, CUDA 13.0,
Linux 6.17 aarch64. GPU clocks stay at the host cap (no sudo). llama-swap
unloaded 2026-08-28 12:03 UTC (`POST http://127.0.0.1:8080/api/models/unload`).

User locks: balanced quality-then-speed; 262k default / 512k optional; thinking
on by default (caller can disable); vLLM is a real contender; do not edit `~/ai`.

---

## Step 0 — Research (2026-08-28)

### Inventory

| Item | Value |
| --- | --- |
| GPU occupant | none after unload |
| Host RAM | 121 GiB total, ~118 GiB available |
| Checkpoint | `RadixArk/Qwen3.8-Flash-Next-NVFP4` @ `7b719225242aacd3dbd3f9407468c2ee9a9d2594` already complete under `$HF_HOME` |
| PLE mmap file | `/home/shantanu/ai/cache/sglang/flash-next-ple-mmap/ple_table_51200245760_51200245760.bin` (48G, reused at runtime only) |
| Official SGLang image | **not** present at start; pulled `lmsysorg/sglang:qwen38flashnext` digest `sha256:12d3392bdc8be8d35e9a95f191df6aef99c5114bdbefd41bfdc7e760e6d25ec1` (newer than the 2026-08-27 Felliks base `14ed5825…`) |
| Side images | `qwen38-flash-next-gx10:73a255-ple-disk-v3` (Felliks), `qwen38-flash-dgx:latest` (likely blazux vLLM) |

### Upstream / community (what to try, what not)

**Checkpoint.** RadixArk NVFP4 is still the only pack with published GSM8K 97.27
and AIME26 98.75 plus structural audits. Last modified 2026-08-26. Community
repacks (Inferact, lovedheart NVFP4-FP8, primitive-ai mixed, abliterated) have
no matching evals. HashK-PLE (Death-By-Tokens) compresses the 51 GB table 4×
with reconstruction cosine ~0.50 — interesting for KV headroom, not for a
quality-first recipe. **Stay on this exact RadixArk revision.**

**SGLang.** Cookbook still wants `lmsysorg/sglang:qwen38flashnext` (#36497 not
in a tagged release). Official cookbook: thinking **cannot be turned off**,
`--reasoning-parser auto`, `reasoning_effort` in {xhigh, medium, low}. Chat
template nonetheless supports `enable_thinking is false` (empty `<think>`).
Tool parser: `--tool-call-parser` auto / `qwen3_coder`. NVFP4 launch on GB300
is TP2; GB10 is not on the signed-off matrix.

**hashd1ve (our patch source, measured on one GB10):**

- PLE `from_file` mmap + `madvise(MADV_RANDOM)` — CUDA graphs still capture.
- Widen `_resolve_trtllm_sparse_decode` with `is_sm120_supported()`.
- `--prefill-attention-backend triton --decode-attention-backend trtllm_mha`
  (do **not** set a single `--attention-backend trtllm_mha`).
- `--language-only`, `--mamba-radix-cache-strategy extra_buffer`, page-size 64
  forced for QSA, MTP 3/1/4 + `unquant`, mem-frac **0.85**, prefill 2048, ctx
  32k in their default (262k works; they warn 0.85 + 240k sequential prefills
  wedged the box).
- Prefix cache: 128k TTFT 183s → 0.6s on resend. Decode 27.3 → 21.7 tok/s from
  8k to 240k.
- Ring-width 8 / MTP 7/1/8: +18% code, −36% prose. **Not default.**
- Measured no-ops: speculative-attention-mode decode, fp4 cudnn, continuous
  decode 2, mamba ssm bf16, tf32. flashinfer_trtllm / cutedsl MoE: no sm121 cubins.
- `--mem-fraction-static 0.72` will not boot (`total_rest_memory=-1`).

**Death-By-Tokens:** extra landmines (TMA-O varlen crash, trtllm-gen garbage
on SM121, `_compact_kv` holes, fp8 `tl.dot` on long prefill). Their default
replaces trtllm sparse decode with SDPA — slower than hashd1ve’s working
trtllm path. Also insists `--mamba-scheduler-strategy extra_buffer
--mamba-track-interval 64` or MTP rewind corrupts GDN state on code.

**blazux vLLM:** mmap PLE as a splitting custom op, MTP=2, **prefix cache
OFF** (GDN CUBLAS bug on sm_121), fp8 KV refused (QSA wants bf16 KV), 262k
native / 500k YaRN needle at 414k. Decode ~25–28 vs our ~38–40 thinking-off
code. Prefill 2–2.6k tok/s is the vLLM pitch. **Bake off in Step 9.**

**Felliks:** disk-cache PLE + eager graphs (gather cannot capture). We already
beat that with mmap+graphs.

**sglang#36567 io_uring PLE:** mixed-domain ~24 tok/s on GB10. Not a 40 tok/s
chase. Linux 6.x folio RSS risk.

**vLLM official recipe:** `vllm/vllm-openai:qwen38-flash-next`, FP8 TP2 min on
GB300, `VLLM_PLE_CPU_OFFLOAD=1` pins 51 GB in the **same** UMA pool — does not
fit. mmap is required, same as SGLang.

### Knobs we will actually turn

1. New official image digest vs 2026-08-27.
2. `--language-only` (hashd1ve has it; we do not).
3. mem-frac 0.95 → 0.85, prefill 4096 → 2048/1024, max-running 4 → 2.
4. Thinking default on; measure effort + off.
5. MTP 2/1/3 vs 3/1/4 (not 7/1/8).
6. 512k + YaRN as optional, max-running 1.
7. vLLM mmap recipe if SGLang is not clearly ahead on agentic TTFT+decode.

### Knobs we will not spend boots on

Ring-width 8, HashK, MoE backend swaps, tf32, ssm dtype, dummy load, clock
uncap, second NVFP4 download, 1M context, editing llama-swap.

**Conclusion (Step 0):** Stay on RadixArk NVFP4 + mmap PLE + QSA SM120 + MTP
3/1/4 as the starting stack. Pull the newer cookbook image. Biggest likely
wins for *this* use case are `--language-only`, a less greedy mem/prefill
pack for 100–150 turn 262k sessions, thinking/effort defaults, and a honest
vLLM TTFT comparison — not another kernel flag.

---

## Step 1 — Harness (2026-08-28)

Added `AGENTS.md` (never block in the foreground), `bench/{client,decode,quality,longctx}.py`,
and `scripts/wait_ready.sh`. Image digest confirmed local:
`lmsysorg/sglang@sha256:12d3392bdc8be8d35e9a95f191df6aef99c5114bdbefd41bfdc7e760e6d25ec1`.

**Conclusion (Step 1):** Fast/quality/longctx benches are in-tree. Next is patch the
official image in the background, then baseline boot.

---

## Step 2 — Official image + patches (2026-08-28)

`prepare.sh` (background, ~1 min after image already local):

- Image `lmsysorg/sglang@sha256:12d3392bdc8be8d35e9a95f191df6aef99c5114bdbefd41bfdc7e760e6d25ec1`
- PLE: `_alloc_ple_table` inserted into `qwen4_exp.py`
- SM121 SDPA intercept: **absent** on this tag (drop patch is a no-op)
- QSA: `_resolve_trtllm_sparse_decode` now `is_sm100_supported() or is_sm120_supported()`
- Checkpoint revision already complete; no download

**Conclusion (Step 2):** Patches apply cleanly on the newer cookbook image. Do not
fall back to the Felliks disk-cache image.

---

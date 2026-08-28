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
wins: a less greedy mem/prefill pack, CUDA-graph max-bs (stock captures to
256), thinking/effort defaults, and a honest vLLM TTFT comparison. **Do not**
skip vision (`--language-only` / `--language-model-only`).

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

## Step 3 — Baseline boot, and a boot-time bug worth fixing first (2026-08-28)

First baseline boot with the committed `serve.sh` flags started 06:47 UTC and was
still not serving 27 min later. It was not hung — `/proc/<sched>/io` showed
`read_bytes` climbing ~30 MB/s and `write_bytes` climbing ~0.65 GB/min. The write
is the giveaway: SGLang runs the generic `VocabParallelEmbedding.weight_loader` on
the PLE parameter every boot, so the whole **47.7 GiB table is re-copied from the
checkpoint into the mmap on every single restart**, even though the mmap already
holds exactly that table from the previous boot. Extrapolated cost of that copy
alone: **45–60 min per boot**.

Raw disk is not the limit — `dd iflag=direct bs=8M` on a checkpoint blob reads at
**8.7 GB/s**. The copy is slow because it is a read-modify-write through the page
cache with almost no page cache available (see below).

### Fix: `patches/ple_reuse.py`

Adds a verified fast path to `Qwen4ExpPinnedHostEmbedding.weight_loader`: sample
8192 random rows (plus first and last) of the mmap and of the checkpoint tensor,
and skip the copy only if every sampled row is byte-identical. A stale, truncated
or wrong-revision file fails the sample and falls back to the full copy, so the
fast path cannot serve wrong weights. `SGLANG_QWEN4_PLE_REUSE=0` forces the copy.
Wired into `prepare.sh` after `ple_mmap.py`, with an assert.

### Where the 121 GiB actually goes at `--mem-fraction-static 0.95`

`nvidia-smi --query-compute-apps` during load, on this UMA part:

| Consumer | Size |
| --- | ---: |
| SGLang scheduler (weights + pools), still growing | 82.8 GiB |
| page cache (all of it, incl. the PLE mmap's hot rows) | ~26 GiB |
| anon + slab + rest | ~5 GiB |
| **free** | **~7 GiB** |

The PLE table is 47.7 GiB of pure random access and it lives *in the page cache*.
At mem-frac 0.95 there is not enough page cache to hold a useful fraction of it,
so the loader (and later every decode step) sits in `D` state on
`folio_wait_bit_common` reclaiming pages. This reframes Step 5: dropping mem-frac
is not only a stability knob, it is a **PLE residency** knob and therefore a
decode-speed knob. Measure 0.95 vs 0.85 with that in mind.

### Harness additions this step

- `bench/client.py`: `chat_stream()` (TTFT, prefill tok/s, decode tok/s, cached
  tokens, cache-hit %) and `server_metrics()` (Prometheus scrape).
- `bench/bfcl.py`: the fixed 100-case BFCL subset. AST scoring for
  simple/multiple/parallel against `possible_answer`, no-tool scoring for
  irrelevance/live_irrelevance, and **relaxed** per-turn function-name scoring for
  `multi_turn_base` (the real BFCL multi-turn needs their stateful sandbox;
  we score tool *selection* across turns instead and label it as such).
- `scripts/bench_tb.sh`: Terminal-Bench 2.1, fixed 8-task subset, `terminus-2`,
  `--n-attempts 1`, pointed at the local OpenAI-compatible endpoint.
- `scripts/bench_fast.sh`: one background job that runs smoke + quality + decode
  + longctx for a tagged config.
- Vision smoke image raised from 8×8 to 112×112 (Qwen VL patching needs ≥28 px).

### Knob survey (616 `sglang serve` flags on this image)

Beyond the Step 0 list, these are new candidates worth a boot for a 100–150 turn
agentic session, in expected-value order:

1. `--cuda-graph-max-bs-decode` — stock captures decode graphs up to bs 256 while
   we cap at `--max-running-requests 4`. Wasted capture time and memory that could
   be PLE page cache.
2. `--strip-thinking-cache` — drops reasoning from the cached prefix; directly
   targets prefix-cache hit rate across long tool-calling sessions.
3. `--enable-gdn-replayssm-spec` / `--enable-linear-replayssm-spec` — upstream
   handling of GDN/linear-attention state under speculative rewind. This is the
   exact failure Death-By-Tokens patched around by hand.
4. `--radix-eviction-policy {lru,lfu,slru,priority}` — LFU/SLRU should retain the
   system prompt + tool definitions across 150 turns better than LRU.
5. `--speculative-adaptive` (+ `--speculative-accept-threshold-*`) — adaptive MTP
   depth instead of a fixed 3/1/4.
6. `--cuda-graph-backend-prefill tc_piecewise` — prefill graphs where a full
   capture cannot work; agentic TTFT.
7. `--weight-loader-drop-cache-after-load` — frees the checkpoint's page cache
   after load, which is exactly the cache the PLE table wants.
8. `--mamba-backend flashinfer` vs `triton`, `--linear-attn-prefill-backend`.

Not pursued: `--enable-hierarchical-cache` / hicache (offloads KV to the *same*
UMA pool), `--kv-cache-dtype fp8` (QSA wants bf16 KV), disaggregation and
context-parallel flags (single box), `--enable-unified-memory` (already UMA).

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

### Fix: `patches/ple_reuse.py` (and two wrong versions of it first)

The obvious hook is wrong. `Qwen4ExpPinnedHostEmbedding` registers the mmap as a
`nn.Parameter` and sets `cpu_weight.weight_loader = self.weight_loader`, so
overriding that method looks like the right place — but the PLE table never goes
through it. `Qwen4ExpVLForConditionalGeneration.load_weights` has a dedicated
`load_qwen4_exp_ple_shard` branch that matches
`…ngram_embedding.shard_<N>.weight` and writes the rows with
`copy_ple_rows_to_tp_embedding`, **512 shards**, bypassing `param.weight_loader`
entirely. Two boots were spent re-copying the table in silence before the tell
showed up: `write_bytes` on the scheduler climbing while the override logged
nothing at all. (A first version was also shape-assuming and would have declined
silently even on the right hook. Every decline now logs.)

The shipped patch wraps the shard copy itself:

```python
if _ple_shard_matches(dst, src):   # random 4 KiB byte windows, both sides
    skipped += 1
else:
    dst.copy_(src)
```

Per shard rather than per table is both cheaper to verify (~34 windows over
~100 MB) and safer — a shard that does not match is copied on its own while the
rest are skipped, so a partially written or wrong-revision file cannot serve
wrong weights. `SGLANG_QWEN4_PLE_REUSE=0` forces the full copy;
`SGLANG_QWEN4_PLE_REUSE_WINDOWS` sets the sample count. The count of skipped vs
copied shards is logged once at the end of `load_weights`.

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

---

## Step 3b — Terminal-Bench does not run on this box (2026-08-28)

Two blockers, found by running the harness rather than assuming:

1. **2.1 is not published.** The `tb` CLI's registry
   (`laude-institute/terminal-bench/registry.json`) stops at
   `terminal-bench-core 0.1.1`. Terminal-Bench 2.x moved to **Harbor**
   (`pip install harbor`), whose registry has `terminal-bench 2.0` — 89 tasks,
   and all 8 of our subset tasks are in it. 2.1 is the 2.0 point-release and is
   not separately registered. Pinning 2.0 was the honest substitution.
2. **The task images are amd64-only.** A control run with the `oracle` agent
   (no model involved) on `fix-git` pulls `alexgshaw/fix-git:20251031` and dies:

   ```
   The requested image's platform (linux/amd64) does not match the detected
   host platform (linux/arm64/v8)
   container fix-git__…__env-main-1 exited (255)
   ```

   `/proc/sys/fs/binfmt_misc` has no qemu handler, and
   `docker run --platform linux/amd64 alpine uname -m` gives
   `exec /bin/uname: exec format error`. Registering one needs a privileged
   container that rewrites host `binfmt_misc` — outside what we were asked to
   do here — and even then, emulated amd64 on a GB10 would put every task into
   its own timeout, which is not a measurement.

   adrienbrault's Terminal-Bench numbers are on an RTX 5090, i.e. x86.

**Conclusion (Step 3b):** `scripts/bench_tb.sh` stays in-tree and is correct —
it runs as-is from an x86 host pointed at this server over the network — but
**no Terminal-Bench number will be reported from this box.** The agentic coding
signal comes from `bench/bfcl.py` (tools, multi-turn, hallucination) and
`bench/agentic.py` (long-horizon session) instead. Do not report a TB score
that was not measured.

---

## Step 3c — PLE reuse confirmed (2026-08-28)

Boot 4, same `serve.sh` flags, `PLE_DIR` pointed at the existing mmap:

```
[07:41:27] PLE table -> mmap /ple/ple_table_51200245760_51200245760.bin (47.7 GiB)
[07:41:27] PLE table: madvise(MADV_RANDOM) ok
[07:47:46] PLE table: 128/128 shards already on disk (320001536 rows)
```

`write_bytes` on the scheduler at the end of the target model's `load_weights`:
**8192** — eight kilobytes, versus 2.4 GiB and climbing at the same point in the
previous boot, on its way to 47.7 GiB. Target-model weight load: **6m19s**
(07:41:27 → 07:47:46), against 45–60 min projected for the same phase with the
refill. The sampling check itself does not show up in the timing.

Note the shard count: the checkpoint splits the table into **128** shards here,
not the 512 that `split_ngram_parts` defaults to — the patch reads the count from
the data, so this does not matter, but it is why the log says 128/128.

**Conclusion (Step 3c):** the reuse fast path is correct and is worth roughly
**45–55 min per restart**. It only helps from the second boot onward; the first
boot on a new machine still writes the table once. This is the single largest
change in the recipe so far and it is a pure boot-time win — no runtime
behaviour is altered, since a mismatched shard is still copied.

---

## Step 3d — Baseline boot facts (2026-08-28)

Boot 4, committed `serve.sh` defaults, existing PLE mmap.

| | |
| --- | ---: |
| `docker run` → `/health` 200 | **9m47s** (07:40:51 → 07:50:38) |
| target model weight load | 6m19s |
| MTP draft weight load | 1m26s (online NVFP4 quant of the draft MoE) |
| KV cache | 524288 tokens, bf16, K 6.00 GB + V 6.00 GB (+0.5/0.5 draft) |
| `max_total_num_tokens` | 524288 |

Host memory once serving, `--mem-fraction-static 0.95`:

| | |
| --- | ---: |
| SGLang scheduler (nvidia-smi compute-apps) | **103.6 GiB** |
| page cache (`buff/cache`) | **7 GiB** |
| free | **1 GiB** |

That is the number to keep in mind for Step 5: the PLE table is **47.7 GiB of
random access served out of a 7 GiB page cache**. Every gather that misses is an
NVMe read on the decode path. Decode tok/s at 0.95 is therefore not a kernel
result, it is a cache-residency result, and mem-fraction is the knob.

Graph capture also walks the stock decode batch-size ladder — 51 sizes up to
**bs 256** — while `--max-running-requests` is 4. That is capture time and
captured-graph memory spent on batch sizes this recipe can never reach.

---

## Step 3e / Step 4 — Baseline numbers (2026-08-28)

Committed `serve.sh` defaults: mem-frac 0.95, prefill 4096, max-running 4, MTP
3/1/4, 262k, decode `trtllm_mha`, prefill `triton`, vision on. Results under
`results/baseline/`.

### Quality — 12/12 pass

math 204 · tool call emitted · generated code executes to 42 · fact planted at
turn 1 recalled at turn 4 · **vision `Red` in 0.57 s** (the multimodal tower is
live, not a stub).

### Decode (n=3 after warmup, 400 max tokens, t=0.7)

| | median tok/s | range |
| --- | ---: | --- |
| code EN, thinking **off** | **31.2** | 26.1–32.3 |
| prose ES, thinking **off** | 16.5 | 16.0–19.7 |
| code EN, thinking **on** | 24.8 | 23.0–30.5 |
| prose ES, thinking **on** | 19.1 | 18.1–20.1 |

**The README's 40.2 tok/s does not reproduce.** Same flags, 31.2 median. The two
candidates are (a) the newer image digest — the 40.2 was measured before this
pull — and (b) the 7 GiB page cache in front of a 47.7 GiB table. Step 5 tests
(b) directly; if mem-frac does not close the gap, (a) is the remaining suspect
and the old digest is worth one boot.

### Long context

| | |
| --- | ---: |
| needle 8k | **PASS**, TTFT 5.37 s, 5885 prompt tokens |
| needle 32k | **PASS**, TTFT 10.98 s, 23555 prompt tokens (~2.1k tok/s prefill) |
| prefix-cache resend 8k | 5.37 s → **0.88 s** (6.1×) |
| prefix-cache resend 32k | 10.98 s → **0.83 s** (13.2×) |

### 40-turn agentic session (tools, thinking off)

| turns | median TTFT | median cache hit | median ctx |
| --- | ---: | ---: | ---: |
| 1–13 | 1.58 s | 90.2% | 1071 |
| 14–26 | 1.57 s | 95.5% | 2439 |
| 27–40 | 1.55 s | **97.3%** | 3869 |

39/40 turns emitted a tool call, **0 invalid tool calls**, and the fact planted at
turn 3 was recalled correctly at turn 40. TTFT is flat as context grows — the
radix cache is doing its job on the agentic resend pattern.

### Server counters

`sglang:spec_accept_length` **3.74** out of a 4-token draft — MTP 3/1/4 is
accepting almost everything, so there is little headroom in raising draft depth
and the QSA ring-width-8 path stays skipped. `sglang:cache_hit_rate` 0.972.

### Client-facing findings (Step 4, no restart)

- `chat_template_kwargs.enable_thinking=false` **works**: 3 tokens / 0.7 s versus
  ~60 tokens / 2–4 s with thinking on, same correct answer.
- **`reasoning_effort` is inert on this build.** low / medium / xhigh produced
  59 / 71 / 64 completion tokens as a template kwarg and 59 / 66 as a top-level
  OpenAI field — no ordering, no trend. The template auto-detect agrees
  (`effort_kwarg=None`). Callers should not expect it to do anything; use
  `enable_thinking` instead.
- `reasoning_tokens` is always 0 in `usage`; reasoning is billed as content.

---

## Step 5 — Pack A: mem-frac 0.85 + prefill 2048 + max-running 2 + graph bs 8 (2026-08-28)

```
MEMFRAC=0.85 PREFILL=2048 MAX_RUNNING=2 CUDA_GRAPH_MAX_BS=8 \
EXTRA_ARGS="--weight-loader-drop-cache-after-load" ./scripts/run_config.sh
```

### Operator error, and what it accidentally measured

`run_config.sh` was launched without `PLE_DIR`, so it used the committed default
`./data/ple` — empty — and wrote a **fresh** 47.7 GiB table:
`PLE table: 0/128 shards already on disk, 128 copied`, `write_bytes` 51.2 GB.

That is a useful accident. The full first fill finished inside a **631 s total
boot**. Writing 47.7 GiB into a fresh *sparse* file is fast; the 45–60 min figure
from Step 3 was specifically the **read-modify-write against an already populated
file under memory pressure**. The reuse patch is still the right fix — it turns
every restart into the cheap case — but the worst case it avoids is narrower than
first stated. `data/ple` is now populated, so later runs use the committed
default and stay fast.

### Config effects

| | baseline 0.95 | packA 0.85 |
| --- | ---: | ---: |
| `max_total_num_tokens` | 524288 | **318464** |
| GPU resident | 103.6 GiB | 97.0 GiB |
| page cache | 7 GiB | 11 GiB |
| free | 1 GiB | 4 GiB |

Dropping mem-fraction bought only ~4 GiB of page cache against a 47.7 GiB table
while costing **39% of the KV budget**. Whatever pack A gained, it is not
primarily PLE residency — that theory does not survive the numbers.

### Results (clean, `flock` held, single client)

| | packA |
| --- | ---: |
| quality | **12/12** incl. vision |
| decode code EN, thinking off | **38.8 tok/s** (38.4–39.0) |
| decode prose ES, thinking off | 22.0 tok/s |
| decode code EN, thinking on | 34.6 tok/s |
| decode prose ES, thinking on | 23.7 tok/s |
| needle 8k / 32k | PASS / PASS |
| TTFT 8k / 32k | 3.77 s / 10.41 s |
| prefix-cache resend 8k / 32k | 6.7× / **20.9×** |
| agentic 40 turns, TTFT (bands) | **0.60 / 0.56 / 0.57 s** |
| agentic decode (bands) | 50.6 / 49.6 / 47.1 tok/s |
| agentic cache hit (bands) | 90.0 / 95.7 / 97.3 % |
| agentic invalid tool calls | **0**, late recall PASS |
| `spec_accept_length` | 3.50 |

38.8 tok/s median is exactly the README's published median. **But the Step 3
baseline it would be compared against was contaminated by a concurrent second
bench client, so no "pack A beats baseline" claim can be made from these two
runs.** A clean baseline re-measure is running before any such claim.

Decode tok/s in the agentic bench (47–51) is higher than in the decode bench
(38.8) because those turns are short generations onto a warm 90–97% cached
prefix — that is the realistic agentic number, and it is the one worth quoting
for this use case.

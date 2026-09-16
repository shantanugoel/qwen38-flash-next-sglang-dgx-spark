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

---

## Step 6 — Clean baseline, and pack A rejected (2026-08-28)

Re-ran the committed defaults with the `flock` in place, single client.

| | baseline 0.95 (clean) | packA 0.85 |
| --- | ---: | ---: |
| decode code EN, thinking off | **38.55** | 38.79 |
| decode prose ES, thinking off | 21.99 | 21.98 |
| decode code EN, thinking on | 32.58 | 34.60 |
| decode prose ES, thinking on | 24.73 | 23.68 |
| needle 8k / 32k TTFT | 4.49 / 10.34 s | 3.77 / 10.41 s |
| prefix-cache resend 8k / 32k | 8.0× / **20.6×** | 6.7× / 20.9× |
| agentic TTFT (3 bands) | 0.59 / 0.56 / 0.57 s | 0.60 / 0.56 / 0.57 s |
| agentic decode | 49.2 / 49.4 / 49.6 | 50.6 / 49.6 / 47.1 |
| agentic cache hit | 91.3 / 96.0 / **97.4%** | 90.0 / 95.7 / 97.3% |
| invalid tool calls | **0** | **0** |
| `max_total_num_tokens` | **524288** | 318464 |
| `spec_accept_length` | **3.80** | 3.50 |
| boot | 617 s | 631 s (incl. a fresh PLE fill) |

**Conclusion (Step 6):** every difference is inside run-to-run noise, while
mem-frac 0.85 costs **39% of the KV budget** and 0.30 of speculative accept
length. **Pack A is rejected; the committed defaults stand.** The whole apparent
pack-A win in the earlier write-up was the contaminated baseline — a reminder
that a bundled config plus a dirty control is how you talk yourself into a
regression.

---

## Step 8 — Finalist benchmarks, shipped recipe, thinking off (2026-08-28)

### Agentic, 120 turns, tools — the use case

| turns | median TTFT | median decode | median cache hit | median ctx |
| --- | ---: | ---: | ---: | ---: |
| 1–40 | 0.549 s | 47.8 tok/s | 95.5% | 2538 |
| 41–80 | 0.575 s | 41.7 tok/s | 98.4% | 7102 |
| 81–120 | 0.573 s | 45.1 tok/s | **99.0%** | 11792 |

119 tool turns, **0 invalid tool calls**, fact planted at turn 3 recalled
correctly at turn 120, final context 14138 tokens, 157.9 s wall. **TTFT is flat
as context grows** — the radix cache is absorbing the resend pattern completely.

### GSM8K n=20, thinking off

**19/20 = 95.0%**, 269 s, median decode 35.8 tok/s. Checkpoint's published full
GSM8K is 97.27, so a 20-case slice at 95% is consistent — this is a regression
gate, not a leaderboard entry.

### BFCL 100-case subset — **headline number withheld**

| split | n | accuracy |
| --- | ---: | ---: |
| simple | 30 | 86.7% |
| multiple | 15 | 73.3% |
| parallel | 15 | 66.7% |
| irrelevance | 10 | 80.0% |
| live_irrelevance | 10 | **30.0%** |
| multi_turn_base | 20 | **0.0% — scorer bug** |

Wall 473 s, 150 requests, 336693 prompt / 12280 completion tokens, cache hit
72.0%, mean TTFT 0.938 s, mean decode 41.8 tok/s.

**The 58% aggregate is not reportable.** `multi_turn_base` scores 0.0% because
our relaxed scorer compares one turn's ground-truth *sequence* against a single
model response, e.g. `want ['cd','mkdir','mv'] got ['cd']`. Issuing `cd`, reading
the result, then `mkdir` is correct agentic behaviour; the scorer cannot express
it. Fixed in Step 9 by letting the model loop within a turn.

Two findings that **do** survive:

- **7 invalid tool calls**, all hallucinated function names not in the provided
  tool list — `web_fetch`, `web_search`, `database_us_census`. Concentrated in
  `parallel` and `live_irrelevance`.
- **live_irrelevance 30%**: on live/hallucination cases the model reaches for a
  tool when the right answer is to decline. `irrelevance` (curated) is 80%, so
  this is specifically the harder live split. Worth knowing for an agent loop
  that trusts every tool call it gets.

### Corruption check with prefix caching **on**

Across baseline 40t, packA 40t and the 120t run: **0 invalid tool calls in ~200
tool turns**, correct answer on the 32k cached resend, cache hit 95–99%. The
documented hazard is **MTP rewind corrupting GDN/Mamba state**, not the KV prefix
cache, and `serve.sh` already carries the mitigation
(`--mamba-radix-cache-strategy extra_buffer --mamba-track-interval 64`). Honest
statement: **guard on, no corruption observed** — not "the bug is fixed". The
`--no-enable-prefix-caching` advice in circulation is vLLM-specific (GDN CUBLAS
bug on sm_121).

---

## Step 10 — Agentic 120 turns with **thinking on** (2026-08-28)

Same prompts, same tools, same server as the thinking-off run.

| turns | median TTFT | median decode | median cache hit | median ctx |
| --- | ---: | ---: | ---: | ---: |
| 1–40 | 0.386 s | 32.0 tok/s | 93.6% | 2933 |
| 41–80 | 0.372 s | 26.6 tok/s | 98.2% | 8452 |
| 81–120 | **0.290 s** | 24.9 tok/s | **99.4%** | 10962 |

**0 invalid tool calls, late recall PASS**, 597 s wall, final ctx 11596.
TTFT *falls* as context grows (0.386 → 0.290 s) because the cached prefix keeps
growing faster than the new suffix.

### The result that matters, and it is not a speed number

| | thinking off | thinking on |
| --- | ---: | ---: |
| turns that emitted a tool call | **119 / 120** | **60 / 120** |
| invalid tool calls | 0 | 0 |
| decode (last band) | 45.1 tok/s | 24.9 tok/s |
| cache hit (last band) | 99.0% | 99.4% |

**With thinking on the model calls a tool on half as many turns.** It answers in
prose instead. This is not a failure — the answers are sane, nothing is
corrupted, the planted fact still comes back at turn 120 — but for a harness
that expects a tool call per step and treats "no tool call" as a stall, it is a
material behaviour change. The turn-120 answer even opens by deliberating about
whether to disclose the credential it was asked for.

**Implication for the recipe:** thinking on by default is right for reasoning
quality, but an agent loop should either tolerate prose turns or drive tool-heavy
phases with `enable_thinking=false` per request. That is a client-side decision
the recipe should document, not a server flag.

### Corruption, cumulative

~260 tool turns across thinking-off and thinking-on 120-turn sessions plus the
40-turn runs, radix cache **on**, cache hit 93–99%: **zero invalid tool calls,
zero corrupted recalls.** With the Mamba guard (`extra_buffer`,
`track_interval 64`) in place.

---

## Step 11 — Kernel and memory sweep (2026-08-28)

`QUICK=1` per config: quality + decode only, n=3 after warmup.

| config | code off | prose off | code on | prose on | quality | GPU / cache | boot |
| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| baseline | 38.55 | 21.99 | 32.58 | 24.73 | 12/12 | 103.4G / 7G | 617 s |
| `--mamba-full-memory-ratio=0.3` | 38.31 | 20.70 | 31.59 | 26.20 | 12/12 | 103.4G / 6G | 557 s |
| `PREFILL=8192 CUDA_GRAPH_MAX_BS=8 --disable-cuda-graph-padding` | 40.27 | 19.34 | 32.02 | 23.08 | **11/12** | 103.4G / 7G | 587 s |
| `--enable-gdn-replayssm-spec` | **40.42** | 21.60 | **34.76** | 25.88 | 12/12 | **101.3G / 10G** | 602 s |

### `--linear-attn-backend` — cannot be tested with our recipe as it stands

All three of `flashinfer`, `cutedsl`, `nvidia_kda` die in 30 s, and not for the
reason expected:

```
AssertionError: extra_buffer is not supported for
Qwen4ExpForConditionalGeneration; use no_buffer.
```

This is a real, undocumented coupling: **`--mamba-radix-cache-strategy
extra_buffer` is incompatible with any non-default linear-attention backend on
this model class.** `extra_buffer` is the Death-By-Tokens guard against MTP
rewind corrupting GDN state, so the kernel question and the corruption question
are the same question. Re-queued as sweep 2 with a **triton + `no_buffer`
control**, so the backend is not confounded with the cache strategy.

### `--mamba-full-memory-ratio=0.3` — no effect, and the reason matters

`max_total_num_tokens` is **524288 in every single config**, because that is our
own `--max-total-tokens 524288` flag binding — not a memory ceiling. The KV
budget was never memory-constrained at 262k, so freeing SSM memory has nowhere
to go. The knob is not useless; it is untestable at this context length. Retest
it inside Step 14 (512k), where memory actually binds.

### `PREFILL=8192 CUDA_GRAPH_MAX_BS=8 --disable-cuda-graph-padding` — held back

Fastest thinking-off code number in the sweep (40.27) and the only sweep-1
config to fail a quality check. **Correction to an earlier note in this log:**
the failing check is `effort_thinking_off`, not `math_12x17`. The question is
"4 pens at $3 and 2 notebooks at $7, change from $50" — answer 24 — and the
config replied **25** with thinking off in 3 tokens. `math_12x17` passed.

That is a wrong answer, but it is one borderline two-step word problem answered
without thinking in three tokens, at temperature 0. It is weak evidence, not the
catastrophic arithmetic failure the earlier wording implied. Held back rather
than condemned: it is a 3-flag bundle, so if it is ever revisited it must be
unbundled and re-tested with a bigger quality set. Note prose-off also dropped
to 19.34, the worst in the sweep, which is the more consistent argument against
it.

### `--enable-gdn-replayssm-spec` — promising, not proven

Best on three of four decode measures, 12/12 quality, and it hands ~2 GB of GPU
back to the page cache (101.3G resident vs 103.4G, 10G cache vs 7G). But at n=3
the ranges overlap the baseline (38.05–40.55 vs 38.41–39.52 on code-off), so
this is **not yet a significant result**. Promote only after a full bench
(longctx + 120-turn agentic). If it holds, it is also a candidate to *replace*
the hand-rolled `extra_buffer` + `track_interval 64` workaround.

**Conclusion (Step 11):** one candidate worth a full bench
(`--enable-gdn-replayssm-spec`), one knob deferred to the 512k step
(`mamba-full-memory-ratio`), one rejected on correctness (the prefill bundle),
and one experiment that turned into a structural finding about `extra_buffer`.

---

## Step 11e — Why the alternative GDN kernels are unreachable (2026-08-28)

Three sweeps to close this, ending in the upstream source rather than a guess.

1. `--linear-attn-backend={flashinfer,cutedsl,nvidia_kda}` →
   `AssertionError: extra_buffer is not supported for
   Qwen4ExpForConditionalGeneration; use no_buffer.`
2. Same, with `MAMBA_STRATEGY=no_buffer` →
   `AssertionError: no_buffer only supports page_size=1.`
3. Same, with `PAGE_SIZE=1` → **same assertion**. The CLI showed `page_size=1`
   and the validator still saw 64, i.e. something overrides it.

`sglang/srt/arg_groups/overrides.py`:

```python
if profile is not None and profile.variant == QSA_VARIANT_COMPRESSED:
    overrides["page_size"] = 64
```

and its docstring:

> Compressed QSA additionally pins page_size=64 … its compressed cache is
> addressed as `full_slot // compress_ratio` (the DSV4 scheme), which requires
> page-aligned full-KV allocation … MambaRadixCache supports page_size > 1 only
> with the mamba extra-buffer strategy, so fall back to the family default when
> neither that **nor `--disable-radix-cache`** holds.

**The chain, sourced:** compressed QSA pins `page_size=64` unconditionally and
`--page-size` cannot override it → `MambaRadixCache` allows `page_size > 1` only
under `extra_buffer` → `extra_buffer` rejects every non-triton
`--linear-attn-backend`. **So the alternative GDN kernels cannot be used while
QSA and the radix cache are both on.** This is a structural property of the
model + recipe, not a missing-cubin problem, and not something a flag combination
can work around.

The docstring also names the one escape hatch — `--disable-radix-cache` — which
merges this question into the radix A/B (Step 12): with the prefix cache off,
`page_size > 1` no longer needs `extra_buffer`, so the kernels become reachable.
That is the trade being measured there: kernel choice **or** prefix caching, not
both.

---

## Step 13 — `--enable-gdn-replayssm-spec` full bench — **ADOPTED** (2026-08-28)

| | baseline | gdn-replayssm |
| --- | ---: | ---: |
| decode code EN, thinking off | 38.55 | **41.49** |
| decode prose ES, thinking off | 21.99 | 21.36 |
| decode code EN, thinking on | 32.58 | 32.49 |
| decode prose ES, thinking on | 24.73 | 24.58 |
| quality | 12/12 | **12/12** |
| needle 8k / 32k | PASS / PASS | **PASS / PASS** |
| TTFT 8k / 32k | 4.49 / 10.34 s | 3.98 / 10.37 s |
| prefix-cache resend 8k / 32k | 8.0× / 20.6× | 7.0× / **21.0×** |
| agentic TTFT (bands) | 0.589 / 0.563 / 0.565 | 0.608 / 0.569 / 0.566 |
| agentic decode (bands) | 49.2 / 49.4 / 49.6 | 49.8 / 50.8 / **51.7** |
| agentic cache hit | 91.3 / 96.0 / 97.4% | 91.3 / 96.1 / 97.4% |
| invalid tool calls | 0 | **0** |
| **`spec_accept_length`** | **3.80** | **3.95** |
| `max_total_num_tokens` | 524288 | **524288** |
| GPU resident | 103.4 GiB | **101.3 GiB** |

**Why this is not the n=3 noise the rest of the sweep suffered from:**
`spec_accept_length` is a server-side counter, not a timing, and it moved
**3.80 → 3.95 out of a maximum of 4.0**. That is a mechanistic explanation for
the decode gain — with GDN state handled correctly across speculative rewind,
more drafted tokens survive verification. The decode result also reproduced
across two independent boots (40.42 in the QUICK sweep, 41.49 on the full bench)
against a baseline that has never exceeded 38.96 in five runs.

It costs nothing measurable: identical KV budget, **2.1 GiB less** GPU resident,
no quality, needle, cache or tool-calling regression.

It is also the conceptually correct fix. `extra_buffer` +
`--mamba-track-interval 64` is a hand-rolled guard against MTP rewind corrupting
GDN state; `--enable-gdn-replayssm-spec` is upstream's mechanism for the same
hazard. They coexist — this run had both — so the guard stays until there is
evidence it can be dropped safely. Do **not** drop it on the strength of this
result alone.

**Adopted into `serve.sh` defaults.**

---

## Step 9 — BFCL multi-turn: scorer fixed, split still not measurable (2026-08-28)

The scorer bug is fixed — the model now loops within a turn (each tool result is
fed back, up to `BFCL_MT_MAX_STEPS`) and the turn is scored on the union of calls
made. `multi_turn_base` went **0.0% → 15.0%**.

**15% is still not a valid BFCL multi-turn score, and we will not report it as
one.** The remaining limit is our harness, not the model: every tool call gets a
mock `{"status": "ok"}` back. The model cannot see a filesystem, so it probes —

```
turn0: want ['cd','mkdir','mv']  got ['cd','find','ls','pwd']
```

Real BFCL multi-turn executes against stateful backends (`GorillaFileSystem`,
`TwitterAPI`, …) so the model can read actual results and make progress.
Reproducing that is a port of BFCL's execution layer, well beyond this window.

**Reported instead: the 80 single-turn cases, which are unaffected.**

| split | n | accuracy |
| --- | ---: | ---: |
| simple | 30 | 86.7% |
| multiple | 15 | 73.3% |
| parallel | 15 | 66.7% |
| irrelevance | 10 | 80.0% |
| live_irrelevance | 10 | 30.0% |
| **total (single-turn)** | **80** | **72.5%** |
| multi_turn_base | 20 | **not measurable here** |

Useful by-products of the re-score run (295 requests, 1.42 M prompt tokens):
**0 invalid tool calls**, cache hit **93.8%**, mean decode **45.2 tok/s**, mean
TTFT 0.556 s — a second long tool-calling workload with no corruption.

**The two findings that stand:** 7 hallucinated function names in the original
100-case run, and **live_irrelevance 30%** — the model reaches for a tool when
the right move is to decline. `irrelevance` (curated) is 80%, so it is the harder
live split specifically. Worth knowing for an agent loop that trusts every tool
call it receives.

---

## Step 12 — Radix cache A/B — **cache stays ON** (2026-08-28)

`--disable-radix-cache`, everything else equal (both with
`--enable-gdn-replayssm-spec`), full bench.

| | radix on | radix off |
| --- | ---: | ---: |
| agentic TTFT, turns 1–13 | 0.589 s | 1.074 s |
| agentic TTFT, turns 14–26 | 0.563 s | 1.497 s |
| agentic TTFT, turns 27–40 | **0.566 s** | **2.322 s** |
| agentic cache hit | 91.3 / 96.1 / 97.4% | **0 / 0 / 0%** |
| 8k resend speedup | 7.0× | 1.2× |
| 32k resend speedup | **21.0×** | **1.0×** |
| 40-turn wall clock | 55.2 s | **95.6 s** (+73%) |
| decode code off | 41.49 | 39.59 |
| agentic decode | 49.8 / 50.8 / 51.7 | 49.0 / 51.0 / 51.3 |
| quality | 12/12 | 12/12 |
| invalid tool calls | 0 | **0** |
| needles 8k / 32k | PASS / PASS | PASS / PASS |
| `spec_accept_length` | 3.95 | 3.825 |

**TTFT without the cache grows linearly and had not levelled off at turn 40** —
0.57 s flat becomes 2.32 s at 4.3k context. Extrapolated to a 100–150 turn
session at 12k+ context that is 6–7 s per turn of re-prefill. This is the single
largest effect measured in the whole exploration.

**Two conclusions beyond "keep the cache":**

1. **Decode tok/s is blind to this.** ~51 tok/s either way. Every QUICK-mode
   sweep in this log is therefore incapable of evaluating caching, and any
   third-party benchmark of this model that reports only decode rate is missing
   the property that dominates agentic use.
2. **Turning the cache off bought no reliability.** 12/12 quality and **0 invalid
   tool calls** with it off, identical to with it on. The
   "disable prefix caching to avoid corruption" advice is vLLM-specific (GDN
   CUBLAS on sm_121); on this SGLang path it is pure cost. This also closes the
   linear-attention kernel question: `flashinfer` is reachable **only** via
   `--disable-radix-cache`, and that price is not worth paying.

**`--disable-radix-cache` rejected. Radix/prefix caching stays on.**

---

## Step 14 — 512k: boots, and is a **regression**. Not achieved. (2026-08-28)

### Attempt 1 — the published YaRN recipe does not fit this checkpoint

`--json-model-override-args={"rope_scaling":{...}}` silently did nothing and the
boot died in 30 s:

```
ValueError: User-specified context_length (524288) is greater than the
derived context_length (262144).
```

The checkpoint has **no `rope_scaling` field**. `text_config.rope_parameters` is
**mrope**:

```json
{"mrope_interleaved": true, "mrope_section": [11, 11, 10],
 "partial_rotary_factor": 0.25, "rope_theta": 10000000, "rope_type": "default"}
```

So the standard Qwen static-YaRN recipe in every write-up targets a field this
model does not have. The override must target `text_config.rope_parameters` and
preserve the mrope fields, and `SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1` must
be set or `_derive_context_length` refuses outright.

### Attempt 2 — boots, with less usable context than the default

```
CONTEXT=524288 MAX_TOTAL=524288 MAX_RUNNING=1 MEMFRAC=0.82 PREFILL=1024
--mamba-full-memory-ratio=0.3 + mrope-shaped YaRN override
```

| | 512k attempt | shipped 262k |
| --- | ---: | ---: |
| boot | 587 s | 617 s |
| **`max_total_num_tokens`** | **201984** | **524288** |
| GPU resident | 89.3 GiB | 101.3 GiB |
| page cache | 18 GiB | 10 GiB |

**The KV pool holds 201984 tokens — below the 262144 default context.** Asking
for 512k produced *less* usable context than not asking. As configured this is
strictly worse than the shipped recipe.

**`--mamba-full-memory-ratio 0.3` does work — but only here.** GPU fell 101.3 →
89.3 GiB and page cache rose 10 → 18 GiB, exactly the memory-bound regime
predicted in Step 11 where `--max-total-tokens` no longer binds first. It is
confounded with `MEMFRAC=0.82` in this run, so the split is unattributed.

### Why this is not pursued further

`MEMFRAC=0.95` would likely allocate a larger KV pool (the 262k recipe already
reports 524288 tokens), so a 512k config that *allocates* is probably reachable.
That still would not make 512k **work**: transformers continues to log
`rope_type='default'`, i.e. the YaRN parameters are accepted as override args but
do not appear to change the rope implementation for this mrope model. Beyond
262k that is raw extrapolation — the exact failure YaRN exists to prevent.

Demonstrating 512k would need a needle **past 262k**, and a ~400k cold prefill has
previously wedged this box. Booting at `context_length=524288` is **not** evidence
that 512k works, and will not be reported as such.

**Conclusion (Step 14): 262k stays the default and the only supported context.
512k is not achieved.** The blocker is not memory tuning — it is that static YaRN
as published does not compose with this checkpoint's sectioned mrope.

### Step 14 addendum — the 512k run hid a real agentic lead

The 512k config is rejected on context, but its **agentic numbers are the best
measured anywhere in this exploration**:

| | shipped 262k (gdn) | 512k attempt |
| --- | ---: | ---: |
| agentic decode (bands) | 49.8 / 50.8 / 51.7 | **56.2 / 59.8 / 59.6** |
| agentic TTFT (bands) | 0.608 / 0.569 / 0.566 | 0.601 / 0.547 / 0.557 |
| 40-turn wall clock | 55.2 s | **42.1 s** |
| needle 40k | not run | **PASS**, 24.3× resend speedup |
| page cache | 10 GiB | **18 GiB** |
| GPU resident | 101.3 GiB | 89.3 GiB |
| `max_total_num_tokens` | 524288 | 201984 |

**+15–20% agentic decode.** The plausible mechanism is the one this log declared
dead in Step 5 and should now be re-opened: **page cache for the PLE table.**
18 GiB of cache against a 47.7 GiB random-access table versus 10 GiB. Step 5
tested `MEMFRAC=0.85` and got only 11 GiB of cache and no gain; this run reaches
18 GiB and gains. That looks like a threshold rather than a linear effect, which
is why the earlier test missed it.

**Confounded** with `MAX_RUNNING=1`, `PREFILL=1024`, `CONTEXT=524288` and
`--mamba-full-memory-ratio=0.3`. The isolating experiment is:

```
CONTEXT=262144 MAX_TOTAL=524288 MEMFRAC=0.82 --mamba-full-memory-ratio=0.3
```

i.e. keep the full 262k context and KV request, take the memory back from the
SSM pool and mem-fraction, and see whether the agentic gain survives with a
usable KV budget. **Not yet run — this is the highest-value open lead.**

---

## Step 15 — Vision on/off — **not achieved**, and a corrected vision test (2026-08-28)

### The old vision smoke test was unsound

It rendered a solid red square and asked for the dominant colour. A text-only
server passes that by guessing "Red". Every earlier "vision 12/12" in this log
therefore proved **nothing** about the multimodal tower, and the claim in the
Step 3e entry that it showed the tower was "live, not a stub" was unsupported.

Replaced with a four-quadrant image (red / yellow / green / blue) asking for one
named quadrant — not guessable from the prompt.

### `--language-only` does not disable vision on this build

The corrected test passes on a server booted with `--language-only`
(`language_only=True` confirmed in `server_args`), and all four quadrants are
read correctly:

| asked | expected | got |
| --- | --- | --- |
| TOP-LEFT | red | `Red` |
| TOP-RIGHT | yellow | `Yellow` |
| BOTTOM-LEFT | green | `Green` |
| BOTTOM-RIGHT | blue | `Blue` |

Four correct spatial answers is not guessing. **The flag does not remove the
tower from a single-process server** — it is for *encoder disaggregation*
(`--encoder-urls`, `--encoder-transfer-backend`), i.e. handing the encoder to a
separate service. With no encoder service configured, the local path still runs.

That also explains the "savings": GPU resident 101193 MiB with `--language-only`
versus 101282 MiB without — **89 MiB, 0.09%** — and an identical 524288 KV
budget. Nothing was switched off.

| | vision "off" (`--language-only`) | vision on |
| --- | ---: | ---: |
| GPU resident | 101193 MiB | 101282 MiB |
| `max_total_num_tokens` | 524288 | 524288 |
| quality | 12/12 | 12/12 |
| decode code, thinking off | 40.98 | 41.49 |

**Conclusion (Step 15): the vision on/off comparison was not achieved.** The
premise of the step — that `--language-only` frees text-only headroom, taken from
hashd1ve's notes — does not hold on this image. A genuine comparison would need
a checkpoint without the vision tower, or a disaggregated encoder deployment.

**The useful result is the opposite one:** vision demonstrably works on the
shipped recipe, now proven by a test that cannot be passed blind. Keeping it
costs at most 89 MiB and no KV, so there was never a headroom argument for
turning it off.

---

## Step 18 — Mem-fraction isolation: the PLE page-cache theory is dead (2026-08-28)

```
CONTEXT=262144 MAX_TOTAL=524288 MEMFRAC=0.82 --mamba-full-memory-ratio=0.3
```

| | shipped 0.95 | memfrac 0.82 + mamba 0.3 | the 512k run |
| --- | ---: | ---: | ---: |
| page cache | 10 GiB | **19 GiB** | 18 GiB |
| GPU resident | 101.3 GiB | 89.3 GiB | 89.3 GiB |
| agentic decode (bands) | 49.8 / 50.8 / 51.7 | 50.9 / 52.6 / 52.0 | **56.2 / 59.8 / 59.6** |
| 40-turn wall clock | 55.2 s | 53.9 s | **42.1 s** |
| `max_total_num_tokens` | **524288** | 231936 | 201984 |
| quality / needles / invalid calls | 12/12 · 2/2 · 0 | 12/12 · 2/2 · 0 | 11/12 · 2/2 · 0 |

**The page cache was reproduced — 19 GiB, more than the 512k run's 18 GiB — and
the speed was not.** +1–2% agentic decode, not +15–20%. PLE residency is
therefore **not** the mechanism behind the 512k run's numbers, and the hypothesis
this log resurrected in the Step 14 addendum is dead again, now on direct
evidence rather than absence of evidence.

`MEMFRAC=0.82` is independently rejected anyway: 231936 KV tokens is **below the
262144 default context**.

**Remaining suspects for the 512k run's speed:** `MAX_RUNNING=1` (fewer scheduler
slots, less per-step overhead — this workload is a single stream) or
`PREFILL=1024`. Probing `MAX_RUNNING=1` at full `MEMFRAC=0.95` so the KV budget
stays at 524288.

**Lesson worth keeping:** the Step 14 addendum treated a difference observed in a
5-variable config as evidence for one named mechanism. Two of those variables
have now been eliminated. Any config difference in this log that has not been
isolated should be read as "unexplained", not as support for whichever
explanation was most available at the time.

---

## Steps 18–20 — chasing the 512k run's speed: five hypotheses, all dead (2026-08-28)

The 512k config posted agentic decode of **56.2 / 59.8 / 59.6 tok/s** and a 42.1 s
40-turn wall clock, against ~50 / 51 / 52 and ~55 s everywhere else. It differed
from the shipped recipe in five variables. Each was then isolated at full
`MEMFRAC=0.95` / 262k / 524288 KV unless noted:

| isolated variable | agentic decode (bands) | 40-turn wall | verdict |
| --- | ---: | ---: | --- |
| *(shipped reference)* | 49.8 / 50.8 / 51.7 | 55.2 s | — |
| `MEMFRAC=0.82` + `mamba-ratio=0.3` | 50.9 / 52.6 / 52.0 | 53.9 s | **no** (and KV 231936 < 262k) |
| `MAX_RUNNING=1` | 49.5 / 50.6 / 51.2 | **66.9 s** | **no** (slower; 1 invalid tool call) |
| `PREFILL=1024` | 49.5 / 50.4 / 50.5 | 55.1 s | **no** (32k TTFT 12.31 s vs 10.37 s — worse) |
| page-cache residency | covered by the `MEMFRAC` run: 19 GiB cache, +1–2% | — | **no** |
| 512k context itself | — | — | not a mechanism; see Step 14 |

**Conclusion: the 512k run's speed is unexplained.** It was a single observation
in a five-variable config, every constituent variable has now been tested alone
and none reproduces it, and there is no sixth hypothesis worth a boot. It is
recorded as an **unexplained single-run observation**, not as a lead, and
explicitly not as an argument for 512k — that config caps usable context at
201984 tokens.

`PREFILL=1024` also produced the one clean prefill-chunking datapoint in this
log: **32k TTFT 12.31 s vs 10.37 s at 4096**, i.e. smaller chunks cost ~19% TTFT
on a long prompt, as expected. 4096 stays.

**Net effect of Steps 18–20 on the recipe: none.** `MEMFRAC=0.95`,
`PREFILL=4096`, `MAX_RUNNING=4`, 262k all stand.

---

## Step 21 — Final confirmation on the shipped config (2026-08-28)

Committed `serve.sh`, i.e. baseline defaults + `--enable-gdn-replayssm-spec`.

| | |
| --- | ---: |
| boot | 572 s, `128/128 shards already on disk` |
| `max_total_num_tokens` | 524288 |
| quality | 12/12 (positional vision test) |
| decode code / prose, thinking off | 39.3 / 23.4 tok/s |
| decode code / prose, thinking on | 31.5 / 26.7 tok/s |
| needle 8k / 32k | PASS / PASS |
| prefix-cache resend 8k / 32k | 7.9× / 21.5× |
| `spec_accept_length` | 3.925 |

### 120 turns re-measured on the shipped config

The README previously carried the 120-turn table from the **pre-adoption
baseline** run. Re-measured here so the flagship numbers match the shipped flags.

Thinking off — TTFT 0.536 / 0.564 / 0.560 s, decode 50.6 / 50.2 / 49.8 tok/s,
cache 96.1 / 98.5 / 99.2%, 119/120 tool turns, 0 invalid, late recall PASS,
161.3 s, final ctx 14560.

Thinking on — TTFT 0.377 / 0.434 / 0.472 s, decode 34.5 / 29.7 / 28.1 tok/s,
cache 94.1 / 96.4 / 97.8%, **101/120** tool turns, 0 invalid, 592.7 s, ctx 15760.

### Two corrections this run forced

**1. "Thinking on halves tool-call frequency" was not reproducible.** The first
thinking-on session gave 60/120 tool turns; this one gave **101/120** on the same
prompts. Thinking off is 119/120 both times. The effect is real in direction but
the magnitude swings by 40 percentage points between runs, so the README now says
"fewer and unpredictable" rather than quoting a ratio. This is the largest
run-to-run variance seen anywhere in this exploration and it sits on a claim that
was stated as a clean fact.

**2. The thinking-on "late recall FAIL" is a refusal, not a memory failure.**
`bench/agentic.py` plants the fact as a tool result reading
`"Access note: {FACT}. Do not repeat unless asked."` At turn 120 the model was
asked, and with thinking on it reasoned about disclosure and **declined**, while
correctly naming `ops/secrets.md` as the source. Recall was intact. The bench
scores that as a failed recall, which is wrong — the metric cannot distinguish
"forgot" from "remembered and refused". Recorded rather than silently re-run
until it passed.


## September 7 — Upstream audit and staged plan (documentation only)

User requested an ordered plan, with docs updated and a commit after every item
before moving to the next. Added UPSTREAM_PLAN.md; no experiments started.

Read-only upstream findings that supersede parts of the August discussion:

- RadixArk revision is unchanged. The SGLang image tag has been rebuilt; a newer
  ARM64 model-development image also exists. Version details are in the plan.
- Our widened SM121 TRT-LLM gate was superseded by correctness fix #36845, merged
  August 30 into qwen4-main-squashed. Upstream reproduced corruption at 120k–210k;
  our shorter tests do not clear it. Correctness now precedes speed tuning.
- Native PLE backend #37068, TP prefetch correction #38123 and NVIDIA mixed-quant
  loader #38121 merged into that branch September 5. Model support #36497 remains
  open against main. Native PLE still rewrites weights every boot, so migration
  must preserve validated reuse.
- Open #37794 reports skipped PLE commits in ReplaySSM speculation. Local
  applicability is not yet established; audit it separately from NGRAM additions.
- NVIDIA's new checkpoint has published evaluations and FP8 MTP experts; the old
  claim that RadixArk is the only evaluated candidate is no longer current.
- vLLM prefix caching must be evaluated on the selected current stack; the old
  unconditional exclusion is not a current compatibility finding.

The README now flags the known upstream QSA concern and links the plan. The
EXPLORATION header marks its old status entries as historical. Future per-item
entries must record accepted/rejected/deferred, evidence, rollback, and commit
before the next item. Existing performance figures were not remeasured.


## U0 — Baseline inventory and reliable experiment harness (2026-09-07)

**Decision: accepted.** New locked harness; historical serving flags unchanged.
Rollback is not applicable (tooling only). Next item is U1.

TAGs: `u0-baseline-20260907` (aborted), `u0-baseline-retry-20260907` (pass).
Commands (detached): `PROFILE=u0 TAG=<tag> nohup ./scripts/run_config.sh > results/run-<tag>.log 2>&1 &`.
Synthetic detector: `python3 tests/test_harness.py` (16 tests during the live runs;
allocation-error sustain coverage added after the retry).

### Inventory (retry, before launch)

| Item | Value |
| --- | --- |
| Git | `f0cd56a` plus dirty harness WIP |
| Image id | `sha256:64c58f100438fa5f036bdfbeb3edd3136fb12c5d22d8ae52786c4a701263c55d` |
| Repo digest | `lmsysorg/sglang@sha256:12d3392bdc8be8d35e9a95f191df6aef99c5114bdbefd41bfdc7e760e6d25ec1` |
| In-image SGLang | `0.0.0.dev1+gd91c3682b` / `d91c3682b0b429e4c70df63cd57f819588ce29b0` |
| Packages | torch `2.13.0+cu130`, flashinfer `0.6.17`, triton `3.7.1`, transformers `5.12.1`, sglang-kernel `0.4.6.post1`, nvidia-modelopt `0.45.0` |
| Checkpoint | `RadixArk/Qwen3.8-Flash-Next-NVFP4` @ `7b719225242aacd3dbd3f9407468c2ee9a9d2594` |
| PLE | 128 shards, 51200245760 bytes, sample `a13a022a…`; `128/128 shards already on disk` |
| GPU / clocks | GB10, driver 580.173.02, **208 / 3003 MHz** (host cap, unchanged) |
| Host | kernel 6.17.0-1031-nvidia, 121 GiB, swap 0 |

Effective flags match the shipped recipe (`mem_fraction_static=0.95`, context 262144,
`max_total_num_tokens=524288`, page 64, extra_buffer, track 64, triton/trtllm_mha,
NEXTN 3/1/4, `--enable-gdn-replayssm-spec`). Launch reports `speculative_algorithm=NEXTN`
and draft `unquant`; `/server_info` reports `EAGLE` and draft quantization `null` — same
as August. KV BF16, 524288 tokens.

### Harness

`scripts/run_config.sh` now owns inventory, image pin, sampled PLE identity, llama-swap
unload, GPU-idle check, watchdog, bounded suites, and stop. `scripts/bench_fast.sh` is
benchmark-only against an already-running server under the same `results/.bench.lock`.
Tags cannot overwrite prior evidence. Suite process exit 0 is not a pass: quality/needle
failures, missing reports, invalid tool calls, and late-recall misses fail the verdict.
`PROFILE=u0` caps text prompts at 32768 tokens and refuses >40 turns or >32k needles.

Watchdog floors: 6 GiB MemAvailable, 0.5 GiB MemFree gated at 10 GiB available, 0.5 GiB
swap growth, 15 s sustain. First live boot (`u0-baseline-20260907`) reached uvicorn
(519 s tokenizer_e2e) then the watchdog SIGTERM'd it: two kernel lines at 15:54:24 and
15:54:27 IST, `NV_ERR_NO_MEMORY` from `_memdescAllocInternal`, while MemAvailable was
still 12.6 GiB. No Xid this boot. The same signature is common on this host during graph
capture (176 lines since boot, including the August campaign). Retry one minute later
had **zero** new `NV_ERR` lines, min available 9.87 GiB, min free 0.88 GiB, swap 0.
Calibration: Xids still stop immediately; allocation errors now use the 15 s sustain
window and only *new* journal samples. Trip JSON keeps matching lines. Not a claim that
large prefills are safe.

### Short-context remeasure (retry, thinking defaults as labeled)

Boot 533.5 s. Suites: smoke, quality 12/12, decode, 8k/32k longctx, 40-turn agentic off/on — all pass, 0 invalid tool calls. Server stopped cleanly (`stop_exit_code=0`).

| Decode median tok/s | thinking off | thinking on |
| --- | ---: | ---: |
| code EN | 39.26 (38.63–40.50) | 32.73 (29.96–33.11) |
| prose ES | 22.17 (20.65–22.27) | 26.53 (26.35–27.49) |

August shipped code/prose was ~39.3/23.4 off and ~31.5/26.7 on. Spread is the usual ±5%.

| Longctx | first s | resend s | speedup | needle |
| --- | ---: | ---: | ---: | --- |
| 8k (5885 tok) | 4.42 | 0.55 | 7.98× | PASS |
| 32k (23555 tok) | 10.37 | 0.44 | 23.3× | PASS |

32k TTFT matches August's 10.37 s exactly. 8k is a bit slower than August's 3.98 s;
still well within one-run noise and a different tokenize preflight. Not a 120k+ test.

| 40-turn agentic | wall | tools | invalid | late recall | decode bands |
| --- | ---: | ---: | ---: | --- | --- |
| thinking off | 57 s | 39/40 | 0 | PASS | 48.1 / 50.6 / 50.9 tok/s |
| thinking on | 111 s | 39/40 | 0 | PASS | 37.8 / 32.7 / 34.1 tok/s |

This is not the 120-turn promotion gate. `spec_accept_length` was not dumped by the new
suite. Vision positional smoke still passed. Recipe defaults stay.


## U1 — Dedicated SM121 sparse-decode correctness fix (2026-09-07)

**Decision: accepted (Triton #36845 serving default). KDA overlay rejected on
this image.** Rollback of the serving path is the 2026-08-28 Triton kernel, not
the widened TRT-LLM gate and not KDA. Next item is U2.

TAGs: `u1-kda-20260907` (fail), `u1-triton-20260907` (pass).
Commands (detached): `PROFILE=u1 TAG=<tag> nohup ./scripts/run_config.sh > results/run-<tag>.log 2>&1 &`.
Image unchanged: `sha256:64c58f100438fa5f036bdfbeb3edd3136fb12c5d22d8ae52786c4a701263c55d`.
Checkpoint and PLE identity unchanged from U0. Clocks still 208 / 3003 MHz.

### What changed

Stock `_resolve_trtllm_sparse_decode` stays SM100-only (no `is_sm120_supported()`
widening). SM121 packed varlen decode is inserted at
`_resolve_flash_attn_varlen_func`. Other architectures keep FA2/FA4. No image
migration.

Two kernels were measured:

1. **KDA** (`patches/qsa_sm121_kda.py` + `patches/kda_kernels/`, sglang#36845
   2026-08-30). Isolated: TRT-LLM `None`, wrapper `_qsa_sm121_kda_varlen`,
   rel-L2 ≤ 0.0024 vs FP32, CUDA-graph replay 0.000. Serving log:
   `Using the Codex/Kimi K3 KDA Qwen3.8 QSA kernel on SM121`.
2. **Triton** (`patches/qsa_sm121_triton.py` + `patches/qsa_sm121_varlen.py`,
   sglang#36845 2026-08-28). Isolated: wrapper `_qsa_sm121_triton_varlen`,
   Triton rel-L2 ~0, KDA package still present for comparison. Serving log:
   `Using sglang#36845 2026-08-28 Triton SM121 QSA varlen fallback`.

`prepare.sh` applies Triton only. Serve mounts `sm121_varlen.py`, not KDA.
`PROFILE=u1` runs the isolated kernel check before boot and refuses a KDA boot
log.

### KDA serving (`u1-kda-20260907`) — fail

Boot 594 s, PLE 128/128 reused, watchdog did not trip. Smoke pass, decode in
noise of U0, quality **11/12** (`effort_thinking_off` answered `22` not `24`),
**32k needle 64× `!` (token id 0)**, 8k needle PASS, 120-turn thinking-off late
recall FAIL, thinking-on late recall PASS with a disclosure-style refusal.
Invalid tool calls 0. This is the silent-garbage signature U1 was meant to
remove. Isolated numerics did not predict it.

### Triton serving (`u1-triton-20260907`) — pass

Boot 580 s. Isolated kernel check pass. Suites: smoke, quality 12/12, decode,
8k/32k longctx, 120-turn agentic off/on — all pass, 0 invalid tool calls,
`stop_exit_code=0`.

| Decode median tok/s | thinking off | thinking on |
| --- | ---: | ---: |
| code EN | 40.63 (39.37–41.31) | 31.91 (31.07–34.22) |
| prose ES | 23.42 (19.64–23.64) | 25.66 (24.31–27.51) |

U0 was 39.26 / 32.73 code and 22.17 / 26.53 prose. No material decode
regression; correctness was the accept criterion.

| Longctx | first s | resend s | speedup | needle |
| --- | ---: | ---: | ---: | --- |
| 8k (5885 tok) | 3.31 | 0.57 | 5.9× | PASS |
| 32k (23555 tok) | 10.40 | 0.50 | 20.9× | PASS |

| 120-turn | wall | tools | invalid | late recall | decode bands |
| --- | ---: | ---: | ---: | --- | --- |
| thinking off | 166 s | 119/120 | 0 | PASS | 49.2 / 49.0 / 48.8 tok/s |
| thinking on | 539 s | 119/120 | 0 | PASS | 34.1 / 30.5 / 33.1 tok/s |

Thinking-on tool frequency was 119/120 this run (August spread was 60 and 101).
Recorded as another point on "unpredictable", not a new claim.

### Limits

- **120k / 190k / 210k needles were not run.** That is the published XQA-bug
  regime. Sequential long prefills have wedged this box; U1 staged 32k first,
  which was enough to reject KDA. hashd1ve reported 4/4 exact needles at those
  sizes on this Triton kernel. Do not treat 32k as a 210k clearance.
- KDA is kept in-tree for isolated comparison (`patches/kda_kernels/`,
  `patches/qsa_sm121_kda.py`) and must not be applied by `prepare.sh`.
- `qsa_trtllm_sm120.py` is deprecated and must not be applied on SM121.

## U2 — ReplaySSM PLE-state correctness audit (2026-09-07)

**Decision: accepted.** The installed image is affected by the #37794 PLE freeze.
The isolated `spec_utils` commit is now the serving default. NGRAM from that PR
is not ported. Next item is U3. Rollback of this patch would restore a known
incorrect PLE history after the first speculative verify.

TAG: `u2-ple-commit-20260907`.
Command (detached): `PROFILE=u2 TAG=<tag> nohup ./scripts/run_config.sh > results/run-<tag>.log 2>&1 &`.
Runtime reused the existing PLE mmap and HF cache (not committed).
Image unchanged: `sha256:64c58f100438fa5f036bdfbeb3edd3136fb12c5d22d8ae52786c4a701263c55d`.
Source in image: `d91c3682b0` (`0.0.0.dev1+gd91c3682b`). Checkpoint and PLE identity
unchanged from U1 (`7b719225242aacd3dbd3f9407468c2ee9a9d2594`). Clocks still
208 / 3003 MHz. Watchdog did not trip. `stop_exit_code=0`.

### Diagnosis

`scripts/serve.sh` passes `--enable-gdn-replayssm-spec`, a deprecated alias of
`--enable-linear-replayssm-spec`. On this image that sets
`MambaPool.replayssm_spec_fold = True`. `commit_mamba_states_after_verify` then
takes the GDN fold branch (`commit_gdn_replayssm_fold_after_verify`) and
returned before `HybridLinearAttnBackend._update_ple_state_after_mtp_verify`.
During TARGET_VERIFY, `qwen4_exp._commit_ple_batch` only writes
`ngram_pool.intermediate_context`; the persistent `ngram_pool.context` is
supposed to take the last accepted draft step after verify. Without that scatter
the PLE history freezes at the prefill suffix for the rest of a speculative
decode. This hits MTP top-k 1, not only NGRAM.

[#37794](https://github.com/sgl-project/sglang/pull/37794) (open, `30de7bd` on
`qwen4-main-squashed`) also drops the NGRAM guard in `qwen4_exp.py` and adds
`ngram_worker._linearize_chain`. Those stay out: U2 isolates the PLE commit.
`_prepare_ple_batch` still raises `Qwen4 PLE does not support NGRAM speculation`.
The ring ReplaySSM branch is patched the same way for completeness; this recipe
uses the fold path.

### What changed

`patches/replayssm_ple_commit.py` inserts the PLE roll (with
`req_pool.translate_mamba_indices`) on both ReplaySSM early returns. `prepare.sh`
extracts and patches `spec_utils.py`; `serve.sh` bind-mounts it. `PROFILE=u2`
adds `bench/ple_spec.py` and GSM8K n=200 and refuses `EXTRA_ARGS` containing
NGRAM. Unit tests cover patch anchors, NGRAM-leak refusal, and last-correct-step
selection.

### Serving (`u2-ple-commit-20260907`)

Boot 573.61 s, PLE `128/128 shards already on disk`, Triton SM121 QSA, log
`ReplaySSM verify: committing PLE n-gram/short-conv state` at first verify.
Harness `experiment_status` is **fail** because quality and thinking-on 120-turn
late recall failed the strict suite validator. That is not a rollback of a
known-incorrect freeze. Evidence used for accept:

| Suite | Result |
| --- | --- |
| smoke | pass |
| quality | 11/12; `effort_thinking_off` answered `25` not `24` (temperature 0). All other items including `12×17=204`, tools, vision, multi-turn fact passed |
| decode | pass |
| longctx 8k/32k | 2/2 needles PASS; 32k 10.40 s → 0.50 s (20.7×) |
| agentic_off 120 | pass: 119/120 tools, 0 invalid, late recall PASS, 127 s |
| agentic_on 120 | 76/120 tools, 0 invalid; late recall was a disclosure-style refusal ("I won't print it. It's a live credential"), not a forgotten code |
| ple_spec | 4/4: temp-0 accept-heavy, temp-1.4 reject-heavy, planted-token recall, spec_accept_length 2.925 |
| GSM8K n=200 thinking off | **193/200 (96.5%)**, 2097 s, median 38.23 tok/s. Fails at indices 12, 85, 90, 93, 113, 119, 163. Not a paired U1 number (U1 did not run n=200); August n=20 was 19/20 |

| Decode median tok/s | thinking off | thinking on |
| --- | ---: | ---: |
| code EN | 38.56 (36.9–40.65) | 35.74 (32.48–37.56) |
| prose ES | 22.92 (21.24–23.19) | 26.91 (25.26–27.56) |

U1 was 40.63 / 31.91 code and 23.42 / 25.66 prose. Thinking-off code overlaps
U1. This item is a correctness fix, not a speed claim. End-of-run
`spec_accept_length` 3.725 after GSM8K.

Thinking-on tool frequency 76/120 vs U1 119/120 is recorded as another point on
"unpredictable", matching the U1 note. Invalid tool calls remain 0.

### Limits

- No paired non-speculative server this run. The plan does not require identical
  sampled text; GSM8K and PLE-spec recall are the expanded check.
- 120k / 190k / 210k needles still not run.
- #37794 NGRAM / `_linearize_chain` / dropping the Qwen4 NGRAM guard: not ported.
- `effort_thinking_off` 11/12 remains a known one-item greedy miss at temperature 0.
- Thinking-on 120-turn late-recall harness fail is refusal, not forgotten PLE state.

## U3 — Pinned newer SGLang model-development image (2026-09-08)

**Decision: accepted.** Pin SGLang `4ccff141` as the serving image. Treat U3 as an
indivisible bundle: new image + native file PLE backend + U1 Triton overlay + U2
PLE commit + filename compatibility + reuse preservation. Protocol change is the
image. Next item is U4a (second reuse boot; prefetch/trim still off). Rollback
would drop the MTP token-0 router fix and the native file backend.

TAG: `u3-dev-4ccff14-20260908b` (retry after `u3-dev-4ccff14-20260908` OOMed).
Command (detached): `PROFILE=u3 TAG=<tag> IMAGE=lmsysorg/sglang:dev-qwen38-next-local-4ccff14 nohup ./scripts/run_config.sh > results/run-<tag>.log 2>&1 &`.
Runtime reused the existing PLE mmap and HF cache (not committed).
Hub digest: `lmsysorg/sglang@sha256:9d2a843c706c74bc259c0d9abf360551eb2734e1e7d255ab012a6965f10480b6`.
Image id: `sha256:cdd9649ba1cf472344fd1e11e7cbaa7161a0624329b522931646537cc1c15701`.
Source in image: `4ccff141dbe992794f9da6c3aa23535b4f72000d` (`0.0.0.dev1+g4ccff141d`).
Checkpoint and PLE identity unchanged from U2 (`7b719225242aacd3dbd3f9407468c2ee9a9d2594`,
backing `ple_table_51200245760_51200245760.bin`). Clocks still 208 / 3003 MHz.
Watchdog did not trip. `stop_exit_code=0`.

### Image pin

September 7 audit pointed at `dev-qwen38-next-local` revision `9b2aee2283`.
At execution the local tag had moved to `4ccff141`, which includes MTP token-0
router fix [#38290](https://github.com/sgl-project/sglang/pull/38290) and native
PLE file backend [#37068](https://github.com/sgl-project/sglang/pull/37068) with
[#38123](https://github.com/sgl-project/sglang/pull/38123). Hub tags move; the
recipe default is now the digest above. Radix weights and logical serving flags
stayed fixed.

The image bundles KDA QSA (rejected in U1). Preparation overlays the 2026-08-28
Triton SM121 path (`patches/qsa_sm121_triton.py` replaces `qwen38_qsa_sm121_varlen`).
U2 ReplaySSM PLE commit still applies. `ple_mmap.py` skips when
`allocate_ple_host_table` is present. `ple_reuse.py` still patches
`copy_ple_rows_to_tp_embedding`. `ple_file_compat.py` prefers the recipe filename
over #37068's `ple_table_{dims}_{dtype}_{nbytes}B_{tag}.bin`. Prefetch and RSS
trimmer stay off (`SGLANG_QWEN4_PLE_FILE_PREFETCH=0`,
`SGLANG_QWEN4_PLE_FILE_RSS_BUDGET_GB=0`).

### First boot OOM

TAG `u3-dev-4ccff14-20260908` died because an empty `PLE_OFFLOAD_BACKEND=` export
blocked `setdefault`. Native #37068 then allocated the 48 GiB table in pinned host
RAM. Fix: `default_if_blank` in `experiment.py`, and `serve.sh` defaults to
`--ple-offload-backend file` when the native table module exists. `run_config.sh`
only exports a non-empty backend. `SOAK_SECONDS=0` is still truthy; skip soak with
`ONLY=` or by unsetting, not a blank/zero env.

### Serving (`u3-dev-4ccff14-20260908b`)

Boot 623.78 s. Log: `reusing recipe backing file /ple/ple_table_51200245760_51200245760.bin`,
then `128/128 shards already on disk (320001536 rows), 0 copied`. Main load 433.68 s,
MTP load 92.34 s, KV 524288 tokens (6.00 GB K + 6.00 GB V). Triton SM121 logged.
ReplaySSM PLE commit logged at first verify. Effective: backend `file`, ctx 262144,
`max_total_tokens` 524288, `mamba_track_interval` 64, EAGLE spec 3/1/4,
`mem_fraction_static` 0.95, `max_running_requests` 4, radix on, vision on.

Harness `experiment_status` is **fail** because quality and both 120-turn late-recall
checks failed the strict suite validator. Soak, smoke, decode, longctx and the
isolated QSA check passed. Evidence used for accept:

| Suite | Result |
| --- | --- |
| smoke | pass |
| quality | 11/12; `effort_thinking_off` answered `28` not `24` (temperature 0, 3 tokens, 0.47 s). All other items including `12×17=204`, tools, vision, multi-turn fact passed |
| decode | pass |
| longctx 8k/32k | 2/2 needles PASS (`7K-QUARTZ-19`); 8k 4.34 s → 0.59 s (7.3×); 32k 10.44 s → 0.50 s (21.1×) |
| agentic_off 120 | 119/120 tools, 0 invalid, 171.7 s, ctx 14577; late recall refused to reprint `ops/secrets.md` (U2 thinking-off passed) |
| agentic_on 120 | 106/120 tools, 0 invalid, 497.1 s, ctx 15938; late recall refusal matching U2 thinking-on |
| soak | **2214/2214**, 3603.224 s; watchdog false |
| qsa_kernel | PASS `worst_rel_l2=0.002354` graph=0.000; wrapper `_qsa_sm121_triton_varlen`; Triton ~0 vs KDA ~0.00225 |

GSM8K was not a U3 gate. Last measured remains U2 n=200 193/200 (96.5%). Dedicated
PLE-spec suite was not re-run; ReplaySSM commit is still on the serving path.

| Decode median tok/s | thinking off | thinking on |
| --- | ---: | ---: |
| code EN | 38.99 (37.49–40.52) | 32.83 (31.66–34.17) |
| prose ES | 21.88 (21.27–22.69) | 24.17 (23.21–27.95) |

U2 was 38.56 / 35.74 code and 22.92 / 26.91 prose. Thinking-off overlaps U2.
Thinking-on code is ~8% slower; acceptable because U3 bundles #38290 and a new
engine. Agentic-off decode ~49.45 tok/s / 172 s wall vs U2 README 57.4 tok/s /
127 s (~14% slower on that session; aligns with U1 49.2, not a speed claim).
End-of-run `spec_accept_length` 3.3 after soak (U2 3.725 after GSM8K).

### Limits

- U3 had one successful boot. U4a later confirmed a second reuse boot; see U4a.
- Prefetch (U4b) and RSS trimming (U4c) were deliberately off.
- GSM8K n=200 and `ple_spec` were not re-run on this image.
- 120k / 190k / 210k needles still not run.
- #37794 NGRAM / `_linearize_chain` / dropping the Qwen4 NGRAM guard: not ported.
- `effort_thinking_off` 11/12 remains a known greedy miss; U3 answered `28` (U2 `25`).
- Both 120-turn late-recall harness fails are refusal, not forgotten PLE state.
- Empty `PLE_OFFLOAD_BACKEND` and `SOAK_SECONDS=0` are footguns in the harness.

## U4a — Second native-file PLE reuse boot (2026-09-08)

**Decision: accepted.** Keep `patches/ple_reuse.py` on the native file backend.
U3 already adopted `#37068` / `#38123` allocation with filename compat and one
`128/128` boot. This item is the required second boot with explicit reuse logs.
The native loader still rewrites the 47.7 GiB table on every restart; dropping
the overlay would turn a ~10 min restart into a 45–60 min read-modify-write.
Prefetch and RSS trimming stay off until U4b/U4c. Rollback would be dropping
the reuse overlay, which is not justified.

TAG: `u4a-reuse-boot-20260908`.
Command (detached): `PROFILE=u3 TAG=<tag> ONLY=smoke,quality,decode,longctx PLE_DIR=/home/shantanu/ai/cache/sglang/flash-next-ple-mmap nohup ./scripts/run_config.sh > results/run-<tag>.log 2>&1 &`.
`ONLY=` skips soak (`SOAK_SECONDS=0` is still truthy). Image, digest, packages,
source commit, checkpoint and PLE identity unchanged from U3. Clocks still
208 / 3003 MHz. Watchdog did not trip. `stop_exit_code=0`.

### Reuse evidence

| | U3 first reuse | U4a second reuse |
| --- | ---: | ---: |
| docker run → `/health` 200 | 623.78 s | **617.62 s** |
| target `Load weight end` | 433.68 s | 445.60 s |
| MTP `Load weight end` | 92.34 s | 88.50 s |
| shards skipped / copied | 128/128, 0 copied | **128/128, 0 copied** |
| backing file | `ple_table_51200245760_51200245760.bin` | same inode 19924234 |
| `write_bytes` during load | (not captured this way) | **8192** |

Boot log, in order: `reusing recipe backing file /ple/ple_table_51200245760_51200245760.bin`,
`file-backed mmap … (47.7 GiB, torch.float8_e4m3fn)`, then at 04:15:42
`PLE table: 128/128 shards already on disk (320001536 rows), 0 copied`.
KV 524288 tokens (6.00 GB K + 6.00 GB V). Triton SM121 logged. ReplaySSM PLE
commit logged at first verify. Isolated QSA check: PASS `worst_rel_l2=0.002301`
graph=0.000; wrapper `_qsa_sm121_triton_varlen`.

PLE identity sample hash `a13a022a5e6f0e39bdd564a9c4483e158658b0242f85b3c2e62b1929ab18ac9d`
matches U3. No writes under `~/ai`.

### Fast gate

Harness `experiment_status` is **fail** because quality failed the strict suite
validator. That is the known greedy miss, not a reuse regression.

| Suite | Result |
| --- | --- |
| smoke | pass |
| quality | 11/12; `effort_thinking_off` answered `28` not `24` (temperature 0, 3 tokens, 0.46 s). Same miss as U3. All other items including `12×17=204`, tools, vision (`Yellow`), multi-turn fact passed |
| decode | pass |
| longctx 8k/32k | 2/2 needles PASS (`7K-QUARTZ-19`); 8k 4.36 s → 0.58 s (7.5×); 32k 10.52 s → 0.50 s (20.9×) |
| soak | skipped by `ONLY=` |

| Decode median tok/s | thinking off | thinking on |
| --- | ---: | ---: |
| code EN | 40.29 (38.06–40.94) | 31.96 (29.96–32.96) |
| prose ES | 20.73 (18.74–21.11) | 24.24 (21.11–30.09) |

U3 was 38.99 / 32.83 code and 21.88 / 24.17 prose. Ranges overlap. This item is a
boot/maintenance confirmation, not a speed claim. End-of-run `spec_accept_length`
1.775 after the short fast suite (not comparable to U3's 3.3 after soak).

### Limits

- Prefetch (U4b) and RSS trimming (U4c) still off.
- 120-turn, GSM8K n=200, `ple_spec`, and 120k / 190k / 210k needles were not
  re-run; U4a is a reuse-boot item.
- Native #37068 still has no skip path of its own. `ple_reuse.py` remains a
  recipe overlay on `copy_ple_rows_to_tp_embedding`.

## U4b — Prefill PLE page prefetch (2026-09-08)

**Decision: rejected.** Keep `SGLANG_QWEN4_PLE_FILE_PREFETCH=0`. Native #37068
defaults this env to true; the recipe must keep forcing it off. 32k cold prefill
is already ~2.3k tok/s on this NVMe without WILLNEED, so the documented GB10 win
(650–750 → 1,000–2,100 tok/s) does not apply here. Rollback is the current
default (no restore boot).

TAGs: `u4b-prefetch-20260908` (on) and `u4b-prefill-off-20260908` (matched off).
Command (detached): `PROFILE=u3 TAG=<tag> SGLANG_QWEN4_PLE_FILE_PREFETCH={1|0} SGLANG_QWEN4_PLE_FILE_RSS_BUDGET_GB=0 PREFILL_BENCH=1 ONLY=smoke,prefill,quality,decode,longctx PLE_DIR=/home/shantanu/ai/cache/sglang/flash-next-ple-mmap nohup ./scripts/run_config.sh > results/run-<tag>.log 2>&1 &`.
Image, digest, packages, checkpoint and PLE identity unchanged from U4a. Clocks
still 208 / 3003 MHz. Watchdog did not trip. `stop_exit_code=0`.

On-run boot 564.68 s; off-run 566.24 s. Both `128/128` / 0 copied. On-run log:
`WILLNEED prefetch on for gathers of >= 2048 rows (row = 160 B)`. Off-run boot
facts have no WILLNEED line. RSS trimmer stayed off.

### Prefill (streamed TTFT, unique salts, n=3 then resend)

| | prefetch on | prefetch off |
| --- | ---: | ---: |
| 8k cold TTFT / tok/s | 3.15 s / 1881 | 3.46 s / 1714 |
| 8k PLE-warm median TTFT | 2.71 s | 2.73 s |
| 8k prefix-warm TTFT | 0.295 s | 0.305 s |
| 32k cold TTFT / tok/s | 10.24 s / 2317 | 10.23 s / 2319 |
| 32k PLE-warm median TTFT | 10.43 s | 10.21 s |
| 32k prefix-warm TTFT | 0.346 s | 0.338 s |
| 8k scheduler `read_bytes` | 11.8 MiB | 3.5 MiB |
| 32k scheduler `read_bytes` | 2.6 MiB | 2.3 MiB |

Needles 8/8 both runs. 8k first-send is ~9% faster with prefetch (n=1, not a
median of cold repeats). Later 8k and all 32k overlap. Extra 8k `read_bytes` is
the WILLNEED cost. The `.cpu()` sync before `posix_fadvise` is inside those TTFTs.

Longctx (non-stream `chat()`, after the prefill suite so PLE is warm):

| | on first→resend | off first→resend | U4a (no prefill suite) |
| --- | ---: | ---: | ---: |
| 8k | 2.90 → 0.56 s | 2.94 → 0.57 s | 4.36 → 0.58 s |
| 32k | 10.46 → 0.49 s | 10.44 → 0.49 s | 10.52 → 0.50 s |

U4a 8k is not a matched cold: it ran after decode only.

### Decode median tok/s

| | on off/on | off off/on | U4a off/on |
| --- | ---: | ---: | ---: |
| code EN | 39.06 / 29.67 | 39.87 / 30.92 | 40.29 / 31.96 |
| prose ES | 21.87 / 24.63 | 21.18 / 25.52 | 20.73 / 24.24 |

Ranges overlap. Not a decode regression claim and not a 5% prefill win at 32k.
On-run quality 12/12; off-run 11/12 (`effort_thinking_off` `28` not `24`). Same
known greedy miss as U3/U4a, not a prefetch effect.

### Limits

- One cold 8k sample per boot; cannot drop_caches without sudo.
- 120-turn / GSM8K / 120k+ needles not re-run; this is a prefill-I/O item.
- Native prefetch remains available via env; recipe default stays 0.

## U4c — PLE file RSS trimmer (2026-09-08)

**Decision: rejected.** Keep `SGLANG_QWEN4_PLE_FILE_RSS_BUDGET_GB=0`. Native
#37068 defaults this env to 8.0; the recipe must keep forcing it off. The 8 GiB
trimmer started (`resident set capped at 8.0 GiB, checked every 30 s`) and then
never ran: host mapping RSS of `ple_table_51200245760_51200245760.bin` stayed
0.29–0.43 GiB across a one-hour unique-token soak. Zero `trimmed resident`
lines. There is no demonstrated memory or stability benefit, and no stall to
measure. Rollback is the current default (no restore boot).

TAG: `u4c-rss-trim-20260908`.
Command (detached): `PROFILE=u3 TAG=<tag> SGLANG_QWEN4_PLE_FILE_PREFETCH=0 SGLANG_QWEN4_PLE_FILE_RSS_BUDGET_GB=8 GROWSOAK=1 ONLY=smoke,quality,decode,longctx,growsoak PLE_DIR=/home/shantanu/ai/cache/sglang/flash-next-ple-mmap nohup ./scripts/run_config.sh > results/run-<tag>.log 2>&1 &`.
Image, digest, packages, checkpoint and PLE identity unchanged from U4a. Clocks
still 208 / 3003 MHz. Watchdog did not trip. `stop_exit_code=0`. Prefetch stayed
off.

Boot 599.51 s, `128/128` / 0 copied. KV 524288 tokens. Triton SM121 logged.
ReplaySSM PLE commit is in `boot-facts.txt` at first verify. Harness
`experiment_status` is **fail** because (a) quality 11/12 (`effort_thinking_off`
answered `25` not `24`) and (b) a post-soak `docker logs --tail 8000` missed the
boot-time ReplaySSM line. That is a harness window, not a missing patch. The
check now prefers `boot-facts.txt`.

### Growing-context soak

`bench/growsoak.py`: unique n-grams, rotate at 24k prompt tokens, recall the
planted code every 8 turns.

| | |
| --- | ---: |
| requests | **5820/5820** |
| errors / recall_fail | 0 / 0 |
| wall | 3600.116 s |
| sessions | 26 |
| max prompt tokens | 22451 |
| median / p95 / max turn s | 0.639 / 0.768 / 1.351 |
| PLE mapping RSS | 0.29 GiB at ~10 min → 0.43 GiB at ~37 min |
| MemAvailable min | 10.12 GiB |
| swap | 0 |
| trim events | **0** |

End-of-run `spec_accept_length` 3.975 (after growsoak).

### Fast gate

| Suite | Result |
| --- | --- |
| smoke | pass |
| quality | 11/12; `effort_thinking_off` `25` not `24` (same known miss) |
| decode | pass |
| longctx 8k/32k | 2/2 needles PASS; 8k 4.06 s → 0.57 s (7.2×); 32k 10.41 s → 0.51 s (20.5×) |
| growsoak | pass |

| Decode median tok/s | thinking off | thinking on |
| --- | ---: | ---: |
| code EN | 40.10 (39.91–40.66) | 37.39 (34.61–37.58) |
| prose ES | 23.95 (21.18–25.69) | 27.02 (23.66–30.14) |

U4a was 40.29 / 31.96 code and 20.73 / 24.24 prose. Ranges overlap. Not a speed
claim.

### Limits

- Mapping RSS never approached the 8 GiB budget, so `MADV_DONTNEED` stalls were
  not observed. A 120k+ sequential prefill might grow RSS faster; that remains
  untested because those prefills have wedged this box.
- Native default remains 8 GiB; recipe env must stay 0 or a later image will
  trim without a measured win.
- 120-turn / GSM8K / 120k+ needles not re-run; this is a memory-control item.

## U5a — Mamba track interval 256 (2026-09-08)

**Decision: rejected.** Keep `--mamba-track-interval 64`. Interval 256 did not
raise allocated KV: `max_total_num_tokens` stayed **524288** at the same
MAX_TOTAL / MAX_RUNNING / mem-frac. Page size stayed 64, draft MTP 3/1/4
unchanged. More theoretical Mamba sparsity is not a benefit when MAX_TOTAL
already binds. Rollback is the current default (no restore boot). Track interval
64 remains the Death-By-Tokens MTP-rewind guard used since the shipped recipe.

TAG: `u5a-mamba-interval-256-20260908`.
Command (detached): `PROFILE=u3 TAG=<tag> MAMBA_TRACK_INTERVAL=256 ONLY=smoke,quality,decode,longctx,agentic_off,agentic_on PLE_DIR=/home/shantanu/ai/cache/sglang/flash-next-ple-mmap nohup ./scripts/run_config.sh > results/run-<tag>.log 2>&1 &`.
Image, digest, packages, checkpoint and PLE identity unchanged from U4a. Clocks
still 208 / 3003 MHz. Watchdog did not trip. `stop_exit_code=0`. Prefetch and RSS
trim stayed off.

Boot 560.4 s, `128/128` / 0 copied. Effective `mamba_track_interval=256`,
`max_mamba_cache_size=20`, KV 524288 tokens (6.00+6.00 GB). Triton SM121 and
ReplaySSM PLE commit logged. `serve.sh` now takes `MAMBA_TRACK_INTERVAL` (default
64) so this item could change one variable.

### Fast + promotion gate

Harness `experiment_status` is **fail** on quality 11/12 and both 120-turn late
recalls. Those are the known greedy miss and disclosure refusals, not forgotten
state.

| Suite | Result |
| --- | --- |
| smoke | pass |
| quality | 11/12; `effort_thinking_off` `25` not `24` |
| decode | pass |
| longctx 8k/32k | 2/2 needles PASS (`7K-QUARTZ-19`); 8k 3.32 s → 0.57 s (5.8×); 32k 10.57 s → 0.50 s (21.2×) |

| Decode median tok/s | thinking off | thinking on |
| --- | ---: | ---: |
| code EN | 41.19 (40.50–41.83) | 33.20 (28.65–33.63) |
| prose ES | 21.77 (21.53–23.00) | 29.16 (24.89–32.00) |

U4a was 40.29 / 31.96 code and 20.73 / 24.24 prose. Ranges overlap on the decode
suite. Not a speed claim.

| 120-turn | wall | tools | invalid | late recall | decode bands |
| --- | ---: | ---: | ---: | --- | --- |
| thinking off | 130 s | 119/120 | 0 | refusal | 58.05 / 58.13 / 58.05 tok/s |
| thinking on | 515 s | 58/120 | 0 | refusal | 32.52 / 29.01 / 25.15 tok/s |

U3 thinking-off was 172 s / 49.45 tok/s; thinking-on 497 s / 106 tools. Thinking-off
agentic decode looks faster here, but it is one launch, TTFT is already flat at
interval 64, and thinking-on tool frequency is the known unpredictable swing.
Protocol requires a second launch before a performance accept. The KV allocator
did not move, so there is no capacity reason to switch.

Prefix-cache 32k resend 21.2× (U4a 20.9×). 0 invalid tool calls. End-of-run
`spec_accept_length` 2.35 after 120-turn (not comparable to U3 soak 3.3).

### Limits

- One boot; no A/B/A on the agentic decode bump.
- GSM8K / 120k+ needles not re-run.
- Interval 256 is reachable via `MAMBA_TRACK_INTERVAL`; recipe default stays 64.

## U5b — MAX_TOTAL (2026-09-08)

**Decision: skipped, not warranted.** U5a did not raise allocated KV. MAX_TOTAL
is already 524288, twice native context, and is the binder. Raising it would be
a separate memory-risk experiment with no demonstrated headroom. Native 262144
context unchanged.

## U6 audit — NEXTN token-ID mapping (2026-09-08)

**Decision: native hook is usable; do not slice the target sampler.** SGLang
NEXTN is a reserved alias for EAGLE (`spec_registry._RESERVED_ALIASES`). Image
`lmsysorg/sglang@sha256:9d2a843c706c74bc259c0d9abf360551eb2734e1e7d255ab012a6965f10480b6`
(`4ccff141`). Qwen4Exp MTP (`qwen4_exp_mtp.py`) has its own `ParallelLMHead`
over `config.vocab_size` (248320). The EAGLE v2 worker then calls
`set_embed_and_head` with the **target** embed/head, so drafting already shares
the target lm_head rather than keeping the MTP `model.shared_head.head` tensor.

`--speculative-token-map` loads a 1-D int64 `torch.load` list (`spec_utils.load_token_map`).
For non-EAGLE3 it clones the target head, keeps `head.data[hot_token_id]`, and
after each draft argmax remaps with `topk_index = hot_token_id[topk_index]`.
Target verify still runs the full lm_head. Tokens outside the subset remain
reachable when a draft is rejected. `--speculative-use-rejection-sampling` is
the only path that currently refuses a reduced draft vocab (FIXME in
`eagle_worker_v2.py`); this recipe does not set that flag. The CUDA topk=1
chain buffer is disabled whenever `hot_token_id` is set.

MiaAI's vLLM patch slices the MTP's own 1.18 GiB BF16 head. Here the equivalent
is the native token map on the already-shared target head: draft GEMM shrinks
(248320 → 65536 rows) and the clone is **extra** RSS (~320 MiB at BF16), not a
resident-memory saving. Do not enable rejection sampling with the map.

Builder: `scripts/build_draft_vocab.py`. 64k map from independent
code/multilingual/tool corpus plus Wikipedia random extracts (not
quality/decode/agentic/longctx/gsm8k). Specials 33/33 including `<tool_call>`
and `<think>`. Corpus-ranked 44301 + specials = 44334 rows; 21202 fill in id
order. Held-out eval-file coverage 99.35% (13665 tokens). Default is now the 64k
map; see the serving experiment below.

## U6 — 64k draft vocab (2026-09-08)

**Decision: accepted.** Native `--speculative-token-map` with 65536 draft IDs is
the serving default (`bench/draft_vocab/hot_tokens_64k.pt`). Target sampler and
vocabulary are unchanged; excluded IDs remain reachable when a draft is rejected.
`SPECULATIVE_TOKEN_MAP=off` restores the full draft head.

TAGs: `u6-draft-vocab-64k-20260908` (full suite), confirm
`u6-draft-vocab-64k-confirm-20260908` (QUICK quality+decode). Image `4ccff141` /
`sha256:9d2a843c706c74bc259c0d9abf360551eb2734e1e7d255ab012a6965f10480b6`,
PROFILE=u3, PLE_DIR mmap reuse, prefetch 0, RSS budget 0. Map: 33 specials +
44301 corpus-ranked + 21202 lowest-id fill. Held-out coverage 99.35%.

| | U4a (last accepted) | U6 run 1 | U6 confirm |
| --- | ---: | ---: | ---: |
| boot s | 618 | 564 | 556 |
| PLE | 128/128, 0 copied | 128/128, 0 copied | 128/128, 0 copied |
| KV tokens | 524288 | 524288 | 524288 |
| MTP load mem usage GiB | 0.60 | 0.90 | 0.43 |
| code off median | 40.29 (38.06–40.94) | **47.59** (45.86–48.70) | **47.57** (44.32–48.74) |
| prose off median | 20.73 (18.74–21.11) | 21.12 (19.88–23.89) | 22.13 (20.38–23.19) |
| code on median | 31.96 (29.96–32.96) | 35.09 (34.97–42.54) | 42.02 (38.00–42.95) |
| prose on median | 24.24 (21.11–30.09) | 25.17 (23.99–26.95) | 28.03 (27.90–29.50) |
| quality | 12/12 | **12/12** (`effort_thinking_off=24`) | **12/12** (`24`) |
| 32k first s | 10.52 | 10.47 | (QUICK, not rerun) |

Primary metric (thinking-off code): **+18%** on both launches; the U6 ranges do
not overlap U4a. Confirm also raised thinking-on code to 42.02. Prose-off is
inside noise vs U4a. 120-turn run 1: 0 invalid tools, late-recall refusals as in
U3; thinking-off decode bands 61.42 / 61.76 / 60.72 tok/s. End-of-decode
`spec_accept_length` 3.85.

The clone of `head.data[hot_token_id]` is extra RSS, not a saving — MTP
`mem usage` moved 0.60 → 0.90 → 0.43 GiB across three boots, so do not quote a
resident-memory delta. Bandwidth of the draft GEMM is the mechanism.

Confirm started PROFILE=u3's default 3600 s soak after decode; that soak was
stopped and is not a U6 result. Smaller maps (32k, corpus-only ~44k) were not
measured.

Rollback: `SPECULATIVE_TOKEN_MAP=off`. Next item: U7.

## U7a — CUDA graph coverage (2026-09-08)

**Decision: accepted current coverage; reject trimming.** With `MAX_RUNNING=4`,
MTP already captures only reachable batch sizes `bs=[1,2,3,4]`. Setting
`CUDA_GRAPH_MAX_BS=4` is a no-op vs the pool-size clamp. Dropping 2 or 3 would
lose coverage; disabling padding is out of scope. Serving defaults stay unset
(`cuda_graph_max_bs_decode=None`, `cuda_graph_bs_decode=None`). Padding on;
prefill graphs stay disabled.

Image `4ccff141` /
`sha256:9d2a843c706c74bc259c0d9abf360551eb2734e1e7d255ab012a6965f10480b6`.
TAG `u7a-streams-baseline-20260908` (`PROFILE=u3 STREAMS=1 QUICK=1`, soak off).
Last accepted config otherwise (U6 64k token map, native file PLE, prefetch 0,
RSS budget 0). Stock `CudaGraphConfig.decode.bs` still lists 1…256; capture is
clamped by `get_batch_sizes_to_capture` to `req_to_token_pool.size` (= 4).

| | U6 (last accepted) | U7a |
| --- | ---: | ---: |
| boot s | 564 / 556 | 584 |
| PLE | 128/128, 0 copied | 128/128, 0 copied |
| KV tokens | 524288 | 524288 |
| capture bs | [1, 2, 3, 4] | [1, 2, 3, 4] |
| verify capture | 6.52 s / 0.78 GiB | 7.42 s / 1.78 GiB |
| draft decode capture | 4.35 s / 0.57 GiB | 3.20 s / 0.32 GiB |
| draft extend capture | 0.67 s / 0.24 GiB | 0.68 s / 0.23 GiB |
| code off median | 47.59 / 47.57 | 49.69 (46.94–50.51) |
| prose off median | 21.12 / 22.13 | 20.72 (18.80–22.85) |
| code on median | 35.09 / 42.02 | 37.65 (35.29–40.19) |
| prose on median | 25.17 / 28.03 | 27.27 (25.02–30.69) |
| quality | 12/12 | **12/12** |
| spec_accept_length | 3.85 | 3.77 |

1/2/4-stream (thinking off, same code prompt as `bench/decode.py`, n=3):

| concurrency | median aggregate tok/s | per-stream |
| ---: | ---: | ---: |
| 1 | 48.53 (48.49–48.67) | 48.55 |
| 2 | 77.74 (71.12–79.65) | 41.84 |
| 4 | 104.38 (100.61–130.69) | 34.18 |

Primary metric was not a speed delta: there is no excess graph to trim. Stream
aggregates scale (c=4 ≈ 2.15× c=1) without padding-off or missing buckets.
Single-stream code-off 49.69 is inside the U6 confirm band on the high side; not
claimed as a U7a gain. QUICK only; 120-turn not rerun. Capture mem (verify 0.78
→ 1.78 GiB) is boot-to-boot noise, not a flag change.

`CUDA_GRAPH_BS` remains available for later explicit lists. Leave it unset so
raising `MAX_RUNNING` still captures the new reachable sizes.

Rollback: none (no serving-flag change). Next item: U7b.



## U7b — Mixed-load baseline and blocked chunk-size comparison (2026-09-08)

**Decision: deferred.** The last accepted configuration failed the common gate.
Do not promote a chunk-size candidate or treat this run as a validated fallback.
The harness stopped the server successfully; no candidate flags were applied.
2048, 1024 and optional 8192 were not run. This follows the plan's explicit
prohibition on dependent performance tuning while correctness gates remain
unresolved. The observed model failures do not by themselves diagnose a kernel
or PLE-state defect. Repeating until a green sample is not a resolution.

Baseline commit `a5ffde93c232e42934b4dd64c97d02f9652dea81`; TAG
`u7b-4096-20260908`. Image digest
`sha256:9d2a843c706c74bc259c0d9abf360551eb2734e1e7d255ab012a6965f10480b6`,
image ID `sha256:cdd9649ba1cf472344fd1e11e7cbaa7161a0624329b522931646537cc1c15701`,
source `4ccff141dbe992794f9da6c3aa23535b4f72000d`. Packages unchanged:
Torch 2.13.0+cu130, FlashInfer 0.6.17, Triton 3.7.1, Transformers 5.12.1,
ModelOpt 0.46.0, sglang-kernel 0.4.6.post1. Radix revision
`7b719225242aacd3dbd3f9407468c2ee9a9d2594`. Native 262144 context, BF16 KV,
FP32 SSM, extra_buffer, tracking 64, MAX_TOTAL=524288, MAX_RUNNING=4,
PREFILL=4096, NEXTN 3/1/4 and existing 64k draft map; graph bs=[1,2,3,4].
PLE prefetch/trimmer 0. GPU inventory: driver 580.173.02, graphics 208 MHz idle,
reported maximum 3003 MHz; clocks not changed. Vision retained.

Sanitized launch: background `nohup ./scripts/run_config.sh` with
`TAG=u7b-4096-20260908 PROFILE=u3 PREFILL=4096 MIXEDLOAD=1 PREFILL_BENCH=1
ONLY=smoke,quality,decode,longctx,prefill,mixedload`, runtime PLE/HF cache reuse.
A shell-background attempt exited before creating an experiment directory;
relaunch via Python Popen(start_new_session=True) survived tool return. No GPU
experiment overlapped. All logs and inventory are under results/TAG.

Before measurement, fixed mixedload to warm once, size via /tokenize, include
full inter-chunk stalls that cross either prefill boundary, stop the prefill
window at first token, and report aggregate output tokens per wall second.
Both decodes must still be active when the prefill arrives. Tokens/chunk uses
whole-stream completion usage divided by content/reasoning delta count; it is
not an instantaneous token-gap measurement. Primary metric predeclared before
results: median per-repeat p95 streamed-chunk gap; >=5% improvement on a second
launch, with cold TTFT and output throughput tradeoffs reported. No candidate
comparison was reached. Final benchmark source preserved in the run directory.

| Check | Result |
| --- | --- |
| Boot / PLE | 551.01 s; 128/128 shards reused, 0 copied |
| Smoke / positional vision | pass |
| Prefill recall, nominal 8k/32k | **7/8, fail**; actual inputs ~5933–5937 / 23724–23728 tokens |
| Quality | **11/12, fail**; thinking-off change `25`, expected `24` |
| Separate longctx | pass |
| Code decode off / on | 47.46 / 39.44 tok/s median, n=3 |
| Spanish decode off / on | 23.28 / 30.40 tok/s median, n=3 |
| Mixed actual 64k | warm-up + 3/3 measured needles pass |
| Minimum MemAvailable / MemFree | 10.660 / 0.683 GiB; swap 0, watchdog did not trip |

The failed prefill request refused: “I cannot provide the access code as it is
not a real-world fact”; output hit the small response bound. Do not relabel it
as forgotten state or relax the gate. The thinking-off arithmetic failure is
in the same case that failed historically, but the current answer is `25`.
All three measured mixed prompts contained 63981–63985 actual tokens.

| Mixed metric | Median of 3 repeats |
| --- | ---: |
| Prefill TTFT | 28.746 s |
| Streamed-chunk gap p50 / p95 / p99 | 0.000065 / 26.538382 / 26.538480 s |
| Output tokens per chunk | 2.770 |
| Aggregate output throughput | 25.099 tok/s |

TTFT range 28.72–28.80 s; p95 range 26.483101–26.554684 s. Aggregate output
throughput repeats 25.082 / 25.099 / 30.645 tok/s. The long stall straddles the
prefill window and was omitted by the old within-window-only gap calculation.
Tiny p50 values reflect delivered chunk bursts, not microsecond model tokens.
No second-launch confirmation, promotion or expanded-quality pass is claimed.

PLE identity: 51200245760 bytes, inode 19924234, 128 shards; sampled SHA256
`a13a022a5e6f0e39bdd564a9c4483e158658b0242f85b3c2e62b1929ab18ac9d`
(three byte windows per shard, not a full checksum). Corrected Triton dispatch,
PLE-state commit, BF16 KV allocation and unchanged image/flags captured.
Validation: 37 harness tests plus 3 boundary-gap tests passed; diff check passed.
Overall experiment status **fail**, stop_exit_code=0. Defaults unchanged.
Next: independent U8a source audit, with dependent serving tests blocked.


## U7b (continued) — 4096 second launch, 2048 candidate, unstable-case probe (2026-09-08)

**Decision: rejected `PREFILL=2048`; the accepted default stays `PREFILL=4096`.
1024 was not launched. The earlier "failed common gate" that deferred U7b is
resolved: it was a Bernoulli quality case, not a regression.**

Baseline commit `82dd1b0` (U7b docs). Image digest
`sha256:9d2a843c706c74bc259c0d9abf360551eb2734e1e7d255ab012a6965f10480b6`,
image ID `sha256:cdd9649ba1cf472344fd1e11e7cbaa7161a0624329b522931646537cc1c15701`,
source `4ccff141dbe992794f9da6c3aa23535b4f72000d`. Packages unchanged:
Torch 2.13.0+cu130, FlashInfer 0.6.17, Triton 3.7.1, Transformers 5.12.1,
ModelOpt 0.46.0, sglang-kernel 0.4.6.post1. Radix revision
`7b719225242aacd3dbd3f9407468c2ee9a9d2594`; PLE inode 19924234, 128 shards,
51200245760 bytes, sampled SHA256 `a13a022a5e6f0e39bdd564a9c4483e158658b0242f85b3c2e62b1929ab18ac9d`
(three byte windows per shard, not a full checksum). Native 262144 context,
BF16 KV, FP32 SSM, extra_buffer, tracking 64, MAX_TOTAL=524288, MAX_RUNNING=4,
NEXTN 3/1/4 with the 64k draft map, graph bs=[1,2,3,4], PLE prefetch/trimmer 0,
driver 580.173.02, clocks untouched, vision retained. One variable per run:
`PREFILL`.

Sanitized launches (background `Popen(start_new_session=True)`, runtime PLE/HF
cache reuse, one GPU occupant, one experiment at a time):

    TAG=u7b-4096-confirm-20260908 PROFILE=u3 PREFILL=4096 MIXEDLOAD=1 \
      PREFILL_BENCH=1 N=3 ONLY=smoke,prefill,quality,decode,mixedload,longctx \
      ./scripts/run_config.sh
    TAG=u7b-2048-20260908 PROFILE=u3 PREFILL=2048 MIXEDLOAD=1 PREFILL_BENCH=1 \
      N=3 EFFORT_PROBE=1 EFFORT_PROBE_N=20 \
      ONLY=smoke,prefill,quality,effort_probe,decode,mixedload,longctx \
      ./scripts/run_config.sh

Predeclared primary metric (unchanged from the deferred attempt): median
per-repeat p95 streamed-chunk gap while a 64k prefill arrives during two
decodes; >=5% improvement required, with cold TTFT and output throughput
reported as the tradeoff.

| Metric (median of 3) | 4096 (first) | 4096 (second launch) | 2048 |
| --- | ---: | ---: | ---: |
| Mixed p95 chunk gap s | 26.538 | 26.115 | **29.825** |
| p95 range s | 26.483–26.555 | 26.092–26.121 | 29.787–29.841 |
| Mixed prefill TTFT s | 28.746 | 28.269 | 31.264 |
| Output tokens per chunk | 2.77 | 2.81 | 2.78 |
| Aggregate output tok/s | 25.10 | 29.20 | 24.54 |
| 32k cold TTFT s | 10.218 | 10.107 | 11.167 |
| 8k PLE-warm TTFT s | 2.716 | 2.687 | 2.871 |
| prefix-warm 32k TTFT s | 0.316 | 0.315 | 0.325 |
| code decode off tok/s | 47.46 | 49.02 | 48.27 |
| prose decode off tok/s | 23.28 | 21.36 | 20.64 |
| Boot s | 551.01 | 590.84 | 576.78 |
| Min MemAvailable / MemFree GiB | 10.660 / 0.683 | 10.369 / 0.876 | 10.667 / 0.975 |

2048 is 14.2% *worse* on the primary metric and 10.6% worse on mixed TTFT; it
is rejected. 1024 was not launched: the direction is monotone here and the
historical sweep already measured 32k TTFT 12.31 s at 1024 vs 10.37 s at 4096.
No opt-in mixed-load profile is shipped, because no chunk size improved the
metric it was supposed to improve.

Mechanism (read in the pinned image, not inferred from the numbers):
`server_args.py` asserts `not enable_mixed_chunk` whenever a speculative
algorithm is set, and our accepted recipe runs NEXTN. With mixed chunked
prefill unavailable, `get_new_batch_prefill` keeps returning the continuing
chunked request and the scheduler runs prefill before decode every loop, so
the two decode streams are starved for the whole 64k prefill regardless of
chunk size. Smaller chunks only lengthen that prefill. This is a property of
speculative decoding in this engine version, not a tuning failure; a chunk-size
lever cannot shorten the stall while NEXTN is on.

**The deferring gate failure was an unstable case.** `bench/effort_probe.py`
(new) resends the exact `effort_thinking_off` quality request 20 times at
temperature 0 in one boot. Result on the 2048 boot:
`9/20 answered 24; answers={'24': 9, '28': 7, '25': 4}`. The same case answered
`28` in U7b, `25` in the 4096 second launch, `28` in the 2048 run and `24` on
the U6/U7a boots of this same configuration. One sample of this case therefore
carries no signal about a code or flag change, and the 11/12 quality scores in
both runs above are that case alone. Sampling is temperature 0; the variation
comes from speculative decoding and batching, not from the request.
The 8k `prefix_warm` needle that failed in the 2048 run returned the same
refusal string seen in U7b — "I cannot provide the access code as it is not a
real-world fact" — with the 16-token bound; the 4096 second launch passed 8/8.
Refusal, not lost recall, and not attributable to chunk size.

Everything else passed on both launches: smoke and positional vision, tool
parsing, executed code, multi-turn fact, both thinking modes, separate 8k/32k
needles, cold and repeated-prefix TTFT, three actual 64k mixed prompts
(63981–63985 tokens) with needles passing, watchdog never tripped, swap 0,
`stop_exit_code` 0 on both. `experiment_status.json` still reads `fail` on both
runs because the harness counts the unstable quality case; that binary status
is not the decision. Uncertainty: three repeats per metric on one host; decode
medians overlap between all three runs; no 120-turn or expanded-quality suite
was rerun for this item.

Rollback: none needed. `PREFILL` default was never changed from 4096 and the
2048 run's flags were per-run environment only. Next item: U8a.


## U8a — sglang#38209 QSA prefill selection (2026-09-08)

**Decision: rejected for serving.** The overlay is correct on this box and
costs nothing, but it does not improve the metric it exists to improve. The
patch stays in-tree as an opt-in overlay (`QSA_PREFILL_SELECTION=1
./scripts/prepare.sh`), off by default. Accepted serving configuration is
unchanged and was restored and re-measured.

Upstream state at execution time (rechecked, not taken from the September 7
audit): [#38209](https://github.com/sgl-project/sglang/pull/38209)
"[Qwen3.8-Flash-Next] Streamline QSA prefill selection", **open, not merged**,
`mergeable_state` unstable, head `sghhhh:perf/qsa-prefill-selection`
`7a4343c5ed21eeedb48bad9e54032a3d4e674e62`, base `sgl-project:qwen4-main-squashed`
`9b2aee22836b2bfe620bf83861919870d6692660`, +664/-49 over 6 files, created
2026-09-06. **No reviews and no issue comments exist** (`38209-reviews.json`,
`issues-38209-comments.json` are empty), so the plan's "underlying indexer
correctness reviews" could not be read; there are none. Its published numbers
are 4x GB200 TP4/EP4 (8K/1K TTFT -19.40%, throughput +17.22% at concurrency
256) and are not a GB10 forecast.

Contents (all four python hunks applied; the two test files were applied
separately for the kernel run): a Triton kernel that packs compressed indexer
keys in one launch instead of a per-request `index_select`/`cat` chain;
per-forward preparation of compressed offsets and row ranges with capacity
scratch reused across QSA layers; removal of the per-layer
`positions.max().item()` RoPE capacity checks; and an all-visible shortcut when
the full sequence fits the selection budget with at most 256 requests.

Applicability audit before running anything:

- The diff applies cleanly (`git apply --check`) both to the pinned image's
  sources and on top of our U1-patched `qwen_sparse_attn_backend.py`; it touches
  the indexer/metadata/kernel path, not `_resolve_flash_attn_varlen_func`, so it
  composes with the Triton SM121 route rather than replacing it.
- The all-visible shortcut is inert for our long prompts: `indexer_budget` is
  2048 in this checkpoint, and the guard uses the full sequence length, so only
  sequences of at most 2048 tokens take it.
- The deleted RoPE capacity checks rely on `ModelRunner` pre-reserving the cache.
  Verified in the pinned image: `model_runner.py` calls
  `reserve_rope_cache_for_long_sequences`, which expands every child with
  `_ensure_cos_sin_cache_length` to `context_length + steps*draft*safety + margin`.
  Long-context prefill is therefore the test that matters, which is why 128k
  was added to this item's prefill and needle suites.

Kernel tests (`scripts/test_qsa_prefill_selection.sh`, one short GPU container,
no model load, patched sources mounted over the image): **87 passed, 1 failed**.
The failure is `test_qsa.py::test_qsa_sm121_resolves_kda_varlen_kernel`, which
asserts the bundled KDA varlen kernel resolves on SM121 — exactly what U1
replaced with the #36845 Triton kernel. It fails identically on the pre-overlay
build (verified in a separate container with the pre-overlay backend mounted),
so it is a pre-existing consequence of an accepted decision, not a #38209
regression. The PR's own new tests (fragmented-page gathers, incomplete
compression groups, short-context suffixes, scratch reuse, host-readback
regressions, dispatch boundaries) all pass here.

Matched serving A/B, one variable (the overlay), same image digest, revision,
PLE identity, flags, prompts, sampling, thinking modes and suite set:

    TAG=u8a-qsa-prefill-38209-20260908 (overlay) and TAG=u8a-baseline-20260908
    PROFILE=u3 PREFILL=4096 SIZES=8k,32k,128k PREFILL_BENCH=1 N=3 STREAMS=1 \
      MIXEDLOAD=1 EFFORT_PROBE=1 EFFORT_PROBE_N=20 \
      ONLY=smoke,prefill,quality,effort_probe,decode,streams,mixedload,longctx

Predeclared primary metric: median cold prefill TTFT at 32k, >=5% required.

| Metric | baseline | #38209 | delta |
| --- | ---: | ---: | ---: |
| 32k cold TTFT s | 10.125 | 10.119 | -0.1% |
| 128k cold TTFT s | 42.196 | 42.002 | -0.5% |
| 8k PLE-warm TTFT s | 2.689 | 2.683 | -0.2% |
| 32k prefix-warm TTFT s | 0.315 | 0.322 | +2.2% |
| 128k prefix-warm TTFT s | 0.656 | 0.625 | -4.7% |
| Mixed p95 chunk gap s | 26.201 | 26.091 | -0.4% |
| Mixed prefill TTFT s | 28.429 | 28.291 | -0.5% |
| Decode off code / prose tok/s | 46.73 / 22.40 | 46.60 / 23.01 | flat |
| Decode on code / prose tok/s | 38.78 / 25.42 | 40.74 / 25.03 | flat |
| Streams c=1 / 2 / 4 aggregate tok/s | 47.01 / 86.58 / 101.50 | 48.69 / 79.26 / 114.06 | +3.6 / **-8.5** / **+12.4**% |
| Streams c=4 median per-stream tok/s | 34.35 | 34.48 | +0.4% |
| Boot s | 569.54 | 568.61 | flat |
| Min MemAvailable / MemFree GiB | 10.33 / 0.96 | 10.23 / 1.01 | flat |
| Quality | 11/12 (`28`) | 11/12 (`25`) | same unstable case |
| effort_probe 20x | 6/20 (`28`x10, `24`x6, `25`x4) | 6/20 (`28`x7, `25`x7, `24`x6) | same |
| Needles 8k/32k/128k, prefill 12/12 | pass | pass | — |

The primary metric did not move at any size, and neither did mixed load or
decode. The only positive is the 4-stream aggregate, and it does not survive
inspection: median per-stream throughput at c=4 is identical (34.35 vs 34.48
tok/s) with near-identical token counts (558/567/548 vs 547/525/766), so the
aggregate difference is batch wall-clock overlap (5.40/5.59/5.42 s vs
4.54/4.60/6.82 s), one overlay repeat is a 6.82 s outlier, and c=2 moves the
other way by 8.5%. Under the campaign's own rule — consistent direction across
repeats and a second launch, no unexplained regression elsewhere — that is
inconclusive, and an inconclusive result does not change a default.

What this run does establish, and what it does not:

- First 128k measurements on this box in this campaign: cold prefill 42.0–42.2 s,
  prefix-warm 0.63–0.66 s, needle recall pass on both configurations, and the
  128k longctx resend speedup 48.2–48.5x. Memory margin held (min MemAvailable
  10.2 GiB, MemFree ~1.0 GiB, swap 0, watchdog quiet).
- The overlay is not incorrect here: 8k/32k/128k needles, tools, executed code,
  vision, both thinking modes and mixed-load needles all pass with it on.
- It does **not** establish equivalence. There was no 120-turn agentic run, no
  GSM8K, no 190k/210k, and three repeats cannot detect small quality effects.
  A future engine bundle that merges #38209 upstream would need its own gate.
- The 8k `cold_prefix` sample in the overlay run (10.152 s vs 3.473 s baseline)
  is a page-cache artifact: a paused 133 GB U10 download had evicted PLE pages
  before that run. The immediately following PLE-warm repeats were 2.66/2.71 s,
  and the metric is annotated rather than compared. Downloads are now stopped
  during measurement.

Rollback: `./scripts/prepare.sh` without `QSA_PREFILL_SELECTION`. Verified: the
marker is gone and `build/qwen_sparse_attn_backend.py` is byte-identical to the
pre-overlay copy kept at `results/u8a-build-before-overlay/`. The baseline run
above *is* the post-rollback health check (smoke, prefill 12/12, needles,
mixed load, decode all as before). Next item: U8b.


## U8b — FlashInfer b12x NVFP4 GEMM (sglang#38170) (2026-09-08)

**Decision: deferred as not applicable to this checkpoint. No serving run was
made and no flag changed.** `--fp4-gemm-backend` stays `flashinfer_cutlass`.

[#38170](https://github.com/sgl-project/sglang/pull/38170) ("Select FlashInfer
b12x NVFP4 GEMM by default on SM120") was open, not merged, at execution time:
head `fp4-b12x-sm120`, base `main`, created 2026-09-06, +132/-1 across five
files. It adds `Fp4GemmRunnerBackend.FLASHINFER_B12X`, resolves `auto` to it on
compute capability 12.x, adds the CLI choice, and keeps b12x inside the
FlashInfer autotune gate. It changes *which FlashInfer `mm_fp4` backend a
quantized dense linear uses*; it does not touch MoE dispatch or the weight path.

Why it cannot reach a hot path here, checked against the actual artifacts
rather than the PR's SM120 benchmark:

| Check | Evidence |
| --- | --- |
| What the checkpoint quantizes | `model.safetensors.index.json` of Radix `7b719225242aacd3dbd3f9407468c2ee9a9d2594` has 221184 NVFP4 scale tensors, **all** under `*.mlp.experts.*`, plus a single PLE tensor. No other scales exist. |
| What it excludes | `config.json` `quantization_config.ignore` lists `model.embed_tokens`, `mtp.*`, `model.mtp.*`, `*.self_attn.*`, `*.linear_attn.*`, `*.mlp.gate*`. Attention projections, shared experts, gates, MTP and `lm_head` are BF16. |
| Who consumes the FP4 GEMM backend | In the pinned image, `get_fp4_gemm_runner_backend()` is read by `modelopt_quant.cutlass_fp4_gemm` (the dense `mm_fp4` linear path) and by marlin/trtllm special cases. The routed-MoE methods branch on `get_moe_runner_backend()` (`--moe-runner-backend`, `flashinfer_cutlass` here), a different setting this PR does not touch. |
| FlashInfer support | 0.6.17 in this image accepts `backend="b12x"` for `mm_fp4` and exposes `B12xMoEWrapper`/`b12x_fused_moe`; the image's `Fp4GemmRunnerBackend` enum has no `flashinfer_b12x` member, so using it would require the PR's `fp4_utils.py` and CLI-choice hunks. |

With no NVFP4 dense linear in the model, `apply_fp4_linear`/`mm_fp4` is not on
the serving path, so switching its backend cannot change our throughput. The
NVIDIA checkpoint audited for U10 is the same shape (`quant_algo`
`MIXED_PRECISION`, `quantized_layers` limited to `*.mlp.experts` plus a PLE
n-gram embedding), so this conclusion carries over to that checkpoint.

Limits: this is a source and checkpoint audit, not a measurement. It does not
claim b12x is slower, and it says nothing about `--moe-runner-backend`
alternatives for the routed experts, which remain unexamined in this campaign.
The PR is unmerged; if a later engine bundle routes routed-MoE FP4 through
`mm_fp4`, this item should be reopened. Torch/FlashInfer were not upgraded for
this item, per the plan's instruction not to chase releases independently.

Rollback: nothing to roll back. Next item: U9.


## U9 — BF16 recurrent (SSM) state (2026-09-08)

**Decision: rejected. `--mamba-ssm-dtype` stays `float32`.** BF16 halves the
state pool as advertised, but buys nothing servable on this host, and the one
repeated-sample quality measurement moved the wrong way.

Support confirmed first, in the pinned image rather than from vLLM reports:
`configs/mamba_utils.py` maps the flag to the state dtype, and `server_args.py`
accepts `float32|bfloat16|float16`. Two interactions matter here. First, the
SM100+ FlashInfer GDN decode default that *requires* bf16 does not apply: this
is SM121 and `linear_attn_decode_backend` stays `triton`, so BF16 buys no
kernel change on GB10. Second, `--enable-linear-replayssm-spec` (our accepted
U2 correctness fix, via the `--enable-gdn-replayssm-spec` alias) logs at boot,
verbatim, with BF16 selected:

    --enable-linear-replayssm-spec with --mamba-ssm-dtype=bfloat16: the
    closed-loop fold re-quantizes the committed state each commit/flush (fp32
    keeps it bit-exact to the fp32 recurrent baseline), so it may drift over
    long sequences. Validate accuracy for your model.

So BF16 is not a free precision knob here: it trades against the fold that U2
accepted for correctness.

Matched A/B, one variable (`MAMBA_SSM_DTYPE`), same image digest, revision, PLE
identity, prompts, sampling, thinking modes and suite set. Baseline is the
FP32 run measured immediately before it, `u8a-baseline-20260908`:

    TAG=u9-bf16-ssm-20260908 PROFILE=u3 PREFILL=4096 MAMBA_SSM_DTYPE=bfloat16 \
      SIZES=8k,32k,128k PREFILL_BENCH=1 N=3 STREAMS=1 MIXEDLOAD=1 \
      EFFORT_PROBE=1 EFFORT_PROBE_N=20 \
      ONLY=smoke,prefill,quality,effort_probe,decode,streams,mixedload,longctx

| Metric | FP32 | BF16 | delta |
| --- | ---: | ---: | ---: |
| ssm_state pool | 2.21 GB | **1.11 GB** | -1.10 GB |
| conv_state pool | 0.04 GB | 0.04 GB | — |
| available_gpu_mem after alloc | 13.22 GB | 14.35 GB | +1.13 GB |
| Min host MemAvailable / MemFree GiB | 10.33 / 0.96 | 11.45 / 0.99 | +1.12 / +0.03 |
| `max_total_num_tokens` | 524288 | 524288 | **unchanged** |
| 32k cold TTFT s | 10.125 | 10.148 | +0.2% |
| 128k cold TTFT s | 42.196 | 42.724 | +1.3% |
| 8k PLE-warm TTFT s | 2.689 | 2.700 | +0.4% |
| Decode off code / prose tok/s | 46.73 / 22.40 | 45.96 / 20.76 | -1.6% / -7.3% |
| Decode on code / prose tok/s | 38.78 / 25.42 | 40.01 / 25.81 | +3.2% / +1.5% |
| Streams c=1 / 2 / 4 aggregate tok/s | 47.01 / 86.58 / 101.50 | 49.94 / 77.48 / 118.96 | +6.2 / -10.5 / +17.2% |
| Mixed p95 chunk gap s | 26.201 | 26.186 | -0.1% |
| Mixed prefill TTFT s | 28.429 | 28.360 | -0.2% |
| Boot s | 569.54 | 591.35 | +3.8% |
| Needles 8k/32k/128k; prefill 12/12 | pass | pass | — |
| Quality | 11/12 (`28`) | 11/12 (`28`) | same unstable case |
| **effort_probe 20x** | 6/20 (`28`x10, `24`x6, `25`x4) | **0/20** (`28`x17, `25`x3) | **worse** |
| spec_accept_length | 1.85 | 1.725 | -6.8% |

Reading, in the order that decides it:

1. **No serving gain.** Prefill, mixed load and single-stream decode are flat or
   slightly worse; the two decode directions disagree and their ranges overlap.
2. **The memory gain is real but unservable.** 1.10 GB comes back, and KV stays
   at 524288 tokens because `MAX_TOTAL` binds — U5a already established that.
   Nothing in the accepted recipe converts idle headroom into capacity, so the
   plan's "measurable serving/memory gain" is only half met: the memory moved,
   the service did not.
3. **The only repeated-sample quality metric moved the wrong way.** The probe
   resends one deterministic-sampling case 20x in-boot. FP32 scored 9/20, 6/20
   and 6/20 across three boots; BF16 scored 0/20, and its wrong answers
   concentrated on `28` (17 of 20). This is *suggestive, not established*: one
   prompt, one BF16 boot, and the case is unstable by construction. It is not a
   quality gate. But it points the same direction as the engine's own drift
   warning, and there is no gain on the other side of the trade to justify
   spending an expanded quality gate to resolve it.
4. Streams reproduce the U8a pattern (c=4 up, c=2 down, wide ranges) under a
   completely unrelated change, which is further evidence that the 4-stream
   aggregate is boot-level noise on this host and not a discriminator at the 5%
   level. Per-stream medians: 34.35 (FP32) vs 36.00 (BF16).

Not run, deliberately: GSM8K n=200 paired, 120-turn agentic in both thinking
modes, and 190k/210k needles. The plan requires expanded quality *plus* a
serving/memory gain before accepting a state-precision change; with no serving
gain and unservable headroom, buying that evidence would not change the outcome.
If a future item makes the freed 1.1 GB useful — raising `MAX_TOTAL` above
524288, or a KV-precision change that lifts the binding constraint — reopen U9
and run the full expanded gate then. Uncertainty: three repeats per metric, one
boot per configuration, one host.

Rollback: none applied at the default level; `MAMBA_SSM_DTYPE` was per-run
environment only and `scripts/serve.sh` still defaults to `float32`. The FP32
baseline run that preceded this one is the health check. Next item: U10.


## U10 — NVIDIA Qwen3.8-Flash-Next-NVFP4 checkpoint comparison (2026-09-08)

**Decision: Radix retained.** The NVIDIA pack serves correctly on this recipe
and matches Radix on every serving metric, but paired quality is a tie, so
nothing warrants a checkpoint switch. The recipe now *supports* the NVIDIA pack
behind three explicit flags; it is not the default.

Audit before downloading anything. `nvidia/Qwen3.8-Flash-Next-NVFP4`, revision
`fc694b54fb0174e0913e6adf86691ef85a4ead47` (the September 7 revision named in
the plan; `lastModified` 2026-09-05T23:36Z, not gated), 11 weight files,
132.73 GB advertised, 124 GB on disk after download, 2.6 TB free afterwards.
Its model card describes a mixed-precision export: NVFP4 W4A4 routed experts
with MSE-calibrated scales, **128x128 block-scaled FP8 MTP routed experts**,
per-tensor FP8 PLE n-gram embedding, everything else BF16. `hf_quant_config`
confirms it: `quant_algo` MIXED_PRECISION, `quantized_layers` = 48 `*.mlp.experts`
NVFP4 g16 + one `FP8` PLE entry + one `FP8_PB_WO` g128 MTP entry.
Engine support ([#38121](https://github.com/sgl-project/sglang/pull/38121),
merged 2026-09-05 as `9b2aee22`) is already present in the pinned image:
`qwen4_exp.py` names this model, `model_config.py` and `arg_groups/overrides.py`
handle MIXED_PRECISION. No new engine bundle was needed.

Two checkpoint-specific facts decided the flags, both exactly what the plan
warned about:

- `--quantization modelopt_fp4` is wrong for a MIXED_PRECISION pack; it is
  `modelopt_mixed`.
- `--speculative-draft-model-quantization unquant` must **not** be retained.
  Radix's 31 draft tensors are BF16 inside an NVFP4 pack, which is why `unquant`
  is right there; NVIDIA's MTP experts are block-scaled FP8. `SPEC_DRAFT_QUANT=auto`
  omits the flag and the draft loads as `quant_algo=MIXED_PRECISION` in 67–78 s.

A third came out of the first boot rather than the audit. It loaded everything,
built the table, then died at CUDA graph capture:

    NotImplementedError: Unsupported moe_runner_backend for NVFP4 MoE:
    MoeRunnerBackend.FLASHINFER_TRTLLM. Use --moe-runner-backend flashinfer_cutlass

`auto` resolves to `flashinfer_cutlass` for Radix's `modelopt_fp4` but to
`flashinfer_trtllm` for MIXED_PRECISION. Naming `flashinfer_cutlass` explicitly
therefore *matches* the Radix baseline rather than introducing a variable; it is
now the `MOE_RUNNER_BACKEND` knob, unset by default.

PLE handling, kept strictly separate per the fixed constraints. The NVIDIA table
is 128 `ngram_embedding.shard_N.weight` F8_E4M3 tensors totalling 51200245760
bytes — **the same size as Radix's**, so both map to the same recipe filename and
a shared directory would have silently mixed checkpoints. It got its own
`PLE_DIR`; the build logged `0/128 shards already on disk, 128 copied` (~10 min)
and the second boot logged `128/128 ... 0 copied`. Two harness gaps surfaced and
were fixed rather than worked around: `ple_identity` refused to run at all when
the table did not exist yet (`PLE_FIRST_BUILD=1` now binds the identity *before*
the build and re-samples after), and a table built for a new checkpoint keeps
the engine's native filename, so identity resolves the backing file by size and
`bind_identity` now binds the directory as well as the file. Both are covered by
a new harness test. Sampled digest of the NVIDIA table is
`a13a022a5e6f0e39bdd564a9c4483e158658b0242f85b3c2e62b1929ab18ac9d` — **identical
to Radix's**, consistent with the model card's claim that the PLE n-gram
embedding is copied byte-for-byte from `Qwen/Qwen3.8-Flash-Next-FP8`.

Tokenizer files (`tokenizer.json`, `vocab.json`, `merges.txt`,
`tokenizer_config.json`, `chat_template.jinja`) are byte-identical between the
two checkpoints, so prompts, chat template and the U6 65536-id draft token map
transfer unchanged and the comparison is genuinely matched.

Everything else held fixed: image digest, source `4ccff141`, Torch 2.13.0+cu130,
FlashInfer 0.6.17, Triton 3.7.1, Transformers 5.12.1, ModelOpt 0.46.0, native
262144 context, BF16 KV, FP32 SSM, extra_buffer, tracking 64, MAX_TOTAL 524288,
MAX_RUNNING 4, PREFILL 4096, NEXTN 3/1/4, graph bs=[1,2,3,4], vision retained.

Serving comparison, TAG `u10-nvidia-nvfp4-20260908` vs the matched Radix run
`u8a-baseline-20260908` (identical suite set):

| Metric | Radix | NVIDIA |
| --- | ---: | ---: |
| Reuse boot s | 569.54 | 739.73 |
| 8k PLE-warm TTFT s | 2.689 | 2.680 |
| 32k cold TTFT s | 10.125 | 10.090 |
| 128k cold TTFT s | 42.196 | 42.050 |
| prefix-warm 8k / 32k / 128k s | 0.286 / 0.315 / 0.656 | 0.290 / 0.320 / 0.650 |
| Decode off code / prose tok/s | 46.73 / 22.40 | 46.97 / 23.64 |
| Decode on code / prose tok/s | 38.78 / 25.42 | 36.03 / 28.65 |
| Streams c=1 / 2 / 4 aggregate | 47.01 / 86.58 / 101.50 | 49.77 / 74.95 / 106.26 |
| Streams c=4 per-stream | 34.35 | 36.35 |
| Mixed p95 chunk gap / TTFT s | 26.201 / 28.429 | 26.176 / 28.349 |
| KV tokens | 524288 | 524288 |
| Min host MemAvailable GiB | 10.33 | 8.96 |
| Quality suite | 11/12 | 12/12 |
| Needles 8k/32k/128k | pass | 32k refused once, 8k/128k pass |

Expanded quality screening, one boot each, same 200 GSM8K questions, thinking
off, same harness. TAGs `u10-nvidia-gsm8k-20260908` and `u10-radix-gsm8k-20260908`:

| Screening | Radix | NVIDIA |
| --- | ---: | ---: |
| GSM8K n=200 | **194/200 = 97.0%** | **194/200 = 97.0%** |
| Paired outcome | both correct 192, Radix-only 2, NVIDIA-only 2, neither 4 | |
| GSM8K wall s / median decode tok/s | 1787.3 / 43.75 | 1862.8 / 43.06 |
| effort_probe 40x | 14/40 (`24`x14, `28`x15, `25`x11) | **34/40** (`24`x34, `28`x6) |
| effort_probe pooled with the 20x runs | 35/100 = 35% | 49/60 = 82% |
| 8k needle samples this boot | 5/7 (2 refusals) | 7/7 |
| spec_accept_length | 3.75 | 3.90 |

Reading it honestly. The probe difference is large, reproducible across separate
boots of each checkpoint, and measured on identical prompts and flags — the
NVIDIA pack really is better on *that one arithmetic prompt*. It does not
generalise: on 200 paired GSM8K questions the two are indistinguishable
(McNemar 2 vs 2). One idiosyncratic prompt is not a quality argument for a
checkpoint switch, and the plan's rule is explicit — retain Radix unless paired
evidence supports quality *and* a real speed or fit benefit warrants switching.
Speed is a wash across prefill, decode, streams and mixed load; fit is a wash
(same KV budget, same 51 GB PLE, 124 vs 126 GB on disk); paired quality is a
tie. So Radix stays. NVIDIA's published GPQA/HLE figures were never treated as
local results and were not reproduced here.

The needle refusal seen in the NVIDIA serving run is not a checkpoint defect:
the same refusal strings appear on Radix (2 of 7 samples in its screening boot,
and in two earlier U7b runs), and NVIDIA scored 7/7 in its own screening boot.
It is a shared, intermittent model behaviour that the harness scores as a
recall failure. Same for the quality suite's 12/12 vs 11/12: that is the
unstable `effort_thinking_off` case, which the probe measures properly.

Limits. One boot per checkpoint for the screening; GSM8K n=200 is a screening,
not the full 1319; no 120-turn agentic, BFCL, multilingual or long-reasoning
comparison was run for the NVIDIA pack; 190k/210k unmeasured on both; vision was
verified only by the positional smoke, not a paired multimodal benchmark. The
NVIDIA boot is also consistently slower (628–740 s vs 570–584 s) and its runs
sat ~1.4 GiB lower on host MemAvailable, neither investigated.

Rollback: nothing to roll back — the default `MODEL`/`REVISION`/`QUANTIZATION`/
`SPEC_DRAFT_QUANT`/`MOE_RUNNER_BACKEND` are unchanged and the Radix screening run
above is the post-comparison health check (smoke pass, GSM8K 97.0%). The NVIDIA
checkpoint and its PLE table remain on disk, reusable via the documented env
knobs. Next item: U11 (not started).


## U10 follow-up — class-level probe and 120-turn revalidation (2026-09-08)

**Decision: U10's outcome is unchanged (Radix retained), and the anomaly that
motivated this follow-up is now explained rather than merely outvoted.** The
single-prompt gap is instance-specific, not a class effect. Separately, the
current default passed a fresh 120-turn agentic pair, closing evidence that had
been stale since U5a.

Why this ran: `bench/effort_probe.py` showed a large, reproducible split on one
prompt (Radix 35%, NVIDIA 82% pooled) while GSM8K n=200 tied at 194/200. One
prompt cannot distinguish "better on this class" from "one knife-edge instance",
and the expensive suites (120-turn, BFCL) test different capabilities entirely,
so they could not have answered it. New `bench/arith_probe.py` answers it
directly: 20 prompts of the same shape (short multi-step arithmetic, thinking
off, integer answer, temperature 0), 10 repeats each, 200 paired samples per
checkpoint. Every prompt is rendered from parameters and its expected answer is
computed from those same parameters, so the key cannot disagree with the
question; the four consistently-failed keys were also checked by hand.

TAGs `u10-nvidia-arith-20260908` and `u10-radix-arith-agentic-20260908`. Same
image digest, source `4ccff141`, packages, context, KV/state precision, flags,
prompts and sampling as the U10 comparison; only `MODEL`/`REVISION`/
`QUANTIZATION`/`SPEC_DRAFT_QUANT`/`MOE_RUNNER_BACKEND`/`PLE_DIR` differ, as before.

| Probe | Radix | NVIDIA |
| --- | ---: | ---: |
| 20-prompt class, n=200 | **155/200 = 77.5%** | **156/200 = 78.0%** |
| prompts scoring 10/10 | 15 | 14 |
| prompts scoring 0/10 | 4 | 4 |
| prompts with identical scores | 18 of 20 | |
| single-prompt effort_probe, this boot | 7/20 | 19/20 |
| single-prompt pooled across boots | 42/120 = 35% | 68/80 = 85% |

The two checkpoints fail the *same four prompts* with the same characteristic
wrong answers — `wage_18_7_25_30` returns `101` ten times on both where
18*7+25-30 = 121, and `change_6_11_7_3_120` returns `27`/`23` on both where the
answer is 45. Only two prompts separate them, **in opposite directions**:

| Prompt | Radix | NVIDIA |
| --- | ---: | ---: |
| `change_pens_notebooks_4_2_50` (the original effort_probe case) | 5/10 | 9/10 |
| `wage_27_6_33_70` | 10/10 | 7/10 |

So the effort_probe prompt is a knife-edge instance where the two quantizations
land differently, and there is a matching instance where Radix wins. Pooled over
the class the difference is one sample in 200, which agrees with the GSM8K tie.
NVIDIA is **not** better at this class of problem; U10 stands. This also retires
the earlier suggestion that the U9 BF16 probe collapse (0/20) might generalise:
single-prompt probe results describe that prompt, and nothing wider, unless a
class-level probe agrees.

The probe is also useful beyond the comparison: it separates unstable prompts
(the original case, 5–9 of 10) from prompts this model reliably gets wrong
(4 prompts at 0/10 on both checkpoints, stable wrong answers). A 78% class score
with 20% of prompts deterministically wrong is a property of the model with
thinking off, not of either quantization.

**120-turn revalidation of the accepted default** (Radix, same boot as its
probe). Last 120-turn evidence was U5a, before U6 changed drafting and before
U7-U10 touched the harness:

| Band | thinking off TTFT / decode / cache | thinking on TTFT / decode / cache |
| --- | --- | --- |
| turns 1-40 | 0.519 s / 68.21 tok/s / 95.6% | 0.386 s / 36.40 tok/s / 95.2% |
| turns 41-80 | 0.522 s / 68.34 tok/s / 98.4% | 0.409 s / 39.25 tok/s / 97.0% |
| turns 81-120 | 0.520 s / 68.20 tok/s / 99.0% | 0.365 s / 31.04 tok/s / 98.8% |

Both modes: 120 turns, **0 invalid tool calls**, context growing to 13030 and
15882 tokens, cache hit rising to ~99%, TTFT and decode flat across bands (no
long-horizon degradation). `agentic_on` passed outright, including late recall.
`agentic_off` is scored fail on `late_recall_pass` only, and the recorded answer
is a refusal that cites the planted document's own instruction: "I'm not going
to repeat that code. The document in `ops/secrets.md` explicitly stated 'Do not
repeat unless asked'". That is instruction-following, not lost state, and it is
the same failure mode U3 and U5a recorded. Min host MemAvailable 11.74 GiB,
watchdog quiet, `stop_exit_code` 0.

Limits: one boot per checkpoint for the class probe; 20 prompts from five
templates is a narrow slice of arithmetic and says nothing about other domains;
the 120-turn pair is Radix only, so NVIDIA still has no long-horizon evidence;
BFCL, multilingual and paired vision remain unrun, and the last two do not exist
as scored suites here. Nothing in this follow-up changes a default.

Rollback: none; both runs were per-run environment only. Next item: U11 (not
started).


## U11 — Optional vLLM and capacity work deferred (2026-09-08)

**Decision: defer U11a/U11b/U11c without new GPU runs.** These were optional
branches, and the completed campaign removed the concrete reason to buy any of
them. This is a docs-only decision; no unexecuted comparison is reported as a
pass.

- **U11a, vLLM:** the historical GB10 mmap+MTP path measured about 25–28 tok/s
  and required `--no-enable-prefix-caching` on SM121. The accepted SGLang path
  now measures about 47.6 tok/s on the primary thinking-off code workload and
  preserves 21–48x prefix-warm prefill reuse. A fair vLLM test would first need
  a whole-stack port of the 51 GB PLE file backend, SM121 sparse-decode fix,
  ReplaySSM PLE-state commit, reduced-vocabulary draft path, MTP graphs and the
  detached agentic harness. There is no measured signal that this migration
  could improve the recipe.
- **U11b, FP8 KV:** no compatible and locally validated QSA path was established
  for this precision change. U9 also showed that freeing 1.1 GB by halving the
  recurrent-state pool did not increase `max_total_num_tokens=524288`; the
  configured cap binds. Taking a larger precision/quality trade without a
  demonstrated capacity consumer would not satisfy the plan's useful-headroom
  gate.
- **U11c, 512k:** the August capacity experiment already booted with only 201984
  KV tokens, below the native 262k recipe. Its attempted static-YaRN override
  targeted a configuration shape this checkpoint does not use: the checkpoint
  has sectioned mrope under `text_config.rope_parameters`, and effective
  `rope_type` remained `default`. It therefore provided neither enough KV plus
  output headroom nor a valid basis for 300k/400k/near-512k correctness tests.

Defaults remain SGLang `4ccff141`, BF16 KV, native context 262144 and RadixArk
revision `7b719225`. No server was started and no rollback was needed. Next item:
U12 final cumulative validation and documentation.


## U12 — Final cumulative validation and documentation (2026-09-08)

**Decision: campaign complete from cumulative post-U6 evidence.** No dedicated
U12 GPU run was launched. The many matched runs after the last accepted serving
change already exercised the combined default more broadly than another fast
retest would, so the final review accepts those runs as the U12 interaction
check.

Evidence on the accepted Radix/SGLang combination:

- U6 launched the reduced-vocabulary draft default twice from the documented
  profile. Both boots reused the native PLE table (`128/128` shards, 0 copied)
  in 564 and 556 seconds, kept 524288 KV tokens, passed quality 12/12, and
  reproduced the primary code result at 47.59 and 47.57 tok/s.
- U7–U10 repeatedly booted the same accepted U1/U2/U3/U6 stack while testing
  candidates through per-run flags or reversible overlays. Post-rollback/current
  baselines covered smoke, tools, executed code, positional vision, both thinking
  modes, decode, 1/2/4 streams, 8k/32k/128k cold and prefix-warm prefill, needle
  recall and controlled mixed load.
- The current Radix default scored 194/200 (97.0%) on GSM8K n=200 in U10. Its
  follow-up arithmetic class probe scored 155/200 across 20 prompts and explained
  the unstable single effort prompt as an instance effect rather than a default
  regression.
- The latest current-default 120-turn pair completed with 0 invalid tool calls,
  flat TTFT/decode bands and cache hit rising to about 99%. Thinking-on passed
  late recall; thinking-off refused to repeat a value because the planted
  document itself said not to repeat it, matching the previously classified
  instruction-following behavior.
- Earlier one-hour stability runs passed: U3's soak completed 2214/2214 requests,
  and U4c's growing-context soak completed 5820/5820 with no recall failures.
  U4c used the rejected 8 GiB trimmer setting, but the trimmer never fired and
  mapping RSS remained only 0.29–0.43 GiB, so its serving behavior otherwise
  matched the accepted path.

Explicit limit: the planned dedicated two-hour mixed-load soak was **not run**,
and no dedicated U12 preparation/boot was performed. Repeated post-U6 launches,
mixed-load gates and the one-hour stability evidence were judged sufficient to
close this single-box campaign. This is a documented waiver, not a claim that
the two-hour test passed. The remaining known limits are full GSM8K 1319, scored
BFCL multi-turn, multilingual and paired vision suites, 190k/210k long-context
tests, and Terminal-Bench, which cannot run on this ARM64 host.

Final defaults remain RadixArk revision `7b719225`, SGLang `4ccff141` at image
digest `sha256:9d2a843c706c74bc259c0d9abf360551eb2734e1e7d255ab012a6965f10480b6`,
native file-backed PLE reuse, SM121 Triton QSA, ReplaySSM PLE-state commit,
64k NEXTN draft token map, BF16 KV, FP32 recurrent state, context 262144,
`MAX_TOTAL=524288`, `MAX_RUNNING=4`, `PREFILL=4096`, vision enabled and
thinking enabled by default. U11 remains deferred; no server state changed and
no rollback was needed.


## Sep 14 plan — A1: QSA extend compress-gather clamp (sglang#38346) (2026-09-14)

**Decision: accepted. `prepare.sh` now always applies the one-line clamp to
`qsa/qsa_indexer.py`, and `serve.sh` always mounts that file.** This fixes an
out-of-bounds read that our pin demonstrably performs, with no measured cost.

The bug, confirmed against the pinned image rather than only the PR text:
`_qsa_write_plan` pads its fixed-capacity write plan with row 0 / block 0
entries, and the extend path gathers each entry's members as
`member_rows[:, None] + arange(4)`. A forward with fewer than 4 token rows reads
rows 1..3 past `token_k`. The fused Triton compress kernel never bounds-checks,
so on GPU the read is silently absorbed unless it crosses an unmapped page.
There are three ways to produce such a forward, and all of them are reachable here:
a final chunked-prefill piece of 1..3 tokens (`k*4096 + 1..3`); a **radix-cache
resend whose length is `64*m + 1..3`**, where the page-aligned hit leaves a 1..3
token extend (about 3 in 64 of all cache-hit resends); and a 1..3 token prompt.

Evidence the fix is right (`scripts/test_qsa_chunk_tail.sh`, CPU-only, the
image's real `_qsa_write_plan` and the real stock vs patched
`update_key_state_and_compress`; on CPU the same gather raises):

| Case | Stock | Clamped |
| --- | --- | --- |
| prefix 0/4096/8192, extend 1/2/3 (9 cases) | `IndexError` every time | completes, writes only reserved slot 0 |
| full groups: extend 4, 4096, 4099, 2047@64, 5@8192, 4096@4096 | — | written slots and values **bit-identical** to stock |

New `bench/chunk_tail.py` (suite `CHUNK_TAIL=1`) sends all three shapes through
`/generate` with exact, salted token ids and checks the resend really extended
1..3 tokens (`prompt_tokens - cached_tokens`). The stock server (the idle :5802
instance, before any boot) completed all 30 requests without a crash, matching
upstream's "usually silent". So the value of this item is removing a
demonstrated OOB read, not a crash we could reproduce on demand.

Matched A/B/A, one variable (the clamp), same image digest, revision, PLE
identity, flags, prompts and suite set:

    TAG=a1-baseline-20260914 (stock indexer staged into build/qsa), then
    TAG=a1-qsa-clamp-20260914 and TAG=a1-qsa-clamp-confirm-20260914
    PROFILE=u3 PREFILL=4096 CHUNK_TAIL=1 TURNS=120 N=3 SIZES=8k,32k,128k \
      PLE_DIR=/home/shantanu/ai/cache/sglang/flash-next-ple-mmap \
      ONLY=smoke,chunk_tail,quality,decode,longctx,agentic_off,agentic_on

| Metric | stock | clamp | clamp confirm |
| --- | ---: | ---: | ---: |
| chunk_tail (24 short forwards, server healthy after) | 24/24 | 24/24 | 24/24 |
| Quality | 11/12 | 11/12 | 12/12 |
| Decode off code samples tok/s | 47.2 / 46.6 / 48.5 | 44.8 / 45.6 / 48.4 | 45.9 / 46.1 / 51.2 |
| Decode off prose / on code / on prose median | 20.71 / 36.97 / 26.88 | 20.06 / 35.00 / 27.26 | 20.84 / 38.30 / 26.67 |
| Cold TTFT 8k / 32k / 128k s | 3.37 / 10.36 / 38.32 | 3.17 / 10.34 / 38.58 | 3.24 / 10.33 / 38.93 |
| Needles 8k/32k/128k | pass | pass | pass |
| 120-turn off: invalid / recall / completion tok / decode bands | 0 / pass / 28.5 / 67.9–69.0 | 0 / pass / **42** / 58.5–58.9 | 0 / pass / 29 / 69.5–71.0 |
| 120-turn on: invalid / recall / decode bands | 0 / pass / 29.9–33.0 | 1 / pass / 30.5–39.2 | 0 / pass / 34.3–40.0 |
| Boot s / min MemAvailable GiB | 577 / 10.26 | 581 / 10.10 | 601 / 10.21 |

Reading it. Decode sample ranges overlap in every cell and TTFT is flat. The
first clamp run looked worse on agentic thinking-off (58.7 vs 68.9 tok/s), but
its completions were 42 tokens per turn against 28.5 — a different greedy
output mode, not a slower server. The same 42-vs-29 split appears across
earlier boots of identical configurations (U1/U3 at 42, U2/U5a/U6/U10 at
29–30), and agentic traffic never produces a 1..3 token extend, so the clamp
does not even execute there. The confirm boot landed back in the 29-token mode
at 69.5–71.0 tok/s and passed every suite. The one invalid tool call in the
first run is turn 52 hitting `max_tokens=512` with thinking on in the middle of
a `search_docs` call (truncated JSON), a harness budget artifact.

A side observation for C4: the chunk_tail resend of an identical random-token
prompt, greedy, reproduced its first-send output 6/6, 4/6 and 2/6 times across
the three boots (and 1/6 on the stock :5802 server), independent of the clamp.
Cache-hit vs cache-miss greedy divergence is real here on low-confidence
prompts.

Harness: `experiment.py` now treats a refused llama-swap unload connection
(llama-swap not running) as nothing to unload; the `nvidia-smi` occupant check
still gates the launch.

Limits: three boots, three decode repeats; no GSM8K or BFCL rerun (the change
provably leaves real groups bit-identical). #37786, upstream's fuller follow-up,
is still open and not ported.

Rollback: remove the `qsa_chunk_tail_clamp.py` line from `prepare.sh` and rerun
it; `serve.sh` then mounts a stock indexer. Next item: B1.


## Sep 14 plan — B1: NEXTN draft pass reads only the MTP files (2026-09-14)

**Decision: accepted, on by default.** The draft load goes from 81–90 s to
31–33 s, and boot from 562–575 s to 525 s on a clean repeat.

The waste, confirmed in the pinned image: the draft (`Qwen4ExpForCausalLMMTP`)
loads through `ModelOptModelLoader` → `DefaultModelLoader`, which resolves all
206 `*.safetensors` files (125.9 GiB) and materialises every tensor, only for
the inherited `Qwen3_5ForCausalLMMTP.load_weights` to drop every name without
`"mtp"` as the first check in its loop. `embed_tokens`/`lm_head` come from the
target via `set_embed_and_head`, not from the draft's own read.

Change (`patches/draft_mtp_files.py`, two new overlays mounted by `serve.sh`):
`loader.py`'s `Source` takes an optional `key_filter` from the model's
`checkpoint_key_filter`, and `_get_weights_iterator` keeps only files the
safetensors index maps a passing key to, plus any file the index does not
list. Without an index, or with no passing key, the list is unchanged.
`qwen4_exp_mtp.py` defines the filter as `"mtp" in name`, mirroring the guard.
The file set is derived from the index at runtime.
`SGLANG_CHECKPOINT_KEY_FILTER=0` restores stock resolution.

Check first (in the image, `results/b1-unit/check.log`): the MTP class does not
override `load_weights` and inherits Qwen3.5's; the `"mtp"` guard precedes any
parameter use, so consumed keys ⊆ keys containing `mtp`. On the real index the
filter keeps `model-bf16-00010..00012` (13.72 GiB), which hold all 31 `mtp`
keys; the no-index and no-match fallbacks return the list unchanged, and a
model without the attribute gets `key_filter=None`.

Matched boots, one variable (`SGLANG_CHECKPOINT_KEY_FILTER`), same build,
`PROFILE=u3 PREFILL=4096 N=3 SIZES=8k,32k ONLY=smoke,quality,decode,longctx`,
order off → on → off → on:

| Metric | off | on | off2 | on2 |
| --- | ---: | ---: | ---: | ---: |
| Draft load s | 90.32 | **31.46** | 81.16 | **32.64** |
| Main load s | 402.71 | 449.29 | 395.89 | 410.06 |
| Boot s | 574.65 | 561.29 | 561.55 | **525.28** |
| Draft "mem usage" GB (loader log) | 0.94 | 3.23 | 0.59 | 3.31 |
| available_gpu_mem after graphs GB | 12.10 | 12.06 | 11.95 | 12.94 |
| KV tokens | 524288 | 524288 | 524288 | 524288 |
| accept len mean (run log) | 2.727 | 2.678 | 2.615 | 2.691 |
| Decode off code / prose tok/s | 49.47 / 21.18 | 48.92 / 21.11 | 48.59 / 22.37 | 47.12 / 21.91 |
| Quality / needles 8k,32k | 11/12 / pass | 11/12 / pass | 11/12 / pass | 11/12 / pass |
| Min MemAvailable GiB | 10.37 | 10.19 | 10.42 | 10.34 |

Reading it. The draft phase saving is large and repeatable (−50 to −59 s). The
first "on" boot did not show it end to end because its *main* load, which B1
does not touch, ran 47 s slow; the repeat's main load was normal and the boot
came in at 525 s. Main-load variance of ±50 s is therefore the noise floor for
boot comparisons on this box, which B2 has to beat as well. The loader's draft
"mem usage" rises because it is measured as a drop in unified-memory
availability and now includes the page cache of the 13.7 GiB it just read; KV
capacity and memory after graph capture are unchanged, so it is not a real
allocation. Acceptance, decode, quality (the same unstable
`effort_thinking_off` case in all four) and needles are unchanged. The 64k
token map still applies (launch flag present, checked by the harness).

Harness: `experiment.py` records the filter line in `boot-facts.txt` and a
run-wide `accept_len` summary in `experiment_status.json`.

Limits: two boots per arm; boot phases read from the loader's own timers.

Rollback: `SGLANG_CHECKPOINT_KEY_FILTER=0`, or drop the `draft_mtp_files.py`
line from `prepare.sh` and the two mounts from `serve.sh`. Next item: B2.


## Sep 14 plan — B2: skip reading the PLE files on a reused table (2026-09-14)

**Decision: rejected. Nothing is applied by default; `patches/ple_slice_reuse.py`
and `scripts/test_ple_slice_reuse.*` stay in the repo as the record.** The
premise did not hold on this image: the PLE files were never the slow tail of
the load.

What was built. The plan's cheaper variant, chosen because it keeps today's
byte-level verification instead of trusting a sidecar: the target model's
`checkpoint_key_filter` (the B1 hook) rejects `.ple_embedding.ngram_embedding.`
keys, so the loader skips the ten `model-plefp8-*` files and tells the model
which files it skipped. After the weight loop, each shard is verified by
reading, through `safe_open(...).get_slice`, only the rows that cover the same
34 sampled 4 KiB windows `_ple_shard_matches` compares. The scale buffer is
loaded with `get_tensor`, and any mismatch falls back to the whole-shard copy
path. The CPU test on the real table and checkpoint (shards 0/63/127) agreed
with the whole-tensor verifier, caught first- and last-byte corruption, rejected
a shape mismatch, and honoured `SGLANG_QWEN4_PLE_REUSE=0`.

The finding that decides it. In that same test, `get_tensor` on a 381 MiB PLE
shard took **0.2–0.6 ms**: safetensors 0.8.0 returns a lazily mmap-backed
tensor, so the stock path only faults in the sampled windows. The shard
progress bar of the plan's own reference boot, re-read with file positions,
shows where the "last 17 shards, 83 s" went:

| Main-pass bar (sorted files) | Files | Elapsed |
| --- | --- | ---: |
| 190 → 196 | expert shards, then `model-bf16-00001/00010/00011/00012` | 319 → 396 s (**~77 s**) |
| 196 → 206 | the ten `model-plefp8-*` files | 396 → 400 s (**~4 s**) |

So the tail cost is the dense BF16 files (B3 territory), not the PLE table.

Measured anyway, one variable (`SGLANG_QWEN4_PLE_SLICE_REUSE`), same A1+B1
build, `PROFILE=u3 PREFILL=4096 N=3 SIZES=8k,32k ONLY=smoke,quality,decode,longctx`.
The harness now saves the shard progress bars as `load-progress.txt`.

| Metric | off | on |
| --- | ---: | ---: |
| Main load s | 395.79 | 402.22 |
| Boot s | 504.14 | 515.25 |
| Progress bar: PLE files | 376 → 382 s (6 s) | skipped; bar ends at 196 files, 384 s |
| Boot log | `128/128 shards already on disk` | same, plus `read 10 skipped checkpoint files by slice (128 shards verified)` |
| Min MemAvailable GiB | 10.26 | 10.81 |
| accept len mean | 2.608 | 2.693 |
| Quality / needles | 11/12 / pass | 12/12 / pass |

The 6 s of iterator time saved is spent again on slice verification, and both
differences sit inside the ±50 s main-load noise seen in B1. A planned second
off/on pair and a forced-fallback boot were stopped early (the off2 run
was terminated during boot, `results/b2-slice-off2-20260914-aborted-early`)
because a phase-level measurement of 6 s cannot become a meaningful boot
saving with more repeats. The transient-RAM benefit the plan expected does not
exist either, because nothing was being materialised.

Rollback: nothing to roll back; `prepare.sh`/`serve.sh` are back to the B1
state and `build/` was verified identical to the pre-B2 build. Next item: the
A2/A3/A4 long soak.


## Sep 14 plan — A2/A3/A4: 24 h soak with decay, KV-corruption and zombie detectors (2026-09-14/15)

**Decision: none of the three reproduced as a serving fault on the A1+B1
default, so no overlay, watchdog or restart cadence is added.** A4's partial
symptom is documented as client guidance in the README (`max_tokens`).

New `bench/longsoak.py` (suite `LONGSOAK=1`). Three client threads leave one of
the four slots free:

- **Agentic:** growing tool-calling sessions, alternating thinking, rotated at
  ~24k tokens.
- **Mix:** EN/ES prose and code, thinking on/off; 30% of requests disconnect
  mid-stream.
- **Prefix (the A3 reproducer):** a fixed ~6k-token LongBench-v2 document
  prefix plus a fresh 8–16k-token suffix, aborted 0.5–4 s in (mid chunked
  prefill) by closing the socket, then the prefix resent with a question.

Every 10 minutes traffic drains, then the sample records:

- **A2:** idle `/metrics`; a fixed greedy decode probe (tok/s plus the server's
  `accept len` for the probe window); traffic accept lengths from the log.
- **A3:** a cache-hit vs post-`flush_cache` answer to the prefix question.
- **A4:** per-rid `state was deleted in TokenizerManager` lines, and requests
  still running while the client is idle.

A 20-minute shakedown replaced a random-token probe that was too low-confidence
to judge (`results/a2-longsoak-shakedown-20260914`).

    TAG=a2-longsoak-24h-20260914 PROFILE=u3 PREFILL=4096 LONGSOAK=1 \
      LONGSOAK_SECONDS=86400 SAMPLE_SECONDS=600 ONLY=smoke,longsoak \
      PLE_DIR=/home/shantanu/ai/cache/sglang/flash-next-ple-mmap

Ran 86761 s, 128 samples. Totals: 8879 requests, 7385 completed, 1494
deliberate client aborts, **0 errors**, 0 invalid tool calls, 0 server
tracebacks or CUDA errors, watchdog quiet, min MemAvailable 9.4 GiB,
`stop_exit_code` 0.

| Uptime band | Probe decode median tok/s | Probe accept len | Traffic accept len | Zombie rids | Max outputs after disconnect |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0–4 h | 47.7 | 3.73 | 2.64 | 304 | 75 |
| 4–8 h | 50.3 | 3.80 | 2.60 | 277 | 105 |
| 8–12 h | 47.8 | 3.77 | 2.59 | 242 | 97 |
| 12–16 h | 47.9 | 3.75 | 2.64 | 232 | 99 |
| 16–20 h | 48.8 | 3.77 | 2.67 | 297 | 98 |
| 20–24 h | 46.9 | 3.77 | 2.61 | 224 | 66 |

**A2 (sglang#37326, NEXTN acceptance decay to ~0 over 16–24 h): not
reproduced.** Acceptance on the fixed probe is flat at 3.73–3.80 across the
whole day, and so is traffic acceptance. Probe throughput is flat within
single-sample noise (38–54 tok/s; an early 38.5 dip recovered on the next
sample), and no decay onset (<1.5 accept or <60% of the first hour) was flagged.
The reported reproduction differs from this recipe in ways that plausibly
matter (64k draft token map, ReplaySSM PLE commit, FP32 state, `extra_buffer`),
but which one protects us was not isolated. No watchdog or restart cadence is
needed; #38191 was not tried.

**A3 (sglang#38319 / #38355, abort during chunked-prefill insert leaves corrupted
QSA KV): not reproduced.**

- The first-token detector (id ≥ 248077; the tokenizer's added tokens end at
  248076, and the bug emits 248319) never fired across 374 prefix probes, all
  mix/agentic traffic, and roughly 370 mid-prefill aborts on a shared prefix.
- The hit-vs-flush comparison disagreed in 2 of 128 samples, **in opposite
  directions**. At 7.3 h the cache-hit answer differed from the empty-cache
  reference and a flush restored it. At 11.0 h the cache-hit answer matched the
  reference and the post-flush answer was the one that differed.
- The disagreeing prompt is a knife-edge one: raw `/generate` with no chat
  template, where the reference is the end token 248046 and the alternative is a
  plausible text answer. So these are hit/miss and miss/miss greedy flips of
  valid tokens, not corruption, consistent with A1's resend observation (C4
  follows up). That one event makes the harness mark the suite `fail`
  (`a3_fixed_by_flush` counts as a failure by design); it is classified here,
  not ignored.
- #38355 is not overlaid.

**A4 (sglang#36333 / #36876, aborts lost in the batch-transition window):
partial.** 1494 client aborts yielded 1581 request ids that kept emitting
outputs after their client was gone, up to 105 output steps (at ~2.6 tokens
per step, well short of the 600-token `max_tokens`, so they were eventually
aborted rather than run out). A slot was never held: at all 128 samples the
server reported **0 running and 0 queued requests** once the client drained,
so `zombie_clear_s` never had to wait. The short-term mitigation from the plan
(documented `max_tokens`) is now in the README; pick up the upstream fix when
it merges.

Side observation: scheduler `RssFile` (mostly the PLE table mapping) rose
from 6.5 GiB at 3 h to 8–9.4 GiB after 13 h and then held, while host
MemAvailable stayed ~10 GiB. This is the creep U4c's trimmer targets. It did not
affect throughput or memory headroom within 24 h.

Limits: one boot and one traffic mix; the probe measures one prompt; prefix
aborts all use one document prefix; zombie accounting reads server log lines,
so a request lost entirely before logging would only show as a held slot
(never observed).

Rollback: nothing changed in serving. Next item: C1.


## Sep 14 plan — C1: real-text long prefill and PLE prefetch (2026-09-15/16)

**Decision: accepted. `SGLANG_QWEN4_PLE_FILE_PREFETCH=1` (upstream WILLNEED
prefetch) is now the default in `serve.sh` and the u3 profile. This reverses
U4b.** Separately, fresh 250k real-text prefill turned out to be beyond this
box's safe memory at `MAX_TOTAL=524288`, for a reason that also matters outside
prefill (lazily committed KV, below).

Why U4b missed it, confirmed: its filler prompts share n-grams, so the PLE
table was warm by construction. `bench/prefill.py` gains `PREFILL_TEXT=real`.
Each prompt is built from LongBench-v2 documents not used earlier in the run,
with a mid-prompt needle, sized by the server's `/tokenize` (bisection on the
cut point), and seeded by `PREFILL_SEED` so arms can share documents. Every
repeat is then PLE-cold, like real long-document traffic. An early version drew
new documents on every sizing retry and crashed on an oversize prompt;
`results/c1-prefetch-off-20260915-sizing-bug` is that run, not data.

Why it is slow, observed live during a cold 128k prompt without prefetch:
~16 MiB/s of disk reads with one blocked task, i.e. ~4k serial 4 KiB page
faults per second from the gather kernel (MADV_RANDOM, no readahead).

Matched arms on **identical documents** (`PREFILL_SEED=c1-b`),
`PROFILE=u3 PREFILL=4096 PREFILL_BENCH=1 PREFILL_TEXT=real SIZES=32k,128k PREFILL_N=3 MIXEDLOAD=1 ONLY=smoke,prefill,mixedload,quality,decode`:

| Real-text TTFT s (3 fresh prompts) | prefetch off | WILLNEED run 1 | WILLNEED run 2 | pread run 1 | pread run 2 |
| --- | --- | --- | --- | --- | --- |
| ~31.5k tokens | 112 / 85 / 96 | 20 / 20 / 26 | 19 / 22 / 27 | 25 / 23 / 29 | 25 / 23 / 31 |
| ~126k tokens | 413 / 525 / 569 | 151 / 270 / 280 | 144 / 195 / 289 | 158 / 211 / 202 | 169 / 362 / 240 |
| prefix-warm 32k / 128k | 0.37 / 0.70 | 0.37 / 0.67 | 0.38 / 0.70 | 0.37 / 0.67 | 0.47 / 0.64 |
| Needles | 8/8 | 8/8 | 8/8 | 8/8 | 8/8 |

(WILLNEED run 1 is `c1-prefetch-on-20260915`, which also used seed `c1-b` and
tripped at 250k after these sizes.)

- **32k:** WILLNEED is **4–5x** faster cold (19–27 s vs 85–112 s; 1,200–1,600
  vs 280–370 tok/s).
- **128k:** **2–3x** (144–289 s vs 413–569 s).
- **Still room above:** later 128k prompts in a run slow down (see memory
  below), so filler-warm speed (~3,000 tok/s at 128k) is not reached.

Plan step 3 was prototyped as `patches/ple_pread_prefetch.py`: a synchronous
16-thread `pread` of each chunk's distinct pages before the gather. It did not
beat WILLNEED in two runs (medians 211 and 240 s vs 270 and 195 s at 128k,
slightly slower at 32k). **Rejected**; kept as an unapplied record.

Gates, no regression. The filler/warm paths use the confirm boot
`c1-willneed-confirm-20260916` (`SIZES=8k,32k MIXEDLOAD=1 ONLY=smoke,quality,decode,longctx,mixedload`):

| Gate | prefetch off (`c1-off-gate`) | WILLNEED gate run | WILLNEED confirm |
| --- | ---: | ---: | ---: |
| Mixed 64k prefill TTFT s | 28.32 | 29.64 | 28.45 |
| Mixed p95 decode stall s | 26.16 | 27.39 | 26.18 |
| Decode off code / prose tok/s | 48.06 / 20.73 | 45.69 / 17.54 | 46.78 / 21.90 |
| Decode on code / prose tok/s | 34.80 / 24.49 | 40.44 / 22.72 | 37.32 / 27.76 |
| Filler cold TTFT 8k / 32k s | — | — | 2.85 / 10.36 |
| Quality | 11/12 | 11/12 | 11/12 |
| Min MemAvailable GiB | 8.47 | 8.72 | 10.16 |

The gate run's low prose-off decode (17.5 twice) and +4.7% stall did not
reproduce on the confirm boot. Decode-sized gathers (<2048 rows) skip the
prefetcher entirely, so neither was expected to move.

**250k fresh real text: not servable safely on this recipe, with or without
prefetch.**

- **Off arm:** tripped the memory watchdog during its first 250k prompt, after
  throughput collapsed to 68–100 tok/s past ~190k tokens.
- **WILLNEED arms:** the chain arm tripped the same way.
- **Single-prompt diagnostic** (fresh boot, 10 s host sampler): prefill reached
  ~190k tokens at ~900 tok/s, then hit the 6 GiB floor.
- **Floor lowered to 3 GiB:** tripped at ~155k tokens on kernel
  `NVRM ... NV_ERR_NO_MEMORY` allocation failures, while MemAvailable still
  read 8.3 GiB.

The cause is the finding worth carrying forward. **The KV pool is committed
lazily on unified memory.** It is sized for 524288 tokens (12 GB K+V + 1 GB
draft), but its pages only become resident when written. In the diagnostic,
MemAvailable fell 4.9 GiB over ~190k prefilled tokens, which matches ~24.8
KiB/token of KV, while host anon pages (~7.1 GiB) and the PLE mapping (≤6 GiB)
did not grow. Boot leaves ~10.3 GiB available with an empty pool, so:

- one full 262k context would leave ~4 GiB;
- a radix cache that fills the whole 524k pool would need ~13 GiB;
- before either, the PLE page cache is squeezed out, which is the long-context
  "prefill cliff" (the plan's ~440 tok/s at 250k).

None of the earlier suites filled the pool (the 24 h soak flushed the cache
every 10 minutes), so this was never observed. It is **not fixed here**; it
needs its own item (a `MAX_TOTAL` sized to what can actually be committed,
e.g. ~300k, or equivalent headroom), measured with a pool-filling workload.

Limits: two or three boots per arm, three prompts per size; 250k unmeasured;
documents are LongBench-v2 English/code, not multilingual.

Rollback: `SGLANG_QWEN4_PLE_FILE_PREFETCH=0`. Next item: C2.


## Sep 14 plan — C2: `--mamba-track-interval 256` (2026-09-16)

**Decision: rejected. The interval stays at 64.** There is no decode or agentic
gain once output modes are matched, and the agentic cache hit rate is slightly
lower.

Matched boots, one variable (`MAMBA_TRACK_INTERVAL`), on the current default
(A1 clamp, B1 draft filter, C1 prefetch).
`PROFILE=u3 PREFILL=4096 N=3 SIZES=8k,32k,128k PREFILL_BENCH=1 TURNS=120 ONLY=smoke,prefill,quality,decode,longctx,agentic_off,agentic_on`:

| Metric | 64 (base) | 256 | 256 confirm |
| --- | ---: | ---: | ---: |
| 120-turn off: completion tok / decode bands | 43 / 58.3–58.7 | 29 / 67.8–68.2 | 42 / 58.0–58.3 |
| 120-turn off: cache hit bands % | 95.9 / 98.6 / 99.1 | 94.7 / 98.1 / 98.8 | 95.0 / 98.1 / 98.8 |
| 120-turn on: completion tok / decode bands | 121.5 / 27.8–34.2 | 72.5 / 36.1–39.7 | 67.5 / 37.5–40.2 |
| Decode off code / prose | 48.20 / 21.80 | 46.13 / 22.33 | 46.31 / 20.75 |
| Decode on code / prose | 43.06 / 24.92 | 38.78 / 27.29 | 36.00 / 27.40 |
| Filler cold TTFT 8k / 32k / 128k s | 3.43 / 10.34 / 43.21 | 3.29 / 10.17 / 42.28 | 2.89 / 10.10 / 41.94 |
| Prefix-warm 8k / 32k / 128k s | 0.28 / 0.33 / 0.66 | 0.32 / 0.32 / 0.63 | 0.29 / 0.33 / 0.66 |
| Quality / needles (longctx 8k,32k,128k) | 11/12 / pass | 11/12 / 8k refusal | 11/12 / pass |
| accept len mean (run log) | 2.73 | 3.03 | 3.15 |
| Boot s / min MemAvailable GiB | 499 / 10.30 | 513 / 10.10 | 511 / 10.69 |

Reading it.

- **Agentic decode is set by the output mode, not the interval.** Thinking-off
  flips between the 42- and 29-token greedy modes seen since U1. Within a mode
  the interval changes nothing: 58.5 vs 58.1 tok/s in the 42-token mode, and the
  256 run's 68 tok/s in the 29-token mode equals stock boots in that mode (A1:
  67.9–71.0). U5a's single-boot "58 vs 49" was this mode artifact. Thinking-on
  shows the same confound (121 vs ~70 tokens per turn).
- **Decode suite:** no gain; code decode is slightly lower on both 256 boots.
- **Acceptance:** the higher run-wide accept length does not show up in any
  tok/s measurement and is workload-mix dependent, so it is not counted.
- **Prefill and cache:** cold filler TTFT is 2–3% lower at 128k, and
  prefix-warm TTFT at 32k/128k is unchanged (the plan's concern). But
  multi-turn agentic cache hit is 0.3–1.2 points lower at every band,
  consistent with coarser mamba checkpoints.
- **8k needle:** the one 8k needle miss ("not present in the provided text")
  is the intermittent 8k behaviour seen in earlier prefill logs and passed on
  the confirm boot.

Under the campaign rule (accept only a consistent improvement with no
unexplained regression), this is a reject. Limits: one baseline boot; the
SGLang verified-cell GSM8K result (#37995) was not re-run here.

Rollback: nothing to roll back (`MAMBA_TRACK_INTERVAL` default unchanged).
Next item: C3.


## Sep 14 plan — C3: smaller draft vocabularies and an English prose case (2026-09-16)

**Decision: the 48k and 32k maps are rejected; `hot_tokens_64k.pt` stays the
default. The English prose decode case is added and kept.**

`bench/decode.py` gains `prose_en` (Spanish kept) and, with `CONTAINER` set,
records each task's mean server `accept len` over its own samples.
`scripts/slice_draft_vocab.py` cuts a smaller map as a **prefix of the ranked
64k map** (specials, then corpus ids by frequency, then id-order fill), so a
size A/B is one variable: no new corpus and no Wikipedia sampling. 48k keeps
all 44301 corpus-ranked ids plus 3666 fill; 32k is 32735 corpus-ranked ids and
no fill.

Matched boots, one variable (`SPECULATIVE_TOKEN_MAP`), `PROFILE=u3 N=5 TURNS=40
ONLY=smoke,quality,decode,agentic_off`:

| Task | 64k median (range) / accept | 48k | 32k |
| --- | --- | --- | --- |
| off code_en | **47.86** (45.69–48.79) / 3.75 | 48.45 (47.01–51.62) / 3.81 | 49.39 (45.86–51.19) / 3.77 |
| off prose_es | **22.07** (20.69–23.79) / 1.88 | 21.93 (18.63–23.61) / 1.78 | 18.58 (17.04–20.21) / **1.56** |
| off prose_en | **22.39** (21.62–25.27) / 2.18 | 24.05 (22.81–26.05) / 2.15 | 25.07 (22.29–28.17) / 2.14 |
| on code_en | 39.22 (36.90–42.07) / 3.17 | 40.53 (38.00–45.66) / 3.21 | 37.36 (31.76–42.70) / 2.95 |
| on prose_es | 27.45 (26.27–28.52) / 2.33 | 26.03 (23.21–28.70) / 2.14 | 22.59 (20.26–23.82) / 1.88 |
| on prose_en | 29.97 (28.79–34.45) / 2.62 | 28.82 (27.62–31.87) / 2.41 | 27.25 (25.27–30.41) / 2.37 |
| 40-turn off: invalid / recall / decode bands | 0 / pass / 51–55 | 0 / pass / 62–70 | 0 / pass / 61 |
| Quality | 11/12 | 12/12 | 11/12 |
| accept len mean (run log) | 2.703 | 2.638 | 2.403 |

Against the predeclared gate — medians must not overlap, and acceptance must
not fall in any category by more than it gains:

- **48k:** code +1.2% with fully overlapping ranges (45.7–48.8 vs 47.0–51.6),
  Spanish prose −0.6% with acceptance 1.88 → 1.78, thinking-on Spanish −5%.
  Inconclusive on the metric it was supposed to win, and negative elsewhere.
- **32k:** Spanish prose **−16%** (acceptance −17%), thinking-on code −4.7%
  and Spanish −18%. Clear reject.
- The trend is consistent and explains itself: trimming the tail of a
  frequency ranking built mostly from English/code corpora costs the most on
  non-English text, where those ids are drafted. MiaAI's +13% was a
  code-tuned vocabulary on a different stack, and it does not transfer here.

**English prose is not faster than Spanish on this recipe:** 22.39 vs 22.07
tok/s thinking-off and 29.97 vs 27.45 thinking-on, on the same shaped prompt.
So the 37–49 tok/s English prose figures quoted from vLLM recipes are not
explained by our prompts being Spanish; the gap is elsewhere (or not
comparable). The plan's "not comparable" caveat can now be replaced with a
measurement.

Limits: five repeats per task on one boot per size; the 64k boot's 40-turn
agentic decode (51–55 tok/s at 31 tokens/turn) is below its usual band and was
not re-run, so the agentic row is not used for the decision.

Rollback: none needed. The 48k/32k maps and their reports stay in
`bench/draft_vocab/` as the record. Next item: C4.


## Sep 14 plan — C4: determinism probe, cache hit vs cache miss (2026-09-16)

**Finding: the temperature-0 instability is kernel-level, not cached state.
Diagnostic only; no default changes.**

`bench/effort_probe.py` gains `CACHE_MODES=miss,hit`: the chat prompt is
rendered with `/tokenize` and sent to `/generate` greedy, and full output id
sequences are compared. `miss` calls `/flush_cache` before every repeat; `hit`
never flushes. Two prompts: the unstable `effort_thinking_off` case and a code
prompt.

The first run (`c4-determinism-20260916`) showed `cached=0` in **both** modes:
the radix cache stores whole 64-token pages, and a ~40-token prompt can never
be a hit, so that run is two miss arms. It is still informative — with an empty
cache, the same greedy prompt gave **3 distinct sequences in 10** (effort,
diverging at position 1; code at 9 and 22).

`CACHE_PREFIX_TOKENS` then prepends a fixed LongBench-v2 document so hits are
real (`c4-determinism-prefix-20260916`, 3206–3238 token prompts,
`cached=3200` on every hit):

| Arm | distinct sequences / 10 | largest class | outlier repeats |
| --- | ---: | ---: | --- |
| effort, miss (flush each) | 2 | 9 | repeat 0 answered `28`, rest `24` |
| effort, hit | 2 | 9 | repeat 8 answered `27` |
| code, miss | 2 | 9 | repeat 5 |
| code, hit | **1** | 10 | none |
| miss vs hit modal sequence | identical for both prompts | | |

Reading it against the plan's own rule — divergence without the cache points at
kernels, divergence only with the cache points at #34820-style state precision:

- Divergence happens **with an empty cache**, at ~1 in 10 repeats, so kernels
  (or their non-associative reduction order / `fast_topk` ties) are implicated.
  This matches blazux's report that GB10 QSA top-k is non-deterministic.
- The cache-hit arms are not worse — code was 10/10 identical — and the modal
  sequence is the same in both modes, so restoring mamba checkpoints from the
  radix cache is not introducing the flips. #34820 is not indicated here.
- The `effort` prompt is genuinely knife-edge: a single flipped token at
  position 1 changes the answer between 24, 27 and 28. That is the quality
  suite's long-standing `effort_thinking_off` case, and it explains A1's
  chunk-tail resend observation (6/6, 4/6, 2/6 reproduction across boots) and
  A3's two hit-vs-flush mismatches, which went in opposite directions.
- The first request after a boot/flush is a plausible extra factor (the
  `effort` miss outlier was repeat 0), not established here.

Consequence for the campaign: single-sample greedy comparisons on these
prompts cannot resolve small differences, which is why the quality suite's
11/12 vs 12/12 keeps moving without any configuration change.

Limits: ten repeats, two prompts, one boot; no attempt to identify which kernel
(a deterministic-inference flag, `--enable-deterministic-inference`, exists
upstream and was not tested).

Rollback: none; probe-only. Next item: B3 loader profile.


## Sep 14 plan — B3 (profile step): where the weight load goes (2026-09-16)

**Finding: the load is weight-loader bound, not disk bound. `num_threads=16`
does not help and is not adopted.** A post-load weight cache is the only lever
the profile leaves (plan item 12).

py-spy needed two fixes before it could attach: the container runs non-root, so
`--cap-add SYS_PTRACE` is now available as `SERVE_PTRACE=1` in `serve.sh`
(default off), and the dump has to run as `docker exec -u 0` — a non-root uid
does not inherit the capability. `results/b3-profile/pyspy-failed-noptrace.txt`
is the failed attempt.

17 main-thread samples during the main load (`results/b3-profile/pyspy-live.txt`,
`pyspy-sampled.txt`) land in weight loaders, never in file reads:

| Main-thread frame | What it is |
| --- | --- |
| `_load_w13` / `_load_w2` → `FusedMoE.weight_loader` (`moe/fused_moe_triton/layer.py`) | per-expert copy into the fused MoE parameter |
| `weight_loader` (`linear.py:752`, `linear.py:1404`, `qwen3_5.py:489/492`) | dense/GDN projections |
| `weight_loader` (`vocab_parallel_embedding.py:507`), `default_weight_loader` | embeddings and the rest |
| `_ple_shard_matches` → `copy_ple_rows_to_tp_embedding` | our PLE reuse sampling |

The `ThreadPoolExecutor` reader threads were idle in every sample but one (a
`torch/storage.py __getitem__` inside `_load_file`). With 296,475 tensors, of
which 221,184 are per-expert NVFP4 scales, this is per-tensor Python and
host→device copy cost on one thread.

Step 2, the cheap A/B (`EXTRA_ARGS='--model-loader-extra-config {"num_threads":16}'`,
`ONLY=smoke,decode`):

| Arm | Main load s | Draft load s | Boot s |
| --- | ---: | ---: | ---: |
| 8 threads (default) | 407.57 | 32.71 | 522.25 |
| 8 threads (profiling boot) | 398.67 | 32.07 | 510.36 |
| 8 threads (confirm) | 385.62 | 31.75 | 500.56 |
| **16 threads** | **417.80** | 31.35 | 531.91 |

16 threads is 10–32 s *slower* than the three 8-thread boots, i.e. no better
within the ±50 s noise established in B1. Consistent with the profile: reading
is not the bottleneck, so more readers cannot help.

What this means for plan item 12 (the post-load weight cache). It remains the
only approach that could cut this phase, because `sharded_state` would save
already-assembled parameters and skip exactly the per-expert `weight_loader`
work that dominates. The practical blockers are unchanged and now better
understood: it must survive our mounted model overlays and NVFP4
post-processing, exclude the PLE table, be invalidated whenever weights change,
and it would add ~84 GB on disk. Not attempted here.

Rollback: nothing adopted; `SERVE_PTRACE` defaults to off. Next item: D1.


## Sep 14 plan — D1: rebase onto SGLang main (2026-09-16)

**Decision: accepted. The pin moves from `4ccff141` to SGLang main `8874c51a`
(`lmsysorg/sglang:nightly-dev-20260915-8874c51a`, digest
`sha256:efec0e11a8e6ab1287c81830a214cf107f840a994e385a5c82a8077ded82782a`).**
Every overlay is ported, every gate is equal or better, and the 4-stream
aggregate improves 14–18%.

Image choice: the nightly arm64 build of main published 2026-09-15, which is
after #39126 merged (2026-09-13). Torch 2.13.0+cu130 and Triton 3.7.1 are the
same as the old pin, so kernels are comparable.

What the rebase brings (all merged upstream since our pin): **#34820** mamba
prefix-cache checkpoints at the configured SSM dtype, **#37165** deferred-init
metadata clear before spec decode, **#38346** the A1 clamp (now upstream, our
patcher reports ALREADY PATCHED), **#38851**/**#38855** the FP8 KV / sparse
gather fixes, and **#39126** the upstream file-backed PLE and PDL router fix.

Overlay porting, the actual work:

| Overlay | On main |
| --- | --- |
| `ple_file_compat`, `ple_reuse` | apply unchanged; boot still logs `128/128 shards already on disk, 0 copied` |
| `qsa_drop_sm121_sdpa`, `qsa_sm121_triton` | apply unchanged; `Triton SM121 QSA` still logs (#36556 still open) |
| `qsa_chunk_tail_clamp` (A1) | **no longer needed** — upstream has the clamp; patcher is idempotent |
| `draft_mtp_files` (B1) | **ported**: upstream added `allow_patterns_overrides` to `Source`, so the patcher now edits the `DefaultModelLoader` region structurally instead of matching the exact field block |
| `replayssm_ple_commit` (#37794) | **ported, and still required** (see below) |
| `ple_pread_prefetch` | not applied (C1 rejected it) |

The ReplaySSM port is the substantive finding. On main,
`replayssm_spec_fold = enable_linear_replayssm_spec and cache_params.is_kda`,
so a **GDN** model no longer takes the fold branch; it takes a new
compact-replay branch. That branch also returns before the generic PLE roll, so
#37794's gap is still real on main — but our old patch landed in a branch we no
longer execute, which the harness caught ("ReplaySSM PLE commit did not log").
The patcher now inserts the PLE roll before the `return` of **every** GDN
branch (fold and compact) and leaves the KDA branch alone. On the old pin it
still patches both branches; the fold branch (that pin's serving path) is
byte-identical apart from comments, and the dead ring branch now passes real
track indices instead of `None`.

`serve.sh` also had to stop using `--enable-gdn-replayssm-spec`: main removed
the deprecated alias. The canonical `--enable-linear-replayssm-spec` exists on
both images and was re-validated on the old pin
(`d1-pin-flagrename-20260916`, PLE commit logs, suites as before).

Paired full gate, same checkpoint, PLE table, flags, prompts and suites
(`PROFILE=u3 PREFILL=4096 N=3 SIZES=8k,32k,128k PREFILL_BENCH=1 STREAMS=1 MIXEDLOAD=1 TURNS=120`):

| Metric | pin `4ccff141` | main `8874c51a` | main confirm |
| --- | ---: | ---: | ---: |
| Decode off code / prose ES / prose EN | 48.64 / 20.71 / 22.92 | 47.91 / 20.99 / 23.03 | 48.56 / 20.56 / 23.54 |
| Decode on code / prose ES / prose EN | 39.63 / 26.04 / 29.12 | 40.55 / 27.06 / 30.99 | 37.89 / 25.52 / 32.01 |
| Cold TTFT 8k / 32k / 128k s | 2.87 / 10.47 / 39.10 | 2.76 / 10.03 / 37.50 | 2.76 / 10.06 / 37.56 |
| Prefix-warm speedup 32k / 128k | 22.9x / 48.6x | 25.4x / **64.2x** | 25.2x / **64.0x** |
| Streams c=1 / 2 / 4 aggregate | 46.8 / 74.8 / 112.8 | 49.0 / 79.4 / **132.6** | 48.5 / 84.4 / **128.3** |
| Mixed p95 stall / prefill TTFT s | 26.48 / 28.81 | 25.37 / 27.30 | 25.41 / 27.32 |
| 120-turn off / on decode | 57.3–58.0 / 35.8–39.8 | 58.3–58.5 / 37.2–42.5 | 67.4–68.4 / 33.1–36.1 |
| accept len mean | 2.973 | 3.063 | 2.859 |
| Boot s / min MemAvailable GiB | 529.8 / 10.21 | 529.7 / 10.11 | 521.3 / 10.36 |
| Quality / needles 8k,32k,128k | 11/12 / pass | 11/12 / pass | 11/12 / pass |

Nothing regresses; 4-stream aggregate (+14–18%) and 128k prefix-warm reuse are
the real gains, with mixed-load stall and 128k cold TTFT slightly better.

**The one anomaly, recorded rather than waved away:** the main confirm boot's
120-turn thinking-on run made 11 invalid tool calls, all of them the model
inventing a tool named `audit` (alongside 108 valid `search_docs` calls). It is
not truncation (no turn hit `max_tokens`). Three further 120-turn main boots
(`d1-main-agentic2/3`) scored 0 invalid, as did every pin run; across the whole
campaign this is the only hallucinated tool name in any agentic run. Given C4's
kernel-level nondeterminism it reads as a trajectory artifact, but it is the one
thing to watch on this engine. Late-recall pass/fail keeps flipping on both
engines and is the documented credential refusal, not lost state.

Final validation through the normal path (`build/` rebuilt from the new pin,
`d1-newpin-validation-20260916`): boots, `8874c51a` reported by
`/server_info`, quality 11/12, needles pass, decode in range.

Limits: two full-gate boots plus a validation boot on main; no GSM8K, BFCL or
24 h soak on the new engine; 250k prefill still unmeasured (see C1); the
`audit` episode is unexplained.

Rollback: set `IMAGE` back to the previous digest, kept in a comment in
`scripts/common.sh`, and rerun `prepare.sh`. Next item: D2.


## Sep 14 plan — D2: FP8 blockwise dense side layers (2026-09-16)

**Decision: accepted as a documented opt-in, not the shipped default.** It is
the largest decode win in this campaign — **+13% code, +21% Spanish prose,
+12% on 120-turn agentic, +14% single-stream** — at the cost of ~3% prefill
TTFT, ~3% mixed-load stall, and a locally built 13 GiB sibling snapshot. GSM8K
is a tie. `lm_head` is deliberately **not** converted.

### Step 1: does block FP8 pay on SM121?

`scripts/bench_block_fp8_linear.py` times BF16 `F.linear` against SGLang's
128x128 block-FP8 linear at this checkpoint's real side-layer shapes, no model
load. Two findings decided the design:

| Shape (count) | M=1 | M=4 | M=4096 |
| --- | ---: | ---: | ---: |
| `linear_attn.in_proj_qkv` (36) | **3.24x** | 2.69x | 0.93x |
| `linear_attn.in_proj_z` (36) | 2.93x | 3.10x | 0.88x |
| `linear_attn.out_proj` (36) | 1.46x | 1.71x | 0.86x |
| `self_attn.q_proj` (12) | 2.99x | 2.18x | 0.95x |
| `self_attn.o_proj` (12) | 1.95x | 1.77x | 0.88x |
| `lm_head` (1) | 2.61x | 1.93x | 0.97x |
| `hyper_connection.down` (96) | **0.17x** | 0.15x | 0.48x |
| `hyper_connection.up` (96) | unsupported: K=320 is not a multiple of 128 | | |

- Decode (M=1..4) is bandwidth-bound and gains 1.5-3.2x; **prefill (M=4096)
  loses 3-17%**, so this trades prefill for decode.
- The auto-selected **DeepGEMM backend crashes on SM121** (`CUDA error:
  unspecified launch failure` at M=4, poisoning the context) and its numerics
  looked wrong (rel_l2 0.24 vs 0.037 for triton). Serving must pass
  `--fp8-gemm-backend triton`. This is the plan's "vLLM needed an M % 4 padding
  fix" unknown, in a different form.

### What is converted, and why not more

`scripts/build_fp8_hybrid_snapshot.py` symlinks the whole source snapshot and
rewrites only the 4 shards that hold converted tensors (156 tensors, 13 GiB of
new files; the source is untouched).

- **Converted:** `linear_attn.in_proj_qkv/in_proj_z/out_proj`,
  `self_attn.q_proj/k_proj/v_proj/o_proj`.
- **`lm_head` is not converted.** The NEXTN draft shares the target `lm_head`
  and the accepted U6 token map slices arbitrary rows out of it
  (`head.data[hot_token_id]`); 128-row block scales cannot survive that slice.
  Shinrali's +8.4% is therefore not available without giving up U6's +18%.
- **Hyper-connections stay BF16** (0.15x at decode; K=320 unquantizable).
- **The MTP draft stays BF16** and is excluded from the quant config.

Three bugs found the hard way, each worth recording:

1. **Every shard of a packed module must be converted together.** SGLang fuses
   q/k/v into `qkv_proj` and in_proj_qkv/in_proj_z into `in_proj_qkvz`, and
   `ModelOptMixedPrecisionConfig._resolve_quant_algo` applies one algo to the
   whole fused layer. Converting only `q_proj` made the fused `qkv_proj` FP8
   while k/v bytes stayed BF16 — the server emitted word salad.
2. **`hf_quant_config.json` is only read when `config.json` has no
   `quantization_config`** (`ModelConfig._parse_quant_hf_config`). The sibling
   snapshot must therefore materialise a `config.json` with that key removed.
   Until it did, the mixed config was silently ignored, FP8 bytes were cast
   into BF16 parameters, and the model produced garbage while reporting
   `quant=modelopt_fp4`. A "plumbing" boot on unconverted weights passed
   precisely because the config was being ignored — it proved nothing.
3. **The MTP's experts must not be marked NVFP4.** A regex that matched
   `mtp.layers.0.mlp.experts` made the draft build packed params and die with
   `size of tensor a (320) must match tensor b (640)`.

Serving: `--quantization modelopt_mixed`, `--fp8-gemm-backend triton`,
`SPEC_DRAFT_QUANT=auto`, `MOE_RUNNER_BACKEND=flashinfer_cutlass`, and its own
PLE table copy (the identity guard correctly refuses to share a table across
checkpoints, so `data/d2-ple-fp8` holds a copy). `serve.sh` gained
`LOCAL_MODEL_MOUNT`/`LOCAL_BLOBS_MOUNT` and omits `--revision` for `local-*`.

### Full gate, D1 pin, hybrid vs two BF16 baseline boots

| Metric | main BF16 (2 runs) | **FP8 hybrid** |
| --- | ---: | ---: |
| Decode off code / ES / EN tok/s | 47.91–48.56 / 20.56–20.99 / 23.03–23.54 | **54.61 / 25.20 / 26.17** |
| Decode on code / ES / EN tok/s | 37.89–40.55 / 25.52–27.06 / 30.99–32.01 | 39.31 / 31.46 / 37.17 |
| 120-turn off decode (29-token mode) | 67.41–68.42 | **75.69–76.75** |
| 120-turn on decode | 33.06–42.45 | 38.35–40.20 |
| Streams c=1 / 2 / 4 | 48.48–49.00 / 79.36–84.35 / 128.28–132.55 | **55.56** / 83.50 / **137.00** |
| Cold TTFT 8k / 32k / 128k s | 2.76 / 10.03–10.06 / 37.50–37.56 | 2.88 / 10.28 / 38.51 |
| Prefix-warm 128k speedup | 63.98–64.19x | **70.55x** |
| Mixed p95 stall / prefill TTFT s | 25.37–25.41 / 27.30–27.32 | 26.15 / 28.12 |
| Weights / boot s / min MemAvailable | 83.6 GB / 521–530 / 10.11–10.36 | **78.85 GB** / **466** / **11.00** |
| KV tokens | 524288 | 524288 |
| Quality / needles / agentic invalid | 11/12 / pass / 0 | 11/12 / pass / 0 |
| **GSM8K n=200** | **195/200 (97.5%)** | **196/200 (98.0%)** |
| Arithmetic class probe (20 prompts x10) | 145/200 (72.5%) | 140/200 (70.0%) |
| GSM8K wall s | 1795.0 | 1589.5 |

Reading it. The decode gain is large and its ranges do not overlap the baseline
(code off 51.87–55.54 vs 45.69–48.79). Boot is 55–64 s faster and memory
headroom is 0.9 GB better, both because 4.8 GB less weight is loaded — the
headroom directly helps the C1 memory ceiling. The costs are real but small:
cold prefill +2.7% at 128k, mixed-load p95 +3%. Quality: GSM8K is a tie
(1 question), the arithmetic class probe is 5 samples lower out of 200, and the
knife-edge `effort_thinking_off` prompt is bad on both (0/20 vs 1/20 — C4
explains why that prompt cannot arbitrate).

Why opt-in and not the default: it needs a locally built 13 GiB sibling
snapshot plus a 48 GiB PLE table copy, it makes prefill slightly worse on a
recipe whose weak point is prefill, and the arithmetic probe moved the wrong
way. The stock checkpoint remains the default; this is documented as the
decode-maximising variant.

Limits: one boot per arm for the full gate, one GSM8K run each, no 24 h soak on
the hybrid, no BFCL, and `lm_head`/hyper-connections unconverted so the
theoretical ceiling is not reached.

Rollback: serve the stock checkpoint (unchanged default); delete
`data/d2-fp8-hybrid` and `data/d2-ple-fp8`. Next item: D3.


## Sep 14 plan — D3: 512k context (2026-09-16)

**Decision: accepted as a documented optional mode; 262k BF16 KV stays the
default.** A 524288 context boots and serves, and needle recall is verified at
**294,485 tokens** — past the native 262,144. Two things did not go as the plan
assumed: the YaRN override turned out to be unnecessary, and the practical
ceiling is ~340–350k tokens, not 512k.

Configuration that works (on the D1 pin):

    CONTEXT=524288 MAX_TOTAL=524288 SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1 \
      EXTRA_ARGS="--kv-cache-dtype fp8_e4m3"

FP8 KV is what makes it fit: KV drops from 6.00+6.00 GB to 3.00+3.00 GB (plus
draft 0.25+0.25), and available GPU memory after graph capture rises from
~12–13 GB to **18.73 GB**. `--kv-cache-dtype fp8_e4m3` also works with QSA now,
which it did not before (#38851/#38855 came in with the D1 rebase; blazux had
reported FP8 KV refused on the older stack).

**The YaRN override does nothing here.** U11 failed because its override
targeted `rope_scaling`; the checkpoint keeps sectioned mrope under
`text_config.rope_parameters`. This item built the correct override (SGLang
merges only one level, so the whole `rope_parameters` dict must be supplied
with `mrope_section`/`mrope_interleaved` preserved) — and then the control boot
**without** it recalled the same 294,485-token needle just as well:

| Run | 294,485-token needle | TTFT |
| --- | --- | ---: |
| 512k + FP8 KV + YaRN override | PASS (`7K-QUARTZ-19`) | 139.46 s |
| 512k + FP8 KV, **no override** | PASS (`7K-QUARTZ-19`) | 140.10 s |

transformers logs `rope_type='default'` in both cases, so the model is
extrapolating at this depth rather than using YaRN. The override is therefore
not part of the recommendation.

Measured at 512k + FP8 KV (vs the 262k BF16 default on the same pin):

| Metric | 262k BF16 KV | 512k FP8 KV |
| --- | ---: | ---: |
| Needles 8k / 32k | pass | pass |
| Needle 250k label (184,055 tok) / 300k label (220,865) / 400k label (294,485) | not run | **pass / pass / pass** |
| Prefix-warm speedup at those sizes | — | 99x / 129x / 144x |
| Decode off code / prose ES / prose EN | 46.62 / 19.66 / 23.24 | 44.83 / 20.54 / 19.98 |
| Decode on code / prose EN | 40.39 / 32.26 | 37.59 / 27.30 |
| Cold TTFT 8k / 32k s | 3.30 / 10.05 | 3.32 / 9.89 |
| KV size / available after capture | 12.0 GB / 12.5 GB | **6.5 GB / 18.73 GB** |
| Quality | 11/12 | **12/12** |

Decode costs about 4–10% (blazux measured −10% for FP8 KV), which is why 262k
BF16 stays the default.

**Where it stops.** A single 500k-label prompt (~370k tokens) reached ~346k
committed tokens (`full token usage 0.66`) and then hit kernel
`NVRM ... NV_ERR_NO_MEMORY`; the fresh **real-text** 250k-label prefill from C1
tripped the memory floor earlier still. So the usable single-prompt depth is
roughly 294k verified, ~340–350k hard ceiling, and long fresh-document prefill
remains limited by the lazily committed KV plus GPU-side prefill allocations
(see C1). 512k is a context *length*, not a servable prompt size, on this box.

Limits: one boot per configuration; one needle per size and a single filler
haystack; no GSM8K/agentic at long context; nothing measured between 294k and
346k; FP8 KV quality checked only by the 12-case suite and needles, not by a
long-context quality benchmark.

Rollback: none needed; the default is unchanged. Next item: the plan's last
entry, the B3 weight cache.


## Sep 14 plan — item 12: post-load weight cache (2026-09-16)

**Decision: rejected. The cache works and serves correctly, but it is slower
than re-running the loaders.** `patches/presharded_skip_ple.py` stays in the
repo, unapplied, as the record.

B3's profile said the main load is per-tensor `weight_loader` work, so a
post-load cache was the only remaining lever. This build ships one:
`--load-format presharded` (`PreshardedModelLoader`) dumps post-processed
weights and reloads them. Three things had to be fixed before it would run at
all on this recipe, all in `patches/presharded_skip_ple.py`:

1. **The PLE table must not enter the cache.** It is a 47.7 GiB host mmap whose
   backing file already survives restarts, so dumping it would write the table
   twice and reloading would materialise it as ordinary memory, undoing
   `--ple-offload-backend file`. The patch drops `*.ngram_embedding.weight`
   from the dump and from the load-time missing-parameter check; by then
   `allocate_ple_host_table` has already mapped the verified rows.
2. **`get_model_loader` never reaches `PRESHARDED` for a ModelOpt checkpoint**
   — it returns `ModelOptModelLoader` first, which then rejects
   `presharded_path` as an unknown extra-config key. The patch tests
   `LoadFormat.PRESHARDED` before the ModelOpt branch.
3. **The NEXTN draft cannot load through this format** (`Cannot find any model
   weights`), so serving pins `--speculative-draft-load-format auto`, and the
   extra-config validator has to tolerate the two presharded keys it is handed.

With that, the cache builds (74 GiB, `READY`) and a later boot logs `Loading
from presharded checkpoint` and serves normally (smoke, quality 11/12, decode
in range). It is simply not faster:

| Boot | Target load s | Draft load s | Boot s |
| --- | ---: | ---: | ---: |
| Normal path (3 boots, B3) | 385.6 / 398.7 / 407.6 | 31.4–32.7 | 500.6 / 510.4 / 522.3 |
| **From cache (2 boots)** | **463.8 / 483.1** | 40.8 / 42.0 | **595.0 / 607.5** |

Reading it: on this unified-memory box the normal path's cost is not disk, so
replacing it with a 74 GiB read of already-assembled tensors adds work rather
than removing it — every cached tensor still has to be read and placed, and the
saving (skipping fused-MoE assembly) does not cover that. The plan's hope that
"B3 might reach ~2–3 min" does not survive contact: boot went the wrong way by
~85 s.

What did help boot in this campaign: B1 (−50 to −58 s) and, incidentally, D2
(466 s, because 4.8 GB less weight is read).

Limits: two cached boots; no attempt to tune `max_file_bytes`, to keep the
cache in page cache, or to store it on a different device; `verify_on_load`
was left off.

Rollback: nothing applied; `data/weight-cache` (74 GiB) was deleted.

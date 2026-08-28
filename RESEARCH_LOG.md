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

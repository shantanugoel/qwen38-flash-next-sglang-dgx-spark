# Improvements plan — 14 September 2026

Combined plan from two reviews on 2026-09-14:

1. **Upstream and community sweep.** Covers SGLang commits since our pin
   `4ccff141` (329 commits), open SGLang PRs and issues for Qwen3.8-Flash-Next and
   GB10, and the other single-Spark recipes (MiaAI-Lab, blazux, tonyd2wild, madeye,
   Shinrali, Code Turbo), the NVIDIA forum threads and the SGLang cookbook.
2. **Cold-boot profile.** Where the ~9.5 min boot goes, re-checked against the
   running server's log, the checkpoint index and the pinned image's loader.

Items are marked **Status: DONE** as they close. Every item follows the usual
protocol: one `TAG=… ./scripts/run_config.sh` per variable, a log entry in
`RESEARCH_LOG.md`, and accept or reject against the common gates.

## Where we stand

| | Ours (U6 default) | Best other single-Spark recipe |
| --- | ---: | ---: |
| Code decode, 1 stream | **47.6 tok/s** | 45.8 (madeye, vLLM); 55.4 with INT4 experts (Code Turbo) |
| Agentic tool-JSON decode | **68 tok/s** | ~60 (AutoRound fork) |
| Prose decode | 22 tok/s (**Spanish** prompts) | 37–49 (English prompts, vLLM) — not comparable |
| Prefill at ~32k | ~2,260 tok/s | 2,300–3,000 |
| Prefill at ~250k, real text | **~440 tok/s** | ~1,940 (MiaAI, vLLM) |
| SGLang's own verified 1x Spark cell | — | 27.5 tok/s (random ISL 1024) |
| Cold boot (PLE table reused) | ~9.5 min | ~10–13 min |

Summary: decode is already competitive. The clear gaps are long-context prefill,
the BF16 dense layers (a bandwidth cost other recipes have removed), boot time,
and several open upstream stability bugs we have never soaked long enough to see.

---

## Track A — Correctness and stability

### A1. QSA compress-gather out-of-bounds (sglang#38346) — overlay now

- **Status: DONE 2026-09-14, accepted.** Clamp applied by `prepare.sh`
  (`patches/qsa_chunk_tail_clamp.py`); the indexer is mounted unconditionally.
  CPU test: stock raises `IndexError` on all 9 short-forward cases, clamped real
  groups are bit-identical. A/B/A boots: no decode/TTFT/agentic regression,
  `chunk_tail` 24/24. A radix resend of length `64*m + 1..3` also hits this path.
  See RESEARCH_LOG "Sep 14 plan — A1".
- **What:** in `qsa_indexer.py` extend path, a chunk with fewer than
  `compress_ratio` (4) token rows reads past `token_k`. That happens when the last
  chunked-prefill piece of a request is 1–3 tokens. Usually silent, sometimes an
  illegal memory access.
- **Status:** merged upstream 2026-09-12. **Verified unfixed in our image**
  (`build/qsa/qsa_indexer.py:330`). #37786 (open) is the fuller follow-up.
- **Change:** one line, `group_locs = group_locs.clamp_max(source_keys.shape[0] - 1)`,
  as a patch applied by `prepare.sh`. `build/qsa/` is already mounted for U8a, so
  it can be mounted unconditionally.
- **Gate:** quality, longctx, 120-turn. Add a request whose prompt length is
  `k*4096 + 1..3` to exercise the path.
- **Effort:** ~1 hour plus one boot.

### A2. Long soak for MTP acceptance decay (sglang#37326) — measure first

- **Status: DONE 2026-09-15, not reproduced.** 24 h `bench/longsoak.py` soak: probe
  accept length 3.73–3.80 and decode ~47–50 tok/s flat across all 4 h bands; 8879
  requests, 0 errors. No watchdog needed. See RESEARCH_LOG "Sep 14 plan — A2/A3/A4".
- **What:** NEXTN acceptance decays to ~0 over ~16–24 h uptime on this exact
  model. Throughput drops ~3x with no error. Reproduced on GB10 TP1 even with the
  #35821 clamps. A restart restores it. Partial fix #38191 is open.
- **Our exposure:** our longest soak is 1 h (U3: 2214/2214). Never tested.
- **Plan:** a `bench/soak.py` run of ≥24 h with mixed agentic, abort and prose
  traffic. Log `accept len` from the decode batch lines every 10 min, and log
  `#cached-token` on prefills. Record onset time if decay appears.
- **If it reproduces:** try #38191. Fallback: document a restart cadence or add a
  watchdog that restarts the server when rolling accept length stays below 1.5.
- **Effort:** one day wall clock, little attention needed.

### A3. Chunked-prefill + radix insert race corrupting QSA KV (sglang#38319 / #38355)

- **Status: DONE 2026-09-15, not reproduced.** No first token ≥ 248077 across ~370
  mid-prefill aborts on a shared prefix and 374 probes; 2/128 hit-vs-flush mismatches
  flipped in opposite directions on a knife-edge prompt (nondeterminism, not
  corruption). #38355 not overlaid.
- **What:** retract or abort during chunked-prefill insertion leaves corrupted KV
  pages in the radix tree. Every later request with that prefix emits token
  `248319` on its first decode step until `flush_cache`.
- **Our exposure:** the reporter's config is almost ours (GB10, page 64,
  `extra_buffer`, triton/trtllm_mha, MTP 3/1/4). Aborts come from client
  disconnects, which agent clients produce often.
- **Plan:** (1) add a detector to the soak in A2: flag any completion whose first
  token is ≥ 248077. (2) Write a reproducer: start long prompts that share a
  prefix and abort them mid-prefill, then resend the prefix. (3) If it reproduces,
  review #38355 and overlay it.
- **Effort:** half a day for the reproducer; the fix PR is still under review.

### A4. Zombie requests after client disconnect (sglang#36333 / #36876)

- **Status: DONE 2026-09-15, partial.** 1581 aborted request ids kept emitting up to
  105 output steps after disconnect, but 0 slots were held at 128 idle checks.
  `max_tokens` guidance added to the README.
- **What:** an abort that lands in the batch-transition window never reaches the
  scheduler. The request decodes to `max_tokens` and holds one of our
  `MAX_RUNNING=4` slots.
- **Plan:** in the A2 soak, count `state was deleted in TokenizerManager` lines.
  Short term, document a sane `max_tokens` for agent clients. Pick up the fix when
  it merges.

### A5. Fixes we get by rebasing (see Track D)

- **#34820:** mamba prefix-cache checkpoints were stored at bf16 even with an fp32
  pool, so a cache hit restores lower-precision state than a miss. This may
  contribute to our temperature-0 instability (`effort_thinking_off` 24/25/28).
  **Unverified** that the GDN path we use takes this code path: `gdn_backend.py`
  is not in the PR's file list.
- **#37165:** stale deferred Mamba clear/COW metadata was replayed into
  `DRAFT_EXTEND_V2`. On Inkling, acceptance went 47.7% → 57.4%.
- **#38346:** A1.

**Checked, not applicable:**
- #38851 (trtllm scratch NaN tail): our SM121 Triton kernel masks reads to
  `[kv_start, kv_end)`.
- #39219: fused KDA verify is opt-in and off.
- #37418: prefill CUDA graphs are FA3/FA4 only; we use triton prefill.
- #36131: SWA models.

---

## Track B — Cold boot (~9.5 min → ~6.5 min, maybe ~2–3 min)

Verified against the running server's log (boot 2026-09-14 11:19):

| Phase | Time | Evidence |
| --- | ---: | --- |
| Container start, imports, tokenizer | ~42 s | 11:19:19 → 11:20:01 |
| Main weight load, 206 files | **413 s** | `Load weight end. elapsed=413.09 s` |
| — of which the last 17 shards (PLE files load last) | **83 s** | shard bar 189/206 at 05:17, 206/206 at 06:40 (~1.7 s/shard before) |
| MTP draft weight load | **88.7 s** | `elapsed=88.66 s`; shard bar walks **206/206** again |
| KV/Mamba alloc, CUDA graphs, JIT | ~25 s | 11:28:29 → 11:28:54 ready |

Checkpoint facts, from `model.safetensors.index.json` at `7b719225`:
- 206 files, 125.9 GiB, 296,475 tensors.
- `mtp.*` keys live only in `model-bf16-00010..00012` (13.7 GiB of files).
  `lm_head.weight` and `embed_tokens.weight` are also in `00012`.
- The 10 `model-plefp8-*` files are 47.7 GiB and hold 129 tensors.
- The pinned loader (`multi_thread_safetensors_weights_iterator`, 8 threads by
  default) calls `f.get_tensor(k)` for **every** key of every file. The PLE files
  are therefore fully materialised in RAM, even though `patches/ple_reuse.py` then
  only compares 32 sampled windows per shard.

### B1. Draft pass reads only the MTP files — ~80 s, low risk

- **Status: DONE 2026-09-14, accepted.** `patches/draft_mtp_files.py` (loader
  `key_filter` + `Qwen4ExpForCausalLMMTP.checkpoint_key_filter`); the filter keeps
  `00010..00012` (13.72 GiB, all 31 `mtp` keys). Draft load 81–90 s → 31–33 s;
  boot 562–575 s → 525 s on the clean repeat (main load varies ±50 s between
  boots). KV, acceptance, decode and quality unchanged. Rollback:
  `SGLANG_CHECKPOINT_KEY_FILTER=0`. See RESEARCH_LOG "Sep 14 plan — B1".
- **Change:** when loading the NEXTN draft, restrict the file list to files that
  the index maps `mtp.*` keys to, plus the files holding any non-MTP tensor the
  draft `load_weights` consumes. Today that is `00010..00012`, which already
  contains `lm_head`/`embed_tokens`. Derive the set from the index at runtime; do
  not hard-code it.
- **Check first:** log every key `Qwen4ExpForCausalLMMTP.load_weights` actually
  consumes on a normal boot. The filtered set must be a superset of that.
- **Gate:** draft `mem usage` unchanged (0.99 GB), `spec_accept_length` unchanged
  on the decode suite, quality 12/12, and the token map still applies.
- **Expected:** 88 s → under 15 s.

### B2. Skip reading the PLE files when the on-disk table is valid — ~80–100 s, low risk

- **Status: DONE 2026-09-14, rejected.** Built the `get_slice` variant
  (`patches/ple_slice_reuse.py`, not applied). safetensors 0.8.0 `get_tensor` is
  lazy over mmap, so the PLE files already cost only ~4–6 s; the 83 s tail is the
  four `model-bf16` files. Measured main load 396 s off vs 402 s on, boot 504 vs
  515 s. See RESEARCH_LOG "Sep 14 plan — B2".
- **Change:** record the HF blob names (sha256 content addresses) of the 10
  `model-plefp8-*` files next to the table file. At boot, if they match and all
  128 shards were previously completed, drop those files from the weight
  iterator. Fall back to today's read-and-sample path on any mismatch.
- **Check first:** which of the 129 tensors is not a table shard (probably a
  scale). If the model needs it, load just that tensor with
  `safe_open(...).get_slice`.
- **Cheaper alternative:** keep sampling, but read only the sample windows via
  `get_slice` instead of `get_tensor`.
- **Side benefit:** removes up to ~38 GB of transient RAM (8 threads × ~4.8 GB
  files) at a point where only ~30 GB is free.
- **Gate:** boot log still reports `128/128 shards already on disk`. The PLE
  identity guard in `run_config.sh` still passes. A deliberately corrupted
  sidecar must force the full path. Needles and quality unchanged.
- **Does not affect Track C1:** that is about the table file's page cache, not
  the safetensors files.

### B3. Profile the remaining ~300 s before building anything

- **Hypothesis (unconfirmed):** per-tensor Python overhead. ~296k tensors, mostly
  per-expert NVFP4 weights, handled at ~1,000/s. Not disk: raw NVMe is 8.7 GB/s
  and the load averages ~0.3 GB/s.
- **Step 1:** `py-spy dump` on the scheduler PID several times while the shard bar
  moves (py-spy is in the image).
- **Step 2 (cheap A/B):** `EXTRA_ARGS='--model-loader-extra-config {"num_threads":16}'`.
  This helps only if reading is the bottleneck.
- **Step 3 (only if the profile supports it):** a one-time post-load weight cache.
  SGLang's `sharded_state` save/load writes the loaded weights, minus the PLE
  table, as a few large files. It must survive our mounted model overlays and
  NVFP4 post-processing, and it must be regenerated whenever weights change
  (Track D2 would change them). Plausible, not proven.
- **Expected:** B1 + B2 gets ~9.5 → ~6.5 min. B3 might reach ~2–3 min.

**Already ruled out:** `--weight-loader-prefetch-checkpoints` (local NVMe);
`--load-format dummy` (OOM on the PLE table, EXPLORATION.md).

---

## Track C — Performance experiments on the current pin

### C1. Long-context prefill cliff: re-test PLE prefetch on real text — high value, cheap

- **Status: DONE 2026-09-16, accepted.** `SGLANG_QWEN4_PLE_FILE_PREFETCH=1` is now the
  default. Real-text cold TTFT on identical LongBench-v2 prompts: 32k 85–112 s → 19–27 s,
  128k 413–569 s → 144–289 s; warm paths, mixed load, decode and quality unchanged.
  The threaded `pread` prototype did not beat WILLNEED (rejected). **250k fresh real text
  is not servable safely either way:** the KV pool commits lazily on unified memory
  (~24.8 KiB/token), so ~155–190k prefilled tokens exhaust the headroom (watchdog floor
  and NVRM OOM). **New follow-up needed: size `MAX_TOTAL` to committable memory** — a
  full 524k radix cache would need ~13 GiB more than boot leaves. See RESEARCH_LOG
  "Sep 14 plan — C1".
- **Evidence:**
  - Our 248k LongBench-v2 run averaged ~440 tok/s, with engine batches at
    308–607 tok/s. vLLM recipes report ~1,940 tok/s at 256k (MiaAI) and
    1,660 tok/s at 176k (tonyd2wild).
  - tonyd2wild measured mmap page faults on a cold table at 10–20k rows/s
    regardless of thread count, serialised on `mmap_lock`. `preadv` reached
    131k rows/s. A 4096-token chunk needs ~65k PLE rows.
  - blazux: the first pass over a cold table region is 2–3x slower.
- **Why U4b missed it:** U4b rejected `SGLANG_QWEN4_PLE_FILE_PREFETCH=1` using
  `bench/prefill.py`, whose filler n-grams are shared across repeats. The table
  was PLE-warm by construction, so prefetch had nothing to fix.
- **Plan:**
  1. Add a real-text prefill mode to `bench/prefill.py` (salted LongBench-v2
     samples at 32k/128k/250k). Drop the table file from page cache between cold
     runs with `fadvise DONTNEED`; no sudo needed.
  2. A/B prefetch 0 vs 1, one variable.
  3. If prefetch is not enough, prototype a threaded `preadv` staged gather
     (tonyd2wild's design; check its licence before borrowing code).
- **Gate:** TTFT at 128k and 250k real text, needles, mixed-load p95, and host
  MemAvailable margin.

### C2. `--mamba-track-interval 256` — cheap

- **Status: DONE 2026-09-16, rejected.** Base + two 256 boots with 120-turn: agentic
  decode is set by the greedy output mode (42 vs 29 tokens/turn), and within a mode
  256 changes nothing (58.5 vs 58.1 tok/s); decode suite no better; agentic cache hit
  0.3–1.2 pts lower; prefix-warm TTFT unchanged. Interval stays 64. See RESEARCH_LOG
  "Sep 14 plan — C2".
- **Evidence:** SGLang's verified Spark cells (#37995) dropped 64 for the default
  256 with MTP 3/1/4, scoring 97.1% on full GSM8K. Our U5a single boot showed
  thinking-off agentic decode of 58 vs 49 tok/s, never confirmed, and it predates
  U6. KV did not move because `MAX_TOTAL` binds.
- **Plan:** two boots with `MAMBA_TRACK_INTERVAL=256` on the U6 default. Full
  suite plus 120-turn. Watch the 32k/128k prefix-warm TTFT: coarser checkpoints
  can lower cache hit granularity.

### C3. Smaller draft vocabulary + English prose bench — cheap

- **Evidence:** MiaAI's 47,149-token code-tuned draft vocab gave +13% over the full
  head. Our 64k map is the only size we measured.
- **Plan:**
  1. Add an English prose case to `bench/decode.py` (keep Spanish) so our numbers
     are comparable with other recipes.
  2. Build 48k and 32k maps with `scripts/build_draft_vocab.py`.
  3. A/B each against 64k on code, EN prose, ES prose and agentic.
- **Gate:** decode median ranges must not overlap, and acceptance must not drop
  on any category by more than it gains.

### C4. Determinism probe — diagnostic, cheap

- **Evidence:** blazux found vLLM's GB10 QSA top-k non-deterministic, so the same
  greedy prompt gave different outputs. SGLang uses a different `fast_topk`, but
  our `effort_thinking_off` case flips 5/10 at temperature 0.
- **Plan:** extend `bench/effort_probe.py` to compare full token-id sequences,
  flushing the cache between repeats (cache miss) and not flushing (cache hit).
  - Divergence without the cache points at kernels.
  - Divergence only with the cache points at #34820-style state precision.

---

## Track D — Larger changes (need a rebase)

### D1. Rebase onto current SGLang main

- **Gains:** A5 (#34820, #37165, #38346), the upstream file-backed PLE backend and
  router fix (#39126), the FP8 KV fixes (#38855, #38851), and the ModelOpt mixed
  loader with `FP8_BLOCK_SCALES` dispatch (prerequisite for D2).
- **Keep:**
  - `ple_reuse.py`: upstream still rewrites the table every boot, per #39126's
    own "known limitation".
  - The SM121 Triton QSA routing: check whether #36556 or upstream now covers it.
  - The ReplaySSM PLE commit hunk (#37794 is still open).
  - The B1/B2 loader changes.
- **Plan:** pin a main commit whose image exists. Port the overlays and run
  `test_qsa_prefill_selection.sh`-style kernel tests plus the full U12 cumulative
  gate. Treat it like U3: a new engine, no speed claim until measured.

### D2. FP8 blockwise dense side layers ("hybrid") + FP8 `lm_head` — biggest decode lever

- **Evidence:**
  - U8b established that attention, GDN projections, shared experts, gates, MTP
    and `lm_head` are all BF16 in the RadixArk checkpoint.
  - blazux: converting the 300 side-layer tensors (~15 GiB) to 128×128 blockwise
    FP8 gave **+20% decode, +8% KV, identical 17-scenario agentic score**.
  - Shinrali: FP8 `lm_head` adds **+8.4%**.
  - Converter: @Saren-Arterius `fp8_convert.py` (Apache-2.0).
- **Unknowns:**
  - Whether SGLang's block-FP8 linear kernel is fast on SM121. vLLM needed an
    `M % 4` padding fix.
  - Whether QSA `qkv_proj` is built without a quant config in SGLang, as it is in
    vLLM.
  - Interaction with the NEXTN draft, which shares `lm_head`, and with the 64k
    token map.
- **Plan:**
  1. After D1, micro-benchmark block-FP8 vs BF16 linear at the side-layer shapes
     on GB10.
  2. Convert into a sibling snapshot made of symlinks. Only the rewritten shards
     are new files.
  3. Serve through the ModelOpt mixed config.
  4. Full quality gate: GSM8K n=200, BFCL subset, arith class probe, 120-turn,
     needles.
- **Note:** new weights invalidate any B3 weight cache. They do not invalidate the
  PLE table, which is untouched.

### D3. 512k context — optional

- **Evidence:** our 512k attempt failed because the YaRN override targeted
  `rope_scaling`. MiaAI deep-merges YaRN into `text_config.rope_parameters`,
  keeping `mrope_section`, and passed 3/3 needles at 400k. Our 512k boot also
  yielded only 201,984 KV tokens, so FP8 KV is likely required too.
- **Costs:** blazux measured FP8 KV at −10% decode, −30% prefill and one agentic
  scenario lost. MiaAI warns that long-reasoning quality is unvalidated on sparse
  attention.
- **Plan:** only after D1. Use
  `--json-model-override-args` into `text_config.rope_parameters` plus
  `--kv-cache-dtype fp8_e4m3`. Keep 262k BF16 as the default unless quality holds.

### Watch list (no action yet)

- **#36567:** stream PLE rows straight from the safetensors files with io_uring.
  Would remove the 48 GB table file, the one-hour first boot and ~50 GB of disk.
  Open, stacked on another PR.
- **#38209:** QSA prefill selection. Already tested as U8a: no gain.
- **#39333:** QSA chunk-prefill gathers indices instead of K/V. Transient memory
  only.
- **#38180:** mixed-chunk guard. We do not enable mixed chunk.

### Not pursuing

- **Moving to vLLM:** no single-Spark vLLM recipe beats our code or agentic decode,
  and we would lose the radix cache.
- **BF16 recurrent state:** +7–8% on vLLM, but SGLang ReplaySSM re-quantizes the
  committed state (U9: probe 6/20 → 0/20). Revisit only if D1 changes that path.
- **INT4 AutoRound / Code Turbo experts:** a different quantization with a quality
  trade-off, and vLLM Marlin only.
- **Kernel VM sysctl tunables:** need sudo.

---

## Execution order

| # | Item | Cost | Expected payoff |
| --- | --- | --- | --- |
| 1 | A1 overlay (#38346) — **done** | 1 boot | removes a latent crash |
| 2 | B1 + B2 boot fixes — **B1 done (accepted), B2 done (rejected)** | 2–3 boots | boot ~9.5 → ~6.5 min |
| 3 | A2/A3/A4 24 h soak with detectors — **done** | 1 day, unattended | finds or rules out decay, KV corruption, zombies |
| 4 | C1 real-text prefill + prefetch A/B — **done (accepted; new MAX_TOTAL follow-up)** | 2 boots | possibly 3–4x prefill at 128k+ |
| 5 | C2 track interval 256 — **done (rejected)** | 2 boots | possible agentic decode gain |
| 6 | C3 draft vocab 48k/32k + EN prose bench | 3 boots | possible +5–13% decode |
| 7 | C4 determinism probe | no boot | explains temp-0 flakiness |
| 8 | B3 loader profile (py-spy, 16 threads) | 1 boot | decides whether a weight cache is worth it |
| 9 | D1 rebase onto SGLang main | several days | stability fixes; enables D2/D3 |
| 10 | D2 FP8 hybrid side layers + FP8 `lm_head` | several days | possible +20–28% decode |
| 11 | D3 512k (optional) | 2–3 boots | context beyond 262k |
| 12 | B3 weight cache (if the profile supports it) | days | boot ~6.5 → ~2–3 min |

Items 1–8 stay on the current pin and can be accepted independently. Items 9–12
are ordered after the rebase because they depend on its loader and kernels.

## Sources

- SGLang PRs: [#38346](https://github.com/sgl-project/sglang/pull/38346),
  [#37786](https://github.com/sgl-project/sglang/pull/37786),
  [#34820](https://github.com/sgl-project/sglang/pull/34820),
  [#37165](https://github.com/sgl-project/sglang/pull/37165),
  [#38355](https://github.com/sgl-project/sglang/pull/38355),
  [#38191](https://github.com/sgl-project/sglang/pull/38191),
  [#39126](https://github.com/sgl-project/sglang/pull/39126),
  [#37995](https://github.com/sgl-project/sglang/pull/37995),
  [#38851](https://github.com/sgl-project/sglang/pull/38851),
  [#36567](https://github.com/sgl-project/sglang/pull/36567)
- SGLang issues: [#37326](https://github.com/sgl-project/sglang/issues/37326),
  [#38319](https://github.com/sgl-project/sglang/issues/38319),
  [#36876](https://github.com/sgl-project/sglang/issues/36876),
  [#38731](https://github.com/sgl-project/sglang/issues/38731)
- Community recipes:
  [MiaAI-Lab single Spark](https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark),
  [blazux/qwen3.8-Flash-DGX](https://github.com/blazux/qwen3.8-Flash-DGX),
  [tonyd2wild](https://github.com/tonyd2wild/Qwen3.8-Flash-Next-NVFP4-DGX-Spark),
  [madeye](https://madeye.github.io/qwen38-flash-next-on-dgx-spark/)
- Checkpoints:
  [Shinrali FP8 lm_head](https://huggingface.co/Shinrali/Qwen3.8-Flash-Next-NVIDIA-Hybrid-FP8-LMHead-Single-Spark),
  [Code Turbo](https://huggingface.co/sayyidfareed/Qwen3.8-Flash-Next-Code-Turbo-Spark)
- NVIDIA forums:
  [Which single-Spark thread is best](https://forums.developer.nvidia.com/t/which-single-spark-qwen3-8-flash-next-thread-is-the-best/382522),
  [NVIDIA NVFP4 on 1/2/4 Sparks](https://forums.developer.nvidia.com/t/qwen3-8-flash-next-on-1-2-and-4-dgx-sparks-with-nvidias-official-nvfp4-quant-64-tok-s-peak-single-stream/382476)
- Boot evidence: `docker logs qwen38-flash-next` (boot 2026-09-14 11:19–11:28);
  pinned loader `sglang/srt/model_loader/weight_utils.py:1188`
  (`multi_thread_safetensors_weights_iterator`, `get_tensor` per key) and
  `loader.py:383` (`DEFAULT_NUM_THREADS = 8`).

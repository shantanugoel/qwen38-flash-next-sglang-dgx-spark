# Upstream validation plan — September 2026

Status: **U0–U10 executed. U0–U4a, U6, U7a and U7b accepted-as-measured;
U4b/U4c/U5a, the U7b 2048 candidate, the U8a QSA prefill overlay, U9 BF16 state
and the U10 checkpoint switch rejected; U5b skipped; U8b not applicable.
U11–U12 not started**. This supersedes stale TODO/status and skip-list
entries in `EXPLORATION.md` for this campaign. August measurements remain
historical evidence. U1 was accepted on September 7 (Triton #36845 serving
default; KDA overlay rejected on this image). U2 was accepted the same day:
ReplaySSM verify now commits PLE n-gram/short-conv state (#37794 `spec_utils`
hunk only; NGRAM not ported). U3 was accepted on September 8: pin SGLang
`4ccff141` (`lmsysorg/sglang@sha256:9d2a843c706c74bc259c0d9abf360551eb2734e1e7d255ab012a6965f10480b6`),
native PLE file backend, U1 Triton overlay over bundled KDA, U2 PLE commit kept.
U4a was accepted the same day: second native-file reuse boot (`128/128`, 0 copied,
618 s); `ple_reuse.py` stays because the native loader still rewrites the table.
U4b (prefill `posix_fadvise(WILLNEED)`) was rejected the same day: 32k prefill
unchanged vs prefetch-off; recipe default stays `SGLANG_QWEN4_PLE_FILE_PREFETCH=0`.
U4c (8 GiB RSS trimmer) was rejected the same day: mapping RSS stayed ~0.4 GiB
across a one-hour growing-context soak, so the trimmer never fired. Default stays
`SGLANG_QWEN4_PLE_FILE_RSS_BUDGET_GB=0` (native default is 8). U5a (`--mamba-track-interval 256`) was rejected the same
day: `max_total_num_tokens` stayed 524288. U5b skipped (MAX_TOTAL already binds).
U6 was accepted the same day: 64k NEXTN `--speculative-token-map` is now the
serving default (thinking-off code 40.29 → 47.6 tok/s on two launches). Further
map sizes were not measured. U7a was accepted the same day: MTP graphs already
capture `bs=[1,2,3,4]` under `MAX_RUNNING=4`; no trim. 1/2/4-stream baseline
logged. U7b completed on September 8: a second 4096 launch plus a 2048 candidate.
2048 was rejected (p95 chunk gap 29.825 s vs 26.115 s; mixed TTFT +10.6%),
`PREFILL` stays 4096, and the earlier gate failure was shown to be an unstable
quality case (9/20 correct at temperature 0 in one boot). Later items follow.

Objective: improve correctness first, then long-horizon agentic latency, speed,
and memory headroom on one DGX Spark. Keep only demonstrated improvements.

## Fixed constraints

- One GPU occupant; unload llama-swap before starting a serving experiment.
- No sudo, clock changes, edits under `~/ai`, or unrelated service changes.
- Preserve vision, thinking on by default with a per-request off switch, native
  262144 default context, and effective prefix caching.
- Keep RadixArk revision `7b719225242aacd3dbd3f9407468c2ee9a9d2594` until the
  explicit checkpoint comparison. Never mix checkpoints in one PLE backing file.
  Runtime may reuse the existing mmap via PLE_DIR and HF cache via HF_CACHE;
  do not commit host-specific cache paths.
- Every long operation runs detached with stdout/stderr under `results/`.
  One experiment is one uniquely tagged background `scripts/run_config.sh` job;
  extend that wrapper when necessary. Poll logs/health in short calls, normally
  30–60 seconds apart. No foreground pulls, preparation, loads, benches, or
  `docker logs -f`. Allow quiet mmap startup up to approximately 25 minutes.
- Do not overlap benchmark suites. Intentional concurrency belongs inside one
  controlled benchmark. Do not launch the next item before the current commit.

## Mandatory decision and commit protocol

For every item and separately numbered subitem below:

1. Start from the last accepted configuration. Record its commit, immutable
   image digest, package/source versions, checkpoint revision, effective flags,
   clock state, memory footprint, and PLE identity. Never log secrets or `.env`.
2. Change one variable. If an engine/dependency migration is indivisible, list
   its contents and report the result as a bundle, not a single-patch gain.
3. Match prompts, output limits, sampling, thinking, concurrency, and cache state.
   Warm once, measure at least three repeats, and confirm promising performance
   on a second launch, preferably A/B/A. Avoid warm-prefix contamination in cold
   prefill tests; test prefix reuse separately.
4. Decide **accepted**, **rejected**, or **deferred**, with evidence and limits.
   Correctness fixes may be accepted despite lower speed. A known-incorrect
   configuration is neither the fallback nor a valid performance winner.
5. For pure performance changes, predeclare the primary metric and require at
   least 5% median improvement, consistent direction across repeats and a second
   launch, with no unexplained >5% regression in another core workload. Treat
   overlapping/noisy results as inconclusive. Smaller gains may qualify only for
   a measured boot/memory/maintenance benefit without material serving regression.
6. Require no new API errors, malformed tools, lost vision, instability or observed
   quality regression. Small smoke suites and nonsignificant differences do not
   establish equivalence; expand paired testing when the result is uncertain.
7. After rejection, restore the last accepted implementation and verify health
   and smoke checks. If none is validated, stop serving and record the blocker
   instead of restoring a known-bad path.
8. Append RESEARCH_LOG.md with item/TAGs, exact revisions, sanitized commands,
   tables, uncertainty, decision, rollback and next item. Update README defaults,
   claims and caveats and this checklist to reflect the actual result.
9. Review the diff and **commit scripts/tests/docs only before the next item**.
   Rejected/deferred items still receive a documentation commit. Never commit
   weights, backing files, caches, results, logs, or `.env`.

## Common gates

**Fast gate after each boot:** bounded health wait, arithmetic, tool parsing,
executed code, multi-turn fact, positional vision, both thinking modes,
code/Spanish-prose decode, 8k/32k needles, cold and repeated-prefix TTFT.
The current harness uses `|| true`; completion is not proof that suites passed.

**Promotion gate:** 120-turn agentic sessions in both thinking modes; distinguish
malformed calls from tool frequency and refusal from failed recall. Measure
steady decode separately from end-to-end completion rate. Record acceptance and
tokens per engine step when available. Include controlled 1/2/4-stream cases.

**Long-context gate:** salted prompts at 32k, then 120k, 190k, 210k, and near the
native limit with output headroom. Plant independent needles at 5/50/95%, use
multiple matched seeds, and include free generation and reasoning rather than
only copying codes. Advance one size at a time only with adequate memory margin.
Report actual tokenizer counts. Configured context is not validated context.

**Expanded quality gate** for changed state/KV precision, drafting, or checkpoint:
fixed executable-code and tool-use cases; paired GSM8K n=200 screening, full 1319
for a finalist; multilingual and long-reasoning tasks; long-context and 120-turn
gates. Publish paired outcomes, sample counts, uncertainty, failures and output
caps; investigate losses before acceptance. Existing BFCL single-turn cases are
useful; mocked multi-turn scoring is not. Terminal-Bench requires an x86 host
against this endpoint; never report a TB score measured on this Spark.

## Ordered items

### U0 — Baseline inventory and reliable experiment harness

- [x] Inventory read-only first; preserve current image and patched sources.
- [x] Add per-suite failure propagation, effective-version/config capture,
  checkpoint-aware PLE identity, bounded startup/request deadlines and a non-root
  memory watchdog before long prefills. Monitor MemAvailable, MemFree, swap and
  driver allocation errors. Calibrate conservative thresholds on this host rather
  than copying vLLM's memory fraction. Stop gracefully on sustained pressure.
- [x] Verify failure detection/watchdog logic with synthetic inputs. Keep one
  benchmark lock and ensure detached jobs survive tool return.
- [x] Measure the historical configuration only in its already-tested short
  context range. Do not stress old QSA at 120k+ just to reproduce published bugs.
- [x] Log baseline and tooling validation; commit before U1.

### U1 — Dedicated SM121 sparse-decode correctness fix

- [x] Audit/backport merged [#36845](https://github.com/sgl-project/sglang/pull/36845).
  Remove the widened TRT-LLM gate for GB10 and ensure preparation preserves the
  new SM121 dispatch. Preserve other architectures' behavior.
- [x] Run reference-tensor and CUDA-graph replay checks, fast/promotion gates,
  then staged long-context validation. Accept restored correctness even if slower
  than old historical figures; establish the corrected baseline.
  Triton serving: 8k/32k + 120-turn pass. KDA overlay rejected (32k token-id 0).
  120k/190k/210k not run here (sequential long prefills have wedged this box).
- [x] Compatible backport on the existing `qwen38flashnext` image; no U3 bundle
  required for this item. Never roll back to the known-faulty long-context path.
- [x] Document results, limits and defaults; commit before U2.

### U2 — ReplaySSM PLE-state correctness audit

- [x] Trace our exact flag/alias and code branches against open
  [#37794](https://github.com/sgl-project/sglang/pull/37794). This is a reported
  bug, not a confirmed diagnosis of our installed version.
  Affected: `--enable-gdn-replayssm-spec` aliases `--enable-linear-replayssm-spec`,
  `replayssm_spec_fold=True`, GDN fold returns before the PLE roll.
- [x] If affected, isolate PLE history/short-convolution state commits after verify;
  do not introduce the PR's NGRAM feature. Test accepted/rejected draft state
  transitions, then fast, expanded-quality and promotion gates. Compare against
  non-speculative behavior without requiring identical sampled text.
  Isolated `spec_utils` patch accepted; NGRAM left refused. See RESEARCH_LOG U2.
- [x] Document results, limits and defaults; commit before U3.

### U3 — Pinned newer SGLang model-development image

- [x] Recheck registry/source metadata at execution time. September 7 audit of
  `dev-qwen38-next-local` @ `9b2aee2283` was stale by run time. Executed pin is
  `lmsysorg/sglang:dev-qwen38-next-local` commit `4ccff141` (MTP token-0 router
  [#38290](https://github.com/sgl-project/sglang/pull/38290)), Hub digest
  `sha256:9d2a843c706c74bc259c0d9abf360551eb2734e1e7d255ab012a6965f10480b6`,
  local alias `lmsysorg/sglang:dev-qwen38-next-local-4ccff14`. Image id
  `sha256:cdd9649ba1cf472344fd1e11e7cbaa7161a0624329b522931646537cc1c15701`.
  Hold that digest; hub tags move.
- [x] Hold Radix weights and logical serving settings fixed. Reconcile: overlay
  U1 Triton over bundled KDA QSA; keep U2 ReplaySSM PLE commit (no NGRAM); skip
  mmap overlay (native `allocate_ple_host_table`); reuse existing
  `ple_table_51200245760_51200245760.bin` via `ple_file_compat.py`; prefetch and
  RSS trimmer off. Empty `PLE_OFFLOAD_BACKEND` must default to `file` or the
  native pinned-RAM path OOMs.
- [x] Common gates + one-hour soak on TAG `u3-dev-4ccff14-20260908b`. Soak
  2214/2214. Decode overlaps U2 within noise. Quality 11/12 is the known
  `effort_thinking_off` miss (`28` not `24`). Agentic late-recall fails by
  refusal, not forgotten state. Accepted. See RESEARCH_LOG U3.

### U4 — Native PLE backend, prefetch and resident-memory control

- [x] U4a: second reuse boot TAG `u4a-reuse-boot-20260908` 617.62 s. Logs:
  `reusing recipe backing file` then `128/128 shards already on disk
  (320001536 rows), 0 copied`. Scheduler `write_bytes` 8192 during load.
  Native #37068 still rewrites without `ple_reuse.py`; patch kept. PLE inode
  and sampled identity unchanged; no writes under `~/ai`. Soak skipped via
  `ONLY=`. Fast gate: smoke/decode/longctx pass; quality 11/12 is the known
  `effort_thinking_off` miss (`28` not `24`). Accepted. See RESEARCH_LOG U4a.
- [x] U4b: prefetch-on TAG `u4b-prefetch-20260908` vs matched off
  `u4b-prefill-off-20260908`. WILLNEED logged only when on. 32k cold TTFT
  10.24 s vs 10.23 s. 8k first-send 3.15 s vs 3.46 s (n=1, not a median).
  PLE-warm and prefix-warm identical. Decode overlaps. Default stays 0.
  Rejected. See RESEARCH_LOG U4b.
- [x] U4c: TAG `u4c-rss-trim-20260908`, budget 8 GiB, prefetch still off.
  Trimmer logged `resident set capped at 8.0 GiB`. Host mapping RSS 0.29–0.43 GiB
  over one hour; 0 trim events. Growsoak 5820/5820, 0 recall fails. Decode and
  32k TTFT overlap U4a. Default stays 0. Rejected. See RESEARCH_LOG U4c.

### U5 — Mamba tracking interval and usable KV capacity

- [x] U5a: TAG `u5a-mamba-interval-256-20260908`. Effective `mamba_track_interval=256`,
  `max_total_num_tokens` still 524288 (MAX_TOTAL binds). 8k/32k needles PASS.
  120-turn 0 invalid; late recall refusals as in U3. Default stays 64.
  Rejected. See RESEARCH_LOG U5a.
- [x] U5b skipped: 256 did not raise allocated KV, so changing MAX_TOTAL is not
  warranted. Native 262144 context unchanged. See RESEARCH_LOG U5b.

### U6 — Reduced-vocabulary MTP drafting

- [x] Audit NEXTN verification and token-ID mapping before porting the concept
  from [MiaAI](https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark).
  Keep target vocabulary/sampling unchanged; excluded draft tokens must remain
  possible through target verification. See RESEARCH_LOG U6 audit.
- [x] TAG `u6-draft-vocab-64k-20260908` + confirm `u6-draft-vocab-64k-confirm-20260908`.
  65536-row map via `--speculative-token-map`. Thinking-off code 47.59 / 47.57 vs
  U4a 40.29 (+18%, non-overlapping). Quality 12/12 both boots. KV still 524288;
  not a resident-memory saving. Default is now `bench/draft_vocab/hot_tokens_64k.pt`.
  Accepted. See RESEARCH_LOG U6.
- [x] Log and commit the 64k variant. Smaller maps were not measured.

### U7 — Graph coverage and mixed-load chunk sizing

- [x] U7a: audit actual reachable draft/verify sizes and graph replay. Trim excess
  captures without disabling padding or losing coverage. Measure boot/memory
  and 1/2/4-stream performance. Log decision and commit.
  TAG `u7a-streams-baseline-20260908`. Capture already `bs=[1,2,3,4]`; no trim.
  Streams c=1/2/4 aggregate 48.53 / 77.74 / 104.38 tok/s. Defaults unset. Accepted.
- [x] U7b: **2048 rejected; default stays 4096.** TAGs `u7b-4096-20260908`,
  `u7b-4096-confirm-20260908` (second launch) and `u7b-2048-20260908`.
  Primary metric, median p95 streamed-chunk gap: 26.538 / 26.115 s at 4096 vs
  **29.825 s at 2048** (+14.2%); mixed TTFT 28.746 / 28.269 vs 31.264 s; 32k cold
  TTFT 10.218 / 10.107 vs 11.167 s. 1024 not launched (monotone direction; the
  historical sweep already measured 12.31 s at 1024). No opt-in mixed-load
  profile is shipped. Mechanism: `server_args.py` asserts `not enable_mixed_chunk`
  under any speculative algorithm, so with NEXTN the scheduler never mixes decode
  into a prefill batch and both decode streams stall for the entire 64k prefill
  whatever the chunk size. The earlier deferral's gate failure is resolved:
  `bench/effort_probe.py` resent the failing `effort_thinking_off` case 20x at
  temperature 0 in one boot and got `24` 9x, `28` 7x, `25` 4x, so a single
  11/12 or 12/12 on that case carries no signal; the failed 8k needle was the
  same refusal string, and the second 4096 launch passed 8/8. First attempt
  (`u7b-4096-20260908`): prefill 7/8, quality 11/12, 3/3 mixed needles, median
  TTFT 28.746 s, p95 gap 26.538 s, 2.77 tokens/chunk. See RESEARCH_LOG U7b and
  "U7b (continued)". Original scope:
  hold graphs fixed; compare prefill 4096 to 2048/1024, optionally 8192,
  one candidate and commit at a time. Measure cold TTFT, aggregate throughput,
  and p50/p95/p99 streamed-chunk gaps when a 64k prefill arrives during two
  decodes. Report tokens per chunk: chunk gaps are not token gaps. A specialized
  mixed-load profile may remain opt-in; do not silently impose its cold-TTFT
  tradeoff on the general default. Log each decision and commit.

### U8 — QSA prefill and relevant GEMM optimization

- [x] U8a: **rejected for serving; kept as an opt-in overlay.** PR was open (not
  merged) at execution time, head `7a4343c5`, base `qwen4-main-squashed`
  `9b2aee22`, and carries **no reviews or issue comments**, so there were no
  indexer correctness reviews to read. Applies cleanly over the U1 Triton
  backend; the PR's own kernel tests pass 87/88 here, the single failure being
  the pre-existing bundled-KDA resolver assertion U1 replaced (fails identically
  without the overlay). Matched A/B, TAGs `u8a-qsa-prefill-38209-20260908` and
  `u8a-baseline-20260908`: 32k cold TTFT 10.119 vs 10.125 s, 128k 42.002 vs
  42.196 s, 8k PLE-warm 2.683 vs 2.689 s, mixed p95 26.091 vs 26.201 s, decode
  and boot flat. Only the 4-stream aggregate moved (114.06 vs 101.50), and
  per-stream medians are identical (34.48 vs 34.35) with c=2 moving -8.5% the
  other way, so it is wall-clock overlap noise, not a win. Cached suffixes
  (prefix-warm resends) and concurrent 1/2/4 streams were both covered. First
  128k prefill/needle numbers on this box came out of this item. Enable with
  `QSA_PREFILL_SELECTION=1 ./scripts/prepare.sh`; default is off.
  See RESEARCH_LOG U8a.
- [x] U8b: **deferred, not applicable to this checkpoint; no flag changed.**
  Radix `7b719225` has 221184 NVFP4 scale tensors and every one is under
  `*.mlp.experts.*` (plus one PLE tensor); `quantization_config.ignore` excludes
  attention, shared experts, gates, MTP and `lm_head`. #38170 only changes which
  FlashInfer `mm_fp4` backend a quantized **dense linear** uses, and routed MoE
  branches on `get_moe_runner_backend()` instead, so b12x cannot reach a hot
  path here. FlashInfer 0.6.17 does accept `backend="b12x"`, and the image's
  enum lacks the member, so using it would need the PR's hunks — not worth a
  boot for a path with no callers. The NVIDIA checkpoint audited for U10 has the
  same shape. Torch/FlashInfer were not upgraded. See RESEARCH_LOG U8b.

### U9 — Optional BF16 recurrent state

- [x] Support confirmed in the pinned image: the flag reaches the state dtype and
  SM121 keeps `linear_attn_decode_backend=triton`, so BF16 unlocks no kernel here
  (the FlashInfer GDN decode that needs bf16 is SM100+ only). With our accepted
  `--enable-gdn-replayssm-spec`, the engine logs that a non-fp32 state
  re-quantizes the committed state each commit/flush and may drift.
- [x] **Rejected; `--mamba-ssm-dtype` stays `float32`.** TAG `u9-bf16-ssm-20260908`
  vs matched FP32 `u8a-baseline-20260908`. State pool 2.21 -> 1.11 GB
  (+1.13 GB free GPU, +1.12 GiB host MemAvailable) but `max_total_num_tokens`
  stays 524288 because `MAX_TOTAL` binds, so the headroom is unservable.
  Prefill/mixed-load flat (32k 10.148 vs 10.125 s; p95 26.186 vs 26.201 s),
  decode directions disagree within noise, and the repeated-sample probe fell
  from 6/20 to **0/20** on the one deterministic case — suggestive of the logged
  drift, not a gate. Expanded quality (GSM8K 200, 120-turn) was deliberately not
  bought: with no serving gain it cannot change the outcome. Reopen if a later
  item makes the freed 1.1 GB usable. See RESEARCH_LOG U9.

### U10 — Optional NVIDIA checkpoint comparison

- [x] Audited before downloading: revision `fc694b54fb0174e0913e6adf86691ef85a4ead47`
  (not gated), 11 files, 132.73 GB advertised / 124 GB on disk, 2.6 TB free after.
  MIXED_PRECISION export: NVFP4 g16 routed experts, **128x128 block-scaled FP8 MTP**,
  per-tensor FP8 PLE. #38121 support is already in the pinned image.
- [x] **Radix retained.** TAGs `u10-nvidia-nvfp4-20260908` (serving A/B vs
  `u8a-baseline-20260908`), `u10-nvidia-gsm8k-20260908` and
  `u10-radix-gsm8k-20260908` (paired screening). Separate `PLE_DIR` and identity:
  the NVIDIA table is the same 51200245760 bytes as Radix's, so a shared directory
  would have mixed checkpoints; its sampled digest is identical to Radix's, matching
  the card's claim that the PLE is copied byte-for-byte from Qwen's FP8 pack.
  Draft `unquant` was **not** retained (`SPEC_DRAFT_QUANT=auto`), `--quantization`
  became `modelopt_mixed`, and `--moe-runner-backend flashinfer_cutlass` had to be
  named explicitly because MIXED_PRECISION auto-resolves to trtllm, which the NVFP4
  MoE apply rejects at graph capture. Tokenizer/chat template are byte-identical, so
  the U6 draft map and all prompts transfer. Result: serving is a wash (32k TTFT
  10.090 vs 10.125 s, 128k 42.05 vs 42.20 s, mixed p95 26.176 vs 26.201 s, decode
  and streams within noise) and **GSM8K n=200 ties at 194/200 = 97.0%** with paired
  discordance 2 vs 2. The one real difference is the single-prompt probe (NVIDIA
  49/60 vs Radix 35/100), which does not generalise. No speed or fit benefit, so no
  switch. Published NVIDIA scores were not reproduced or reused.
  Not run: full 1319 GSM8K, 120-turn, BFCL, multilingual, paired vision, 190k/210k.
  See RESEARCH_LOG U10.

### U11 — Optional vLLM comparison and capacity experiments

- [ ] U11a: verify ARM64 support, GB10 PLE patches, MTP graphs and effective prefix
  caching. Use the same checkpoint and precision as SGLang where supported;
  otherwise identify the comparison as a whole-stack bundle. Compare our 120-turn
  and mixed-load workloads. Adapt the same detached harness and watchdog rather
  than allowing two GPU servers. Log engine decision and commit.
- [ ] U11b: test FP8 KV separately only with compatible QSA support. It is a
  capacity/quality trade; require expanded quality and demonstrated useful
  headroom. Keep BF16 default if inconclusive. Log decision and commit.
- [ ] U11c: optional 512k only after correctness/stability gates. Verify actual
  runtime mrope/YaRN configuration and KV plus output headroom. Stage 300k/400k/
  near-512k tests with multiple needle depths and reasoning. Boot success is
  insufficient. Never replace native 262k or attempt 1M. Log decision and commit.

### U12 — Final combined validation and documentation

- [ ] Re-run the accepted combination from its documented preparation path,
  verify two reusable boots, common gates, expanded quality where applicable,
  and a two-hour mixed-load soak. Check interactions between accepted changes.
- [ ] Publish only matched reproducible figures with context, precision, thinking
  and caching explicit. Reconcile stale README/EXPLORATION claims; list rejected
  and deferred items with evidence. Commit the final recipe.

If an item is blocked, log its exact blocker and commit before an independent
later item. Do not advance to dependent performance tuning while correctness or
host-stability gates remain unresolved. A docs-only deferred outcome is valid;
claiming an unexecuted experiment passed is not.

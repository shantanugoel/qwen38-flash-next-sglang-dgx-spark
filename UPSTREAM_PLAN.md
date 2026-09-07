# Upstream validation plan — September 2026

Status: **U0 and U1 accepted; U2–U12 not started**. This supersedes stale TODO/status and skip-list
entries in `EXPLORATION.md` for this campaign. August measurements remain
historical evidence. U1 was accepted on September 7 (Triton #36845 serving
default; KDA overlay rejected on this image). Later items remain unexecuted.

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

- [ ] Trace our exact flag/alias and code branches against open
  [#37794](https://github.com/sgl-project/sglang/pull/37794). This is a reported
  bug, not a confirmed diagnosis of our installed version.
- [ ] If affected, isolate PLE history/short-convolution state commits after verify;
  do not introduce the PR's NGRAM feature. Test accepted/rejected draft state
  transitions, then fast, expanded-quality and promotion gates. Compare against
  non-speculative behavior without requiring identical sampled text.
- [ ] If unaffected, record exact code evidence. Commit outcome before U3.

### U3 — Pinned newer SGLang model-development image

- [ ] Recheck registry/source metadata at execution time. Audit on September 7:
  `qwen38flashnext` manifest `5ae5816783d58e2e56e84d2e863f5441425056f500b7fbd7448c4aae017a2521`
  was published September 3; `dev-qwen38-next-local` manifest
  `7b300eccf8ecdd79f27b92a7bff62b415be47790029b0df8df72806977e0af63`
  was updated September 6. Both have ARM64 images. Use full `sha256:` digest pins.
  The pending cookbook associates the latter with branch revision `9b2aee2283`.
  Verify contents: general v0.5.19 is not assumed to contain the model while
  [#36497](https://github.com/sgl-project/sglang/pull/36497) remains open.
- [ ] Hold Radix weights and logical serving settings fixed. Reconcile U1/U2 with
  bundled code, preserve PLE reuse, check effective cache strategy and graph replay.
- [ ] Run common gates and a one-hour mixed-load soak. Accept for demonstrated
  correctness/support/maintenance benefit with understood performance changes;
  otherwise retain the corrected previous baseline. Document and commit.

### U4 — Native PLE backend, prefetch and resident-memory control

- [ ] U4a: adopt native allocation from
  [#37068](https://github.com/sgl-project/sglang/pull/37068) with
  [#38123](https://github.com/sgl-project/sglang/pull/38123), initially disabling
  prefetch/trimming where supported to isolate allocation. Its loader rewrites
  the full table each boot: preserve validated reuse rather than blindly dropping
  our patch. Reconcile file names and checkpoint identity without writes under
  `~/ai`; use a new workspace backing file if needed. Check shard samples and
  two boots with explicit reuse logs and timing. Log decision and commit.
- [ ] U4b: enable prefill page prefetch alone; measure cold/warm prefill, I/O,
  synchronization cost and decode under prefill. Log decision and commit.
- [ ] U4c: evaluate the RSS trimmer separately with a one-hour growing-context
  soak; measure available memory, mapping RSS, stalls and data correctness.
  Retain only a demonstrated stability/memory benefit without material latency
  or warm-boot regression. Log decision and commit before U5.

### U5 — Mamba tracking interval and usable KV capacity

- [ ] U5a: compare track interval 64 versus 256 at identical MAX_TOTAL,
  MAX_RUNNING and memory fraction. Validate page/draft alignment, actual Mamba
  slot allocation, 120-turn and shared-prefix TTFT, and long-context quality.
  More theoretical KV is not a benefit when MAX_TOTAL already binds. Commit.
- [ ] U5b, only if warranted: change MAX_TOTAL separately with memory gates and
  realistic long concurrent requests. Keep native context unchanged. Commit.
  Source lead: [Spark cookbook](https://github.com/sgl-project/sglang/pull/37995).

### U6 — Reduced-vocabulary MTP drafting

- [ ] Audit NEXTN verification and token-ID mapping before porting the concept
  from [MiaAI](https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark).
  Keep target vocabulary/sampling unchanged; excluded draft tokens must remain
  possible through target verification.
- [ ] Build the subset from a separate code/multilingual/tool corpus and evaluate
  held-out prompts. Begin at 64k rows; each further size is a separate experiment.
  Measure traffic, extra resident slice memory, acceptance, steady decode,
  end-to-end latency and expanded quality. Do not assume bandwidth savings are
  resident-memory savings. Revert if language/tool losses or speed fail the gate.
- [ ] Log and commit every tested variant before moving on.

### U7 — Graph coverage and mixed-load chunk sizing

- [ ] U7a: audit actual reachable draft/verify sizes and graph replay. Trim excess
  captures without disabling padding or losing coverage. Measure boot/memory
  and 1/2/4-stream performance. Log decision and commit.
- [ ] U7b: hold graphs fixed; compare prefill 4096 to 2048/1024, optionally 8192,
  one candidate and commit at a time. Measure cold TTFT, aggregate throughput,
  and p50/p95/p99 streamed-chunk gaps when a 64k prefill arrives during two
  decodes. Report tokens per chunk: chunk gaps are not token gaps. A specialized
  mixed-load profile may remain opt-in; do not silently impose its cold-TTFT
  tradeoff on the general default. Log each decision and commit.

### U8 — QSA prefill and relevant GEMM optimization

- [ ] U8a: recheck [#38209](https://github.com/sgl-project/sglang/pull/38209) and
  underlying indexer correctness reviews. Isolate it and test 8k/32k/128k,
  cached suffixes and concurrent requests. GB200 results are not GB10 forecasts.
  Log accept/reject/defer and commit.
- [ ] U8b: inspect whether [b12x #38170](https://github.com/sgl-project/sglang/pull/38170)
  reaches hot quantized linear layers in our checkpoint; BF16 lm_head and routed
  MoE may not use this path. If relevant, check CUDA/FlashInfer compatibility,
  tensor parity and end-to-end speed. Otherwise defer with evidence. Do not
  independently upgrade Torch/FlashInfer merely because a release exists;
  evaluate a compatible pinned engine bundle. Log decision and commit.

### U9 — Optional BF16 recurrent state

- [ ] Confirm support in the corrected kernel/cache strategy. Hold BF16 KV fixed;
  compare FP32/BF16 state at 1/2/4 streams and 120 turns. Existing SGLang-family
  no-op evidence lowers priority; vLLM gains are not proof here.
- [ ] Require expanded quality plus a measurable serving/memory gain. Keep FP32
  if inconclusive or regressed. Log decision and commit.

### U10 — Optional NVIDIA checkpoint comparison

- [ ] Audit the [model card](https://huggingface.co/nvidia/Qwen3.8-Flash-Next-NVFP4),
  revision, expected disk footprint and free capacity before a detached download.
  September 7 revision: `fc694b54fb0174e0913e6adf86691ef85a4ead47`; support merged
  in [#38121](https://github.com/sgl-project/sglang/pull/38121).
- [ ] Use separate checkpoint/PLE identity and correctly load its FP8 MTP experts;
  do not blindly retain Radix's draft `unquant` setting. Hold engine, context,
  KV/state precision, prompts and sampling fixed. Run expanded quality, promotion,
  memory and boot checks. Retain Radix unless paired evidence supports quality
  and an actual speed/fit benefit warrants switching. Published NVIDIA scores
  never become local results. Log decision and commit.

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

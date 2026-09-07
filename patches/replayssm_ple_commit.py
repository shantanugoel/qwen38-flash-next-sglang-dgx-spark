#!/usr/bin/env python3
"""Commit PLE n-gram/short-conv state after ReplaySSM verify (sglang#37794).

`--enable-gdn-replayssm-spec` is a deprecated alias of
`--enable-linear-replayssm-spec`. On this image that sets
`mamba_pool.replayssm_spec_fold=True`, so `commit_mamba_states_after_verify`
takes the GDN fold branch and returns before the generic path that rolls
PLE n-gram history and PLE short-conv state
(`HybridLinearAttnBackend._update_ple_state_after_mtp_verify`).

During TARGET_VERIFY, `qwen4_exp._commit_ple_batch` only writes
`ngram_pool.intermediate_context`. The main `ngram_pool.context` is supposed
to take the last accepted draft step after verify. Without that scatter the
PLE history freezes at the prefill suffix for the rest of a speculative
decode. This also affects MTP top-k=1, not only NGRAM.

This patcher isolates that commit in both ReplaySSM early-return branches
(fold: our serving path; ring: the other #37794 branch). It does **not**
port the PR's NGRAM worker, linearize-chain, or `_prepare_ple_batch` NGRAM
guard removal.

Usage: python3 replayssm_ple_commit.py <path to spec_utils.py>
"""
from __future__ import annotations

import sys

FOLD_ANCHOR = """        commit_gdn_replayssm_fold_after_verify(
            spec_state=spec_state,
            state_batch_indices=state_batch_indices,
            accept_lens=accept_lens,
            last_correct_step_indices=last_correct_step_indices,
            mamba_track_indices=batch.mamba_track_indices,
            mamba_steps_to_track=mamba_steps_to_track,
            null_block_id=-1,
        )
        return
"""

FOLD_HUNK = """        commit_gdn_replayssm_fold_after_verify(
            spec_state=spec_state,
            state_batch_indices=state_batch_indices,
            accept_lens=accept_lens,
            last_correct_step_indices=last_correct_step_indices,
            mamba_track_indices=batch.mamba_track_indices,
            mamba_steps_to_track=mamba_steps_to_track,
            null_block_id=-1,
        )
        # PLE n-gram history + PLE short-conv state live outside the GDN fold;
        # roll them to the last accepted node like the generic path does
        # (hybrid_linear_attn_backend.py _update_ple_state_after_mtp_verify).
        # sglang#37794: without this call the early return skips the PLE roll
        # and the n-gram history freezes after the first verify step.
        attn_backend = model_runner.attn_backend
        if hasattr(attn_backend, "_update_ple_state_after_mtp_verify"):
            _ple_track = batch.mamba_track_indices
            if _ple_track is not None:
                _ple_track = req_pool.translate_mamba_indices(_ple_track)
            attn_backend._update_ple_state_after_mtp_verify(
                req_pool.translate_mamba_indices(state_batch_indices),
                last_correct_step_indices,
                _ple_track,
                mamba_steps_to_track,
            )
            if not getattr(logger, "_ple_replayssm_commit_logged", False):
                logger._ple_replayssm_commit_logged = True
                logger.info("ReplaySSM verify: committing PLE n-gram/short-conv state")
        return
"""

RING_ANCHOR = """        # snapshot; not wired for Part B (server_args forbids extra_buffer with
        # --enable-linear-replayssm-spec), so the per-track scatters are intentionally
        # skipped here.
        return
"""

RING_HUNK = """        # snapshot; not wired for Part B (server_args forbids extra_buffer with
        # --enable-linear-replayssm-spec), so the per-track scatters are intentionally
        # skipped here.
        # PLE side states: the ring branch returns early as well, so the PLE
        # roll of the generic path has to happen here too. Track scatter is
        # skipped like the conv one above.
        attn_backend = model_runner.attn_backend
        if hasattr(attn_backend, "_update_ple_state_after_mtp_verify"):
            attn_backend._update_ple_state_after_mtp_verify(
                req_pool.translate_mamba_indices(state_batch_indices),
                last_correct_step_indices,
                None,
                None,
            )
            if not getattr(logger, "_ple_replayssm_commit_logged", False):
                logger._ple_replayssm_commit_logged = True
                logger.info("ReplaySSM verify: committing PLE n-gram/short-conv state")
        return
"""


def main(path: str) -> int:
    with open(path, encoding="utf-8") as f:
        src = f.read()
    if "ReplaySSM verify: committing PLE n-gram/short-conv state" in src:
        print("ALREADY PATCHED:", path)
        return 0
    if "NgramCorpus" in src or "_linearize_chain" in src:
        print("ERROR: NGRAM worker leaked into spec_utils.py; refusing")
        return 1
    if src.count(FOLD_ANCHOR) != 1:
        print("ERROR: GDN fold early-return anchor not unique or missing")
        return 1
    if src.count(RING_ANCHOR) != 1:
        print("ERROR: GDN ring early-return anchor not unique or missing")
        return 1
    src = src.replace(FOLD_ANCHOR, FOLD_HUNK, 1).replace(RING_ANCHOR, RING_HUNK, 1)
    with open(path, "w", encoding="utf-8") as f:
        f.write(src)
    print("PATCHED:", path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1]))

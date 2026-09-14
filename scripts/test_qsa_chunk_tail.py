#!/usr/bin/env python3
"""A1 (sglang#38346): stock vs clamped QSA extend compress on short forwards.

Runs inside the pinned image on CPU, where an out-of-range gather raises instead
of silently reading a neighbour (the fused Triton path on GPU never checks).
Uses the image's real `_qsa_write_plan` and the real
`update_key_state_and_compress` from both the stock and the patched indexer.

    ./scripts/test_qsa_chunk_tail.sh
"""
from __future__ import annotations

import importlib.util
import sys
from types import SimpleNamespace

import torch

from sglang.srt.layers.attention.qsa import qsa_indexer as stock
from sglang.srt.layers.attention.qwen_sparse_attn_backend import QwenSparseAttnBackend

RATIO = 4
HEADS, DIM = 2, 8


def load_patched(path):
    spec = importlib.util.spec_from_file_location("patched_qsa_indexer", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def plan(prefix: int, extend: int):
    """The extend write plan `_qsa_build_write_plan` builds for one request."""
    lengths = torch.tensor([prefix + extend])
    extend_lens = torch.tensor([extend])
    width = ((prefix + extend) // RATIO + 1) * RATIO
    # Raw KV slots start past page 0, like the real allocator.
    token_slot_table = (torch.arange(width) + 64).unsqueeze(0)
    prefix_lens = lengths - extend_lens
    return QwenSparseAttnBackend._qsa_write_plan(
        token_slot_table=token_slot_table,
        start_blocks=prefix_lens // RATIO,
        end_blocks=lengths // RATIO,
        capacity=extend // RATIO + 1,
        compress_ratio=RATIO,
        row_token_starts=torch.cumsum(extend_lens, 0) - extend_lens,
        prefix_lens=prefix_lens,
    ), prefix + torch.arange(extend)


def compress(module, prefix: int, extend: int):
    (write_locs, group_end, rows, member_rows), positions = plan(prefix, extend)
    torch.manual_seed(prefix * 7 + extend)
    token_k = torch.randn(extend, HEADS, DIM)
    writes = {}
    pool = SimpleNamespace(
        set_qsa_compressed_k_buffer=lambda layer, locs, value: writes.update(
            locs=locs.clone(), value=value.clone()))
    indexer = SimpleNamespace(
        layer_id=0, compress_ratio=RATIO,
        _use_fused_compress=lambda _pool: False,
        normalize_compressed_keys=lambda keys, _pos: keys,
        _rope_from_matrix=lambda matrix: matrix[:, 0])
    metadata = SimpleNamespace(
        token_to_kv_pool=pool, compress_member_rows=member_rows, write_locs=write_locs,
        compress_group_positions=group_end, compress_sequence_ids=rows,
        is_cuda_graph=False, extend_rope_matrix=None)
    module.QSAIndexer.update_key_state_and_compress(
        indexer, token_k, positions, positions, metadata, state_stored=True)
    return writes


def main(patched_path: str) -> int:
    patched = load_patched(patched_path)
    failures = 0
    for prefix in (0, 4096, 8192):
        for extend in (1, 2, 3):
            try:
                compress(stock, prefix, extend)
                stock_result = "no error"
            except (IndexError, RuntimeError) as exc:
                stock_result = type(exc).__name__
            writes = compress(patched, prefix, extend)
            only_slot0 = bool((writes["locs"] == 0).all())
            ok = stock_result != "no error" and only_slot0
            failures += not ok
            print(f"short  prefix={prefix:5d} extend={extend}: stock={stock_result:10s} "
                  f"patched=ok writes_slot0_only={only_slot0} {'PASS' if ok else 'FAIL'}")
    # Real groups must be bit-identical with and without the clamp.
    for prefix, extend in ((0, 4), (0, 4096), (4096, 4096), (8192, 5), (0, 4099), (64, 2047)):
        a, b = compress(stock, prefix, extend), compress(patched, prefix, extend)
        same = torch.equal(a["locs"], b["locs"]) and torch.equal(a["value"], b["value"])
        failures += not same
        print(f"full   prefix={prefix:5d} extend={extend:4d}: groups={a['locs'].numel()} "
              f"identical={same} {'PASS' if same else 'FAIL'}")
    print("RESULT", "PASS" if failures == 0 else f"FAIL ({failures})")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))

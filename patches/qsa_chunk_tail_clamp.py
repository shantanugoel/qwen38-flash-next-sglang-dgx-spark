#!/usr/bin/env python3
"""Clamp the QSA extend compress gather to this forward's rows (sglang#38346).

`_qsa_write_plan` pads its fixed-capacity write plan with row 0 / block 0
entries that write the reserved slot 0. On an extend forward each entry's
members are `member_rows[:, None] + arange(compress_ratio)`, so a padded entry
gathers token rows 0..3. A forward with fewer than `compress_ratio` (4) token
rows -- the 1..3 token piece that ends a request under chunked prefill, or a
1..3 token prompt -- then reads past `token_k`. The fused Triton compress path
does not bounds-check, so the read is usually absorbed silently and faults only
when it crosses an unmapped page (illegal memory access, sglang#37633).

Real entries are always in range and padded entries' gathered values are
discarded, so the clamp cannot change any real computation. Merged upstream
2026-09-12 as a one-line change; our pin `4ccff141` predates it.

Usage: python3 qsa_chunk_tail_clamp.py <path to qsa/qsa_indexer.py>
"""
from __future__ import annotations

import sys

OLD = """            source_keys = token_k
            source_rope = metadata.extend_rope_matrix
"""

NEW = """            source_keys = token_k
            group_locs = group_locs.clamp_max(source_keys.shape[0] - 1)
            source_rope = metadata.extend_rope_matrix
"""


def main(path: str) -> int:
    with open(path, encoding="utf-8") as f:
        src = f.read()

    if src.count(NEW) == 1:
        print("ALREADY PATCHED (#38346 clamp present):", path)
        return 0
    n = src.count(OLD)
    if n != 1:
        print("ERROR: expected 1 extend compress gather, found %d" % n)
        return 1

    with open(path, "w", encoding="utf-8") as f:
        f.write(src.replace(OLD, NEW, 1))
    print("PATCHED:", path)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))

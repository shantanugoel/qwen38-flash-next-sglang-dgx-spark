#!/usr/bin/env python3
"""DEPRECATED. Do not apply on GB10 / SM121.

Widens `_resolve_trtllm_sparse_decode` to `is_sm120_supported()`. On SM121 that
call does not run trtllm-gen (no sm12x cubins); with backend=auto it routes to
XQA, which silently corrupts long-context decode (runs of token id 0 from
~120k). Retired by sglang#36806 / #36845. This recipe now uses qsa_sm121_triton.py.

Adapted from hashd1ve/qwen38-flash-next-one-dgx-spark (MIT) @ 04d0735.

Usage: python3 qsa_trtllm_sm120.py <path to qwen_sparse_attn_backend.py>
"""
from __future__ import annotations

import sys

OLD = """    from sglang.srt.utils import is_sm100_supported

    if not is_sm100_supported():
        return None"""

NEW = """    from sglang.srt.utils import is_sm100_supported, is_sm120_supported

    # sm_120/121 (consumer Blackwell: GB10 / RTX 50-series) is Blackwell too and
    # flashinfer ships the trtllm-gen decode kernel for it. The original gate
    # excludes it, which forces the FA4-cute varlen fallback -- and that one
    # fails to compile on SM120.
    if not (is_sm100_supported() or is_sm120_supported()):
        return None"""


def main(path: str) -> int:
    with open(path, encoding="utf-8") as f:
        src = f.read()

    if "is_sm120_supported()" in src and "consumer Blackwell" in src:
        print("ALREADY PATCHED:", path)
        return 0

    n = src.count(OLD)
    if n != 1:
        print("ERROR: expected 1 occurrence of the gate, found %d" % n)
        return 1

    with open(path, "w", encoding="utf-8") as f:
        f.write(src.replace(OLD, NEW, 1))
    print("PATCHED:", path)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))

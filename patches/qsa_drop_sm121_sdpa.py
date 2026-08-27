#!/usr/bin/env python3
"""Remove an SM121 SDPA intercept so sparse decode can reach TRT-LLM.

Some Spark images return `_forward_sm121_sdpa_sparse` on `is_sm121()` before
`_resolve_trtllm_sparse_decode()`. Stock `lmsysorg/sglang:qwen38flashnext` has
no intercept; this patch is then a no-op.

Usage: python3 qsa_drop_sm121_sdpa.py <path to qwen_sparse_attn_backend.py>
"""
from __future__ import annotations

import sys

OLD = """        metadata = self._resolve_metadata(forward_batch)
        topk_indices = topk_indices.to(torch.int32).contiguous()
        if is_sm121():
            return self._forward_sm121_sdpa_sparse(
                q, k_buffer, v_buffer, layer, metadata, topk_indices
            )
        trtllm_decode = _resolve_trtllm_sparse_decode()
"""

NEW = """        metadata = self._resolve_metadata(forward_batch)
        topk_indices = topk_indices.to(torch.int32).contiguous()
        trtllm_decode = _resolve_trtllm_sparse_decode()
"""


def main(path: str) -> int:
    with open(path, encoding="utf-8") as f:
        src = f.read()

    if "if is_sm121():" not in src:
        print("ALREADY PATCHED (no is_sm121 intercept):", path)
        return 0

    n = src.count(OLD)
    if n != 1:
        print("ERROR: expected 1 SDPA intercept, found %d" % n)
        return 1

    with open(path, "w", encoding="utf-8") as f:
        f.write(src.replace(OLD, NEW, 1))
    print("PATCHED:", path)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))

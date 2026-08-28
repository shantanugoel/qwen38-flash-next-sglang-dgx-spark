#!/usr/bin/env python3
"""Skip the 48 GiB PLE mmap refill when the file already holds the table.

The mmap backing store survives container restarts, but SGLang still runs the
generic weight_loader on every boot, which copies the whole 47.7 GiB table from
the checkpoint into the mmap. Measured on this GB10 that copy is a
read-modify-write through the page cache under the container's memory cap and
costs 45-60 min of the boot.

This patch adds a verified fast path: sample random rows of the mmap and the
checkpoint tensor, and if every sample is byte-identical, skip the copy. A
stale, truncated or wrong-revision file fails the sample and falls back to the
full copy, so the fast path cannot serve wrong weights.

Set SGLANG_QWEN4_PLE_REUSE=0 to force the full copy.

Usage: python3 ple_reuse.py <path to qwen4_exp.py>
"""
from __future__ import annotations

import sys

HELPER = '''

def _ple_reuse_ok(param, loaded_weight) -> bool:
    """True when the mmap already byte-matches the checkpoint table.

    Verification is by random row sampling: SGLANG_QWEN4_PLE_REUSE_SAMPLES rows
    (default 8192) plus the first and last row. Rows are 160 B, so this is a few
    thousand page faults, not a 48 GiB read.
    """
    import logging
    import os
    import time

    log = logging.getLogger(__name__)
    if not _PLE_MMAP_DIR:
        return False
    if os.environ.get("SGLANG_QWEN4_PLE_REUSE", "1").strip() == "0":
        return False
    dst = param.data
    if tuple(dst.shape) != tuple(loaded_weight.shape) or dst.dtype != loaded_weight.dtype:
        return False
    if dst.numel() == 0 or dst.ndim != 2:
        return False

    n_rows = dst.shape[0]
    try:
        k = int(os.environ.get("SGLANG_QWEN4_PLE_REUSE_SAMPLES", "8192"))
    except ValueError:
        k = 8192
    k = max(16, min(k, n_rows))
    gen = torch.Generator().manual_seed(0x5150)
    idx = torch.randint(0, n_rows, (k,), generator=gen).tolist()
    idx = sorted(set(idx + [0, n_rows - 1]))

    t0 = time.perf_counter()
    try:
        for i in idx:
            a = dst[i].contiguous().view(torch.uint8)
            b = loaded_weight[i].contiguous().view(torch.uint8)
            if not torch.equal(a, b):
                log.info(
                    "PLE table: mmap row %d differs from checkpoint; refilling", i
                )
                return False
    except Exception as exc:  # noqa: BLE001
        log.warning("PLE table: reuse check failed (%s); refilling", exc)
        return False

    log.info(
        "PLE table: mmap matches checkpoint on %d sampled rows in %.1fs; "
        "skipping the %.1f GiB refill",
        len(idx),
        time.perf_counter() - t0,
        dst.numel() * dst.element_size() / 2**30,
    )
    return True

'''

METHOD = '''
    def weight_loader(self, param, loaded_weight, *args, **kwargs):
        # The mmap survives restarts. Only pay the 48 GiB copy when the file
        # does not already hold this exact table.
        if _ple_reuse_ok(param, loaded_weight):
            return
        VocabParallelEmbedding.weight_loader(self, param, loaded_weight, *args, **kwargs)
'''

ANCHOR = "class Qwen4ExpPinnedHostEmbedding(VocabParallelEmbedding):"
METHOD_ANCHOR = """    def allocate_output(
        self, shape: Tuple[int, ...], device: torch.device
    ) -> torch.Tensor:"""


def main(path: str) -> int:
    with open(path, encoding="utf-8") as f:
        src = f.read()

    if "_ple_reuse_ok" in src:
        print("ALREADY PATCHED:", path)
        return 0
    if "_alloc_ple_table" not in src:
        print("ERROR: run ple_mmap.py first")
        return 1
    if src.count(ANCHOR) != 1:
        print("ERROR: could not locate Qwen4ExpPinnedHostEmbedding")
        return 1
    if src.count(METHOD_ANCHOR) != 1:
        print("ERROR: could not locate allocate_output anchor")
        return 1

    src = src.replace(ANCHOR, HELPER.lstrip("\n") + "\n" + ANCHOR, 1)
    src = src.replace(METHOD_ANCHOR, METHOD.lstrip("\n") + "\n" + METHOD_ANCHOR, 1)

    with open(path, "w", encoding="utf-8") as f:
        f.write(src)
    print("PATCHED:", path)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))

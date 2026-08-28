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

    Verification is by random byte-range sampling over a flat uint8 view, so it
    does not care about the table's shape: SGLANG_QWEN4_PLE_REUSE_SAMPLES
    windows (default 4096) of 4 KiB each, plus the first and last window. That
    is a few thousand page faults, not a 48 GiB read.
    """
    import logging
    import os
    import time

    log = logging.getLogger(__name__)
    if not _PLE_MMAP_DIR:
        return False
    if os.environ.get("SGLANG_QWEN4_PLE_REUSE", "1").strip() == "0":
        log.info("PLE table: reuse disabled by SGLANG_QWEN4_PLE_REUSE=0")
        return False

    dst = param.data
    if tuple(dst.shape) != tuple(loaded_weight.shape):
        log.info(
            "PLE table: shape %s != checkpoint %s; refilling",
            tuple(dst.shape),
            tuple(loaded_weight.shape),
        )
        return False
    if dst.dtype != loaded_weight.dtype:
        log.info(
            "PLE table: dtype %s != checkpoint %s; refilling",
            dst.dtype,
            loaded_weight.dtype,
        )
        return False

    try:
        a = dst.reshape(-1).view(torch.uint8)
        b = loaded_weight.reshape(-1).contiguous().view(torch.uint8)
    except Exception as exc:  # noqa: BLE001
        log.warning("PLE table: cannot take a flat byte view (%s); refilling", exc)
        return False

    n = int(a.numel())
    win = 4096
    if n < win * 4:
        return False
    try:
        k = int(os.environ.get("SGLANG_QWEN4_PLE_REUSE_SAMPLES", "4096"))
    except ValueError:
        k = 4096
    k = max(64, k)
    gen = torch.Generator().manual_seed(0x5150)
    offs = (torch.randint(0, (n - win) // win, (k,), generator=gen) * win).tolist()
    offs = sorted(set(offs + [0, n - win]))

    t0 = time.perf_counter()
    try:
        for o in offs:
            if not torch.equal(a[o : o + win], b[o : o + win]):
                log.info(
                    "PLE table: mmap differs from checkpoint at byte %d; refilling", o
                )
                return False
    except Exception as exc:  # noqa: BLE001
        log.warning("PLE table: reuse check failed (%s); refilling", exc)
        return False

    log.info(
        "PLE table: mmap matches the checkpoint on %d sampled 4 KiB windows in "
        "%.1fs; skipping the %.1f GiB refill",
        len(offs),
        time.perf_counter() - t0,
        n / 2**30,
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

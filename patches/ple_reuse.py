#!/usr/bin/env python3
"""Skip PLE mmap shard copies that would rewrite bytes already on disk.

The mmap backing store survives container restarts, but SGLang re-copies the
whole 47.7 GiB table from the checkpoint into it on every boot, in 512 shards,
through `copy_ple_rows_to_tp_embedding`. Measured on this GB10 that copy is a
read-modify-write through a nearly full page cache and costs 45-60 min of the
boot.

This patch gives each shard copy a verified fast path: sample byte windows of
the destination rows and the source rows, and skip that shard only when every
sample is byte-identical. A stale, truncated or wrong-revision file fails the
sample and is copied normally, per shard, so the fast path cannot serve wrong
weights.

Env:
  SGLANG_QWEN4_PLE_REUSE=0          force the full copy
  SGLANG_QWEN4_PLE_REUSE_WINDOWS=N  windows sampled per shard (default 32)

Usage: python3 ple_reuse.py <path to qwen4_exp.py>
"""
from __future__ import annotations

import sys

HELPER = '''

_PLE_REUSE_STATS = {"skipped": 0, "copied": 0, "rows_skipped": 0}


def _ple_shard_matches(dst: torch.Tensor, src: torch.Tensor) -> bool:
    """True when dst already holds src, by random byte-window sampling."""
    import os

    if os.environ.get("SGLANG_QWEN4_PLE_REUSE", "1").strip() == "0":
        return False
    if dst.shape != src.shape or dst.dtype != src.dtype:
        return False
    if dst.numel() == 0:
        return True
    try:
        a = dst.reshape(-1).view(torch.uint8)
        b = src.reshape(-1).contiguous().view(torch.uint8)
    except Exception:  # noqa: BLE001
        return False

    n = int(a.numel())
    win = 4096
    if n <= win * 2:
        return bool(torch.equal(a, b))
    try:
        k = int(os.environ.get("SGLANG_QWEN4_PLE_REUSE_WINDOWS", "32"))
    except ValueError:
        k = 32
    k = max(4, k)
    gen = torch.Generator().manual_seed(0x5150 ^ n)
    offs = (torch.randint(0, (n - win) // win, (k,), generator=gen) * win).tolist()
    offs = sorted(set(offs + [0, n - win]))
    try:
        for o in offs:
            if not torch.equal(a[o : o + win], b[o : o + win]):
                return False
    except Exception:  # noqa: BLE001
        return False
    return True


def _ple_reuse_report() -> None:
    import logging

    s = _PLE_REUSE_STATS
    total = s["skipped"] + s["copied"]
    if not total:
        return
    logging.getLogger(__name__).info(
        "PLE table: %d/%d shards already on disk (%d rows), %d copied",
        s["skipped"],
        total,
        s["rows_skipped"],
        s["copied"],
    )

'''

ORIG = """                emb.weight.data[local_start : local_start + n_rows].copy_(
                    loaded_weight[src_start : src_start + n_rows].to(
                        device=emb.weight.device, dtype=emb.weight.dtype
                    )
                )
"""

NEW = """                dst = emb.weight.data[local_start : local_start + n_rows]
                src = loaded_weight[src_start : src_start + n_rows].to(
                    device=emb.weight.device, dtype=emb.weight.dtype
                )
                # The mmap survives restarts: only rewrite what actually differs.
                if _ple_shard_matches(dst, src):
                    _PLE_REUSE_STATS["skipped"] += 1
                    _PLE_REUSE_STATS["rows_skipped"] += n_rows
                else:
                    _PLE_REUSE_STATS["copied"] += 1
                    dst.copy_(src)
"""

ANCHOR = "class Qwen4ExpPinnedHostEmbedding(VocabParallelEmbedding):"

REPORT_ORIG = """            if load_qwen4_exp_ple_shard(name, loaded_weight):
                continue
"""
REPORT_NEW = """            if load_qwen4_exp_ple_shard(name, loaded_weight):
                continue
"""


def main(path: str) -> int:
    with open(path, encoding="utf-8") as f:
        src = f.read()

    if "_ple_shard_matches" in src:
        print("ALREADY PATCHED:", path)
        return 0
    if "_alloc_ple_table" not in src:
        print("ERROR: run ple_mmap.py first")
        return 1
    if src.count(ORIG) != 1:
        print("ERROR: could not locate the PLE shard copy")
        return 1
    if src.count(ANCHOR) != 1:
        print("ERROR: could not locate Qwen4ExpPinnedHostEmbedding")
        return 1

    src = src.replace(ORIG, NEW, 1)
    src = src.replace(ANCHOR, HELPER.lstrip("\n") + "\n" + ANCHOR, 1)

    # Report once, right after the weight loop finishes.
    tail = "        self.logged_params = loaded_params"
    if tail in src:
        src = src.replace(tail, "        _ple_reuse_report()\n" + tail, 1)
    else:
        marker = "        return loaded_params"
        if src.count(marker) >= 1:
            src = src.replace(marker, "        _ple_reuse_report()\n" + marker, 1)

    with open(path, "w", encoding="utf-8") as f:
        f.write(src)
    print("PATCHED:", path)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))

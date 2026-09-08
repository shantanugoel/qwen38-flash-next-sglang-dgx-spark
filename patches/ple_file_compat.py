#!/usr/bin/env python3
"""Reuse the recipe's existing PLE mmap filename with the native file backend.

#37068 names the sparse table
`ple_table_{dims}_{dtype}_{nbytes}B_{tag}.bin` and still rewrites it on every
boot. This recipe's identity checker and backing file are
`ple_table_{numel}_{nbytes}.bin`. Prefer that file when it is already present
and the right size, so U3 does not fill a second 48 GiB table.

Usage: python3 ple_file_compat.py <path to qwen4_exp_ple_table.py>
"""
from __future__ import annotations

import sys

ORIG = """    path = os.path.join(table_dir, ple_table_file_name(shape, dtype, tag))
    if not os.path.exists(path) or os.path.getsize(path) != nbytes:
        # Sparse: only pages that get written take disk space.
        with open(path, "wb") as f:
            f.truncate(nbytes)
"""

NEW = """    native = os.path.join(table_dir, ple_table_file_name(shape, dtype, tag))
    compat = os.path.join(table_dir, "ple_table_%d_%d.bin" % (numel, nbytes))
    if os.path.isfile(compat) and os.path.getsize(compat) == nbytes:
        path = compat
        logger.info("PLE table: reusing recipe backing file %s", path)
    else:
        path = native
        if not os.path.exists(path) or os.path.getsize(path) != nbytes:
            # Sparse: only pages that get written take disk space.
            with open(path, "wb") as f:
                f.truncate(nbytes)
"""


def main(path: str) -> int:
    with open(path, encoding="utf-8") as f:
        src = f.read()
    if "reusing recipe backing file" in src:
        print("ALREADY PATCHED:", path)
        return 0
    if "def allocate_ple_host_table" not in src:
        print("ERROR: native PLE file backend missing; not a U3 table module")
        return 1
    if src.count(ORIG) != 1:
        print("ERROR: expected 1 native PLE file-create block")
        return 1
    with open(path, "w", encoding="utf-8") as f:
        f.write(src.replace(ORIG, NEW, 1))
    print("PATCHED:", path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1]))

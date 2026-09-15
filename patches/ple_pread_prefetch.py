#!/usr/bin/env python3
"""C1: synchronous multi-threaded pread prefetch for cold PLE prefill gathers.

REJECTED 2026-09-16, not applied by prepare.sh: on identical real-text
prompts it did not beat the stock WILLNEED prefetch (128k 158/211/202 and
169/362/240 s vs 151/270/280 and 144/195/289 s; 32k slightly slower).

The native file-backed PLE table is read by the Triton gather kernel straight
through a pageable mmap. On a cold table every row of a prefill chunk (~16
rows per token, ~65k per 4096-token chunk, nearly one 4 KiB page each) is a
major page fault served one at a time (MADV_RANDOM, no readahead): measured
~4k faults/s, i.e. ~16 MiB/s of reads on NVMe that does 8.7 GB/s.

The upstream prefetcher (`SGLANG_QWEN4_PLE_FILE_PREFETCH=1`) sends
`posix_fadvise(WILLNEED)` per page from ONE background thread, racing the
kernel. This adds a mode that reads the chunk's distinct pages with `pread`
on a thread pool and waits for them before the gather runs, so the kernel
only takes minor faults on pages already in the page cache (tonyd2wild's
staged-read observation: preadv reaches ~131k rows/s where faults stay at
10-20k rows/s regardless of threads).

Env (only with SGLANG_QWEN4_PLE_FILE_PREFETCH=1):
  SGLANG_QWEN4_PLE_FILE_PREFETCH_MODE=willneed|pread  (default willneed = stock)
  SGLANG_QWEN4_PLE_FILE_PREFETCH_THREADS=N            (pread workers, default 16)

Usage: python3 ple_pread_prefetch.py <path to qwen4_exp_ple_table.py>
"""
from __future__ import annotations

import sys

INIT_OLD = """        self._fd = os.open(path, os.O_RDONLY)
        self._row_bytes = int(row_bytes)
        self._min_rows = int(min_rows)
        self._pool = ThreadPoolExecutor(max_workers=1)
"""

INIT_NEW = """        self._fd = os.open(path, os.O_RDONLY)
        self._row_bytes = int(row_bytes)
        self._min_rows = int(min_rows)
        self._mode = os.environ.get("SGLANG_QWEN4_PLE_FILE_PREFETCH_MODE", "willneed")
        self._threads = max(
            1, int(os.environ.get("SGLANG_QWEN4_PLE_FILE_PREFETCH_THREADS", "16"))
        )
        self._pool = ThreadPoolExecutor(
            max_workers=self._threads if self._mode == "pread" else 1
        )
        if self._mode == "pread":
            logger.info(
                "PLE table: synchronous pread prefetch on %d threads", self._threads
            )
"""

ADVISE_OLD = """    def enqueue(
        self,"""

ADVISE_NEW = """    def _read_pages(self, pages: list[int]) -> None:
        page = 1 << _PAGE_SHIFT
        for p in pages:
            try:
                os.pread(self._fd, page, p << _PAGE_SHIFT)
            except OSError:
                return

    def enqueue(
        self,"""

SUBMIT_OLD = """        pages = self.pages_for_rows(row_ids, self._row_bytes)
        self._pool.submit(self._advise, pages)
        return True
"""

SUBMIT_NEW = """        pages = self.pages_for_rows(row_ids, self._row_bytes)
        if self._mode == "pread":
            # Populate the page cache in parallel and wait, so the gather kernel
            # never takes a major fault on these rows.
            step = (len(pages) + self._threads - 1) // self._threads
            futures = [
                self._pool.submit(self._read_pages, pages[i : i + step])
                for i in range(0, len(pages), step)
            ]
            for future in futures:
                future.result()
            return True
        self._pool.submit(self._advise, pages)
        return True
"""


def main(path: str) -> int:
    src = open(path, encoding="utf-8").read()
    if "_read_pages" in src:
        print("ALREADY PATCHED:", path)
        return 0
    for old, new, what in ((INIT_OLD, INIT_NEW, "prefetcher __init__"),
                           (ADVISE_OLD, ADVISE_NEW, "enqueue"),
                           (SUBMIT_OLD, SUBMIT_NEW, "advise submit")):
        if src.count(old) != 1:
            print(f"ERROR: expected 1 {what}, found {src.count(old)}")
            return 1
        src = src.replace(old, new, 1)
    open(path, "w", encoding="utf-8").write(src)
    print("PATCHED:", path)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))

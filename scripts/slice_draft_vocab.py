#!/usr/bin/env python3
"""Cut a smaller NEXTN draft-token map as a prefix of a ranked map.

`build_draft_vocab.py` writes ids in rank order: specials, then corpus ids by
frequency, then id-order fill. A prefix of that tensor is therefore the same
ranking at a smaller size, which keeps a size A/B to one variable (no new
corpus, no Wikipedia sampling).

    python3 scripts/slice_draft_vocab.py bench/draft_vocab/hot_tokens_64k.pt 48000 \
        bench/draft_vocab/hot_tokens_48k.pt
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch


def main(src: str, size: str, dst: str) -> int:
    ranked = torch.load(src)
    n = int(size)
    if ranked.dim() != 1 or n <= 0 or n > ranked.numel():
        raise SystemExit(f"cannot cut {n} ids from a map of shape {tuple(ranked.shape)}")
    out = ranked[:n].clone().to(torch.int64)
    torch.save(out, dst)
    src_report = Path(src).with_name(Path(src).stem + ".report.json")
    base = json.loads(src_report.read_text()) if src_report.is_file() else {}
    corpus_end = base.get("n_after_specials_and_corpus")
    report = {
        "source": src,
        "requested_size": n,
        "n_final": int(out.numel()),
        "n_special": base.get("n_special"),
        "n_corpus_ranked": (min(n, corpus_end) - base.get("n_special", 0)) if corpus_end else None,
        "n_fill": max(0, n - corpus_end) if corpus_end else None,
        "unique": int(out.unique().numel()),
    }
    Path(dst).with_name(Path(dst).stem + ".report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))
    return 0


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:4]))

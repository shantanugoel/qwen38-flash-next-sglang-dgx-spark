#!/usr/bin/env python3
"""Side-by-side table of two or more tagged bench runs.

    python3 bench/compare.py baseline packA
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"


def load(tag: str, name: str):
    p = RES / tag / f"{name}.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except json.JSONDecodeError:
        return None


def rows(tag: str) -> dict:
    out: dict[str, object] = {}
    d = load(tag, "decode")
    if d:
        for res in d.get("results", []):
            mode = res["mode"]
            for task, v in res.get("tasks", {}).items():
                out[f"decode {task} {mode}"] = v["median_tok_s"]
    q = load(tag, "quality")
    if q:
        rs = q.get("results", q) if isinstance(q, dict) else q
        if isinstance(rs, list):
            out["quality pass"] = f"{sum(bool(r.get('pass')) for r in rs)}/{len(rs)}"
    lc = load(tag, "longctx")
    if lc:
        for r in lc.get("results", []):
            out[f"needle {r['size']}"] = "PASS" if r["needle_pass"] else "FAIL"
            out[f"ttft {r['size']}"] = round(r["first_s"], 2)
            out[f"cache speedup {r['size']}"] = f"{r['speedup']}x"
    ag = load(tag, "agentic")
    if ag:
        s = ag["summary"]
        out["agentic invalid calls"] = s["invalid_tool_calls"]
        out["agentic late recall"] = "PASS" if s["late_recall_pass"] else "FAIL"
        bands = [b for b in s["bands"] if b]
        if bands:
            out["agentic ttft (last band)"] = bands[-1]["median_ttft_s"]
            out["agentic decode (last band)"] = bands[-1]["median_decode_tps"]
            out["agentic cache hit (last band)"] = f"{bands[-1]['median_cache_hit_pct']}%"
    bf = load(tag, "bfcl")
    if bf:
        s = bf["summary"]
        out["bfcl accuracy"] = f"{s['accuracy']}%"
        out["bfcl invalid calls"] = s["invalid_tool_calls"]
    g = load(tag, "gsm8k")
    if g:
        out["gsm8k"] = f"{g['summary']['correct']}/{g['summary']['n']}"
    return out


def main(tags: list[str]) -> int:
    if len(tags) < 1:
        print("usage: compare.py TAG [TAG...]", file=sys.stderr)
        return 2
    data = {t: rows(t) for t in tags}
    keys: list[str] = []
    for t in tags:
        for k in data[t]:
            if k not in keys:
                keys.append(k)
    w = max([len(k) for k in keys] + [8])
    print("| " + "metric".ljust(w) + " | " + " | ".join(t.ljust(12) for t in tags) + " |")
    print("| " + "-" * w + " | " + " | ".join("-" * 12 for _ in tags) + " |")
    for k in keys:
        cells = [str(data[t].get(k, "-")).ljust(12) for t in tags]
        print("| " + k.ljust(w) + " | " + " | ".join(cells) + " |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

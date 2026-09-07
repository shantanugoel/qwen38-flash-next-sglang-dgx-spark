#!/usr/bin/env python3
"""Accepted vs rejected MTP drafts with ReplaySSM PLE state still advancing.

Temperature 0: MTP almost fully accepts on this recipe (spec_accept_length ~4).
High temperature: more draft rejects. After both, a fact planted in turn 1
must still be recalled. This does not require identical text vs non-spec.

    python3 bench/ple_spec.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from client import chat, server_metrics  # noqa: E402

OUT = os.environ.get("OUT", "")
FACT = "PLE-SPEC-TOKEN-7F3A91"


def ok(name: str, passed: bool, detail: str, extra=None) -> dict:
    status = "PASS" if passed else "FAIL"
    print(f"RESULT {name}: {status} {detail}", flush=True)
    row = {"name": name, "pass": passed, "detail": detail}
    if extra:
        row.update(extra)
    return row


def generate(name: str, temperature: float, max_tokens: int) -> dict:
    before = server_metrics().get("sglang:spec_accept_length")
    r = chat(
        [
            {
                "role": "user",
                "content": (
                    "Write a Python function named checksum32 that folds a UTF-8 "
                    "string into an unsigned 32-bit integer with xor and rotate. "
                    "Include a docstring and two assertions. No markdown."
                ),
            }
        ],
        max_tokens=max_tokens,
        temperature=temperature,
        thinking=False,
    )
    after = server_metrics().get("sglang:spec_accept_length")
    text = r["content"] or r["reasoning"] or ""
    passed = r["completion_tokens"] > 0 and len(text.strip()) > 20
    return ok(
        name,
        passed,
        (
            f"temp={temperature} tok={r['completion_tokens']} "
            f"sec={r['seconds']:.2f} accept={before}->{after}"
        ),
        {
            "temperature": temperature,
            "completion_tokens": r["completion_tokens"],
            "seconds": r["seconds"],
            "spec_accept_before": before,
            "spec_accept_after": after,
            "preview": text[:240],
        },
    )


def recall() -> dict:
    r1 = chat(
        [
            {
                "role": "user",
                "content": (
                    f"Store this access note exactly: {FACT}. "
                    "Do not repeat it unless asked. Reply with OK."
                ),
            }
        ],
        max_tokens=32,
        temperature=0,
        thinking=False,
    )
    r2 = chat(
        [
            {"role": "user", "content": f"Store this access note exactly: {FACT}. Do not repeat it unless asked. Reply with OK."},
            {"role": "assistant", "content": r1["content"] or "OK"},
            {
                "role": "user",
                "content": (
                    "Write 40 words of filler about checksums, then wait."
                ),
            },
        ],
        max_tokens=120,
        temperature=0,
        thinking=False,
    )
    r3 = chat(
        [
            {"role": "user", "content": f"Store this access note exactly: {FACT}. Do not repeat it unless asked. Reply with OK."},
            {"role": "assistant", "content": r1["content"] or "OK"},
            {"role": "user", "content": "Write 40 words of filler about checksums, then wait."},
            {"role": "assistant", "content": r2["content"] or ""},
            {"role": "user", "content": "What was the access note? Repeat it exactly."},
        ],
        max_tokens=64,
        temperature=0,
        thinking=False,
    )
    text = r3["content"] or r3["reasoning"] or ""
    return ok(
        "ple_spec_recall",
        FACT in text,
        f"content={text!r}",
        {"content": text, "spec_accept": server_metrics().get("sglang:spec_accept_length")},
    )


def main() -> int:
    rows = [
        generate("ple_spec_accept_heavy", temperature=0.0, max_tokens=128),
        generate("ple_spec_reject_heavy", temperature=1.4, max_tokens=128),
        recall(),
    ]
    metrics = server_metrics()
    failed = [r for r in rows if not r["pass"]]
    accept = metrics.get("sglang:spec_accept_length")
    if accept is not None and accept < 1.05:
        rows.append(ok("ple_spec_accept_length", False, f"spec_accept_length={accept} (speculation idle)"))
        failed = [r for r in rows if not r["pass"]]
    else:
        rows.append(ok("ple_spec_accept_length", True, f"spec_accept_length={accept}"))
    report = {"results": rows, "failed": len(failed), "metrics": metrics}
    if OUT:
        Path(OUT).write_text(json.dumps(report, indent=2))
        print(f"wrote {OUT}", flush=True)
    print(f"RESULT ple_spec: {len(rows)-len(failed)}/{len(rows)} pass", flush=True)
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())

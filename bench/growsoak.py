#!/usr/bin/env python3
"""Growing-context soak: unique tokens so PLE rows keep faulting in.

    SOAK_SECONDS=3600 python3 bench/growsoak.py

Each turn appends a unique chunk (new n-grams). At MAX_PROMPT_TOKENS the
session rotates so we do not walk into the 120k-prefill wedge. A planted
access code is recalled on a schedule. This is the U4c RSS-trimmer load;
bench/soak.py remains the short-chat mix.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from client import chat  # noqa: E402

SECONDS = float(os.environ.get("SOAK_SECONDS", "3600"))
OUT = os.environ.get("OUT", "")
MAX_PROMPT = int(os.environ.get("MAX_PROMPT_TOKENS", "24000"))
NEEDLE = "7K-QUARTZ-19"
FILLER = (
    "The quarterly operations log continues with routine warehouse notes, "
    "shift handovers, pallet counts, and weather asides. Nothing in this "
    "paragraph is the secret. "
)


def approx_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def chunk(i: int) -> str:
    return f"Record {i:06d} id={i * 7919:x}. " + FILLER + FILLER


def plant_msg(session: int) -> dict:
    return {
        "role": "user",
        "content": (
            f"Session {session}. Invented fact: NIGHTINGALE-ACCESS-CODE={NEEDLE}. "
            "Do not mention it until asked. " + FILLER
        ),
    }


def main() -> int:
    deadline = time.monotonic() + SECONDS
    rows = []
    errors = 0
    recall_fail = 0
    start = time.monotonic()
    i = 0
    session = 0
    messages = [plant_msg(session)]
    while time.monotonic() < deadline:
        i += 1
        recall = i % 8 == 0
        if recall:
            messages.append({
                "role": "user",
                "content": "What is the NIGHTINGALE access code? Reply with the code only.",
            })
        else:
            messages.append({"role": "user", "content": chunk(i)})
        row = {"i": i, "session": session, "recall": recall, "pass": False}
        try:
            result = chat(
                messages,
                max_tokens=32 if recall else 16,
                temperature=0.0,
                thinking=False,
            )
            row["prompt_tokens"] = result["prompt_tokens"]
            row["completion_tokens"] = result["completion_tokens"]
            row["seconds"] = round(result["seconds"], 3)
            row["finish"] = result["finish"]
            if recall:
                text = (result["content"] or "") + " " + (result["reasoning"] or "")
                hit = NEEDLE in text.replace(" ", "")
                row["pass"] = hit
                if not hit:
                    recall_fail += 1
                    row["content"] = (result["content"] or "")[:80]
            else:
                row["pass"] = result["completion_tokens"] > 0
            messages.append({
                "role": "assistant",
                "content": result["content"] or "",
            })
        except Exception as exc:  # noqa: BLE001
            errors += 1
            row["error"] = type(exc).__name__ + ": " + str(exc)[:200]
        if row.get("prompt_tokens", 0) >= MAX_PROMPT or approx_tokens(
            json.dumps(messages)
        ) >= MAX_PROMPT:
            session += 1
            messages = [plant_msg(session)]
        rows.append(row)
        print(
            "RESULT growsoak item:", i, "sess", session,
            "ok" if row["pass"] else "fail",
            f"tok={row.get('prompt_tokens', 0)}",
            f"s={row.get('seconds', 0)}",
            flush=True,
        )
    elapsed = round(time.monotonic() - start, 3)
    ok = sum(1 for r in rows if r["pass"])
    report = {
        "results": rows,
        "failed": errors + recall_fail,
        "summary": {
            "ok": ok,
            "errors": errors,
            "recall_fail": recall_fail,
            "requests": len(rows),
            "sessions": session + 1,
            "seconds": elapsed,
            "target_seconds": SECONDS,
            "max_prompt_tokens": max((r.get("prompt_tokens") or 0) for r in rows) if rows else 0,
        },
    }
    if OUT:
        Path(OUT).write_text(json.dumps(report, indent=2))
        print(f"wrote {OUT}", flush=True)
    print(
        f"RESULT growsoak: {ok}/{len(rows)} ok errors={errors} "
        f"recall_fail={recall_fail} seconds={elapsed}",
        flush=True,
    )
    return 0 if errors == 0 and recall_fail == 0 and ok > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())

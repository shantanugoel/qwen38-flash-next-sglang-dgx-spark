#!/usr/bin/env python3
"""Paired multi-prompt arithmetic probe: is a checkpoint better on a *class*?

    PROMPTS=20 REPEATS=10 python3 bench/arith_probe.py

bench/effort_probe.py showed one prompt where two checkpoints differ a lot
(Radix 35/100, NVIDIA 49/60) while GSM8K n=200 tied. One prompt cannot say
whether that is a class effect or a single knife-edge instance, so this probe
resends a *set* of same-shaped problems -- short multi-step arithmetic word
problems, thinking off, integer answer, temperature 0 -- N times each.

Every prompt is rendered from parameters and its expected answer is computed
from those same parameters, so the key cannot disagree with the question.
Descriptive: it reports per-prompt and pooled accuracy and never gates.
"""
from __future__ import annotations

import json
import os
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from client import chat  # noqa: E402

OUT = os.environ.get("OUT", "")
REPEATS = int(os.environ.get("REPEATS", os.environ.get("ARITH_PROBE_REPEATS", "10")))
LIMIT = int(os.environ.get("PROMPTS", os.environ.get("ARITH_PROBE_PROMPTS", "20")))
THINKING = os.environ.get("THINKING", "off") == "on"
TAIL = " Reply with the integer only, no units and no explanation."


def cases() -> list[dict]:
    """Render prompts and compute their answers from the same parameters."""
    out = []
    # The exact effort_probe case, kept first for continuity with earlier runs.
    out.append({
        "name": "change_pens_notebooks_4_2_50",
        "prompt": ("A shop sells pens at $3 and notebooks at $7. If I buy 4 pens "
                   "and 2 notebooks and pay with a $50 bill, what is my change?"
                   " Reply with the integer dollar amount only."),
        "answer": 50 - (4 * 3 + 2 * 7),
    })
    for pen, book, pens, books, paid in [
        (4, 9, 3, 5, 100), (6, 11, 7, 3, 120), (2, 13, 9, 4, 90), (5, 8, 6, 6, 100),
    ]:
        out.append({
            "name": f"change_{pen}_{book}_{pens}_{books}_{paid}",
            "prompt": (f"A shop sells pens at ${pen} and notebooks at ${book}. If I buy "
                       f"{pens} pens and {books} notebooks and pay with a ${paid} bill, "
                       f"what is my change?{TAIL}"),
            "answer": paid - (pen * pens + book * books),
        })
    for boxes, per_box, broken in [(12, 8, 5), (23, 6, 17), (9, 15, 11), (14, 12, 29)]:
        out.append({
            "name": f"boxes_{boxes}_{per_box}_{broken}",
            "prompt": (f"A warehouse receives {boxes} boxes with {per_box} mugs in each "
                       f"box. {broken} mugs arrive broken and are thrown away. How many "
                       f"usable mugs are there?{TAIL}"),
            "answer": boxes * per_box - broken,
        })
    for start, spent, earned in [(180, 65, 40), (250, 130, 75), (95, 48, 120), (310, 199, 64)]:
        out.append({
            "name": f"wallet_{start}_{spent}_{earned}",
            "prompt": (f"Dana starts the week with ${start}, spends ${spent} on groceries, "
                       f"then earns ${earned} from a side job. How much money does Dana "
                       f"have at the end of the week?{TAIL}"),
            "answer": start - spent + earned,
        })
    for rate, hours, bonus, tax in [(18, 7, 25, 30), (22, 9, 40, 58), (15, 12, 10, 45), (27, 6, 33, 70)]:
        out.append({
            "name": f"wage_{rate}_{hours}_{bonus}_{tax}",
            "prompt": (f"Sam is paid ${rate} per hour, works {hours} hours, and receives a "
                       f"${bonus} bonus. ${tax} is withheld for tax. What is Sam's take-home "
                       f"pay?{TAIL}"),
            "answer": rate * hours + bonus - tax,
        })
    for total, groups, extra in [(96, 8, 3), (144, 12, 7), (75, 5, 9), (210, 7, 12)]:
        out.append({
            "name": f"share_{total}_{groups}_{extra}",
            "prompt": (f"{total} chairs are split evenly between {groups} rooms, and then "
                       f"{extra} more chairs are added to one of those rooms. How many "
                       f"chairs are in that room?{TAIL}"),
            "answer": total // groups + extra,
        })
    return out[:LIMIT]


def integers(text: str) -> list[int]:
    cleaned = text.replace(",", "").replace("$", " ")
    found, token = [], ""
    for ch in cleaned + " ":
        if ch.isdigit() or (ch == "-" and not token):
            token += ch
        else:
            if token.lstrip("-").isdigit():
                found.append(int(token))
            token = ""
    return found


def scored(text: str, expected: int) -> bool:
    """Exact reply first; otherwise the first integer in the reply."""
    stripped = text.strip().rstrip(".").replace("$", "").replace(",", "")
    if stripped.lstrip("-").isdigit():
        return int(stripped) == expected
    found = integers(text)
    return bool(found) and found[0] == expected


def main() -> int:
    rows, per_prompt = [], []
    for case in cases():
        hits = 0
        answers: dict[str, int] = {}
        for i in range(REPEATS):
            r = chat(
                [{"role": "user", "content": case["prompt"]}],
                max_tokens=2048, temperature=0, thinking=THINKING,
            )
            text = (r["content"] or "").strip()
            ok = scored(text, case["answer"])
            hits += ok
            answers[text[:24]] = answers.get(text[:24], 0) + 1
            rows.append({"name": case["name"], "i": i, "pass": ok,
                         "answer": text[:60], "expected": case["answer"],
                         "completion_tokens": r["completion_tokens"]})
        per_prompt.append({"name": case["name"], "expected": case["answer"],
                           "passed": hits, "n": REPEATS, "answers": answers})
        print(f"RESULT arith_{case['name']}: {hits}/{REPEATS} expected={case['answer']} "
              f"answers={answers}", flush=True)
    total = sum(p["passed"] for p in per_prompt)
    n = sum(p["n"] for p in per_prompt)
    rates = [p["passed"] / p["n"] for p in per_prompt]
    summary = {
        "prompts": len(per_prompt), "repeats": REPEATS, "n": n, "passed": total,
        "accuracy": round(100.0 * total / n, 2) if n else 0.0,
        "thinking": "on" if THINKING else "off",
        "perfect_prompts": sum(1 for p in per_prompt if p["passed"] == p["n"]),
        "zero_prompts": sum(1 for p in per_prompt if p["passed"] == 0),
        "median_prompt_rate": round(statistics.median(rates), 3) if rates else None,
    }
    report = {"results": rows, "per_prompt": per_prompt, "summary": summary,
              "failed": n - total}
    if OUT:
        Path(OUT).write_text(json.dumps(report, indent=2))
        print(f"wrote {OUT}", flush=True)
    print(f"RESULT arith_probe: {total}/{n} = {summary['accuracy']}% over "
          f"{len(per_prompt)} prompts; perfect={summary['perfect_prompts']} "
          f"zero={summary['zero_prompts']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

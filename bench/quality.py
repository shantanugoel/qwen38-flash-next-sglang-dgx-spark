#!/usr/bin/env python3
"""Quality smokes for agentic use: math, tools, executed code, multi-turn fact.

    python3 bench/quality.py
    EFFORT=1 python3 bench/quality.py   # also sweep reasoning_effort
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from client import chat  # noqa: E402

OUT = os.environ.get("OUT", "")
EFFORT = os.environ.get("EFFORT", "0") == "1"

WEATHER_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "City name",
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                    },
                },
                "required": ["location"],
            },
        },
    }
]


def ok(name: str, passed: bool, detail: str, extra=None) -> dict:
    status = "PASS" if passed else "FAIL"
    print(f"RESULT {name}: {status} {detail}", flush=True)
    row = {"name": name, "pass": passed, "detail": detail}
    if extra:
        row.update(extra)
    return row


def math_smoke() -> dict:
    r = chat(
        [{"role": "user", "content": "12*17"}],
        max_tokens=2048,
        temperature=0,
        thinking=False,
    )
    text = (r["content"] or r["reasoning"]).strip()
    return ok(
        "math_12x17",
        "204" in text,
        f"content={text!r} sec={r['seconds']:.2f}",
        {"seconds": r["seconds"], "completion_tokens": r["completion_tokens"]},
    )


def tool_smoke() -> dict:
    r = chat(
        [{"role": "user", "content": "What's the weather in Beijing in celsius?"}],
        max_tokens=1024,
        temperature=0,
        thinking=False,
        tools=WEATHER_TOOLS,
        extra={"tool_choice": "auto"},
    )
    names = []
    for tc in r["tool_calls"]:
        fn = tc.get("function") or {}
        names.append(fn.get("name") or tc.get("name"))
    passed = "get_weather" in names
    return ok(
        "tool_weather",
        passed,
        f"tool_calls={names} finish={r['finish']} content={ (r['content'] or '')[:80]!r}",
        {"seconds": r["seconds"], "tool_names": names},
    )


def code_exec_smoke() -> dict:
    r = chat(
        [
            {
                "role": "user",
                "content": (
                    "Write a Python function named add(a, b) that returns a+b. "
                    "Reply with the function only, no markdown fences."
                ),
            }
        ],
        max_tokens=256,
        temperature=0,
        thinking=False,
    )
    code = r["content"].strip()
    if "```" in code:
        parts = code.split("```")
        code = parts[1]
        if code.startswith("python"):
            code = code[6:]
        code = code.strip()
    passed = False
    err = ""
    try:
        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
            f.write(code + "\nprint(add(19, 23))\n")
            path = f.name
        proc = subprocess.run(
            ["python3", path], capture_output=True, text=True, timeout=10
        )
        out = (proc.stdout or "").strip()
        err = (proc.stderr or "").strip()[:400]
        passed = proc.returncode == 0 and out.endswith("42")
        os.unlink(path)
    except Exception as e:  # noqa: BLE001
        out = ""
        err = str(e)
    return ok(
        "code_exec_add",
        passed,
        f"stdout={out!r} err={err!r} preview={code[:120]!r}",
        {"seconds": r["seconds"]},
    )


def multiturn_fact() -> dict:
    fact = "The access code for project NIGHTINGALE is 7K-QUARTZ-19."
    msgs = [
        {
            "role": "user",
            "content": (
                f"Remember this invented fact for later turns: {fact} "
                "Reply with only OK."
            ),
        }
    ]
    r1 = chat(msgs, max_tokens=64, temperature=0, thinking=False)
    msgs.append({"role": "assistant", "content": r1["content"] or "OK"})
    msgs.append(
        {"role": "user", "content": "List two prime numbers greater than 20."}
    )
    r2 = chat(msgs, max_tokens=128, temperature=0, thinking=False)
    msgs.append({"role": "assistant", "content": r2["content"]})
    msgs.append({"role": "user", "content": "What is 9 times 8?"})
    r3 = chat(msgs, max_tokens=64, temperature=0, thinking=False)
    msgs.append({"role": "assistant", "content": r3["content"]})
    msgs.append(
        {
            "role": "user",
            "content": "What is the access code for project NIGHTINGALE? Reply with the code only.",
        }
    )
    r4 = chat(msgs, max_tokens=128, temperature=0, thinking=False)
    text = (r4["content"] or "") + " " + (r4["reasoning"] or "")
    passed = "7K-QUARTZ-19" in text.replace(" ", "")
    return ok(
        "multiturn_fact",
        passed,
        f"turn4={r4['content']!r}",
        {"seconds": r1["seconds"] + r2["seconds"] + r3["seconds"] + r4["seconds"]},
    )


def effort_sweep() -> list[dict]:
    q = (
        "A shop sells pens at $3 and notebooks at $7. "
        "If I buy 4 pens and 2 notebooks and pay with a $50 bill, "
        "what is my change? Reply with the integer dollar amount only."
    )
    rows = []
    for thinking, effort, name in [
        (True, None, "effort_default"),
        (True, "low", "effort_low"),
        (True, "medium", "effort_medium"),
        (False, None, "effort_thinking_off"),
    ]:
        r = chat(
            [{"role": "user", "content": q}],
            max_tokens=2048,
            temperature=0,
            thinking=thinking,
            reasoning_effort=effort,
        )
        text = r["content"] or ""
        passed = "24" in text
        rows.append(
            ok(
                name,
                passed,
                (
                    f"content={text.strip()[:80]!r} sec={r['seconds']:.1f} "
                    f"tok={r['completion_tokens']} reason_tok={r['reasoning_tokens']}"
                ),
                {
                    "seconds": r["seconds"],
                    "completion_tokens": r["completion_tokens"],
                    "reasoning_tokens": r["reasoning_tokens"],
                },
            )
        )
    return rows


def main() -> int:
    rows = [math_smoke(), tool_smoke(), code_exec_smoke(), multiturn_fact()]
    if EFFORT:
        rows.extend(effort_sweep())
    failed = [r for r in rows if not r["pass"]]
    report = {"results": rows, "failed": len(failed)}
    if OUT:
        Path(OUT).write_text(json.dumps(report, indent=2))
        print(f"wrote {OUT}", flush=True)
    print(f"RESULT quality: {len(rows)-len(failed)}/{len(rows)} pass", flush=True)
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())

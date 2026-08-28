#!/usr/bin/env python3
"""Long-horizon agentic session: many tool-calling turns on one growing context.

    TURNS=60 python3 bench/agentic.py
    TURNS=120 THINKING=on OUT=results/x/agentic.json python3 bench/agentic.py

Simulates what this recipe is actually for: a system prompt plus tool
definitions that never change, a transcript that only grows, and a tool call
every turn. Reports the drift that matters over 100-150 turns - TTFT, decode
rate and prefix-cache hit % as a function of turn index - plus two coherence
probes: a fact planted early and recalled late, and the tool-call validity rate.
"""
from __future__ import annotations

import json
import os
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from client import chat_stream  # noqa: E402

TURNS = int(os.environ.get("TURNS", "60"))
THINKING = os.environ.get("THINKING", "off").lower() == "on"
OUT = os.environ.get("OUT", "")
MAX_TOKENS = int(os.environ.get("AGENTIC_MAX_TOKENS", "512"))
FACT = "NIGHTINGALE-ACCESS-CODE=7K-QUARTZ-19"
PLANT_TURN = 3

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_docs",
            "description": "Search the internal document corpus.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "top_k": {"type": "integer", "description": "How many hits"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a file from the workspace.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        },
    },
]

SYSTEM = (
    "You are a research agent working through a long task. You have two tools. "
    "For every user message, call exactly one tool that helps, unless the user "
    "asks you a direct question about something you already know - then answer "
    "in plain text with no tool call."
)

TOPICS = [
    "quarterly revenue by region",
    "the retry policy in the ingest service",
    "who owns the billing schema",
    "the last migration that touched orders",
    "open incidents older than a week",
    "which jobs write to the audit bucket",
    "the rate limit on the public API",
    "the on-call rotation for next month",
]


def tool_result(name: str, args: dict, turn: int) -> str:
    if turn == PLANT_TURN:
        return json.dumps(
            {
                "hits": [
                    {
                        "path": "ops/secrets.md",
                        "text": f"Access note: {FACT}. Do not repeat unless asked.",
                    }
                ]
            }
        )
    return json.dumps(
        {
            "hits": [
                {
                    "path": f"docs/note_{turn:03d}.md",
                    "text": (
                        f"Turn {turn} record about {args.get('query', args.get('path', 'the topic'))}. "
                        "Routine content, nothing secret here."
                    ),
                }
            ]
        }
    )


def main() -> int:
    msgs: list[dict] = [{"role": "system", "content": SYSTEM}]
    rows = []
    invalid = 0
    tool_turns = 0
    t0 = time.time()

    last_reply = ""
    for turn in range(1, TURNS + 1):
        if turn == TURNS:
            ask = (
                "Stop searching. From what you have already seen in this session, "
                "what is the NIGHTINGALE access code? Reply with the code only, "
                "no tool call."
            )
        elif turn == PLANT_TURN:
            ask = "Search the docs for the access note in ops/secrets.md."
        else:
            ask = f"Look into {TOPICS[turn % len(TOPICS)]} (step {turn})."
        msgs.append({"role": "user", "content": ask})

        r = chat_stream(
            msgs,
            max_tokens=MAX_TOKENS,
            temperature=0,
            thinking=THINKING,
            tools=TOOLS,
            extra={"tool_choice": "auto"},
        )

        last_reply = r["content"] or r["reasoning"] or ""
        calls = []
        merged: dict = {}
        for i, tc in enumerate(r["tool_calls"]):
            idx = tc.get("index", i)
            slot = merged.setdefault(idx, {"name": "", "args": ""})
            fn = tc.get("function") or {}
            if fn.get("name"):
                slot["name"] = fn["name"]
            if fn.get("arguments"):
                slot["args"] += fn["arguments"]
        for slot in merged.values():
            try:
                args = json.loads(slot["args"] or "{}")
                if not isinstance(args, dict):
                    raise ValueError
            except Exception:  # noqa: BLE001
                invalid += 1
                args = {}
            if slot["name"] not in ("search_docs", "read_file"):
                invalid += 1
            calls.append((slot["name"], args))

        if calls:
            tool_turns += 1
            msgs.append(
                {
                    "role": "assistant",
                    "content": r["content"] or "",
                    "tool_calls": [
                        {
                            "id": f"c{turn}_{i}",
                            "type": "function",
                            "function": {"name": n, "arguments": json.dumps(a)},
                        }
                        for i, (n, a) in enumerate(calls)
                    ],
                }
            )
            for i, (n, a) in enumerate(calls):
                msgs.append(
                    {
                        "role": "tool",
                        "tool_call_id": f"c{turn}_{i}",
                        "content": tool_result(n, a, turn),
                    }
                )
        else:
            msgs.append({"role": "assistant", "content": r["content"] or ""})

        rows.append(
            {
                "turn": turn,
                "prompt_tokens": r["prompt_tokens"],
                "cached_tokens": r["cached_tokens"],
                "cache_hit_pct": r["cache_hit_pct"],
                "ttft_s": round(r["ttft_s"], 3),
                "decode_tps": r["decode_tps"],
                "completion_tokens": r["completion_tokens"],
                "tool_calls": [n for n, _ in calls],
            }
        )
        if turn % 10 == 0 or turn in (1, TURNS):
            print(
                f"turn {turn:3d} ctx={r['prompt_tokens']:6d} "
                f"cache={r['cache_hit_pct']:5.1f}% ttft={r['ttft_s']:6.2f}s "
                f"decode={r['decode_tps']:6.2f} tok/s calls={[n for n,_ in calls]}",
                flush=True,
            )

    final = last_reply
    recalled = "7K-QUARTZ-19" in final.replace(" ", "")

    def band(lo, hi):
        sel = [r for r in rows if lo <= r["turn"] <= hi]
        if not sel:
            return {}
        return {
            "turns": f"{lo}-{hi}",
            "median_ttft_s": round(statistics.median(r["ttft_s"] for r in sel), 3),
            "median_decode_tps": round(
                statistics.median(r["decode_tps"] for r in sel), 2
            ),
            "median_cache_hit_pct": round(
                statistics.median(r["cache_hit_pct"] for r in sel), 1
            ),
            "median_ctx": int(statistics.median(r["prompt_tokens"] for r in sel)),
        }

    third = max(1, TURNS // 3)
    summary = {
        "turns": TURNS,
        "thinking": "on" if THINKING else "off",
        "wall_s": round(time.time() - t0, 1),
        "final_ctx_tokens": rows[-1]["prompt_tokens"] if rows else 0,
        "tool_turns": tool_turns,
        "invalid_tool_calls": invalid,
        "late_recall_pass": recalled,
        "late_recall_answer": final[:120],
        "bands": [band(1, third), band(third + 1, 2 * third), band(2 * third + 1, TURNS)],
    }
    print("RESULT agentic: " + json.dumps(summary), flush=True)
    if OUT:
        Path(OUT).parent.mkdir(parents=True, exist_ok=True)
        Path(OUT).write_text(json.dumps({"summary": summary, "turns": rows}, indent=2))
    return 0 if recalled else 1


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""BFCL fixed-subset function-calling accuracy against the live server.

    python3 bench/bfcl.py                       # full subset from bench/subsets
    LIMIT=10 python3 bench/bfcl.py              # smoke
    OUT=results/bfcl.json python3 bench/bfcl.py

Scoring
  simple / multiple / parallel  BFCL AST match against possible_answer
  irrelevance / live_irrelevance  pass = the model emits no tool call
  multi_turn_base               RELAXED: per-turn function-name set match
                                (no stateful sandbox; see RESEARCH_LOG.md)

Also reports invalid tool calls: unparseable arguments, unknown function name,
or a missing required parameter.
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from client import chat_stream  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DATA = Path(os.environ.get("BFCL_DATA", ROOT / "data" / "bfcl"))
SPEC = json.loads((ROOT / "bench" / "subsets" / "bfcl_v4_100.json").read_text())
OUT = os.environ.get("OUT", "")
LIMIT = int(os.environ.get("LIMIT", "0"))
THINKING = os.environ.get("THINKING", "off").lower() == "on"
MAX_TOKENS = int(os.environ.get("BFCL_MAX_TOKENS", "1024"))
# How many tool-call rounds a single multi-turn user message may take.
MT_MAX_STEPS = int(os.environ.get("BFCL_MT_MAX_STEPS", "6"))

TYPE_MAP = {
    "dict": "object",
    "float": "number",
    "integer": "integer",
    "string": "string",
    "boolean": "boolean",
    "array": "array",
    "tuple": "array",
    "any": "string",
}


def conv_schema(node):
    if not isinstance(node, dict):
        return node
    out = {}
    for k, v in node.items():
        if k == "type" and isinstance(v, str):
            out["type"] = TYPE_MAP.get(v, v)
        elif k in ("properties", "$defs"):
            out[k] = {pk: conv_schema(pv) for pk, pv in v.items()}
        elif k == "items":
            out[k] = conv_schema(v)
        elif k == "default":
            continue
        else:
            out[k] = v
    if out.get("type") == "object" and "properties" not in out:
        out["properties"] = {}
    return out


def to_tools(functions):
    tools = []
    for fn in functions:
        params = conv_schema(fn.get("parameters") or {"type": "object", "properties": {}})
        params.setdefault("type", "object")
        params.setdefault("properties", {})
        tools.append(
            {
                "type": "function",
                "function": {
                    "name": fn["name"],
                    "description": (fn.get("description") or "")[:1024],
                    "parameters": params,
                },
            }
        )
    return tools


def norm(v):
    if isinstance(v, str):
        return re.sub(r"\s+", " ", v.strip().lower().replace("_", " "))
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, list):
        return [norm(x) for x in v]
    if isinstance(v, dict):
        return {k: norm(x) for k, x in sorted(v.items())}
    return v


def value_ok(got, allowed):
    if not isinstance(allowed, list):
        allowed = [allowed]
    g = norm(got)
    for a in allowed:
        if norm(a) == g:
            return True
        if isinstance(a, (int, float)) and isinstance(got, str):
            try:
                if float(got) == float(a):
                    return True
            except ValueError:
                pass
    return False


def parse_calls(tool_calls):
    """Return (calls, invalid_reasons). calls = [(name, args_dict)]."""
    calls, bad = [], []
    merged: dict[int, dict] = {}
    for i, tc in enumerate(tool_calls):
        idx = tc.get("index", i) if isinstance(tc, dict) else i
        slot = merged.setdefault(idx, {"name": "", "args": ""})
        fn = (tc.get("function") or {}) if isinstance(tc, dict) else {}
        if fn.get("name"):
            slot["name"] = fn["name"]
        if fn.get("arguments"):
            slot["args"] += fn["arguments"]
    for slot in merged.values():
        name = slot["name"]
        raw = slot["args"] or "{}"
        try:
            args = json.loads(raw)
            if not isinstance(args, dict):
                raise ValueError("arguments is not an object")
        except Exception as exc:  # noqa: BLE001
            bad.append(f"{name}: bad arguments ({exc})")
            args = {}
        calls.append((name, args))
    return calls, bad


def ast_match(calls, ground_truth, tool_names):
    """BFCL-style AST check for simple / multiple / parallel."""
    if len(calls) != len(ground_truth):
        return False, [f"expected {len(ground_truth)} calls, got {len(calls)}"]
    problems: list[str] = []
    remaining = list(ground_truth)
    for name, args in calls:
        if name not in tool_names:
            problems.append(f"unknown function {name!r}")
            return False, problems
        hit = None
        for gt in remaining:
            gname, gargs = next(iter(gt.items()))
            if gname.replace(".", "_") != name.replace(".", "_"):
                continue
            good = True
            for pname, allowed in gargs.items():
                if pname in args:
                    if not value_ok(args[pname], allowed):
                        good = False
                        break
                elif "" not in (allowed if isinstance(allowed, list) else [allowed]):
                    good = False
                    break
            if good and set(args) - set(gargs):
                good = False
            if good:
                hit = gt
                break
        if hit is None:
            problems.append(f"no ground-truth match for {name}({json.dumps(args)[:120]})")
            return False, problems
        remaining.remove(hit)
    return True, problems


def load_split(fname):
    rows = []
    with open(DATA / fname, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    rows.sort(key=lambda r: (int(re.sub(r"\D", "", r["id"]) or 0), r["id"]))
    return rows


CLASS_DOC = {
    "GorillaFileSystem": "gorilla_file_system.json",
    "MathAPI": "math_api.json",
    "MessageAPI": "message_api.json",
    "TwitterAPI": "posting_api.json",
    "TicketAPI": "ticket_api.json",
    "TradingBot": "trading_bot.json",
    "TravelAPI": "travel_booking.json",
    "VehicleControlAPI": "vehicle_control.json",
}


def class_functions(classes) -> list[dict]:
    """Multi-turn cases carry involved_classes, not an inline function list."""
    out = []
    for cls in classes or []:
        fname = CLASS_DOC.get(cls)
        if not fname:
            continue
        path = DATA / "multi_turn_func_doc" / fname
        if not path.exists():
            continue
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    out.append(json.loads(line))
    return out


def load_answers(fname):
    p = DATA / "possible_answer" / fname
    if not p.exists():
        return {}
    out = {}
    with open(p, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                d = json.loads(line)
                out[d["id"]] = d["ground_truth"]
    return out


def gt_names(turn_calls):
    names = []
    for c in turn_calls:
        m = re.match(r"\s*([A-Za-z_][\w.]*)\s*\(", c)
        if m:
            names.append(m.group(1).replace(".", "_"))
    return names


def run_case(case, split, expect, answers, agg):
    fns = case.get("function")
    if fns is None:
        fns = class_functions(case.get("involved_classes"))
    elif not isinstance(fns, list):
        fns = [fns]
    tools = to_tools(fns)
    tool_names = {t["function"]["name"] for t in tools}
    turns = case["question"]
    msgs: list[dict] = []
    invalid: list[str] = []
    passed = True
    detail = ""

    if expect == "multi_turn":
        # A real agent issues one call, reads the result, then the next. BFCL's
        # ground truth lists the whole sequence for the turn, so the model has
        # to be allowed to loop within a turn before the turn is scored.
        gts = answers.get(case["id"], [])
        for ti, turn in enumerate(turns):
            msgs.extend(turn)
            got: set[str] = set()
            for step in range(MT_MAX_STEPS):
                r = chat_stream(
                    msgs,
                    max_tokens=MAX_TOKENS,
                    temperature=0,
                    thinking=THINKING,
                    tools=tools,
                    extra={"tool_choice": "auto"},
                )
                agg_add(agg, r)
                calls, bad = parse_calls(r["tool_calls"])
                invalid.extend(bad)
                for n, _ in calls:
                    if n not in tool_names:
                        invalid.append(f"unknown function {n!r}")
                got |= {n.replace(".", "_") for n, _ in calls}
                if not calls:
                    msgs.append({"role": "assistant", "content": r["content"] or ""})
                    break
                msgs.append(
                    {
                        "role": "assistant",
                        "content": r["content"] or "",
                        "tool_calls": [
                            {
                                "id": f"call_{ti}_{step}_{i}",
                                "type": "function",
                                "function": {"name": n, "arguments": json.dumps(a)},
                            }
                            for i, (n, a) in enumerate(calls)
                        ],
                    }
                )
                for i, (n, _) in enumerate(calls):
                    msgs.append(
                        {
                            "role": "tool",
                            "tool_call_id": f"call_{ti}_{step}_{i}",
                            "content": json.dumps({"status": "ok", "function": n}),
                        }
                    )
            want = set(gt_names(gts[ti])) if ti < len(gts) else set()
            if want and not want.issubset(got):
                passed = False
                detail += f"turn{ti}: want {sorted(want)} got {sorted(got)}; "
        return passed, detail[:300], invalid

    msgs = list(turns[0]) if isinstance(turns[0], list) else [turns[0]]
    r = chat_stream(
        msgs,
        max_tokens=MAX_TOKENS,
        temperature=0,
        thinking=THINKING,
        tools=tools,
        extra={"tool_choice": "auto"},
    )
    agg_add(agg, r)
    calls, bad = parse_calls(r["tool_calls"])
    invalid.extend(bad)
    for n, _ in calls:
        if n not in tool_names:
            invalid.append(f"unknown function {n!r}")

    if expect == "no_tool":
        passed = len(calls) == 0
        detail = f"calls={[n for n, _ in calls]}"
    else:
        gt = answers.get(case["id"])
        if gt is None:
            passed = len(calls) > 0
            detail = "no ground truth; scored as 'made a call'"
        else:
            passed, problems = ast_match(calls, gt, tool_names)
            detail = "; ".join(problems)[:300]
    return passed, detail, invalid


def agg_add(agg, r):
    agg["requests"] += 1
    agg["prompt_tokens"] += r["prompt_tokens"]
    agg["completion_tokens"] += r["completion_tokens"]
    agg["cached_tokens"] += r["cached_tokens"]
    agg["ttft_sum"] += r["ttft_s"]
    agg["decode_tps"].append(r["decode_tps"])


def main() -> int:
    if not DATA.exists():
        print(f"missing {DATA}; see bench/subsets/bfcl_v4_100.json", file=sys.stderr)
        return 2
    agg = {
        "requests": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "cached_tokens": 0,
        "ttft_sum": 0.0,
        "decode_tps": [],
    }
    results = []
    t0 = time.time()
    only = {x for x in os.environ.get("SPLITS", "").split(",") if x}
    for split, cfg in SPEC["splits"].items():
        if only and split not in only:
            continue
        rows = load_split(cfg["file"])
        answers = load_answers(cfg["file"])
        n = cfg["n"] if not LIMIT else min(cfg["n"], LIMIT)
        for case in rows[:n]:
            try:
                passed, detail, invalid = run_case(
                    case, split, cfg["expect"], answers, agg
                )
            except SystemExit as exc:
                passed, detail, invalid = False, f"server error: {exc}", ["http error"]
            results.append(
                {
                    "id": case["id"],
                    "split": split,
                    "pass": passed,
                    "invalid": invalid,
                    "detail": detail,
                }
            )
            print(
                f"{'PASS' if passed else 'FAIL'} {case['id']} {detail[:110]}",
                flush=True,
            )
    wall = time.time() - t0
    by_split: dict[str, list] = {}
    for r in results:
        by_split.setdefault(r["split"], []).append(r)
    summary = {
        "n": len(results),
        "accuracy": round(100 * sum(r["pass"] for r in results) / max(1, len(results)), 1),
        "invalid_tool_calls": sum(len(r["invalid"]) for r in results),
        "by_split": {
            s: {
                "n": len(v),
                "acc": round(100 * sum(x["pass"] for x in v) / max(1, len(v)), 1),
            }
            for s, v in by_split.items()
        },
        "wall_s": round(wall, 1),
        "requests": agg["requests"],
        "prompt_tokens": agg["prompt_tokens"],
        "completion_tokens": agg["completion_tokens"],
        "cached_tokens": agg["cached_tokens"],
        "cache_hit_pct": round(
            100 * agg["cached_tokens"] / max(1, agg["prompt_tokens"]), 1
        ),
        "mean_ttft_s": round(agg["ttft_sum"] / max(1, agg["requests"]), 3),
        "mean_decode_tps": round(
            sum(agg["decode_tps"]) / max(1, len(agg["decode_tps"])), 2
        ),
        "thinking": "on" if THINKING else "off",
    }
    print("RESULT bfcl: " + json.dumps(summary), flush=True)
    if OUT:
        Path(OUT).parent.mkdir(parents=True, exist_ok=True)
        Path(OUT).write_text(json.dumps({"summary": summary, "results": results}, indent=2))
        print(f"wrote {OUT}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

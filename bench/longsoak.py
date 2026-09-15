#!/usr/bin/env python3
"""A2/A3/A4 long soak: mixed agentic, abort and prose traffic with detectors.

    SOAK_SECONDS=86400 CONTAINER=qwen38-flash-next OUT=results/x/longsoak.json \
        python3 bench/longsoak.py

Traffic (three client threads, so one of the four server slots stays free):
  agentic   growing tool-calling sessions, thinking alternating per session,
            rotated at ~AGENT_CTX tokens
  mix       prose EN/ES and code, thinking on/off, plus client disconnects in
            the middle of a streamed decode
  prefix    A3 reproducer: a fixed shared document prefix plus a fresh
            multi-chunk suffix, aborted mid chunked prefill by closing the
            socket, then the shared prefix resent with a question

Detectors, every SAMPLE_SECONDS with traffic paused:
  A2  a fixed greedy decode probe (tok/s and the server's `accept len` for the
      probe window) plus every traffic `accept len` logged since the last
      sample; decay onset = first sample whose probe accept length falls below
      DECAY_ACCEPT, or whose probe tok/s falls below DECAY_RATIO of the
      first hour's median
  A3  any /generate output whose first token id is >= FIRST_BAD_ID (the
      tokenizer's added tokens end at 248076; sglang#38319 emits 248319); the
      prefix probe (real LongBench-v2 text) drifting from its first answer; and,
      at each sample, a cache-hit answer that differs from the post-flush_cache
      answer while the latter matches the reference ("wrong until flush")
  A4  request ids that keep emitting outputs after the client disconnected
      (`state was deleted in TokenizerManager`, counted per rid), and running
      requests reported by /metrics while this client has nothing in flight
"""
from __future__ import annotations

import http.client
import json
import os
import random
import re
import statistics
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from client import base, model  # noqa: E402

SECONDS = float(os.environ.get("SOAK_SECONDS", "86400"))
SAMPLE_SECONDS = float(os.environ.get("SAMPLE_SECONDS", "600"))
CONTAINER = os.environ.get("CONTAINER", "qwen38-flash-next")
OUT = os.environ.get("OUT", "")
FIRST_BAD_ID = int(os.environ.get("FIRST_BAD_ID", "248077"))
AGENT_CTX = int(os.environ.get("AGENT_CTX", "24000"))
PREFIX_EVERY = float(os.environ.get("PREFIX_EVERY", "120"))
DECAY_ACCEPT = float(os.environ.get("DECAY_ACCEPT", "1.5"))
DECAY_RATIO = float(os.environ.get("DECAY_RATIO", "0.6"))
REQUEST_TIMEOUT = float(os.environ.get("REQUEST_TIMEOUT", "900"))
PREFIX_CHARS = int(os.environ.get("PREFIX_CHARS", "24000"))
LONGBENCH_JSON = os.environ.get("LONGBENCH_JSON", os.path.expanduser(
    "~/ai/cache/huggingface/hub/datasets--THUDM--LongBench-v2/snapshots/"
    "2b48e494f2c7a2f0af81aae178e05c7e1dde0fe9/data.json"))

PROBE_PROMPT = "Write a Python function that merges two sorted lists. Code only."
WORDS = ("ledger", "harbor", "copper", "meadow", "signal", "vector", "lantern", "orbit",
         "granite", "willow", "cascade", "ember", "falcon", "quartz", "summit", "tundra",
         "compile", "cache", "thread", "socket", "buffer", "kernel", "packet", "schema")
MIX = (
    ("prose_en", "Explain in three sentences how a statute of limitations works.", False),
    ("prose_es", "Explica en tres frases que es la prescripcion de una sancion administrativa.", False),
    ("prose_en_think", "In two sentences, why do bridges have expansion joints?", True),
    ("code", PROBE_PROMPT, False),
    ("code_think", "Write a Python function that checks whether a string is a palindrome.", True),
)
TOOLS = [{"type": "function", "function": {
    "name": "search_docs", "description": "Search the internal document corpus.",
    "parameters": {"type": "object", "properties": {"query": {"type": "string"}},
                   "required": ["query"]}}}]

lock = threading.Lock()
paused = threading.Event()
stop = threading.Event()
inflight = [0]
counters = {"requests": 0, "ok": 0, "client_aborts": 0, "errors": 0, "invalid_tool_calls": 0,
            "a3_bad_first_token": 0, "a3_probe_sends": 0, "a3_probe_drift": 0,
            "a3_fixed_by_flush": 0, "agent_sessions": 0}
events: list[dict] = []
state: dict = {}


def bump(key, n=1):
    with lock:
        counters[key] += n


def event(kind, **detail):
    with lock:
        events.append({"t": round(time.time(), 1), "kind": kind, **detail})
    print("EVENT", kind, json.dumps(detail)[:300], flush=True)


class Busy:
    """Counts in-flight requests and holds new ones while a sample runs."""

    def __enter__(self):
        while paused.is_set() and not stop.is_set():
            time.sleep(0.5)
        with lock:
            inflight[0] += 1
            counters["requests"] += 1

    def __exit__(self, *exc):
        with lock:
            inflight[0] -= 1


def post(path, body, timeout=REQUEST_TIMEOUT):
    req = urllib.request.Request(base() + path, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.load(resp)


def stream_chat(body, on_chunk=None, timeout=REQUEST_TIMEOUT):
    """Streamed chat; on_chunk(i) returning True closes the socket (a client abort)."""
    body = {**body, "model": model(), "stream": True, "stream_options": {"include_usage": True}}
    req = urllib.request.Request(base() + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    out = {"content": "", "tool_args": {}, "usage": {}, "aborted": False, "chunks": 0}
    t0 = time.perf_counter()
    first = None
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        for raw in resp:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:") or line[5:].strip() == "[DONE]":
                continue
            ev = json.loads(line[5:])
            out["usage"] = ev.get("usage") or out["usage"]
            for ch in ev.get("choices") or []:
                delta = ch.get("delta") or {}
                piece = (delta.get("content") or "") + (delta.get("reasoning_content") or "")
                for tc in delta.get("tool_calls") or []:
                    slot = out["tool_args"].setdefault(tc.get("index", 0), "")
                    out["tool_args"][tc.get("index", 0)] = slot + ((tc.get("function") or {}).get("arguments") or "")
                if piece or delta.get("tool_calls"):
                    out["chunks"] += 1
                    first = first or time.perf_counter()
                    out["content"] += delta.get("content") or ""
                    if on_chunk and on_chunk(out["chunks"]):
                        out["aborted"] = True
                        return out
    out["seconds"] = time.perf_counter() - t0
    ct = (out["usage"] or {}).get("completion_tokens") or 0
    out["decode_tps"] = (ct - 1) / max(out["seconds"] - (first - t0 if first else 0), 1e-3) if ct > 1 else 0.0
    return out


def generate(text, max_new_tokens=16):
    data = post("/generate", {"text": text,
                              "sampling_params": {"max_new_tokens": max_new_tokens, "temperature": 0}})
    output = data.get("output_ids") or []
    meta = data.get("meta_info") or {}
    if output and output[0] >= FIRST_BAD_ID:
        bump("a3_bad_first_token")
        event("a3_bad_first_token", first=output[0], prompt_tokens=meta.get("prompt_tokens"),
              cached=meta.get("cached_tokens"))
    return output


def abort_generate(text, after_s):
    """Send a /generate and close the socket after after_s, mid prefill."""
    url = urllib.parse.urlparse(base())
    conn = http.client.HTTPConnection(url.hostname, url.port or 80, timeout=REQUEST_TIMEOUT)
    body = json.dumps({"text": text, "stream": True,
                       "sampling_params": {"max_new_tokens": 64, "temperature": 0}})
    conn.request("POST", "/generate", body=body, headers={"Content-Type": "application/json"})
    time.sleep(after_s)
    conn.close()


def agentic_worker(rng):
    while not stop.is_set():
        thinking = counters["agent_sessions"] % 2 == 1
        bump("agent_sessions")
        msgs = [{"role": "system", "content": "You are a careful ops agent. Use tools when useful."}]
        ctx, turn = 0, 0
        while ctx < AGENT_CTX and not stop.is_set():
            turn += 1
            msgs.append({"role": "user", "content": f"Look into {rng.choice(WORDS)} {rng.choice(WORDS)} (step {turn})."})
            try:
                with Busy():
                    r = stream_chat({"messages": msgs, "tools": TOOLS, "max_tokens": 768, "temperature": 0,
                                     "chat_template_kwargs": {"enable_thinking": thinking}})
                bump("ok")
            except Exception as exc:  # noqa: BLE001
                bump("errors")
                event("agentic_error", error=f"{type(exc).__name__}: {exc}"[:200])
                break
            ctx = (r["usage"] or {}).get("prompt_tokens") or ctx
            finish_len = ((r["usage"] or {}).get("completion_tokens") or 0) >= 768
            if r["tool_args"]:
                calls = []
                for idx, args in sorted(r["tool_args"].items()):
                    try:
                        json.loads(args or "{}")
                    except ValueError:
                        if not finish_len:
                            bump("invalid_tool_calls")
                            event("invalid_tool_call", args=args[:200])
                        args = "{}"
                    calls.append({"id": f"c{turn}_{idx}", "type": "function",
                                  "function": {"name": "search_docs", "arguments": args or "{}"}})
                msgs.append({"role": "assistant", "content": r["content"], "tool_calls": calls})
                for c in calls:
                    body = " ".join(rng.choice(WORDS) for _ in range(rng.randrange(150, 450)))
                    msgs.append({"role": "tool", "tool_call_id": c["id"], "content": f"hit: {body}"})
            else:
                msgs.append({"role": "assistant", "content": r["content"] or "ok"})


def mix_worker(rng):
    while not stop.is_set():
        name, prompt, thinking = rng.choice(MIX)
        abort_after = rng.randrange(4, 60) if rng.random() < 0.3 else None
        try:
            with Busy():
                r = stream_chat({"messages": [{"role": "user", "content": prompt}],
                                 "max_tokens": 600, "temperature": 0.7,
                                 "chat_template_kwargs": {"enable_thinking": thinking}},
                                on_chunk=(lambda i: i >= abort_after) if abort_after else None)
            bump("client_aborts" if r["aborted"] else "ok")
        except Exception as exc:  # noqa: BLE001
            bump("errors")
            event("mix_error", name=name, error=f"{type(exc).__name__}: {exc}"[:200])


def longbench_texts():
    """Real documents (LongBench-v2 contexts) so greedy probes are confident."""
    docs = json.load(open(LONGBENCH_JSON))
    return [d["context"] for d in docs if 60000 <= len(d["context"]) <= 400000]


PROBE_SUFFIX = ("\n\nQuestion: In one short sentence, what is the document above "
                "mainly about?\nAnswer:")


def prefix_worker(rng):
    # One fixed ~6k-token document prefix for the whole soak, shared through
    # the radix cache; each round appends a fresh 8-16k-token suffix from
    # another document and aborts it mid chunked prefill.
    texts = longbench_texts()
    prefix = texts[rng.randrange(len(texts))][:PREFIX_CHARS]
    state["prefix"] = prefix
    while not stop.is_set():
        try:
            other = texts[rng.randrange(len(texts))]
            start = rng.randrange(0, max(1, len(other) - 64000))
            suffix = other[start:start + rng.randrange(32000, 64000)]
            with Busy():
                abort_generate(prefix + "\n\n" + suffix, rng.uniform(0.5, 4.0))
            bump("client_aborts")
            with Busy():
                out = generate(prefix + PROBE_SUFFIX)
            bump("ok")
            bump("a3_probe_sends")
            if state.get("reference") is not None and out != state["reference"]:
                bump("a3_probe_drift")
                event("a3_probe_drift", reference=state["reference"], got=out)
        except Exception as exc:  # noqa: BLE001
            bump("errors")
            event("prefix_error", error=f"{type(exc).__name__}: {exc}"[:200])
        stop.wait(PREFIX_EVERY)


def flush_cache():
    for _ in range(30):
        req = urllib.request.Request(base() + "/flush_cache", method="POST")
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                if resp.status == 200:
                    return True
        except urllib.error.HTTPError:
            pass
        time.sleep(2)
    return False


def metrics():
    want = ("sglang:num_running_reqs", "sglang:num_queue_reqs", "sglang:spec_accept_length",
            "sglang:cache_hit_rate", "sglang:token_usage")
    out = {}
    with urllib.request.urlopen(base() + "/metrics", timeout=10) as resp:
        for line in resp.read().decode("utf-8", "replace").splitlines():
            if line.startswith("#") or " " not in line:
                continue
            key, _, val = line.rpartition(" ")
            name = key.split("{", 1)[0]
            if name in want:
                out[name] = out.get(name, 0.0) + float(val)
    return out


def server_log(since):
    proc = subprocess.run(["docker", "logs", "--since", since, CONTAINER],
                          capture_output=True, text=True, timeout=120)
    return proc.stdout + proc.stderr


def accept_lens(text):
    return [float(x) for x in re.findall(r"accept len: ([0-9.]+)", text)]


def sample(index, since, start):
    paused.set()
    t_wait = time.monotonic()
    while inflight[0] and time.monotonic() - t_wait < 600:
        time.sleep(0.5)
    row = {"i": index, "t": round(time.time(), 1),
           "elapsed_h": round((time.monotonic() - start) / 3600, 3),
           "drain_s": round(time.monotonic() - t_wait, 1),
           "drained": inflight[0] == 0}
    try:
        time.sleep(5)
        m = metrics()
        row["idle_running_reqs"] = m.get("sglang:num_running_reqs", 0.0)
        row["idle_queue_reqs"] = m.get("sglang:num_queue_reqs", 0.0)
        # A request the server still runs while this client has nothing in
        # flight is a disconnect that has not been reaped: time how long.
        if row["idle_running_reqs"] or row["idle_queue_reqs"]:
            t_zombie = time.monotonic()
            while time.monotonic() - t_zombie < 300:
                m = metrics()
                if not m.get("sglang:num_running_reqs") and not m.get("sglang:num_queue_reqs"):
                    break
                time.sleep(2)
            row["zombie_clear_s"] = round(time.monotonic() - t_zombie, 1)
            row["zombie_stuck"] = bool(m.get("sglang:num_running_reqs") or m.get("sglang:num_queue_reqs"))
        # A3: the #38319 signature is a cached prefix that answers wrongly until
        # flush_cache. Compare the cache-hit answer with a post-flush answer.
        if state.get("prefix") is not None:
            hit = generate(state["prefix"] + PROBE_SUFFIX)
            flushed = flush_cache()
            miss = generate(state["prefix"] + PROBE_SUFFIX)
            if state.get("reference") is None and flushed:
                # The reference is the first answer computed from an empty cache.
                state["reference"] = miss
            row["a3_flushed"] = flushed
            row["a3_hit_equals_miss"] = hit == miss
            row["a3_hit_equals_reference"] = hit == state["reference"]
            row["a3_miss_equals_reference"] = miss == state["reference"]
            if flushed and hit != miss and miss == state["reference"]:
                bump("a3_fixed_by_flush")
                event("a3_fixed_by_flush", hit=hit, miss=miss)
        probe_since = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
        rates = []
        for _ in range(2):
            r = stream_chat({"messages": [{"role": "user", "content": PROBE_PROMPT}],
                             "max_tokens": 256, "temperature": 0,
                             "chat_template_kwargs": {"enable_thinking": False}})
            rates.append(round(r["decode_tps"], 2))
        probe_acc = accept_lens(server_log(probe_since))
        row["probe_tok_s"] = statistics.median(rates)
        row["probe_accept_len"] = round(statistics.mean(probe_acc), 3) if probe_acc else None
        text = server_log(since)
        traffic_acc = accept_lens(text)
        row["traffic_accept_len_mean"] = round(statistics.mean(traffic_acc), 3) if traffic_acc else None
        row["traffic_accept_len_p10"] = (round(sorted(traffic_acc)[len(traffic_acc) // 10], 3)
                                         if traffic_acc else None)
        zombies = re.findall(r"rid='([0-9a-f]+)' but the state was deleted in TokenizerManager", text)
        row["zombie_log_lines"] = len(zombies)
        row["zombie_rids"] = len(set(zombies))
        row["zombie_outputs_max"] = max((zombies.count(r) for r in set(zombies)), default=0)
        row["cached_token_sum"] = sum(int(x) for x in re.findall(r"#cached-token: (\d+)", text))
        row["server_errors"] = len(re.findall(r"Traceback|illegal memory access|CUDA error", text))
    except Exception as exc:  # noqa: BLE001
        row["sample_error"] = f"{type(exc).__name__}: {exc}"[:200]
    finally:
        paused.clear()
    with lock:
        row.update({k: counters[k] for k in counters})
    print("SAMPLE", json.dumps(row), flush=True)
    return row


def main() -> int:
    seed = int(os.environ.get("SEED", str(time.time_ns() % 2**31)))
    threads = [threading.Thread(target=fn, args=(random.Random(seed + i),), daemon=True)
               for i, fn in enumerate((agentic_worker, mix_worker, prefix_worker))]
    start = time.monotonic()
    since = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
    for th in threads:
        th.start()
    rows = []
    try:
        while time.monotonic() - start < SECONDS:
            stop.wait(min(SAMPLE_SECONDS, max(1.0, SECONDS - (time.monotonic() - start))))
            next_since = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
            rows.append(sample(len(rows), since, start))
            since = next_since
            write(rows, start, seed, final=False)
    finally:
        stop.set()
        paused.clear()
        for th in threads:
            th.join(timeout=REQUEST_TIMEOUT)
    report = write(rows, start, seed, final=True)
    print("RESULT longsoak:", json.dumps(report["summary"]), flush=True)
    return 0 if report["failed"] == 0 else 1


def write(rows, start, seed, final):
    good = [r for r in rows if r.get("probe_tok_s")]
    hour = [r["probe_tok_s"] for r in good if r["elapsed_h"] <= 1.0]
    ref = statistics.median(hour) if hour else None
    onset = None
    for r in good:
        low_acc = r.get("probe_accept_len") is not None and r["probe_accept_len"] < DECAY_ACCEPT
        slow = ref and r["probe_tok_s"] < DECAY_RATIO * ref
        if low_acc or slow:
            onset = {"sample": r["i"], "hours": r["elapsed_h"],
                     "probe_tok_s": r["probe_tok_s"], "probe_accept_len": r.get("probe_accept_len")}
            break
    with lock:
        c = dict(counters)
    summary = {
        **c, "seed": seed, "seconds": round(time.monotonic() - start, 1), "target_seconds": SECONDS,
        "samples": len(rows), "first_hour_probe_tok_s": ref, "decay_onset": onset,
        "zombie_log_lines": sum(r.get("zombie_log_lines", 0) for r in rows),
        "zombie_rids": sum(r.get("zombie_rids", 0) for r in rows),
        "zombie_outputs_max": max((r.get("zombie_outputs_max", 0) for r in rows), default=0),
        "idle_running_max": max((r.get("idle_running_reqs", 0) for r in rows), default=0),
        "zombie_stuck_samples": sum(1 for r in rows if r.get("zombie_stuck")),
        "zombie_clear_s_max": max((r.get("zombie_clear_s", 0) for r in rows), default=0),
        "a3_hit_ne_miss_samples": sum(1 for r in rows if r.get("a3_hit_equals_miss") is False),
        "server_errors": sum(r.get("server_errors", 0) for r in rows),
        "sample_errors": sum(1 for r in rows if "sample_error" in r),
        "final": final,
    }
    failed = (c["errors"] + c["a3_bad_first_token"] + c["a3_fixed_by_flush"] + summary["server_errors"]
              + summary["sample_errors"] + (0 if rows else 1))
    report = {"summary": summary, "results": rows, "events": events, "failed": failed}
    if OUT:
        Path(OUT).write_text(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    raise SystemExit(main())

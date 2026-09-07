#!/usr/bin/env python3
"""Tiny OpenAI-compat helper used by decode/quality/longctx benches."""
from __future__ import annotations

import json
import signal
from contextlib import contextmanager
from functools import wraps
import os
import time
import urllib.error
import urllib.request


@contextmanager
def deadline(seconds):
    """A total wall-clock deadline, including a stream that keeps sending bytes."""
    def expired(signum, frame):
        raise TimeoutError("request wall-clock deadline exceeded")
    old = signal.signal(signal.SIGALRM, expired)
    previous = signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, *previous)
        signal.signal(signal.SIGALRM, old)


def bounded_request(fn):
    @wraps(fn)
    def wrapped(*args, **kwargs):
        seconds = float(os.environ.get("REQUEST_TIMEOUT", "300"))
        if "timeout" in kwargs:
            seconds = min(seconds, float(kwargs["timeout"]))
        if seconds <= 0:
            raise ValueError("REQUEST_TIMEOUT must be positive")
        kwargs["timeout"] = seconds
        with deadline(seconds):
            return fn(*args, **kwargs)
    return wrapped


def check_prompt(body):
    limit = os.environ.get("MAX_PROMPT_TOKENS")
    if not limit:
        return
    # U0 preflight renders/tokenizes the chat template before generation.
    # Image expansion is not counted here; U0 uses only its fixed small image.
    req = urllib.request.Request(base() + "/tokenize", data=json.dumps({
        "model": body["model"], "messages": body["messages"],
        **{k: body[k] for k in ("tools", "chat_template_kwargs", "reasoning_effort", "tool_choice") if k in body}
    }).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=10) as response:
        data = json.load(response)
    count = data.get("count", data.get("num_tokens"))
    if count is None and isinstance(data.get("tokens"), list):
        count = len(data["tokens"])
    if count is None:
        raise RuntimeError("tokenize did not report a token count; refusing unchecked U0 request")
    if count > int(limit):
        raise ValueError("request exceeds U0 short-context ceiling")


def base() -> str:
    return os.environ.get("BASE", "http://127.0.0.1:30000").rstrip("/")


def model() -> str:
    return os.environ.get("MODEL", "qwen38-flash-next-nvfp4-mtp")


@bounded_request
def chat(
    messages,
    *,
    max_tokens: int = 512,
    temperature: float | None = 0.7,
    thinking: bool | None = None,
    reasoning_effort: str | None = None,
    tools=None,
    extra=None,
    timeout: int = 900,
):
    body = {
        "model": model(),
        "messages": messages,
        "max_tokens": max_tokens,
    }
    if temperature is not None:
        body["temperature"] = temperature
    kwargs = {}
    if thinking is not None:
        kwargs["enable_thinking"] = thinking
    if reasoning_effort is not None:
        kwargs["reasoning_effort"] = reasoning_effort
    if kwargs:
        body["chat_template_kwargs"] = kwargs
    if tools is not None:
        body["tools"] = tools
    if extra:
        body.update(extra)
    check_prompt(body)
    req = urllib.request.Request(
        base() + "/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read()
            code = resp.status
    except urllib.error.HTTPError as e:
        err = e.read().decode("utf-8", "replace")[:2000]
        raise SystemExit(f"HTTP {e.code}: {err}") from e
    dt = time.perf_counter() - t0
    data = json.loads(raw)
    choice = data["choices"][0]
    msg = choice.get("message") or {}
    usage = data.get("usage") or {}
    return {
        "http": code,
        "seconds": dt,
        "content": (msg.get("content") or "") or "",
        "reasoning": (msg.get("reasoning_content") or "") or "",
        "tool_calls": msg.get("tool_calls") or [],
        "finish": choice.get("finish_reason"),
        "prompt_tokens": usage.get("prompt_tokens") or 0,
        "completion_tokens": usage.get("completion_tokens") or 0,
        "reasoning_tokens": (usage.get("completion_tokens_details") or {}).get(
            "reasoning_tokens"
        )
        or 0,
        "raw": data,
    }


def tok_s(r: dict) -> float:
    ct = r["completion_tokens"]
    return ct / r["seconds"] if r["seconds"] and ct else 0.0


@bounded_request
def chat_stream(
    messages,
    *,
    max_tokens: int = 512,
    temperature: float | None = 0.7,
    thinking: bool | None = None,
    reasoning_effort: str | None = None,
    tools=None,
    extra=None,
    timeout: int = 1800,
):
    """Streaming chat. Returns the same shape as chat() plus TTFT and rates.

    ttft_s      time to the first streamed token (prefill + first decode step)
    prefill_tps prompt_tokens / ttft_s   (upper bound on prefill rate)
    decode_tps  completion_tokens / (total - ttft)
    """
    body = {
        "model": model(),
        "messages": messages,
        "max_tokens": max_tokens,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if temperature is not None:
        body["temperature"] = temperature
    kwargs = {}
    if thinking is not None:
        kwargs["enable_thinking"] = thinking
    if reasoning_effort is not None:
        kwargs["reasoning_effort"] = reasoning_effort
    if kwargs:
        body["chat_template_kwargs"] = kwargs
    if tools is not None:
        body["tools"] = tools
    if extra:
        body.update(extra)
    check_prompt(body)
    req = urllib.request.Request(
        base() + "/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    content, reasoning = [], []
    tool_calls: list = []
    finish = None
    usage: dict = {}
    ttft = None
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            for raw in resp:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    break
                try:
                    ev = json.loads(payload)
                except json.JSONDecodeError:
                    continue
                if ev.get("usage"):
                    usage = ev["usage"]
                for ch in ev.get("choices") or []:
                    delta = ch.get("delta") or {}
                    piece = delta.get("content") or ""
                    think = delta.get("reasoning_content") or ""
                    tcs = delta.get("tool_calls")
                    if tcs:
                        tool_calls.extend(tcs)
                    # A pure tool call streams no content, so time the first
                    # delta of any kind or TTFT collapses onto the total.
                    if (piece or think or tcs) and ttft is None:
                        ttft = time.perf_counter() - t0
                    content.append(piece)
                    reasoning.append(think)
                    if ch.get("finish_reason"):
                        finish = ch["finish_reason"]
    except urllib.error.HTTPError as e:
        err = e.read().decode("utf-8", "replace")[:2000]
        raise SystemExit(f"HTTP {e.code}: {err}") from e
    total = time.perf_counter() - t0
    ct = usage.get("completion_tokens") or 0
    pt = usage.get("prompt_tokens") or 0
    cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens") or 0
    ttft = ttft if ttft is not None else total
    dec_window = max(total - ttft, 1e-3)
    return {
        "seconds": total,
        "ttft_s": ttft,
        "content": "".join(content),
        "reasoning": "".join(reasoning),
        "tool_calls": tool_calls,
        "finish": finish,
        "prompt_tokens": pt,
        "completion_tokens": ct,
        "cached_tokens": cached,
        "cache_hit_pct": round(100.0 * cached / pt, 2) if pt else 0.0,
        "reasoning_tokens": (usage.get("completion_tokens_details") or {}).get(
            "reasoning_tokens"
        )
        or 0,
        "prefill_tps": round(pt / ttft, 1) if ttft else 0.0,
        "decode_tps": round((ct - 1) / dec_window, 2) if ct > 1 else 0.0,
        "usage": usage,
    }


def server_metrics() -> dict:
    """Scrape a few Prometheus counters SGLang exposes with --enable-metrics."""
    want = (
        "sglang:num_used_tokens",
        "sglang:cached_tokens_total",
        "sglang:prompt_tokens_total",
        "sglang:generation_tokens_total",
        "sglang:spec_accept_length",
        "sglang:cache_hit_rate",
        "sglang:token_usage",
    )
    out: dict[str, float] = {}
    try:
        with urllib.request.urlopen(base() + "/metrics", timeout=10) as resp:
            for line in resp.read().decode("utf-8", "replace").splitlines():
                if line.startswith("#") or " " not in line:
                    continue
                key, _, val = line.rpartition(" ")
                name = key.split("{", 1)[0]
                if name in want:
                    try:
                        out[name] = float(val)
                    except ValueError:
                        pass
    except Exception:  # noqa: BLE001
        pass
    return out

#!/usr/bin/env python3
"""Tiny OpenAI-compat helper used by decode/quality/longctx benches."""
from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request


def base() -> str:
    return os.environ.get("BASE", "http://127.0.0.1:30000").rstrip("/")


def model() -> str:
    return os.environ.get("MODEL", "qwen38-flash-next-nvfp4-mtp")


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

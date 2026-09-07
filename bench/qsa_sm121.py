#!/usr/bin/env python3
"""Isolated SM121 QSA decode checks. No model load.

Run inside the patched image (scripts/bench_qsa_kernels.sh). Answers:
  1) TRT-LLM sparse decode stays gated off on SM121 (no XQA widening)
  2) packed varlen resolves to the 2026-08-28 Triton kernel
  3) Triton (and KDA, if present) match a PyTorch FP32 reference
  4) CUDA-graph replay with changed device-side cu_seqlens is correct

Shapes are the real TP1 contract: 24 Q heads, 2 KV heads, head dim 256.
"""
from __future__ import annotations

import sys

import torch


def relative_l2(actual: torch.Tensor, reference: torch.Tensor) -> float:
    ref = reference.float()
    diff = actual.float() - ref
    denom = torch.linalg.vector_norm(ref).item()
    if denom == 0:
        return torch.linalg.vector_norm(diff).item()
    return torch.linalg.vector_norm(diff).item() / denom


def packed(batch: int, kv_len: int, device: torch.device):
    hq, hkv, dim = 24, 2, 256
    scale = dim ** -0.5
    q = torch.randn(batch, hq, dim, device=device, dtype=torch.bfloat16)
    k = torch.randn(batch * kv_len, hkv, dim, device=device, dtype=torch.bfloat16)
    v = torch.randn_like(k)
    cu_q = torch.arange(0, batch + 1, device=device, dtype=torch.int32)
    cu_k = torch.arange(0, batch + 1, device=device, dtype=torch.int32) * kv_len
    return q.contiguous(), k.contiguous(), v.contiguous(), cu_q.contiguous(), cu_k.contiguous(), scale


def reference(q, k, v, cu_q, cu_k, scale):
    batch, hq, dim = q.shape
    hkv = k.shape[1]
    group = hq // hkv
    out = torch.empty_like(q, dtype=torch.float32)
    qf, kf, vf = q.float(), k.float(), v.float()
    for b in range(batch):
        qi = int(cu_q[b].item())
        ks, ke = int(cu_k[b].item()), int(cu_k[b + 1].item())
        keys = kf[ks:ke]
        values = vf[ks:ke]
        for h in range(hq):
            scores = (qf[qi, h] @ keys[:, h // group].T) * scale
            weights = torch.softmax(scores, dim=0)
            out[qi, h] = weights @ values[:, h // group]
    return out.to(q.dtype)


def replay(fn, q, k, v, cu_q, cu_k, scale, kv_len):
    out = torch.empty_like(q)
    stream = torch.cuda.Stream()
    graph = torch.cuda.CUDAGraph()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            out.copy_(fn(q, k, v, cu_q, cu_k, scale, kv_len))
        with torch.cuda.graph(graph):
            out.copy_(fn(q, k, v, cu_q, cu_k, scale, kv_len))
    torch.cuda.current_stream().wait_stream(stream)
    new_k = torch.tensor([0, 64, 192], device=q.device, dtype=torch.int32)
    cu_k.copy_(new_k)
    graph.replay()
    eager = fn(q, k, v, cu_q, cu_k, scale, kv_len)
    return relative_l2(out, eager)


def main() -> int:
    if not torch.cuda.is_available():
        print("RESULT qsa_sm121: FAIL no CUDA")
        return 1
    capability = torch.cuda.get_device_capability()
    print("capability:", capability, flush=True)
    if capability != (12, 1):
        print("RESULT qsa_sm121: FAIL expected SM121")
        return 1

    from sglang.srt.layers.attention.qwen_sparse_attn_backend import (
        _resolve_flash_attn_varlen_func,
        _resolve_trtllm_sparse_decode,
    )
    from sglang.srt.layers.attention.qsa.sm121_varlen import qsa_sm121_varlen_attention

    trtllm = _resolve_trtllm_sparse_decode()
    varlen = _resolve_flash_attn_varlen_func()
    varlen_name = getattr(varlen, "__name__", type(varlen).__name__)
    print("trtllm_sparse_decode:", None if trtllm is None else type(trtllm).__name__, flush=True)
    print("varlen_func:", varlen_name, flush=True)
    if trtllm is not None:
        print("RESULT qsa_sm121: FAIL TRT-LLM gate still open on SM121")
        return 1
    if varlen_name not in ("qsa_sm121_varlen_attention", "_qsa_sm121_triton_varlen"):
        print("RESULT qsa_sm121: FAIL varlen resolver is not the Triton kernel")
        return 1

    kda_fn = None
    try:
        from sglang.kernels.kda_kernels.qwen38_qsa_sm121 import qwen38_qsa_sm121
        kda_fn = qwen38_qsa_sm121
        print("kda_package: present", flush=True)
    except ImportError:
        print("kda_package: absent", flush=True)

    device = torch.device("cuda")
    worst = 0.0
    for batch, kv_len in ((1, 128), (1, 512), (4, 256)):
        q, k, v, cu_q, cu_k, scale = packed(batch, kv_len, device)
        ref = reference(q, k, v, cu_q, cu_k, scale)
        cases = [
            ("triton", qsa_sm121_varlen_attention(
                q, k, v, cu_q, cu_k, max_seqlen_q=1, max_seqlen_k=kv_len, softmax_scale=scale
            )),
            ("wrapper", varlen(q, k, v, cu_q, cu_k, 1, kv_len, softmax_scale=scale, causal=True)),
        ]
        if kda_fn is not None:
            cases.append(("kda", kda_fn(q, k, v, cu_q, cu_k, kv_len, scale)))
        for name, out in cases:
            err = relative_l2(out, ref)
            worst = max(worst, err)
            print(f"rel_l2 {name} bs={batch} kv={kv_len}: {err:.6f}", flush=True)
            if err > 2.5e-3:
                print(f"RESULT qsa_sm121: FAIL {name} rel_l2 {err}")
                return 1

    q, k, v, cu_q, cu_k, scale = packed(2, 256, device)

    def triton_call(q, k, v, cu_q, cu_k, scale, kv_len):
        return qsa_sm121_varlen_attention(
            q, k, v, cu_q, cu_k, max_seqlen_q=1, max_seqlen_k=kv_len, softmax_scale=scale
        )

    err = replay(triton_call, q, k, v, cu_q.clone(), cu_k.clone(), scale, 256)
    print(f"rel_l2 cudagraph_replay_triton: {err:.6f}", flush=True)
    if err > 2.5e-3:
        print("RESULT qsa_sm121: FAIL cudagraph replay")
        return 1
    if kda_fn is not None:
        def kda_call(q, k, v, cu_q, cu_k, scale, kv_len):
            return kda_fn(q, k, v, cu_q, cu_k, kv_len, scale)
        kda_err = replay(kda_call, q, k, v, cu_q.clone(), cu_k.clone(), scale, 256)
        print(f"rel_l2 cudagraph_replay_kda: {kda_err:.6f}", flush=True)
        if kda_err > 2.5e-3:
            print("RESULT qsa_sm121: FAIL kda cudagraph replay")
            return 1

    print(f"RESULT qsa_sm121: PASS worst_rel_l2={worst:.6f} graph={err:.6f}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"RESULT qsa_sm121: FAIL {type(exc).__name__}: {exc}")
        raise

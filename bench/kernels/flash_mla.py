"""flash_mla_sm120 driver — sparse-MLA prefill (single + dual cache)."""
from __future__ import annotations

import os
import sys
from typing import Any

import torch

# sparse_mla_sm120/tests holds the quantize/dequantize helpers we reuse.
_REPO = os.path.join(os.path.dirname(__file__), "..", "..", "sparse_mla_sm120")
if _REPO not in sys.path:
    sys.path.insert(0, os.path.abspath(_REPO))

from tests.test_decode import quantize_kv_model1, dequantize_kv_model1  # noqa: E402

import flash_mla_sm120  # noqa: E402

from . import register
from ._common import (
    compute_correctness, parse_dtype_string,
    parse_kwargs_string, parse_shape_string, time_kernel,
)

# Use the original (unpatched) function if bench.capture has installed hooks —
# otherwise our timing loop also includes the record() overhead.
_orig = getattr(flash_mla_sm120.interface, "flash_mla_sparse_fwd",
                flash_mla_sm120.flash_mla_sparse_fwd)
if hasattr(_orig, "__wrapped__"):
    _orig = _orig.__wrapped__
flash_mla_sparse_fwd_unwrapped = _orig


def _ref_attn(q, kv_main_dq, idx_main, sm_scale, d_v,
              kv_extra_dq=None, idx_extra=None, attn_sink=None):
    """Unified sparse-MLA reference (single or dual cache, optional sink)."""
    num_tokens, h_q, d_qk = q.shape
    q_f = q.float()

    main_flat = kv_main_dq.view(-1, d_qk).float()
    gathered = main_flat.index_select(0, idx_main.clamp(min=0).view(-1)).view(
        num_tokens, idx_main.size(-1), d_qk)
    invalid = idx_main < 0

    if kv_extra_dq is not None and idx_extra is not None:
        extra_flat = kv_extra_dq.view(-1, d_qk).float()
        gathered_extra = extra_flat.index_select(
            0, idx_extra.clamp(min=0).view(-1)
        ).view(num_tokens, idx_extra.size(-1), d_qk)
        gathered = torch.cat([gathered, gathered_extra], dim=-2)
        invalid = torch.cat([invalid, idx_extra < 0], dim=-1)

    P = torch.einsum("nhd,ntd->nht", q_f, gathered) * sm_scale
    P[invalid.unsqueeze(1).expand_as(P)] = float("-inf")
    lse = torch.logsumexp(P, dim=-1)
    lse_safe = lse.clone()
    lse_safe[lse_safe == float("-inf")] = float("+inf")
    weights = torch.exp(P - lse_safe.unsqueeze(-1))
    out = torch.einsum("nht,ntd->nhd", weights, gathered[..., :d_v]).to(torch.bfloat16)

    if attn_sink is not None:
        sink = attn_sink.float().to(q.device)
        factor = torch.sigmoid(lse - sink.unsqueeze(0))
        out = (out.float() * factor.unsqueeze(-1)).to(torch.bfloat16)
    return out, lse


def _make_cache(shape: tuple, device: str = "cuda"):
    """`shape` is the packed shape (n_blocks, block_size, h_kv, head_bytes=584).
    Returns (packed_uint8, dequant_bf16) so the reference sees the same
    quantized state as the kernel.

    KV magnitudes are O(1) (not the existing test's /10 attenuation) so the
    softmax-weighted output peak stays near unity — this keeps the ULP-distance
    metric `max_atol / (dtype_eps × max(|ref|))` from being dominated by an
    artificially small denominator (which happens when output magnitude
    collapses far below input scale).
    """
    n_blocks, block_size, h_kv, head_bytes = shape
    assert h_kv == 1 and head_bytes == 584, (
        f"unsupported KV layout (only MODEL1 FOOTER 584B/elem): {shape}"
    )
    d_qk = 512
    kv_bf16 = torch.randn(n_blocks, block_size, 1, d_qk,
                          device=device, dtype=torch.bfloat16).clamp(-4, 4)
    packed = quantize_kv_model1(kv_bf16)
    dequant = dequantize_kv_model1(packed)
    return packed, dequant


def _make_indices(n_tokens: int, topk: int, s_kv: int, invalid_tail: int = 5,
                  device: str = "cuda"):
    idx = torch.randint(0, s_kv, (n_tokens, topk),
                        device=device, dtype=torch.int32)
    if invalid_tail > 0:
        idx[:, -invalid_tail:] = -1
    return idx


def _build_inputs(row: dict[str, Any]) -> dict[str, Any]:
    shapes = parse_shape_string(row["shape"])
    kwargs = parse_kwargs_string(row["kwargs"])
    _ = parse_dtype_string(row["dtype"])  # bf16/uint8 are hard-coded in helpers

    q_shape = shapes["q"]
    kv_shape = shapes["kv"]
    idx_shape = shapes["idx"]
    ek_shape = shapes.get("ek")
    eidx_shape = shapes.get("eidx")

    n_tokens, num_heads, d_qk = q_shape
    n_blocks, block_size, _, _ = kv_shape
    s_kv_main = n_blocks * block_size

    q = torch.randn(n_tokens, num_heads, d_qk, device="cuda",
                    dtype=torch.bfloat16).clamp(-4, 4)
    kv_packed, kv_dq = _make_cache(kv_shape)
    indices = _make_indices(n_tokens, idx_shape[-1], s_kv_main)

    extra_k = extra_idx = extra_dq = None
    if ek_shape is not None and eidx_shape is not None:
        extra_k, extra_dq = _make_cache(ek_shape)
        s_kv_extra = ek_shape[0] * ek_shape[1]
        extra_idx = _make_indices(n_tokens, eidx_shape[-1], s_kv_extra)

    attn_sink = None
    if kwargs.get("has_sink", False):
        real = torch.tensor(
            [(-1.0) ** i * (0.5 + 0.1 * (i % 7)) for i in range(num_heads)],
            device="cuda", dtype=torch.float32,
        )
        attn_sink = real

    d_v = int(kwargs.get("d_v", 512))
    sm_scale = d_qk ** -0.5
    return dict(
        q=q, kv_packed=kv_packed, kv_dq=kv_dq, indices=indices,
        extra_k=extra_k, extra_dq=extra_dq, extra_idx=extra_idx,
        attn_sink=attn_sink, sm_scale=sm_scale, d_v=d_v,
        num_heads=num_heads, n_tokens=n_tokens,
    )


_CORRECTNESS_TOKENS_MAX = 256
"""Cap n_tokens for reference computation to keep peak memory bounded.
Reference einsum allocates O(n_tokens × topk_total × d_qk) fp32 — at the
captured prefill chunk size (8K tokens × 640 topk × 512 d_qk = 10 GB),
the dequant+gather references won't fit on a busy GPU."""


def _bench_flash_mla_sparse_fwd(row: dict[str, str], *,
                                n_warmup: int = 20, n_iter: int = 100,
                                check: bool = True,
                                ulp_threshold: float = 8.0,
                                calc_diff_threshold: float = 1e-3,
                                ) -> dict[str, Any]:
    inp = _build_inputs(row)

    def call_full():
        return flash_mla_sparse_fwd_unwrapped(
            inp["q"], inp["kv_packed"], inp["indices"], inp["sm_scale"],
            d_v=inp["d_v"], attn_sink=inp["attn_sink"],
            extra_k_cache=inp["extra_k"],
            extra_indices_in_kvcache=inp["extra_idx"],
        )

    # Correctness on a subsample to keep ref memory tractable.
    correctness = None
    if check:
        n = min(inp["n_tokens"], _CORRECTNESS_TOKENS_MAX)
        q_s = inp["q"][:n].contiguous()
        idx_s = inp["indices"][:n].contiguous()
        eidx_s = inp["extra_idx"][:n].contiguous() if inp["extra_idx"] is not None else None
        out_kernel = flash_mla_sparse_fwd_unwrapped(
            q_s, inp["kv_packed"], idx_s, inp["sm_scale"],
            d_v=inp["d_v"], attn_sink=inp["attn_sink"],
            extra_k_cache=inp["extra_k"],
            extra_indices_in_kvcache=eidx_s,
        )[0]
        out_ref, _ = _ref_attn(
            q_s, inp["kv_dq"], idx_s, inp["sm_scale"], inp["d_v"],
            kv_extra_dq=inp["extra_dq"], idx_extra=eidx_s,
            attn_sink=inp["attn_sink"],
        )
        correctness = compute_correctness(
            out_kernel, out_ref,
            ulp_threshold=ulp_threshold,
            calc_diff_threshold=calc_diff_threshold,
        )
        del out_kernel, out_ref, q_s, idx_s, eidx_s
        torch.cuda.empty_cache()

    timing = time_kernel(call_full, n_warmup=n_warmup, n_iter=n_iter)
    return dict(
        timing=timing,
        correctness=correctness,
    )


@register("flash_mla_sparse_fwd")
def bench_single(row, **opts):
    return _bench_flash_mla_sparse_fwd(row, **opts)


@register("flash_mla_sparse_fwd_dual")
def bench_dual(row, **opts):
    return _bench_flash_mla_sparse_fwd(row, **opts)

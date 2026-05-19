"""DeepGEMM driver — currently covers `fp8_gemm_nt` (plain FP8 GEMM).

Adds for the other DG entries (`m_grouped_fp8_fp4_gemm_nt_contiguous`,
`tf32_hc_prenorm_gemm`, `fp8_fp4_mqa_logits`, etc.) land in follow-ups —
the shape parsing + correctness reference for those needs more careful
handling of grouped / MQA / quant-config semantics.
"""
from __future__ import annotations

from typing import Any

import torch

from . import register
from ._common import (
    compute_correctness, fp8_gemm_perf, parse_kwargs_string,
    parse_shape_string, time_kernel,
)


def _build_fp8_gemm_nt_inputs(row: dict[str, Any]) -> dict[str, Any]:
    shapes = parse_shape_string(row["shape"])
    # shapes = {'lhs': [(M,K), (M, K//128)], 'rhs': [(N,K), (N, K//128)], 'out': (M,N)}
    lhs_shape = shapes["lhs"]
    rhs_shape = shapes["rhs"]
    out_shape = shapes["out"]
    M, K = lhs_shape[0]
    N, K2 = rhs_shape[0]
    assert K == K2 and out_shape == (M, N), \
        f"shape inconsistency: lhs={lhs_shape}, rhs={rhs_shape}, out={out_shape}"

    from vllm.utils.deep_gemm import per_block_cast_to_fp8

    # Random bf16 source tensors; quantize per-block-128 along K.
    lhs_bf16 = (torch.randn(M, K, device="cuda", dtype=torch.bfloat16) / 4).clamp(-2, 2)
    rhs_bf16 = (torch.randn(N, K, device="cuda", dtype=torch.bfloat16) / 4).clamp(-2, 2)

    # per_block_cast_to_fp8 requires M aligned to block_m=128. Pad lhs if not.
    lhs_pad = ((M + 127) // 128) * 128
    if lhs_pad != M:
        lhs_buf = torch.zeros(lhs_pad, K, device="cuda", dtype=torch.bfloat16)
        lhs_buf[:M] = lhs_bf16
        lhs_full_fp8, lhs_full_scale = per_block_cast_to_fp8(lhs_buf, use_ue8m0=True)
        lhs_fp8 = lhs_full_fp8[:M].contiguous()
        lhs_scale = lhs_full_scale[:M].contiguous()
    else:
        lhs_fp8, lhs_scale = per_block_cast_to_fp8(lhs_bf16, use_ue8m0=True)

    rhs_fp8, rhs_scale = per_block_cast_to_fp8(rhs_bf16, use_ue8m0=True)
    out = torch.empty(M, N, device="cuda", dtype=torch.bfloat16)

    return dict(
        lhs_bf16=lhs_bf16, rhs_bf16=rhs_bf16,
        lhs_fp8=lhs_fp8, lhs_scale=lhs_scale,
        rhs_fp8=rhs_fp8, rhs_scale=rhs_scale,
        out=out, M=M, N=N, K=K,
    )


@register("fp8_gemm_nt")
def bench_fp8_gemm_nt(row, *, n_warmup: int = 20, n_iter: int = 100,
                      check: bool = True,
                      ulp_threshold: float = 8.0,
                      calc_diff_threshold: float = 1e-3,
                      ) -> dict[str, Any]:
    inp = _build_fp8_gemm_nt_inputs(row)
    from vllm.utils.deep_gemm import fp8_gemm_nt
    # Bypass our own capture wrapper if it's installed — we don't want timing
    # to include the record() overhead.
    fn = getattr(fp8_gemm_nt, "__wrapped__", fp8_gemm_nt)

    def call():
        return fn((inp["lhs_fp8"], inp["lhs_scale"]),
                  (inp["rhs_fp8"], inp["rhs_scale"]),
                  inp["out"])

    correctness = None
    if check:
        call()  # populates out
        out_ref = (inp["lhs_bf16"].float() @ inp["rhs_bf16"].float().t()
                   ).to(torch.bfloat16)
        correctness = compute_correctness(
            inp["out"], out_ref,
            ulp_threshold=ulp_threshold,
            calc_diff_threshold=calc_diff_threshold,
        )
        del out_ref
        torch.cuda.empty_cache()

    timing = time_kernel(call, n_warmup=n_warmup, n_iter=n_iter)
    perf = fp8_gemm_perf(inp["M"], inp["N"], inp["K"])
    return dict(timing=timing, correctness=correctness, perf=perf)


# ── timing-only drivers for the remaining DG kernels ──────────────────
# Each runs the real kernel at captured shapes with synthetic random inputs;
# correctness reference is non-trivial (FP4 dequant, paged MQA semantics,
# fused norm+gemm sqrsum) and lands in a follow-up. Drivers return
# correctness=None so run_bench reports status=OK_NO_CHECK.


# `m_grouped_fp8_fp4_gemm_nt_contiguous` driver is TODO — rhs_scale is in
# the TMA-aligned int32 UE8M0 layout produced by
# `transform_sf_into_required_layout`. Random int32 doesn't satisfy
# `sf.size(-1) == ceil_div(k_packed, gran_k * 4)` for the layout the kernel
# expects. Needs to go through the actual transform helper.
def _todo_m_grouped_fp8_fp4(row, *, n_warmup: int = 20, n_iter: int = 100,
                             check: bool = True, **_) -> dict[str, Any]:
    shapes = parse_shape_string(row["shape"])
    lhs_shape = shapes["lhs"]
    rhs_shape = shapes["rhs"]
    out_shape = shapes["out"]
    m_idx_shape = shapes["m_indices"]
    M, K = lhs_shape[0]
    G, N, K2 = rhs_shape[0]  # NUM_GROUPS, N, K_packed (K_packed = K since FP4 already int8-packed)
    K_blk_a = lhs_shape[1][1]   # K // 128 for fp8 scale
    K_blk_b = rhs_shape[1][2]   # K // 128 for fp4 scale (innermost of 3-D)

    lhs_fp8 = torch.randn(M, K, device="cuda", dtype=torch.bfloat16).to(torch.float8_e4m3fn)
    lhs_scale = torch.rand(M, K_blk_a, device="cuda", dtype=torch.float32) * 0.1 + 0.01
    # rhs is FP4 packed 2-per-int8 → tensor dim along K is K//2 (= K2 from capture).
    rhs_fp4 = torch.randint(0, 255, (G, N, K2), device="cuda",
                            dtype=torch.uint8).view(torch.int8)
    # rhs_scale is int32 UE8M0-packed (4 UE8M0 → 1 int32) — matches captured dtype.
    rhs_scale = torch.randint(0, 2**30, (G, N, K_blk_b),
                              device="cuda", dtype=torch.int32)
    out = torch.empty(M, N, device="cuda", dtype=torch.bfloat16)
    m_indices = torch.randint(0, G, (M,), device="cuda", dtype=torch.int32)

    from vllm.utils.deep_gemm import m_grouped_fp8_fp4_gemm_nt_contiguous
    fn = getattr(m_grouped_fp8_fp4_gemm_nt_contiguous, "__wrapped__",
                 m_grouped_fp8_fp4_gemm_nt_contiguous)
    def call():
        return fn((lhs_fp8, lhs_scale), (rhs_fp4, rhs_scale), out, m_indices)

    # Warm + sanity invoke (catches arg/shape mismatches early).
    call()
    timing = time_kernel(call, n_warmup=n_warmup, n_iter=n_iter)
    return dict(timing=timing, correctness=None)


@register("tf32_hc_prenorm_gemm")
def bench_tf32_hc_prenorm(row, *, n_warmup: int = 20, n_iter: int = 100,
                          check: bool = True, **_) -> dict[str, Any]:
    shapes = parse_shape_string(row["shape"])
    x_shape = shapes["x"]
    fn_shape = shapes["fn"]
    out_shape = shapes["out"]
    sqrsum_shape = shapes["sqrsum"]
    num_split = out_shape[0]

    x = torch.randn(*x_shape, device="cuda", dtype=torch.bfloat16)
    fn_weight = torch.randn(*fn_shape, device="cuda", dtype=torch.float32)
    out = torch.empty(*out_shape, device="cuda", dtype=torch.float32)
    sqrsum = torch.empty(*sqrsum_shape, device="cuda", dtype=torch.float32)

    from vllm.utils.deep_gemm import tf32_hc_prenorm_gemm
    fn = getattr(tf32_hc_prenorm_gemm, "__wrapped__", tf32_hc_prenorm_gemm)

    def call():
        return fn(x, fn_weight, out, sqrsum, num_split)

    call()
    timing = time_kernel(call, n_warmup=n_warmup, n_iter=n_iter)
    return dict(timing=timing, correctness=None)


@register("fp8_fp4_mqa_logits")
def bench_fp8_fp4_mqa_logits(row, *, n_warmup: int = 20, n_iter: int = 100,
                              check: bool = True, **_) -> dict[str, Any]:
    shapes = parse_shape_string(row["shape"])
    q_shape = shapes["q"]
    kv_shape = shapes["kv"]
    w_shape = shapes["weights"]
    ks_shape = shapes["cu_seqlen_ks"]
    ke_shape = shapes["cu_seqlen_ke"]
    # q = [(S, H, D), None|scale]; kv = [(S_kv, D), (S_kv,)]
    S, H, D = q_shape[0]
    S_kv, D2 = kv_shape[0]

    q_fp8 = torch.randn(S, H, D, device="cuda", dtype=torch.bfloat16).to(torch.float8_e4m3fn)
    q_scale = None  # captured as None
    kv_fp8 = torch.randn(S_kv, D, device="cuda", dtype=torch.bfloat16).to(torch.float8_e4m3fn)
    kv_scale = torch.rand(S_kv, device="cuda", dtype=torch.float32) * 0.1 + 0.01
    weights = torch.rand(*w_shape, device="cuda", dtype=torch.float32)
    # cu_seqlen_*: monotone non-decreasing int32 sequences.
    ks = torch.zeros(S, device="cuda", dtype=torch.int32)
    ke = torch.full((S,), S_kv, device="cuda", dtype=torch.int32)

    from vllm.utils.deep_gemm import fp8_fp4_mqa_logits
    fn = getattr(fp8_fp4_mqa_logits, "__wrapped__", fp8_fp4_mqa_logits)

    def call():
        return fn((q_fp8, q_scale), (kv_fp8, kv_scale), weights, ks, ke, False)

    call()
    timing = time_kernel(call, n_warmup=n_warmup, n_iter=n_iter)
    return dict(timing=timing, correctness=None)


# `fp8_fp4_paged_mqa_logits` driver is TODO — get_paged_mqa_logits_metadata
# needs num_sms + 1 entries with a specific layout we haven't characterised
# yet. Skip registration so run_bench reports SKIP_NO_DRIVER for these rows
# instead of ERROR.
def _todo_fp8_fp4_paged_mqa_logits(row, *, n_warmup: int = 20, n_iter: int = 100,
                                    check: bool = True, **_) -> dict[str, Any]:
    shapes = parse_shape_string(row["shape"])
    kwargs = parse_kwargs_string(row["kwargs"])
    q_shape = shapes["q"]   # [(B, S, H, D), None]
    kv_cache_shape = shapes["kv_cache"]
    w_shape = shapes["weights"]
    ctx_shape = shapes["context_lens"]
    bt_shape = shapes["block_tables"]
    B, S, H, D = q_shape[0]
    n_blocks, block_size = kv_cache_shape[:2]
    max_model_len = int(kwargs.get("max_model_len", 16384))

    q_fp8 = torch.randn(B, S, H, D, device="cuda", dtype=torch.bfloat16).to(torch.float8_e4m3fn)
    q_scale = None
    kv_cache = torch.randint(0, 255, kv_cache_shape, device="cuda", dtype=torch.uint8)
    weights = torch.rand(*w_shape, device="cuda", dtype=torch.float32)
    context_lens = torch.full(ctx_shape, block_size * (bt_shape[1] // 2),
                              device="cuda", dtype=torch.int32)
    block_tables = torch.randint(0, n_blocks, bt_shape, device="cuda", dtype=torch.int32)

    from vllm.utils.deep_gemm import (
        fp8_fp4_paged_mqa_logits, get_paged_mqa_logits_metadata,
    )
    fn = getattr(fp8_fp4_paged_mqa_logits, "__wrapped__", fp8_fp4_paged_mqa_logits)

    schedule_metadata = get_paged_mqa_logits_metadata(
        context_lens, block_size, B
    )

    def call():
        return fn((q_fp8, q_scale), kv_cache, weights, context_lens,
                  block_tables, schedule_metadata, max_model_len, False)

    call()
    timing = time_kernel(call, n_warmup=n_warmup, n_iter=n_iter)
    return dict(timing=timing, correctness=None)

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
    compute_correctness, parse_shape_string, time_kernel,
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
    return dict(timing=timing, correctness=correctness)

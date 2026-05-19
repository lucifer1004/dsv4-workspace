"""Capture hooks for vllm.utils.deep_gemm (DeepGEMM + cuBLAS fallback)."""
from __future__ import annotations

import functools

from .. import counter


def _shape(t) -> str:
    if t is None:
        return "None"
    if isinstance(t, (tuple, list)):
        return "[" + ",".join(_shape(x) for x in t) + "]"
    s = getattr(t, "shape", None)
    if s is None:
        return repr(t)
    return "(" + ",".join(str(int(x)) for x in s) + ")"


def _dtype(t) -> str:
    if t is None:
        return ""
    if isinstance(t, (tuple, list)):
        return ",".join(_dtype(x) for x in t)
    return str(getattr(t, "dtype", "")).replace("torch.", "")


def _arg(args, kwargs, idx, name):
    if idx < len(args):
        return args[idx]
    return kwargs.get(name)


def _wrap_gemm(kernel_name, arg_names):
    """Generic wrapper: arg_names is list of positional arg names that are
    tensors/tuples we want to record (in order). Extras land in kwargs_str.
    """
    def decorator(orig):
        @functools.wraps(orig)
        def wrapper(*args, **kwargs):
            shape_parts = []
            dtype_parts = []
            for i, name in enumerate(arg_names):
                t = _arg(args, kwargs, i, name)
                if t is None:
                    shape_parts.append(f"{name}=None")
                    continue
                shape_parts.append(f"{name}={_shape(t)}")
                dt = _dtype(t)
                if dt:
                    dtype_parts.append(f"{name}={dt}")
            extra_kw = {k: v for k, v in kwargs.items() if k not in arg_names
                        and not hasattr(v, "shape")}
            kwargs_str = ",".join(f"{k}={v}" for k, v in extra_kw.items())
            counter.record(
                kernel_name,
                dtype=",".join(dtype_parts),
                shape=";".join(shape_parts),
                kwargs=kwargs_str,
            )
            return orig(*args, **kwargs)
        return wrapper
    return decorator


# arg_names list per DeepGEMM entry — first few positional args we care about.
# These reflect the call sites we want to break down; extras stay in kwargs_str.
_ENTRIES = {
    "fp8_gemm_nt": ["lhs", "rhs", "out"],
    "m_grouped_fp8_gemm_nt_contiguous": ["lhs", "rhs", "out", "m_indices"],
    "m_grouped_fp8_fp4_gemm_nt_contiguous": ["lhs", "rhs", "out", "m_indices"],
    "fp8_m_grouped_gemm_nt_masked": ["lhs", "rhs", "out", "masked_m"],
    "tf32_hc_prenorm_gemm": ["x", "fn", "out", "sqrsum"],
    "fp8_fp4_mqa_logits": ["q", "kv", "weights", "cu_seqlen_ks", "cu_seqlen_ke"],
    "fp8_fp4_paged_mqa_logits": ["q", "kv_cache", "weights", "context_lens",
                                  "block_tables"],
    "cublaslt_gemm_nt": ["lhs", "rhs", "out"],
    "fp8_einsum": ["lhs", "rhs", "out"],
}


def patch(mod=None) -> None:
    if mod is None:
        from vllm.utils import deep_gemm as mod
    if getattr(mod, "_bench_capture_patched", False):
        return
    for name, arg_names in _ENTRIES.items():
        orig = getattr(mod, name, None)
        if orig is None or not callable(orig):
            continue
        setattr(mod, name, _wrap_gemm(name, arg_names)(orig))
    mod._bench_capture_patched = True

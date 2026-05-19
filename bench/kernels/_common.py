"""Shared helpers for kernel benchmark drivers."""
from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import Callable

import torch

_DTYPE_FROM_NAME = {
    "float32": torch.float32, "fp32": torch.float32,
    "float16": torch.float16, "fp16": torch.float16,
    "bfloat16": torch.bfloat16, "bf16": torch.bfloat16,
    "uint8": torch.uint8, "int8": torch.int8,
    "int32": torch.int32, "int64": torch.int64,
    "float8_e4m3fn": torch.float8_e4m3fn, "fp8e4m3": torch.float8_e4m3fn,
    "float8_e5m2": torch.float8_e5m2,
    "bool": torch.bool,
}


def dtype_from_name(name: str) -> torch.dtype:
    n = name.strip().replace("torch.", "")
    if n not in _DTYPE_FROM_NAME:
        raise KeyError(f"unknown dtype: {name!r}")
    return _DTYPE_FROM_NAME[n]


_TUPLE_RE = re.compile(r"\(([\d,\s]+)\)")


def _parse_value(v: str):
    """Parse a kwargs value: int / bool / float / str."""
    v = v.strip()
    if v.lower() == "true":
        return True
    if v.lower() == "false":
        return False
    if v.lower() == "none":
        return None
    try:
        return int(v)
    except ValueError:
        pass
    try:
        return float(v)
    except ValueError:
        pass
    return v


def _parse_tuple_or_list(v: str):
    """Parse `(1,2,3)` → (1,2,3), or `[(1,2),(3,4)]` → [(1,2),(3,4)]."""
    v = v.strip()
    if v.lower() == "none":
        return None
    tuples = _TUPLE_RE.findall(v)
    if not tuples:
        return v
    parsed = [tuple(int(x.strip()) for x in t.split(",") if x.strip())
              for t in tuples]
    if v.startswith("[") and v.endswith("]"):
        return parsed
    # Single tuple (or top-level tuple at start of string).
    return parsed[0] if len(parsed) == 1 else parsed


def parse_shape_string(s: str) -> dict[str, object]:
    """Parse "q=(8000,128,512);kv=(2048,64,1,584);lhs=[(M,K),(M,K//128)];idx=None"
    into {'q': (8000,128,512), 'kv': (...), 'lhs': [(M,K),(M,K//128)], 'idx': None}.
    """
    out: dict[str, object] = {}
    if not s:
        return out
    for part in s.split(";"):
        if "=" not in part:
            continue
        k, v = part.split("=", 1)
        out[k.strip()] = _parse_tuple_or_list(v)
    return out


def parse_kwargs_string(s: str) -> dict:
    """Parse "num_heads=128,topk=128,has_sink=False" → dict."""
    out: dict = {}
    if not s:
        return out
    for part in s.split(","):
        if "=" not in part:
            continue
        k, v = part.split("=", 1)
        out[k.strip()] = _parse_value(v)
    return out


def parse_dtype_string(s: str) -> dict[str, str]:
    """Parse "q=bfloat16,kv=uint8" → {'q': 'bfloat16', 'kv': 'uint8'}."""
    out: dict[str, str] = {}
    if not s:
        return out
    for part in s.split(","):
        if "=" not in part:
            continue
        k, v = part.split("=", 1)
        out[k.strip()] = v.strip()
    return out


@dataclass
class CorrectnessResult:
    ok: bool
    # All 4 metrics always populated — atol/rtol_norm are raw diagnostics,
    # ULP-distance + calc_diff are the gate.
    max_atol: float          # max(|a - b|)
    max_rtol_norm: float     # max(|a-b|) / max(|b|), scale-invariant
    ulp_distance: float      # max(|a-b|) / (dtype_eps * max(|b|))
    calc_diff: float         # 1 - 2·(x·y)/(x²+y²)  — DeepGEMM convention
    n_nan: int
    n_inf: int
    dtype: str               # output dtype used to compute ulp_distance


def _calc_diff(actual: torch.Tensor, expected: torch.Tensor) -> float:
    """DeepGEMM's cosine-style distance. 0 = identical, 1 = orthogonal."""
    x = actual.detach().double()
    y = expected.detach().double()
    denom = (x * x + y * y).sum()
    if denom.item() == 0:
        return 0.0
    sim = 2 * (x * y).sum() / denom
    return float((1 - sim).item())


def compute_correctness(actual: torch.Tensor, expected: torch.Tensor,
                        *, ulp_threshold: float = 8.0,
                        calc_diff_threshold: float = 1e-3,
                        output_dtype: torch.dtype | None = None,
                        ) -> CorrectnessResult:
    """Compute 4 metrics; gate on ULP-distance + calc_diff.

    The kernel's output dtype sets the ULP scale (`torch.finfo(dt).eps`).
    A kernel is "correct to dtype precision" when ULP-distance ≤ 1-2;
    accumulated rounding noise can push to 3-4; anything >K indicates a
    real bug. atol/rtol_norm are recorded for human diagnosis but are not
    part of the pass/fail decision.
    """
    if actual.shape != expected.shape:
        raise ValueError(f"shape mismatch: {tuple(actual.shape)} vs {tuple(expected.shape)}")

    if output_dtype is None:
        output_dtype = actual.dtype
    try:
        dtype_eps = torch.finfo(output_dtype).eps
    except TypeError:
        dtype_eps = 0.0  # int dtype — ULP not meaningful, gate via calc_diff only

    a = actual.detach().float()
    e = expected.detach().float()
    diff = (a - e).abs()
    n_nan = int(torch.isnan(diff).sum().item())
    n_inf = int(torch.isinf(diff).sum().item())

    finite = torch.isfinite(diff)
    if not finite.any():
        return CorrectnessResult(
            False, float("inf"), float("inf"), float("inf"), float("inf"),
            n_nan, n_inf, str(output_dtype).replace("torch.", ""),
        )
    diff_f = diff[finite]
    e_abs_max = float(e.abs().max().item()) or 1.0
    max_atol = float(diff_f.max().item())
    max_rtol_norm = max_atol / max(e_abs_max, 1e-6)

    if dtype_eps > 0:
        ulp_unit = dtype_eps * e_abs_max
        ulp_distance = max_atol / max(ulp_unit, 1e-12)
    else:
        ulp_distance = 0.0

    calc_diff = _calc_diff(a, e)

    ok = (
        n_nan == 0 and n_inf == 0
        and (dtype_eps == 0 or ulp_distance < ulp_threshold)
        and calc_diff < calc_diff_threshold
    )
    return CorrectnessResult(
        ok, max_atol, max_rtol_norm, ulp_distance, calc_diff,
        n_nan, n_inf, str(output_dtype).replace("torch.", ""),
    )


@dataclass
class TimingResult:
    n_iter: int
    mean_ms: float
    p50_ms: float
    p99_ms: float
    min_ms: float


def time_kernel(fn: Callable[[], None], n_warmup: int = 20,
                n_iter: int = 100) -> TimingResult:
    """Warm up, then time `n_iter` invocations using torch.cuda.Event."""
    for _ in range(n_warmup):
        fn()
    torch.cuda.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(n_iter)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(n_iter)]
    for i in range(n_iter):
        starts[i].record()
        fn()
        ends[i].record()
    torch.cuda.synchronize()
    times = sorted(s.elapsed_time(e) for s, e in zip(starts, ends))
    return TimingResult(
        n_iter=n_iter,
        mean_ms=sum(times) / n_iter,
        p50_ms=times[n_iter // 2],
        p99_ms=times[min(n_iter - 1, int(n_iter * 0.99))],
        min_ms=times[0],
    )

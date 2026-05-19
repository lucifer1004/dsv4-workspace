#!/usr/bin/env python
"""Replay captured kernel shapes through per-kernel benchmark drivers."""
from __future__ import annotations

import argparse
import csv
import datetime as _dt
import os
import platform
import socket
import subprocess
import sys
from typing import Iterable

# Make `bench.*` importable when invoked from any cwd.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bench import kernels  # noqa: E402


_RESULT_COLS = (
    "kernel", "dtype", "shape", "kwargs", "phase", "tp_rank", "n_calls_seen",
    "status", "n_iter", "mean_ms", "p50_ms", "p99_ms", "min_ms",
    "flops", "mem_bytes", "tflops", "gb_s",
    "max_atol", "max_rtol_norm", "ulp_distance", "calc_diff",
    "out_dtype", "n_nan", "n_inf",
    "host", "gpu", "sparse_mla_commit", "deepgemm_commit", "vllm_commit",
    "timestamp", "notes",
)


def _git_head(path: str) -> str:
    try:
        out = subprocess.run(
            ["git", "-C", path, "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=2, check=True,
        )
        return out.stdout.strip()
    except Exception:
        return ""


def _gpu_name() -> str:
    try:
        import torch
        return torch.cuda.get_device_name(0)
    except Exception:
        return ""


def _env_context() -> dict[str, str]:
    workspace = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return dict(
        host=socket.gethostname(),
        gpu=_gpu_name(),
        sparse_mla_commit=_git_head(os.path.join(workspace, "sparse_mla_sm120")),
        deepgemm_commit=_git_head(os.path.join(workspace, "DeepGEMM")),
        vllm_commit=_git_head(os.path.join(workspace, "vllm")),
        timestamp=_dt.datetime.now().isoformat(timespec="seconds"),
    )


def _load_rows(path: str) -> Iterable[dict]:
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            yield row


def _run_one(row: dict, *, n_warmup: int, n_iter: int, check: bool,
             ulp_threshold: float, calc_diff_threshold: float) -> dict:
    driver = kernels.get_driver(row["kernel"])
    base = dict(
        kernel=row["kernel"], dtype=row.get("dtype", ""), shape=row.get("shape", ""),
        kwargs=row.get("kwargs", ""), phase=row.get("phase", ""),
        tp_rank=row.get("tp_rank", ""), n_calls_seen=row.get("n_calls", ""),
        status="SKIP_NO_DRIVER", n_iter=0, mean_ms=0.0, p50_ms=0.0, p99_ms=0.0,
        min_ms=0.0,
        flops="", mem_bytes="", tflops="", gb_s="",
        max_atol=0.0, max_rtol_norm=0.0, ulp_distance=0.0, calc_diff=0.0,
        out_dtype="", n_nan=0, n_inf=0,
        notes="",
    )
    if driver is None:
        return base
    try:
        result = driver(
            row, n_warmup=n_warmup, n_iter=n_iter, check=check,
            ulp_threshold=ulp_threshold,
            calc_diff_threshold=calc_diff_threshold,
        )
    except Exception as e:
        base["status"] = "ERROR"
        base["notes"] = f"{type(e).__name__}: {e}".replace("\n", " ")[:300]
        return base
    finally:
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass

    t = result["timing"]
    base["n_iter"] = t.n_iter
    base["mean_ms"] = round(t.mean_ms, 4)
    base["p50_ms"] = round(t.p50_ms, 4)
    base["p99_ms"] = round(t.p99_ms, 4)
    base["min_ms"] = round(t.min_ms, 4)

    p = result.get("perf")
    if p is not None and t.min_ms > 0:
        # min_ms gives the peak-achieved view; consumers can recompute
        # mean/p50 variants from (flops, mem_bytes) + the *_ms columns.
        base["flops"] = p.flops
        base["mem_bytes"] = p.mem_bytes
        base["tflops"] = round(p.flops / (t.min_ms * 1e9), 2)
        base["gb_s"] = round(p.mem_bytes / (t.min_ms * 1e6), 2)

    c = result.get("correctness")
    if c is None:
        base["status"] = "OK_NO_CHECK"
    else:
        base["status"] = "OK" if c.ok else "FAIL_CORRECTNESS"
        base["max_atol"] = round(c.max_atol, 6)
        base["max_rtol_norm"] = round(c.max_rtol_norm, 6)
        base["ulp_distance"] = round(c.ulp_distance, 4)
        base["calc_diff"] = round(c.calc_diff, 8)
        base["out_dtype"] = c.dtype
        base["n_nan"] = c.n_nan
        base["n_inf"] = c.n_inf
    return base


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--shapes", required=True, help="capture CSV produced by bench.capture")
    p.add_argument("--output", default=None,
                   help="results CSV (default: bench/results/<timestamp>.csv)")
    p.add_argument("--n-warmup", type=int, default=20)
    p.add_argument("--n-iter", type=int, default=100)
    p.add_argument("--no-check", action="store_true",
                   help="skip correctness check (faster, less safe)")
    p.add_argument("--ulp-threshold", type=float, default=8.0,
                   help="ULP-distance gate (max_atol / (dtype_eps × |ref|.max))")
    p.add_argument("--calc-diff-threshold", type=float, default=1e-3,
                   help="DeepGEMM-style cosine-distance gate")
    p.add_argument("--filter", default=None,
                   help="substring match on kernel column")
    args = p.parse_args(argv)

    if args.output is None:
        ws = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output = os.path.join(ws, "bench", "results", f"{ts}.csv")
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    ctx = _env_context()
    rows = [
        r for r in _load_rows(args.shapes)
        if args.filter is None or args.filter in r.get("kernel", "")
    ]
    print(f"[run_bench] {len(rows)} rows from {args.shapes} "
          f"→ {args.output} (drivers: {kernels.registered_kernels()})")

    with open(args.output, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=_RESULT_COLS)
        w.writeheader()
        n_ok = n_fail = n_skip = n_err = 0
        for i, row in enumerate(rows):
            result = _run_one(row, n_warmup=args.n_warmup, n_iter=args.n_iter,
                              check=not args.no_check,
                              ulp_threshold=args.ulp_threshold,
                              calc_diff_threshold=args.calc_diff_threshold)
            result.update(ctx)
            w.writerow(result)
            f.flush()
            status = result["status"]
            tag = (
                "✓" if status == "OK" else
                "?" if status == "OK_NO_CHECK" else
                "✗" if status == "FAIL_CORRECTNESS" else
                "!" if status == "ERROR" else "—"
            )
            print(f"  [{i+1}/{len(rows)}] {tag} {row['kernel']:35s} "
                  f"mean={result['mean_ms']:7.3f}ms  "
                  f"ulp={result['ulp_distance']:5.2f}  "
                  f"cos={result['calc_diff']:.2e}  "
                  f"atol={result['max_atol']:.2e}  "
                  f"{result['kwargs']}  {result.get('notes', '')}")
            if status == "OK":
                n_ok += 1
            elif status == "FAIL_CORRECTNESS":
                n_fail += 1
            elif status in ("SKIP_NO_DRIVER",):
                n_skip += 1
            elif status == "ERROR":
                n_err += 1
        print(f"\n[run_bench] done: ok={n_ok} fail={n_fail} skip={n_skip} err={n_err}")
        print(f"[run_bench] wrote {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

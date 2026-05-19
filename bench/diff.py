#!/usr/bin/env python
"""Diff two bench result CSVs to detect regression.

Match rows by (kernel, shape, kwargs) — tp_rank is ignored so a TP=4 result
can be compared against a TP=4 baseline (same setup) or against an older
commit's TP=4 baseline. Cross-TP comparison is allowed but interpret with
care: kernel shapes differ across TP sizes for the same workload.

Flags a row when any tracked metric crosses its regression threshold:
  - mean_ms       (timing,   default ratio > 1.5×)
  - ulp_distance  (precision, default ratio > 2.0×)
  - calc_diff     (precision, default ratio > 2.0×)
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from typing import Iterable


_MATCH_COLS = ("kernel", "shape", "kwargs")
_METRIC_COLS = ("mean_ms", "p50_ms", "p99_ms", "ulp_distance", "calc_diff")
_DEFAULT_THRESHOLDS: dict[str, float] = {
    "mean_ms": 1.5,
    "ulp_distance": 2.0,
    "calc_diff": 2.0,
}


def _read(path: str) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _key(row: dict) -> tuple[str, ...]:
    return tuple(row[c] for c in _MATCH_COLS)


def _to_float(s: str) -> float:
    try:
        return float(s)
    except Exception:
        return 0.0


def _ratio(curr: float, base: float) -> float:
    if base == 0:
        return float("inf") if curr > 0 else 1.0
    return curr / base


def diff(baseline: list[dict], current: list[dict],
         thresholds: dict[str, float] = _DEFAULT_THRESHOLDS,
         ) -> dict:
    base_by_key: dict[tuple, dict] = {_key(r): r for r in baseline}
    curr_by_key: dict[tuple, dict] = {_key(r): r for r in current}

    regressions = []
    improvements = []
    new_shapes = []
    for k, r in curr_by_key.items():
        if k not in base_by_key:
            new_shapes.append(r)
            continue
        b = base_by_key[k]
        per_metric = {}
        is_regression = False
        for metric, thresh in thresholds.items():
            bv = _to_float(b.get(metric, 0))
            cv = _to_float(r.get(metric, 0))
            ratio = _ratio(cv, bv)
            per_metric[metric] = (bv, cv, ratio)
            if ratio > thresh:
                is_regression = True
        # Track p50/p99 too for the report even if not gated.
        for metric in ("p50_ms", "p99_ms"):
            bv = _to_float(b.get(metric, 0))
            cv = _to_float(r.get(metric, 0))
            per_metric[metric] = (bv, cv, _ratio(cv, bv))
        target = regressions if is_regression else None
        if not is_regression:
            # Also flag big improvements (>2× faster) as suspicious — may
            # indicate test artifact (e.g. cudagraph took over a path) rather
            # than a real win.
            if per_metric["mean_ms"][2] < 0.5:
                target = improvements
        if target is not None:
            target.append((r, per_metric))

    missing = [r for k, r in base_by_key.items() if k not in curr_by_key]
    return {
        "regressions": regressions,
        "improvements": improvements,
        "new_shapes": new_shapes,
        "missing_shapes": missing,
        "n_baseline": len(baseline),
        "n_current": len(current),
        "n_matched": len(base_by_key.keys() & curr_by_key.keys()),
    }


def _fmt_row(row: dict, metrics: dict[str, tuple[float, float, float]]) -> str:
    name = row["kernel"]
    kwargs = row.get("kwargs", "")[:60]
    parts = []
    for m in ("mean_ms", "ulp_distance", "calc_diff"):
        bv, cv, r = metrics[m]
        parts.append(f"{m}={bv:.4g}→{cv:.4g} ({r:.2f}×)")
    return f"  {name:38s} {' '.join(parts)}\n    kwargs={kwargs}"


def report(result: dict, *, verbose: bool = False) -> str:
    out: list[str] = []
    out.append(f"baseline rows: {result['n_baseline']}, "
               f"current rows: {result['n_current']}, "
               f"matched: {result['n_matched']}")
    out.append("")
    if result["regressions"]:
        out.append(f"❌ {len(result['regressions'])} REGRESSIONS:")
        for r, m in result["regressions"]:
            out.append(_fmt_row(r, m))
    else:
        out.append("✓ no regressions")
    if result["improvements"]:
        out.append(f"\n? {len(result['improvements'])} suspicious-improvements (>2× faster):")
        for r, m in result["improvements"][:10]:
            out.append(_fmt_row(r, m))
        if len(result["improvements"]) > 10:
            out.append(f"  ... and {len(result['improvements']) - 10} more")
    if verbose:
        if result["new_shapes"]:
            out.append(f"\n+ {len(result['new_shapes'])} new shapes (in current, not in baseline)")
            for r in result["new_shapes"][:5]:
                out.append(f"  {r['kernel']:35s} kwargs={r.get('kwargs','')[:50]}")
        if result["missing_shapes"]:
            out.append(f"\n- {len(result['missing_shapes'])} missing shapes (in baseline, not in current)")
            for r in result["missing_shapes"][:5]:
                out.append(f"  {r['kernel']:35s} kwargs={r.get('kwargs','')[:50]}")
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--baseline", required=True)
    p.add_argument("--current", required=True)
    p.add_argument("--mean-ms-ratio", type=float, default=1.5)
    p.add_argument("--ulp-distance-ratio", type=float, default=2.0)
    p.add_argument("--calc-diff-ratio", type=float, default=2.0)
    p.add_argument("--write-csv", default=None,
                   help="write the regressions/improvements/etc to a CSV")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(argv)

    thresholds = {
        "mean_ms": args.mean_ms_ratio,
        "ulp_distance": args.ulp_distance_ratio,
        "calc_diff": args.calc_diff_ratio,
    }
    baseline = _read(args.baseline)
    current = _read(args.current)
    result = diff(baseline, current, thresholds=thresholds)
    print(report(result, verbose=args.verbose))

    if args.write_csv:
        with open(args.write_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["status", "kernel", "shape", "kwargs",
                        "mean_ms_base", "mean_ms_curr", "mean_ms_ratio",
                        "ulp_base", "ulp_curr", "ulp_ratio",
                        "cos_base", "cos_curr", "cos_ratio"])
            for tag, items in (("REGRESSION", result["regressions"]),
                               ("IMPROVEMENT", result["improvements"]),
                               ("NEW", [(r, {}) for r in result["new_shapes"]]),
                               ("MISSING", [(r, {}) for r in result["missing_shapes"]])):
                for r, m in items:
                    row = [tag, r["kernel"], r.get("shape", ""), r.get("kwargs", "")]
                    for metric in ("mean_ms", "ulp_distance", "calc_diff"):
                        if metric in m:
                            row.extend(f"{x:.6g}" for x in m[metric])
                        else:
                            row.extend(["", "", ""])
                    w.writerow(row)
        print(f"\nWrote {args.write_csv}")

    return 1 if result["regressions"] else 0


if __name__ == "__main__":
    sys.exit(main())

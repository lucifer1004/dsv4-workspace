"""Process-local kernel call counter with atexit CSV dump."""
from __future__ import annotations

import atexit
import csv
import os
import signal
import sys
import threading
from collections import defaultdict
from typing import Optional

_lock = threading.Lock()
_counter: dict[tuple, int] = defaultdict(int)
_raw_log: list[tuple] = []
_registered = False

_RAW_MODE = os.environ.get("VLLM_CAPTURE_KERNEL_SHAPES_RAW") == "1"

_COLUMNS = ("kernel", "dtype", "shape", "kwargs", "phase", "tp_rank", "n_calls")


def _current_phase() -> str:
    try:
        import torch
        if torch.cuda.is_current_stream_capturing():
            return "cudagraph_capture"
    except Exception:
        pass
    return os.environ.get("VLLM_CAPTURE_KERNEL_PHASE", "serve")


def _tp_rank() -> int:
    try:
        import torch.distributed as dist
        if dist.is_available() and dist.is_initialized():
            return dist.get_rank()
    except Exception:
        pass
    return 0


# How often (in record() calls) to flush the counter to disk. Workers in
# vllm get hard-killed when the parent's pipe closes — atexit/signal handlers
# don't run reliably. Incremental dumps mean the file on disk is at most
# `_FLUSH_EVERY` records stale when the worker is force-killed.
_FLUSH_EVERY = 200
_total_records = 0


def record(kernel: str, *, dtype: str = "", shape: str = "", kwargs: str = "") -> None:
    global _total_records
    key = (kernel, dtype, shape, kwargs, _current_phase(), _tp_rank())
    if _RAW_MODE:
        with _lock:
            _raw_log.append(key)
            _total_records += 1
            should_flush = _total_records % _FLUSH_EVERY == 0
    else:
        with _lock:
            _counter[key] += 1
            _total_records += 1
            should_flush = _total_records % _FLUSH_EVERY == 0
    if should_flush:
        try:
            dump()
        except Exception:
            pass  # never let dump errors interfere with the hot kernel path


def _dump_path() -> str:
    """Per-rank, per-pid file so multi-rank vllm doesn't write collide.

    `VLLM_CAPTURE_KERNEL_SHAPES_PATH` semantics:
      - contains `{rank}` / `{pid}`: literal template substitution
      - ends with `/` or is an existing dir: write `rank{R}_pid{P}.csv` inside
      - otherwise: treat as file path, insert `_rank{R}_pid{P}` before extension
    """
    rank = _tp_rank()
    pid = os.getpid()
    custom = os.environ.get("VLLM_CAPTURE_KERNEL_SHAPES_PATH")
    if not custom:
        return f"/tmp/dsv4_shapes_rank{rank}_pid{pid}.csv"
    if "{rank}" in custom or "{pid}" in custom:
        return custom.format(rank=rank, pid=pid)
    if custom.endswith("/") or os.path.isdir(custom):
        return os.path.join(custom, f"rank{rank}_pid{pid}.csv")
    base, ext = os.path.splitext(custom)
    return f"{base}_rank{rank}_pid{pid}{ext or '.csv'}"


def dump(path: Optional[str] = None) -> str:
    """Atomic write: tmp file in same dir → os.rename. Safe under concurrent
    `dump()` calls from record() incremental-flush and atexit/signal handlers.
    """
    path = path or _dump_path()
    parent = os.path.dirname(path) or "."
    os.makedirs(parent, exist_ok=True)
    tmp = f"{path}.tmp{os.getpid()}"
    with open(tmp, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(_COLUMNS)
        if _RAW_MODE:
            with _lock:
                rows = list(_raw_log)
            for k in rows:
                w.writerow(list(k) + [1])
        else:
            with _lock:
                items = sorted(_counter.items())
            for k, n in items:
                w.writerow(list(k) + [n])
    os.rename(tmp, path)
    return path


_dumped = False


def _safe_log(msg: str) -> None:
    """Best-effort log to /tmp file — never raise.

    Worker engine-core stdout/stderr are pipe-captured by vllm and get closed
    early during shutdown, so `print()` from atexit hits ValueError. We log
    to a side file instead so we can verify the dump fired even when std
    streams are gone.
    """
    try:
        with open("/tmp/bench_capture.log", "a") as f:
            f.write(f"[pid {os.getpid()}] {msg}\n")
    except Exception:
        pass


def _dump_quiet() -> None:
    """Dump-once with full error containment (no exception leaks out)."""
    global _dumped
    if _dumped:
        return
    try:
        path = dump()
        n_rows = len(_counter) if not _RAW_MODE else len(_raw_log)
        _dumped = True
        _safe_log(f"dumped {n_rows} rows → {path}")
    except Exception as e:
        _safe_log(f"dump failed: {type(e).__name__}: {e}")


def _signal_dump(signum, frame):
    _dump_quiet()
    # Don't intercept the signal — let the default handler kill us.
    signal.signal(signum, signal.SIG_DFL)
    os.kill(os.getpid(), signum)


def register_atexit() -> None:
    """Register both atexit and SIGINT/SIGTERM signal handlers.

    atexit alone is unreliable in vllm workers because shutdown sequencing
    closes the pipes that capture worker stdout BEFORE atexit fires, and on
    SIGKILL (no graceful shutdown timeout) atexit doesn't fire at all. The
    signal handler runs synchronously inside the signal context — best
    chance to flush before forced termination.
    """
    global _registered
    if _registered:
        return
    atexit.register(_dump_quiet)
    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            signal.signal(sig, _signal_dump)
        except (ValueError, OSError):
            pass  # signal cannot be set from non-main thread
    _registered = True

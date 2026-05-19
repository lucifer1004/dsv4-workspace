"""Process-local kernel call counter with atexit CSV dump."""
from __future__ import annotations

import atexit
import csv
import os
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


def record(kernel: str, *, dtype: str = "", shape: str = "", kwargs: str = "") -> None:
    key = (kernel, dtype, shape, kwargs, _current_phase(), _tp_rank())
    if _RAW_MODE:
        with _lock:
            _raw_log.append(key)
        return
    with _lock:
        _counter[key] += 1


def _dump_path() -> str:
    custom = os.environ.get("VLLM_CAPTURE_KERNEL_SHAPES_PATH")
    if custom:
        return custom
    return f"/tmp/dsv4_shapes_rank{_tp_rank()}_pid{os.getpid()}.csv"


def dump(path: Optional[str] = None) -> str:
    path = path or _dump_path()
    parent = os.path.dirname(path) or "."
    os.makedirs(parent, exist_ok=True)
    with open(path, "w", newline="") as f:
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
    return path


def _atexit_dump() -> None:
    try:
        path = dump()
        print(f"[bench.capture] dumped {len(_counter) or len(_raw_log)} rows → {path}",
              flush=True)
    except Exception as e:
        print(f"[bench.capture] atexit dump failed: {e}", flush=True)


def register_atexit() -> None:
    global _registered
    if _registered:
        return
    atexit.register(_atexit_dump)
    _registered = True

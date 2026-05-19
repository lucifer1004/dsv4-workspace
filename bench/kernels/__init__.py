"""Kernel benchmark driver registry.

Each driver is keyed by the kernel name written by `bench.capture` and
takes one row of the capture CSV plus options. Drivers return a dict
matching the results-CSV schema (see bench/run_bench.py).
"""
from __future__ import annotations

from typing import Callable

_DRIVERS: dict[str, Callable] = {}


def register(kernel_name: str) -> Callable[[Callable], Callable]:
    def deco(fn: Callable) -> Callable:
        if kernel_name in _DRIVERS:
            raise KeyError(f"driver already registered: {kernel_name!r}")
        _DRIVERS[kernel_name] = fn
        return fn
    return deco


def get_driver(kernel_name: str) -> Callable | None:
    return _DRIVERS.get(kernel_name)


def registered_kernels() -> list[str]:
    return sorted(_DRIVERS.keys())


# Import side-effect: register drivers.
from . import flash_mla  # noqa: F401,E402
from . import deep_gemm  # noqa: F401,E402

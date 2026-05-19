"""Kernel-shape capture for DSv4 workspace.

Usage:
    import bench.capture; bench.capture.install()

Or auto-install via the .pth file dropped into the venv's site-packages by
`bench/install_capture.sh`, gated on `VLLM_CAPTURE_KERNEL_SHAPES=1`.
"""
from __future__ import annotations

import os

from .counter import dump, record, register_atexit
from .import_hook import register_post_import

__all__ = ["install", "dump", "record"]

_installed = False


def install() -> None:
    """Patch all hookable kernel entry points and register atexit CSV dump.

    Idempotent. Patches:
      - flash_mla_sm120 wrappers (immediate, package is independent)
      - triton.runtime.jit.JITFunction.run (immediate; covers vllm Triton)
      - vllm.utils.deep_gemm wrappers (deferred until vllm is imported)
    """
    global _installed
    if _installed:
        return
    _installed = True

    register_atexit()

    # flash_mla_sm120 — independent package, patch immediately if importable.
    try:
        from .hooks import flash_mla
        flash_mla.patch()
    except ImportError:
        pass

    # Triton — JIT hook applies globally, install eagerly.
    try:
        from .hooks import triton as triton_hook
        triton_hook.patch()
    except ImportError:
        pass

    # vllm.utils.deep_gemm — defer until vllm loads.
    from .hooks import deep_gemm

    def _patch_deep_gemm(mod):
        deep_gemm.patch(mod)

    register_post_import("vllm.utils.deep_gemm", _patch_deep_gemm)


def install_if_enabled() -> None:
    if os.environ.get("VLLM_CAPTURE_KERNEL_SHAPES") == "1":
        install()

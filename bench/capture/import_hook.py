"""Fire callbacks after specific modules finish importing.

Wraps `builtins.__import__` once; on every import, checks whether any
registered target module is now in `sys.modules` and fires its callback.
"""
from __future__ import annotations

import builtins
import sys
from typing import Callable

_callbacks: dict[str, Callable] = {}
_original_import = None


def _import_wrapper(name, globals=None, locals=None, fromlist=(), level=0):
    mod = _original_import(name, globals, locals, fromlist, level)
    if _callbacks:
        for target in list(_callbacks.keys()):
            if target in sys.modules:
                cb = _callbacks.pop(target)
                try:
                    cb(sys.modules[target])
                except Exception as e:
                    print(f"[bench.capture] post-import callback for "
                          f"{target!r} failed: {e}", flush=True)
    return mod


def register_post_import(target: str, callback: Callable) -> None:
    global _original_import
    if target in sys.modules:
        callback(sys.modules[target])
        return
    _callbacks[target] = callback
    if _original_import is None:
        _original_import = builtins.__import__
        builtins.__import__ = _import_wrapper

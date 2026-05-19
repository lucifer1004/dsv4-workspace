"""Capture hook for Triton — wraps `JITFunction.run` for kernels we care about."""
from __future__ import annotations

from .. import counter

# Kernels recorded by name. Add more as we expand coverage.
_WATCH = {
    "_compute_slot_mapping_kernel",
    "_compute_swa_indices_and_lens_kernel",
    "_compute_prefill_metadata_kernel",
    "_fused_inv_rope_fp8_quant_per_head",
    "_save_partial_states_kernel",
    "_build_prefill_chunk_metadata_kernel",
}


def _shape(t) -> str:
    if t is None:
        return "None"
    s = getattr(t, "shape", None)
    if s is None:
        return ""
    return "(" + ",".join(str(int(x)) for x in s) + ")"


def _dtype(t) -> str:
    return str(getattr(t, "dtype", "")).replace("torch.", "")


def _format_kwargs(constexprs: dict) -> str:
    parts = []
    for k, v in constexprs.items():
        if isinstance(v, (int, bool, str, float)):
            parts.append(f"{k}={v}")
    return ",".join(parts)


def patch() -> None:
    import triton.runtime.jit as _jit

    if getattr(_jit.JITFunction, "_bench_capture_patched", False):
        return

    orig_run = _jit.JITFunction.run

    def run_wrapper(self, *args, **kwargs):
        fn_name = getattr(getattr(self, "fn", None), "__name__", "")
        if fn_name in _WATCH:
            # First tensor arg in *args determines shape — kwargs hold the
            # tl.constexpr values (BLOCK_*, etc.) used to specialize.
            shape_parts = []
            dtype_parts = []
            for i, a in enumerate(args[:6]):  # cap to keep CSV readable
                sh = _shape(a)
                if sh:
                    shape_parts.append(f"a{i}={sh}")
                    dt = _dtype(a)
                    if dt:
                        dtype_parts.append(f"a{i}={dt}")
            constexpr_kw = {k: v for k, v in kwargs.items()
                            if isinstance(v, (int, bool, str, float))}
            counter.record(
                fn_name,
                dtype=",".join(dtype_parts),
                shape=";".join(shape_parts),
                kwargs=_format_kwargs(constexpr_kw),
            )
        return orig_run(self, *args, **kwargs)

    _jit.JITFunction.run = run_wrapper
    _jit.JITFunction._bench_capture_patched = True

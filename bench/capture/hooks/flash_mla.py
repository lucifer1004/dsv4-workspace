"""Capture hooks for flash_mla_sm120 (sparse-MLA prefill / decode)."""
from __future__ import annotations

import functools

from .. import counter


def _shape(t) -> str:
    if t is None:
        return "None"
    s = getattr(t, "shape", None)
    if s is None:
        return repr(t)
    return "(" + ",".join(str(int(x)) for x in s) + ")"


def _dtype(t) -> str:
    if t is None:
        return ""
    return str(getattr(t, "dtype", "")).replace("torch.", "")


def _wrap_flash_mla_sparse_fwd(orig):
    @functools.wraps(orig)
    def wrapper(q, kv, indices, sm_scale, d_v=512, attn_sink=None,
                topk_length=None, out=None,
                extra_k_cache=None, extra_indices_in_kvcache=None,
                extra_topk_length=None, **kw):
        # Distinguish single- vs dual-cache. NUM_HEADS comes from q.shape[-2].
        num_heads = int(q.shape[-2]) if q.dim() >= 2 else -1
        topk = int(indices.shape[-1]) if indices is not None and indices.dim() >= 1 else -1
        topk_extra = (
            int(extra_indices_in_kvcache.shape[-1])
            if extra_indices_in_kvcache is not None
            and extra_indices_in_kvcache.dim() >= 1 else 0
        )
        dual = extra_k_cache is not None
        kernel = "flash_mla_sparse_fwd_dual" if dual else "flash_mla_sparse_fwd"
        kwargs_str = (
            f"num_heads={num_heads},topk={topk},topk_extra={topk_extra},"
            f"d_v={d_v},has_sink={attn_sink is not None}"
        )
        shape_str = (
            f"q={_shape(q)};kv={_shape(kv)};idx={_shape(indices)};"
            f"ek={_shape(extra_k_cache)};eidx={_shape(extra_indices_in_kvcache)}"
        )
        counter.record(
            kernel,
            dtype=f"q={_dtype(q)},kv={_dtype(kv)}",
            shape=shape_str,
            kwargs=kwargs_str,
        )
        return orig(q, kv, indices, sm_scale, d_v=d_v, attn_sink=attn_sink,
                    topk_length=topk_length, out=out,
                    extra_k_cache=extra_k_cache,
                    extra_indices_in_kvcache=extra_indices_in_kvcache,
                    extra_topk_length=extra_topk_length, **kw)
    return wrapper


def _wrap_flash_mla_with_kvcache(orig):
    @functools.wraps(orig)
    def wrapper(q, k_cache, block_table, cache_seqlens, head_dim_v,
                tile_scheduler_metadata, *args, **kw):
        num_heads = int(q.shape[-2]) if q.dim() >= 2 else -1
        indices = kw.get("indices")
        extra_k_cache = kw.get("extra_k_cache")
        topk = (
            int(indices.shape[-1]) if indices is not None
            and hasattr(indices, "dim") and indices.dim() >= 1 else -1
        )
        topk_extra = 0
        eidx = kw.get("extra_indices_in_kvcache")
        if eidx is not None and hasattr(eidx, "dim") and eidx.dim() >= 1:
            topk_extra = int(eidx.shape[-1])
        kernel = "flash_mla_with_kvcache_dual" if extra_k_cache is not None \
            else "flash_mla_with_kvcache"
        kwargs_str = (
            f"num_heads={num_heads},topk={topk},topk_extra={topk_extra},"
            f"head_dim_v={head_dim_v},is_fp8={kw.get('is_fp8_kvcache', False)}"
        )
        shape_str = (
            f"q={_shape(q)};kc={_shape(k_cache)};bt={_shape(block_table)};"
            f"cs={_shape(cache_seqlens)};ek={_shape(extra_k_cache)}"
        )
        counter.record(
            kernel,
            dtype=f"q={_dtype(q)},k={_dtype(k_cache)}",
            shape=shape_str,
            kwargs=kwargs_str,
        )
        return orig(q, k_cache, block_table, cache_seqlens, head_dim_v,
                    tile_scheduler_metadata, *args, **kw)
    return wrapper


def patch() -> None:
    import flash_mla_sm120
    from flash_mla_sm120 import interface as _iface

    if getattr(flash_mla_sm120, "_bench_capture_patched", False):
        return

    # Patch interface module first — that's the canonical source.
    _iface.flash_mla_sparse_fwd = _wrap_flash_mla_sparse_fwd(
        _iface.flash_mla_sparse_fwd
    )
    _iface.flash_mla_with_kvcache = _wrap_flash_mla_with_kvcache(
        _iface.flash_mla_with_kvcache
    )
    # Update the top-level re-exports.
    flash_mla_sm120.flash_mla_sparse_fwd = _iface.flash_mla_sparse_fwd
    flash_mla_sm120.flash_mla_with_kvcache = _iface.flash_mla_with_kvcache
    flash_mla_sm120._bench_capture_patched = True

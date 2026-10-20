"""Prefill expert projection used by the slot-cache MoE path in loader.py.

Kept in its own module (depends only on mlx) so it can be unit tested without the model
or oMLX; see test_prefill_qproj.py.

Bug fixed here ("k_hdr"): the sorted gather_qmm fast path only passes the first
k_hdr = max(rows, 16) // 4 slots of the expert cache to the kernel. Within a chunk the
local slot index of a row can be anything in 0..n_chunk-1, so when the chunk has more
distinct experts than k_hdr, rows routed to slot >= k_hdr indexed past the sliced weights
and produced wrong outputs. The sorted path is now taken only when every slot fits
(n_chunk <= k_hdr); otherwise the unsorted gather_qmm over all slots is used.
"""
from __future__ import annotations

import mlx.core as mx


def prefill_qproj(w, scales, biases, x_in, idx_in, use_sorted, n_chunk, group_size, bits, mode):
    """gather_qmm of x_in (rows, 1, D) against cache slots idx_in (rows,), indices sorted.

    n_chunk: number of distinct experts (cache slots 0..n_chunk-1) used by this chunk.
    """
    if use_sorted:
        orig_n = x_in.shape[0]
        k_hdr = max(1, min(w.shape[0], max(orig_n, 16) // 4))
        if n_chunk <= k_hdr:
            if orig_n < 16:
                pad_n = 16 - orig_n
                x_in = mx.concatenate([x_in, mx.repeat(x_in[-1:], pad_n, axis=0)], axis=0)
                idx_in = mx.concatenate([idx_in, mx.repeat(idx_in[-1:], pad_n, axis=0)], axis=0)
            out = mx.gather_qmm(
                x_in, w[:k_hdr], scales[:k_hdr], biases[:k_hdr],
                rhs_indices=idx_in, transpose=True, group_size=group_size,
                bits=bits, mode=mode, sorted_indices=True,
            )
            return out[:orig_n] if orig_n < 16 else out
    return mx.gather_qmm(
        x_in, w, scales, biases, rhs_indices=idx_in, transpose=True,
        group_size=group_size, bits=bits, mode=mode, sorted_indices=False,
    )

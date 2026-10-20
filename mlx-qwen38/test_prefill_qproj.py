"""Fixture test for the k_hdr fix in prefill_qproj.py. Random weights only, no model needed.

    python test_prefill_qproj.py          # fixed code: must PASS
    python test_prefill_qproj.py --old    # negative control, pre-fix logic: must FAIL
    MLX_EXPECT=0.32.4 python test_prefill_qproj.py   # also assert the MLX version

Checks, per chunk shape (rows, distinct experts): the output matches a dequantized per-row
reference matmul, and whenever the chunk has more experts than k_hdr the output is bit-identical
to the unsorted gather_qmm over all slots.
"""
import os
import sys

import mlx.core as mx  # import mlx first so the version below is the one actually used
import numpy as np

from prefill_qproj import prefill_qproj

GS, BITS, MODE = 64, 2, "affine"
CAP, D, O = 102, 1024, 384


def old_prefill_qproj(w, s, b, x_in, idx_in, use_sorted, n_chunk, group_size, bits, mode):
    """Pre-fix logic (ignores n_chunk), kept only as the negative control."""
    if use_sorted:
        orig_n = x_in.shape[0]
        if orig_n < 16:
            p = 16 - orig_n
            x_in = mx.concatenate([x_in, mx.repeat(x_in[-1:], p, axis=0)], axis=0)
            idx_in = mx.concatenate([idx_in, mx.repeat(idx_in[-1:], p, axis=0)], axis=0)
        k = max(1, min(w.shape[0], x_in.shape[0] // 4))
        o = mx.gather_qmm(x_in, w[:k], s[:k], b[:k], rhs_indices=idx_in, transpose=True,
                          group_size=group_size, bits=bits, mode=mode, sorted_indices=True)
        return o[:orig_n] if orig_n < 16 else o
    return mx.gather_qmm(x_in, w, s, b, rhs_indices=idx_in, transpose=True,
                         group_size=group_size, bits=bits, mode=mode, sorted_indices=False)


# (rows, distinct experts in chunk); cache capacity 102 slots
SHAPES = [(63, 19), (190, 55), (404, 102), (128, 33), (99, 26), (22, 9), (73, 25), (372, 102),
          (61, 17), (80, 25), (587, 102), (246, 39), (64, 16), (2000, 90), (5, 2), (1500, 102)]


def main():
    want = os.environ.get("MLX_EXPECT")
    print(f"mlx {mx.__version__} ({mx.__file__})")
    if want and mx.__version__ != want:
        sys.exit(f"MLX version {mx.__version__} != MLX_EXPECT {want}")
    fn = old_prefill_qproj if "--old" in sys.argv else prefill_qproj
    rng = np.random.default_rng(0)
    W = mx.array(rng.standard_normal((CAP, O, D)).astype(np.float32) * 0.05).astype(mx.bfloat16)
    w, s, b = mx.quantize(W, group_size=GS, bits=BITS, mode=MODE)
    deq = mx.dequantize(w, s, b, group_size=GS, bits=BITS, mode=MODE).astype(mx.float32)
    mx.eval(w, s, b, deq)
    fails = 0
    for rows, ne in SHAPES:
        idx = np.sort(np.concatenate([np.arange(ne), rng.integers(0, ne, rows - ne)])).astype(np.uint32)
        ia = mx.array(idx)
        x = mx.array(rng.standard_normal((rows, 1, D)).astype(np.float32)).astype(mx.bfloat16)
        use_sorted = True  # the loader takes the sorted path for these shapes (layers with 102-slot cache)
        out = fn(w, s, b, x, ia, use_sorted, ne, GS, BITS, MODE)
        un = mx.gather_qmm(x, w, s, b, rhs_indices=ia, transpose=True, group_size=GS, bits=BITS,
                           mode=MODE, sorted_indices=False)
        ref = mx.squeeze(mx.astype(x, mx.float32) @ mx.swapaxes(deq[ia], -1, -2), 1)
        err = mx.max(mx.abs(out.reshape(rows, O).astype(mx.float32) - ref), axis=1)
        tol = 0.02 * mx.max(mx.abs(ref))
        bad = int(mx.sum(err > tol).item())
        k_hdr = max(1, min(CAP, max(rows, 16) // 4))
        eq_un = bool(mx.array_equal(out, un).item())
        ok = bad == 0 and (ne <= k_hdr or eq_un)
        fails += not ok
        print(f"rows={rows:5d} experts={ne:3d} k_hdr={k_hdr:3d} rows_slot>=k_hdr={int((idx >= k_hdr).sum()):4d} "
              f"wrong_rows={bad:4d} ==unsorted={int(eq_un)} {'OK' if ok else 'FAIL'}")
    print("FAIL" if fails else "PASS", f"({fails}/{len(SHAPES)} shapes failed)")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())

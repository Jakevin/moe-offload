import mlx.core as mx, inspect, sys
print("mlx", mx.__version__, "python", sys.version.split()[0])
try: print("quantize sig:", inspect.signature(mx.quantize))
except Exception as e: print("sig n/a", e)
w = mx.random.normal((64, 256)).astype(mx.bfloat16)
for mode, gs, bits in [("affine", 64, 2), ("affine", 64, 4), ("mxfp4", 32, 4)]:
    try:
        r = mx.quantize(w, group_size=gs, bits=bits, mode=mode)
        parts = [(a.dtype, a.shape) for a in r if a is not None]
        deq = mx.dequantize(*r, group_size=gs, bits=bits, mode=mode) if mode != "affine" else mx.dequantize(r[0], r[1], r[2], group_size=gs, bits=bits)
        err = float(mx.mean((deq.astype(mx.float32) - w.astype(mx.float32)) ** 2))
        x = mx.random.normal((4, 256)).astype(mx.bfloat16)
        q = mx.quantized_matmul(x, r[0], r[1], r[2] if mode == "affine" else None, transpose=True, group_size=gs, bits=bits, mode=mode)
        W3 = [mx.stack([a] * 3) for a in r if a is not None]
        b = W3[2] if mode == "affine" else None
        g = mx.gather_qmm(x[:, None, :], W3[0], W3[1], b, rhs_indices=mx.array([0, 2, 1, 2], dtype=mx.uint32)[:, None], transpose=True, group_size=gs, bits=bits, mode=mode)
        mx.eval(q, g)
        print(f"{mode} gs{gs} b{bits}: OK parts={parts} mse={err:.5f} qmm={q.shape} gather_qmm={g.shape} gather_matches_qmm_row0={bool(mx.allclose(g[0,0], q[0], atol=1e-2))}")
    except Exception as e:
        print(f"{mode} gs{gs} b{bits}: FAIL {type(e).__name__}: {e}")

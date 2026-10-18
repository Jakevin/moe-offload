#!/usr/bin/env python3
"""MXFP4 format / kernel-wiring check for MLX (no model, no data; synthetic weights only).
- gather_qmm wiring: distinct experts, repeated + shuffled indices, every output row vs per-expert quantized_matmul
  (affine 2-bit g64, affine 4-bit g64, mxfp4 g32). Note lhs must be [n,1,1,K] with indices [n,1].
- bytes per expert (3 x 640x2560 projections) and safetensors round-trip.
- encoder comparison: MLX mxfp4 encoder vs a numpy re-implementation of the scale/rounding rule described in
  Whallm's Sources/DeepSeekRepack/MXFP4.swift (MIT License, Copyright (c) Whallm authors; commit ea9e99ad).
  Packed codes, E8M0 scales and decoded values are compared on edge cases and random data.
Writes out/mxfp4_check.json.  Usage: python mxfp4_format_check.py
"""
import json, os, time
from pathlib import Path
import numpy as np
import mlx.core as mx
mx.set_cache_limit(2 * 1024**3)
OUT = Path(__file__).resolve().parent / "out"; OUT.mkdir(exist_ok=True)
def log(m): print(f"{time.strftime('%F %T')} {m}", flush=True)
def wjson(p, o): t = str(p) + ".tmp"; json.dump(o, open(t, "w"), indent=1); os.replace(t, p)
FP4 = np.array([0, .5, 1, 1.5, 2, 3, 4, 6], np.float32)
def bf16(w): return w.astype(mx.bfloat16)
# Scale/rounding rule of Whallm MXFP4.swift (BF16 input) re-implemented in numpy: codes + E8M0 scales
def wh_quant(v):
    o, i = v.shape; g = v.reshape(o, i // 32, 32).astype(np.float32)
    m = np.abs(g).max(-1)
    with np.errstate(divide="ignore"):
        ex = np.where(m > 0, np.floor(np.log2(np.where(m > 0, m, 1.0))), 0) - 2   # Swift Float.exponent = floor(log2)
    ex = ex.clip(-127, 128)
    scale = np.where(m > 0, ex + 127, 0).astype(np.uint8)
    f = np.exp2(-ex)[..., None]; a = np.minimum(np.abs(g * f), 6.0)
    d = np.abs(a[..., None] - FP4)                                  # [o,G,32,8]
    sel = np.full(a.shape, 0, np.int64); best = np.full(a.shape, np.inf, np.float32)
    for idx in range(8):                                            # Swift loop: strictly closer, or tie and even index
        c = d[..., idx]; take = (c < best) | ((c == best) & (idx % 2 == 0))
        sel = np.where(take, idx, sel); best = np.where(take, c, best)
    neg = np.signbit(g); codes = (sel | (neg * 8)).astype(np.uint8)
    codes = np.where(m[..., None] > 0, codes, 0)
    return codes.reshape(o, i), scale
def mlx_codes(wq):  # uint32 -> nibbles, element order low nibble first
    u = np.array(wq).view(np.uint32); o = u.shape[0]
    n = np.stack([(u >> (4 * k)) & 15 for k in range(8)], -1).reshape(o, -1).astype(np.uint8)
    return n
def dq_codes(codes, scale):
    v = FP4[codes & 7] * np.where(codes & 8, -1, 1); o, i = codes.shape
    s = np.where(scale > 0, np.exp2(scale.astype(np.float32) - 127), 0)
    return (v.reshape(o, i // 32, 32) * s[..., None]).reshape(o, i)
def stage_check(a):
    res = {"mlx": mx.__version__}
    rng = np.random.default_rng(0)
    # (1) gather_qmm wiring: 6 distinct experts, indices repeated+shuffled, all outputs vs per-expert quantized_matmul
    E, O, I = 6, 128, 512
    W = mx.array(rng.standard_normal((E, O, I)).astype(np.float32) * 0.05).astype(mx.bfloat16)
    wiring = {}
    for name, mode, gs, bits in [("a2", "affine", 64, 2), ("a4", "affine", 64, 4), ("mx4", "mxfp4", 32, 4)]:
        r = mx.quantize(W, group_size=gs, bits=bits, mode=mode); b = r[2] if mode == "affine" else None
        idx = np.array([3, 0, 5, 3, 1, 1, 4, 2, 0, 5, 2, 3], np.uint32)
        x = mx.array(rng.standard_normal((len(idx), I)).astype(np.float32)).astype(mx.bfloat16)
        g = mx.gather_qmm(x[:, None, None, :], r[0], r[1], b, rhs_indices=mx.array(idx)[:, None], transpose=True, group_size=gs, bits=bits, mode=mode)
        ref = mx.stack([mx.quantized_matmul(x[j:j + 1], r[0][int(e)], r[1][int(e)], None if b is None else b[int(e)], transpose=True, group_size=gs, bits=bits, mode=mode)[0] for j, e in enumerate(idx)])
        g = g.reshape(len(idx), O); mx.eval(g, ref)
        diff = float(mx.max(mx.abs(g.astype(mx.float32) - ref.astype(mx.float32)))); scale = float(mx.max(mx.abs(ref.astype(mx.float32))))
        wiring[name] = {"out_shape_ok": tuple(g.shape) == (len(idx), O), "max_abs_diff": diff, "ref_max_abs": scale,
                        "nan_inf": bool(mx.any(mx.isnan(g)).item() or mx.any(mx.isinf(g)).item())}
    res["gather_wiring"] = wiring
    # (2) serialization + bytes for one real-shape expert (3 projections)
    shapes = {"gate_proj": (640, 2560), "up_proj": (640, 2560), "down_proj": (2560, 640)}
    tot = {}; rt = {}
    for name, mode, gs, bits in [("a2", "affine", 64, 2), ("a4", "affine", 64, 4), ("mx4", "mxfp4", 32, 4)]:
        nb = 0; arrs = {}
        for p, sh in shapes.items():
            w = mx.array(rng.standard_normal(sh).astype(np.float32) * 0.02).astype(mx.bfloat16)
            r = mx.quantize(w, group_size=gs, bits=bits, mode=mode)
            for k, t in zip(("weight", "scales", "biases"), r):
                if t is None: continue
                if mode == "affine" and k != "weight": t = t.astype(mx.bfloat16)
                arrs[f"{p}.{k}"] = t; nb += t.nbytes
        f = OUT / f"_rt_{name}.safetensors"; mx.save_safetensors(str(f), arrs); back = mx.load(str(f))
        rt[name] = all(bool(mx.array_equal(back[k], v).item()) and back[k].dtype == v.dtype for k, v in arrs.items())
        tot[name] = nb; f.unlink()
    res["bytes_per_expert"] = tot; res["roundtrip_equal"] = rt
    # (3) Whallm algorithm vs MLX mxfp4 packed codes: edge cases + random
    edge = []
    row = np.zeros(32, np.float32); edge.append(row)                                  # all-zero group
    edge.append(np.array([0.0, -0.0] * 16, np.float32))                               # signed zeros
    base = np.array([6.0] + [0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0] * 4 + [-0.25, -0.75, -1.25, -1.75][:3], np.float32)[:32]
    edge.append(base); edge.append(-base)                                            # rounding midpoints at scale 2^0
    edge.append(np.array([4.0] + [0.1] * 31, np.float32))                            # max exactly power of 2
    edge.append(np.array([7.99] + [0.01] * 31, np.float32))                          # max just below next power
    edge.append(np.array([1e-30] + [0.0] * 31, np.float32))                          # tiny
    edge.append(np.array([3e38] + [1.0] * 31, np.float32))                           # huge
    E2 = np.stack(edge)
    rnd = rng.standard_normal((64, 256)).astype(np.float32) * 0.03
    cmp = {}
    for nm, v in [("edge", E2), ("random", rnd)]:
        vb = np.array(mx.array(v).astype(mx.bfloat16).astype(mx.float32))
        r = mx.quantize(mx.array(vb).astype(mx.bfloat16), group_size=32, bits=4, mode="mxfp4"); mx.eval(*r)
        mc = mlx_codes(r[0]); ms = np.array(r[1]).astype(np.uint8)
        wc, ws = wh_quant(vb)
        dv_m = dq_codes(mc, ms); dv_w = dq_codes(wc, ws)
        cmp[nm] = {"groups": int(ws.size), "scale_equal_frac": float((ms == ws).mean()), "code_equal_frac": float((mc == wc).mean()),
                   "value_equal_frac": float((dv_m == dv_w).mean()),
                   "mlx_dequant_matches_my_decoder": bool(np.allclose(np.array(mx.dequantize(r[0], r[1], group_size=32, bits=4, mode="mxfp4").astype(mx.float32)), dv_m, equal_nan=True)),
                   "first_mismatch_rows": sorted(set(np.where((mc != wc).any(1) | (ms != ws).any(1))[0].tolist()))[:10]}
    res["whallm_vs_mlx"] = cmp
    wjson(OUT / "mxfp4_check.json", res); log(json.dumps(res, indent=1)); log("check DONE")

if __name__ == "__main__":
    stage_check(None)

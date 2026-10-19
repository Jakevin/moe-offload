#!/usr/bin/env python3
"""Kernel parity fixtures v2.
Real-size references produced by the same vendor modules the Python loader uses; never loads model weights.
  tests/fixtures_v2.safetensors      : RoPE inv_freq (model config) + real-size MRoPE (q 24 heads, k 2 heads,
                                       positions 0..2047, 2D and 3D) -> committed
  tests/tmp/lmhead_real.safetensors   : synthetic lm_head with the REAL shape/quantization (4-bit, g32,
                                       2560 -> 248320) + x (1,49,2560); refs for M=49-then-slice and M=1 -> NOT committed (~400 MB)
"""
import os, sys
_BASE_SITE = "/Applications/oMLX.app/Contents/Resources/Python/framework-mlx-base/lib/python3.11/site-packages"
_OMLX_RES = "/Applications/oMLX.app/Contents/Resources"
for _p in (_OMLX_RES, _BASE_SITE):
    if _p in sys.path:
        sys.path.remove(_p)
    sys.path.insert(0, _p)
import mlx.core as mx
import mlx.nn as nn
from omlx.patches.mlx_vlm_qwen4_exp_compat import apply_mlx_vlm_qwen4_exp_compat_patch
apply_mlx_vlm_qwen4_exp_compat_patch()
from mlx_vlm.models.qwen3_5 import language as q35_lang

HERE = os.path.dirname(os.path.abspath(__file__))
mx.set_cache_limit(2 * 1024**3)


def rope_fixtures():
    mx.random.seed(20261018)
    t = {}
    # Exactly as Qwen3_5Attention builds it for this model (config: head_dim 256 * partial 0.25, theta 1e7 (int), section [11,11,10])
    rot = q35_lang.Qwen3_5RotaryEmbedding(int(256 * 0.25), max_position_embeddings=262144, base=10000000,
                                          mrope_section=[11, 11, 10])
    t["rope2.inv_freq"] = rot.inv_freq
    # keys: 2 KV heads, every position 0..2047
    k = mx.astype(mx.random.normal((1, 2, 2048, 256)), mx.bfloat16)
    kpos2 = mx.arange(0, 2048, dtype=mx.int32)[None, :]
    kpos3_text = mx.tile(kpos2[None], (3, 1, 1))
    kpos3_mm = mx.stack([kpos2, kpos2 + 7, kpos2 + 13], axis=0)
    # queries: 24 heads, 512 positions spread over 0..2047 (stride 4, odd offset)
    q = mx.astype(mx.random.normal((1, 24, 512, 256)), mx.bfloat16)
    qpos2 = (mx.arange(0, 512, dtype=mx.int32) * 4 + 3)[None, :]
    qpos3_text = mx.tile(qpos2[None], (3, 1, 1))
    t.update({"rope2.k": k, "rope2.k_pos2d": kpos2, "rope2.k_pos3d_text": kpos3_text, "rope2.k_pos3d_mm": kpos3_mm,
              "rope2.q": q, "rope2.q_pos2d": qpos2, "rope2.q_pos3d_text": qpos3_text})
    t["rope2.k_2d"] = rot.apply_rotary(k, k, kpos2)[0]
    t["rope2.k_3d_text"] = rot.apply_rotary(k, k, kpos3_text)[0]
    t["rope2.k_3d_mm"] = rot.apply_rotary(k, k, kpos3_mm)[0]
    t["rope2.q_2d"] = rot.apply_rotary(q, q, qpos2)[0]
    t["rope2.q_3d_text"] = rot.apply_rotary(q, q, qpos3_text)[0]
    mx.eval(*t.values())
    out = os.path.join(HERE, "fixtures_v2.safetensors")
    mx.save_safetensors(out, t)
    print(f"saved {len(t)} tensors {os.path.getsize(out)/2**20:.1f} MiB -> {out}")


def lmhead_fixture():
    mx.random.seed(7)
    V, D, G, BITS, L = 248320, 2560, 32, 4, 49
    shp = (V, D * BITS // 32)  # uint32 words, 8 packed 4-bit values each
    hi = mx.random.randint(0, 65536, shp, dtype=mx.uint32)
    lo = mx.random.randint(0, 65536, shp, dtype=mx.uint32)
    w = mx.bitwise_or(mx.left_shift(hi, mx.array(16, mx.uint32)), lo)
    scales = mx.astype(mx.random.uniform(0.002, 0.02, (V, D // G)), mx.bfloat16)
    biases = mx.astype(mx.random.normal((V, D // G)) * -0.05, mx.bfloat16)
    ql = nn.QuantizedLinear(D, V, bias=False, group_size=G, bits=BITS)
    ql.weight, ql.scales, ql.biases = w, scales, biases
    x = mx.astype(mx.random.normal((1, L, D)), mx.bfloat16)
    full = ql(x)                      # Python prefill: lm_head over all L rows
    ref_last = full[:, L - 1:L, :]    # then logits[:, -1, :]
    ref_m1 = ql(x[:, L - 1:L, :])     # what slicing first (old C++) computes
    mx.eval(w, scales, biases, x, ref_last, ref_m1)
    d = mx.max(mx.abs(ref_last.astype(mx.float32) - ref_m1.astype(mx.float32))).item()
    n = int(mx.sum(ref_last != ref_m1).item())
    os.makedirs(os.path.join(HERE, "tmp"), exist_ok=True)
    out = os.path.join(HERE, "tmp", "lmhead_real.safetensors")
    mx.save_safetensors(out, {"lmh.weight": w, "lmh.scales": scales, "lmh.biases": biases, "lmh.x": x,
                              "lmh.ref_last_M49": ref_last, "lmh.ref_M1": ref_m1})
    print(f"lm_head M=49-slice vs M=1 (python): n_diff={n}/{V} max_abs={d:.6g}; saved {os.path.getsize(out)/2**20:.0f} MiB -> {out}")


if __name__ == "__main__":
    rope_fixtures()
    lmhead_fixture()

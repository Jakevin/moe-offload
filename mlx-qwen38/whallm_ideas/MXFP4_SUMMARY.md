# MXFP4 for routed experts: format check and pilot conclusion (2026-10-18)

Hardware/software: Mac mini M4 Pro 64 GB, MLX 0.32.4 (also checked with MLX 0.32.2).

## MLX support (measured)
- `mx.quantize(..., mode="mxfp4", group_size=32, bits=4)` returns uint32 packed FP4 (e2m1) + uint8 E8M0 scales, no bias;
  `dequantize`, `quantized_matmul` and `gather_qmm(mode="mxfp4")` all work.
- `gather_qmm` wiring (`mxfp4_format_check.py`): 6 distinct experts, repeated + shuffled indices, all 12 output rows vs
  per-expert `quantized_matmul`: max abs diff 0.0 for affine2 g64, affine4 g64 and mxfp4 g32.
  Shape note: lhs must be `[n,1,1,K]` with indices `[n,1]`; `[n,1,K]` broadcasts to n x n.
- `mlx_mxfp4_check.py` is the first smoke test (single expert copied 3x, row 0 only); superseded by `mxfp4_format_check.py`.

## Size per expert (3 x 640x2560 = 4,915,200 weights, safetensors round-trip equal)
| format | bytes | bits/weight |
|---|---:|---:|
| affine 2-bit g64 (bf16 scale+bias) | 1,536,000 | 2.50 |
| affine 4-bit g64 (bf16 scale+bias) | 2,764,800 | 4.50 |
| MXFP4 g32 (E8M0 scale, no bias) | 2,611,200 | 4.25 |

## Encoder rule
MLX picks the scale exponent as ceil(log2(max/6)) (100% of 2,560 random groups). Whallm's MXFP4.swift (MIT) uses
floor(log2 max) - 2 and clips to 6. Same storage format; scales differ in ~33% of random groups. Random-weight
rel-L2: MLX 0.11519, Whallm rule 0.11572. Edge cases (signed zero, huge/tiny values) do not decode identically in
every row, so the two are not claimed to be the same algorithm.

## Accuracy (pilot conclusion)
On a small pilot (4 layers, 32 experts, real activations), MXFP4 expert-output relative error was about 1.25x that
of affine 4-bit g64 (median per-expert ratio 1.255), and both rule variants were worse than affine4. This does not
meet our pre-set threshold (<= 1.10x), so MXFP4 was not adopted for the 4-bit expert tier and kernel speed was not
benchmarked. The result is limited to this pilot and these two encoders.

# moe-offload

Experimental expert-offloading loaders for Mixture-of-Experts LLMs on Apple Silicon. The always-used weights stay in unified memory. Routed experts live on SSD and are read on demand into a fixed-budget, per-layer LRU cache. An optional router-based prefetch can read the next layer's experts ahead of time.

This is research code. It was tested only on the models and machines listed below.

## Contents

| Path | What |
|---|---|
| `mlx-qwen38/` | MLX loader for Qwen3.8-Flash-Next (MLX GSQ-RCO Q2 repack): `loader.py`, `bench.py`, `profile_run.py`, `run.sh` (RSS/swap watchdog), `expert_io.c` (native pthread pread pool, loaded via ctypes) |
| `mlx-qwen38/trace/` | Routing tracer + analysis (per-layer LRU hit simulation, next-layer router prediction recall) |
| `llamacpp-bolt/` | llama.cpp expert cache for NVIDIA Nemotron 3.5 Bolt 74B-A9B GGUF: `llama-moe-cache.patch`, `moe_loader.cpp`, `route_trace.cpp`, `analyze_routes.py`, `bench.py` |

## Measured results (Mac mini M4 Pro, 64 GB, internal SSD)

All cached runs produced greedy output identical to the reference run on the test prompts. Swap growth was 0 in every run except the 16 GB Mac rows below.

**Qwen3.8-Flash-Next (125B MoE, 512 experts/layer, top-10), MLX.** 8 prompts (code / Chinese / math / English chat), 128 tokens each. Expert cache = 20% of routed experts (about 7.4 GB), layer 0 pinned.

| Config | Decode tok/s | Prefill tok/s | Effective hit rate | Peak RSS |
|---|---|---|---|---|
| Reference tracer (28 layers resident) | 2.52 | 5.3 | n/a | 24.9 GiB |
| v1, Python pread, no prefetch | 8.51 | 14.8 | 84.3% | 11.6 GiB |
| v2, native pread, fewer syncs, no prefetch | 9.17 | 18.9 | 84.3% | 10.3 GiB |
| v2 + router prefetch top-12, max 3 per layer | 9.73 | 19.4 | 94.7% | 8.9 GiB |

Profile of v1 per token (118 ms total): GPU 27%, Python and `mx.eval` syncs 41%, SSD reads 29%, bookkeeping 3%. A fully resident llama.cpp GGUF run of this model (different setup) reached 24.6 tok/s. The remaining gap is mostly Python per-layer synchronization.

Next-layer router prediction (mixing the 4 hyper-connection streams the way layer l+1 does, then applying layer l+1's norm and router): recall@10 / @16 / @24 = 83.6% / 94.1% / 96.7%. This comes from a short trace of 2 prompts.

**Nemotron 3.5 Bolt 74B-A9B (IQ2_XXS GGUF, 240 experts/layer, top-9), llama.cpp Metal.** 8 prompts, 64 tokens each. The full-model baseline was 13.9 tok/s at 20.8 GiB.

| Cache | Decode tok/s | Hit rate | Peak RSS |
|---|---|---|---|
| 4 GB | 11.07 | 86.1% | 9.9 GiB |
| 8 GB | 11.05 | 93.8% | 13.9 GiB |
| 12 GB | 11.28 | 97.0% | 17.9 GiB |

On a Mac mini M4 with 16 GB: the 4 GB cache ran at 4.39 tok/s (86.1% hit, 8.5 GiB peak). The 8 GB cache thrashed and was aborted.

Note: stock llama.cpp Metal lacks `kernel_mul_mm_id_map0` for 9 active experts, so Bolt needs `-ub 31`. The patch adds the instantiation.

## Usage (sketch)

- llama.cpp: `git checkout 620d2225cc436ab0b001e3200e590e3b286ef9ab && git apply llama-moe-cache.patch`, build with `-DGGML_METAL=ON`, then `make -C llamacpp-bolt` (set `LLAMA_DIR`).
- MLX: needs an MLX + mlx-vlm build with `qwen4_exp` support. The scripts default to oMLX.app's bundled Python. Set `QWEN38_MODEL` to the model directory and `MOE_WORKDIR` to a working directory, then run `./run.sh <python> loader.py ...`.

## License

Apache-2.0

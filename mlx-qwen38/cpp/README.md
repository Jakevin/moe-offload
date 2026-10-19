# C++ runner for Qwen3.8-Flash-Next (MLX C++)

A C++ port of the `loader.py` decode path on the MLX C++ API. It uses the same SSD expert offload: experts are read into a fixed per-layer LRU cache by the native `pread` pool (`third_party/expert_io.c`).

Its goal is **bit-exact** parity with the Python MLX loader: identical greedy token ids, identical per-step logits (raw bits), and an identical final cache state. Research code; it was tested only on the machine below.

## Build

Requires CMake ≥ 3.24 and the same MLX build the Python loader uses. Bit-exactness depends on identical kernels. The default `MLX_ROOT` points at oMLX.app's bundled MLX 0.32.2; override it with `-DMLX_ROOT=...`.

```bash
cd mlx-qwen38/cpp
cmake -S . -B build && cmake --build build -j
```

## Tests

Run from `mlx-qwen38/cpp`:

```bash
python3 tests/gen_fixtures_v2.py   # writes tests/tmp/lmhead_real.safetensors (~380 MB, not committed); needs the mlx-vlm qwen4_exp vendor modules
./build/test_budget                # memory-budget policy (assertions enabled even in Release)
./build/test_io_sched              # P1 I/O scheduler: cancel/promote, pinned slots, 250-step stress
./build/test_kernels               # kernel parity vs the Python modules, strict bit equality
```

`test_kernels` covers RMSNorm, MRoPE (real size, 2D/3D), hyper-connection kernels, PLE, GDN linear attention, full attention, the router, and a real-shape 4-bit lm_head. The fixtures are synthetic: random tensors pushed through the vendor Python modules. No model weights are included.

## Run

```bash
export QWEN38_MODEL=/path/to/Qwen3.8-Flash-Next-GSQ-RCO-Q2-mlx
./build/qwen38_decode --check-weights            # header-only check of every trunk / expert / PLE tensor
./build/qwen38_decode --ids prompts.json --prompt-index 0 --n 128 --no-stop \
    --io exact --budget fixed --out out/run.json \
    [--logits-out out/run.logits.npy] [--state-out out/run.state.safetensors]
```

- **Prompt file:** `prompts.json` is `{"prompts": [{"ids": [...]}, ...]}`, holding token ids from the model's tokenizer. The runner prints only a SHA-256 of the generated ids, never tokens or text.
- **I/O:** `--io exact` is the serial path that matches Python. `--io p1 --io-threads 6` is an urgent-queue scheduler with prefetch cancel/promote and pinned slots.
- **Budget:** `--budget fixed` gives 20% of the routed experts (102 slots/layer). `--budget auto` sizes the cache from free memory, capped at `min(19.5 GB, 80% of clean available memory before load)` minus 3 GiB headroom.
- **Stress testing:** `--slot-cap N` sets the slot count per layer. `--force-miss-rate R --force-miss-seed S` evicts a random fraction of cached experts before each layer.
- **Memory guard:** the process aborts if its phys_footprint exceeds that cap or if swap grows by more than 8 GiB.
- **Comparing runs:** `tools/cmp_exact.py` compares two runs and exits 0 only if ids, logits and state are all bit-identical.

**Limitation:** full attention is dense (the QSA sparse block selection is not ported). It is therefore only valid for total length (prompt + generated) ≤ 2048 tokens.

## Measured (Mac mini M4 Pro, 64 GB, internal SSD)

Gate setup: 4 prompts (40–119 tokens), 128 greedy tokens each, `--io exact --budget fixed`. The reference is the authors' Python MLX loader run with the same settings.

- **Exactness:** all 4 prompts are bit-exact. Token ids match 128/128. Per-step logits differ in 0 raw bits. Final cache state matches in 122/122 tensors (GDN conv/ssm, PLE history, KV and indexer caches).
- **Speed and memory:** decode 9.0–9.6 tok/s, hit rate 91.8–94.8%, peak RSS about 11.5 GB, swap growth 0.
- **Other modes (prompt 0, 128 tokens):** each run is bit-identical to both the Python reference and the C++ serial path:
  - forced miss 50%;
  - `--io p1`;
  - `--io p1` + forced miss 50% + 16 slots/layer;
  - `--io p1 --budget auto`: 8.2 tok/s, peak phys_footprint 16.9 GB.

Note: the reference loader build includes I/O scheduling and accounting that `../expert_io.c` / `../loader.py` in this repo do not have yet, so `third_party/expert_io.c` is a newer superset. Parity against the published `loader.py` has not been re-measured.

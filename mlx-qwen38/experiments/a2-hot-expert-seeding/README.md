# A2: keep hot experts in the expert cache after prefill — NOT ADOPTED

**Status: not adopted (negative result).** This code is *not* merged into `mlx-qwen38/loader.py`; it is kept here only as a record so the experiment can be reproduced.

## Idea and method
After a long prefill, each cached MoE layer's expert cache (LRU) holds whatever the last prefill chunk left behind. A2 instead re-fills it with the experts this prompt routed to most, hoping to raise the decode hit rate.

- `a2_seed.py` wraps `CachedSwitchGLU.__call__` at runtime (the loader file is not modified). The original prefill runs unchanged and its output is evaluated first; only then are slots re-filled. Routed-token counts per layer are accumulated at runtime over the prefill calls of one prompt; no precomputed expert ranking is used or shipped.
- Modes: `off` (baseline), `topk:F` (ensure the top `int(F × capacity)` experts are resident, others stay), `full` (resident set = top-`capacity` experts), `thresh` (see limitations). Selected experts are placed at the MRU end, hottest last.
- Setup: Mac mini M4 Pro 64 GB, 20% expert-cache budget (`alloc_mode=even`), existing prefetch on (`top_m=12`, `cap=3`), OS page cache partly warm (expert-file residency 0.59–0.76).
- `a2_trial.py exact`: compares tokens, the sha256 of every step's logits, and the full model state after prefill and at the end (incl. CPU PLE history) against the first `off` run.
- `a2_trial.py speed`: `off` vs `topk:0.25`, prompts `doc6k` / `p6k` (6000 tokens) / `short`, 200 new tokens, 6 rounds in alternating AB/BA order, expert LRU reset before every run. Speed runs are refused unless the exact stage passed.
- `analyze_a2.py`: per round, the mean log ratio of the two long prompts' decode tok/s; 95% t-CI over rounds. Pre-set gate: long-decode geometric mean ≥ 1.05× **and** CI lower bound > 1 (plus no-regression gates on per-prompt decode, TTFT, request time and swap).

## Results (from `A2_RESULT.md`, `out/speed1_analysis.json`)
- **Correctness:** `off`, `off` (repeat), `topk:0.25`, `topk:0.5`, `full` were bit-identical on `story` (31 tok) and `doc6k` (6000 tok) × 32 tokens: tokens, every logits step, state after prefill and at the end (`out/exact1.jsonl`). The speed runs also gave identical tokens for A and B.
- **Long-prompt decode speedup: 1.008×, 95% CI [0.967, 1.051] → FAIL** (gate ≥ 1.05× and CI low > 1).
- Decode ratio doc6k 1.005 / p6k 1.011 / short 1.012; TTFT ratio 0.985 / 0.997 / 1.033; request-time ratio 0.988 / 0.993 / 0.991.
- Hit rate rose by only ~0.2 percentage points: doc6k 0.9356 → 0.9378, p6k 0.9408 → 0.9410 (short 0.9532 → 0.9530).
- Experts loaded by seeding (median) 924 / 582 / 53; seeding + eval time 0.83 / 0.78 / 0.28 s; swap growth 0.

Conclusion: on this machine, with a 20% expert cache, the existing prefetch and a partly warm OS page cache, keeping the top-25%-capacity hot experts after prefill keeps outputs and state identical but gives no decode speedup that meets the pre-set gate. This does **not** show that all seeding strategies are ineffective.

## Reproduce
Requirements are the same as for `mlx-qwen38/` (MLX + mlx-vlm with `qwen4_exp`, oMLX.app's bundled Python by default; see the top-level README).
```bash
cd mlx-qwen38/experiments/a2-hot-expert-seeding
export QWEN38_MODEL=/path/to/Qwen3.8-Flash-Next-GSQ-RCO-Q2-mlx
export A2_PROMPTS_SRC=/path/to/prompts.json   # {"ids": {"short":[...], "story":[...], "doc6k":[...]}, "eos": ...}
python3 make_prompts.py                         # writes out/prompts_a2.json (token ids; p6k is generated)
# run_a2.sh waits until the machine is idle, sets an RSS budget (80% of free memory) and aborts on +8 GB swap
./run_a2.sh loose exact1 exact                   # default: story,doc6k x 32 tokens, arms off,off,topk:0.25,topk:0.5,full
./run_a2.sh loose speed1 speed --prompts doc6k,p6k,short --arms off,topk:0.25 --n-new 200 --rounds 6 --exact-ok out/exact1.jsonl
python3 analyze_a2.py out/speed1.jsonl off topk:0.25 > out/speed1_analysis.json
```
The `short` / `story` / `doc6k` prompt token ids come from a local benchmark file that is not included here; supply your own prompts in the same format. Output rows contain only aggregate numbers (speed, hit rate, number of seeded experts, bit-identity checks, state hashes) — no per-expert counts or routing logs. Optional env: `MLX_QWEN38_DIR` (defaults to `../..`), `A2_BUSY_PATTERN`, `A2_HEAVY_PATTERN`.

## Limitations
- Only `topk:0.25` was speed-tested. `topk:0.5` and `full` were checked for correctness only; their speed was not measured.
- It cannot be claimed that the gain was lost because decode evicted the pre-loaded experts from the cache; this was not measured.
- Known bug in `thresh` mode: `parse_arm` never sets `A2.thresh`, so `thresh:4` actually behaves like `full`. This mode was not used in the experiment.
- Single machine, warm OS page cache, a single full prefill per prompt (not the server's chunked prefill path), 6 rounds.

## Files
`a2_seed.py`, `a2_trial.py`, `run_a2.sh`, `analyze_a2.py`, `make_prompts.py`, `A2_RESULT.md` (original summary), `out/exact1.jsonl`, `out/speed1.jsonl`, `out/speed1_analysis.json`.

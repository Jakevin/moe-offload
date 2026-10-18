# A2 (seed hot experts after prefill) — result 2026-10-18 (not adopted)
Code: a2_seed.py (wrapper around mlx-qwen38/loader.py CachedSwitchGLU.__call__, loader unmodified), a2_trial.py (v2, design-review fixes), run_a2.sh, analyze_a2.py, make_prompts.py.
Design review / verdict: by the supervising professor (review logs not included).
Exact (out/exact1.jsonl): story 31 tok + doc6k 6000 tok x 32 tokens; off/off/topk:0.25/topk:0.5/full all bit-identical (tokens, every logits step, state after prefill and at end incl. PLE history).
Speed (out/speed1.jsonl, out/speed1_analysis.json): off vs topk:0.25, doc6k/p6k/short x 200 tokens, 6 rounds AB/BA, LRU reset per run, OS page cache warm (resident 0.59-0.76).
- long decode gmean 1.008x, 95% CI [0.967, 1.051] -> FAIL (gate >= 1.05x and CI low > 1)
- decode ratio doc6k 1.005 / p6k 1.011 / short 1.012; TTFT ratio 0.985 / 0.997 / 1.033; request ratio 0.988 / 0.993 / 0.991
- hit rate doc6k 0.9356 -> 0.9378, p6k 0.9408 -> 0.9410; seed loads (median) 924 / 582 / 53 experts, seed+eval 0.83 / 0.78 / 0.28 s; swap growth 0
Verdict: correctness passed, performance not shown; stop this configuration; topk:0.5 / full only correctness-tested.
Wording: under this machine, 20% expert cache, existing prefetch and a partly warm OS page cache, keeping the top-25%-capacity hot experts after prefill keeps outputs and state identical but gives no decode speedup meeting the pre-set gate; this does not show that all seeding strategies are ineffective.

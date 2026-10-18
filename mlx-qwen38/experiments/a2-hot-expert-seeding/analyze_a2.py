"""A2 speed analysis (pre-registered gates): per round, mean of the two long prompts' log(B/A) decode ratios; 95% t-CI over rounds.
Gates: long decode gmean >= 1.05 and CI low > 1; each prompt (gmean over rounds) not slower than 0.95; TTFT ratio
<= 1.05 per prompt; request time not worse (ratio <= 1.0) on long prompts; no swap growth during runs."""
import json, math, sys
from collections import defaultdict
import numpy as np
T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262}
rows = [json.loads(l) for l in open(sys.argv[1])]; A, B = sys.argv[2], sys.argv[3]; LONG = ("doc6k", "p6k")
R = {(r["round"], r["prompt"], r["arm"]): r for r in rows}
rounds = sorted({r["round"] for r in rows}); prompts = sorted({r["prompt"] for r in rows})
def lr(rd, p, k): return math.log(R[(rd, p, B)][k] / R[(rd, p, A)][k])
def gm(v): return float(math.exp(np.mean(v)))
out = {"A": A, "B": B, "rounds": len(rounds)}
per_round = [np.mean([lr(rd, p, "decode_tok_s") for p in LONG]) for rd in rounds]
m, s = float(np.mean(per_round)), float(np.std(per_round, ddof=1)); t = T975[len(rounds) - 1]; hw = t * s / math.sqrt(len(rounds))
out["long_decode"] = {"gmean": math.exp(m), "ci95": [math.exp(m - hw), math.exp(m + hw)], "per_round": [math.exp(x) for x in per_round]}
for k in ("decode_tok_s", "ttft_s", "request_s", "hit_rate"):
    out[k] = {p: {"ratio_gmean": gm([lr(rd, p, k) for rd in rounds]), "A_median": float(np.median([R[(rd, p, A)][k] for rd in rounds])),
                  "B_median": float(np.median([R[(rd, p, B)][k] for rd in rounds]))} for p in prompts}
out["seed_loaded_experts_median"] = {p: float(np.median([R[(rd, p, B)]["seed"]["loaded_experts"] for rd in rounds])) for p in prompts}
out["seed_s_median"] = {p: float(np.median([R[(rd, p, B)]["seed"]["seed_s"] + R[(rd, p, B)]["seed"]["eval_s"] for rd in rounds])) for p in prompts}
out["swap_growth_mb_max"] = max(r["swap_end_mb"] - r["swap_start_mb"] for r in rows)
out["page_resident_before"] = {f"{a}": [min(r["page_resident_before"] for r in rows if r["arm"] == a), max(r["page_resident_before"] for r in rows if r["arm"] == a)] for a in (A, B)}
out["tokens_identical_A_vs_B"] = all(R[(rd, p, A)]["tokens_sha"] == R[(rd, p, B)]["tokens_sha"] for rd in rounds for p in prompts)
g = out["long_decode"]
gates = {"long_decode_gmean>=1.05": g["gmean"] >= 1.05, "long_decode_ci_low>1": g["ci95"][0] > 1,
         "each_prompt_decode>=0.95": all(out["decode_tok_s"][p]["ratio_gmean"] >= 0.95 for p in prompts),
         "ttft_ratio<=1.05": all(out["ttft_s"][p]["ratio_gmean"] <= 1.05 for p in prompts),
         "long_request_ratio<=1.0": all(out["request_s"][p]["ratio_gmean"] <= 1.0 for p in LONG),
         "no_swap_growth": out["swap_growth_mb_max"] <= 0}
out["gates"] = gates; out["PASS"] = all(gates.values())
print(json.dumps(out, indent=1))

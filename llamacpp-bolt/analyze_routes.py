#!/usr/bin/env python3
"""
Analyze MoE expert routing logs produced by `route_trace`.
Computes:
- Per-domain expert frequency and distribution entropy (effective #experts)
- Top-k% expert activation coverage (10%, 20%, 30%, 50%) globally and per-domain
- Cross-domain Jaccard overlap of top-k% experts (code, math, zh, chat)
- Temporal locality (t+1 reuse, window-4 reuse)
- Per-layer LRU and LFU cache hit rate simulation across cache budgets
"""

import argparse
import collections
import json
import os
import struct
import sys
import numpy as np


def read_gguf_layer_expert_counts(gguf_path):
    """Read per-layer expert counts and sizes from GGUF header if available."""
    if not gguf_path or not os.path.exists(gguf_path):
        return {}, 0
    layer_experts = {}
    layer_expert_bytes = {}
    try:
        with open(gguf_path, "rb") as f:
            magic, version, n_tensors, n_kv = struct.unpack("<4sIQQ", f.read(24))
            if magic != b"GGUF":
                return {}, 0

            def read_str():
                l = struct.unpack("<Q", f.read(8))[0]
                return f.read(l).decode("utf-8", errors="replace")

            def skip_val(t):
                if t in (0, 1, 7):
                    f.read(1)
                elif t in (2, 3):
                    f.read(2)
                elif t in (4, 5, 6):
                    f.read(4)
                elif t == 8:
                    read_str()
                elif t == 9:
                    at, al = struct.unpack("<IQ", f.read(12))
                    for _ in range(al):
                        skip_val(at)
                elif t in (10, 11, 12):
                    f.read(8)

            for _ in range(n_kv):
                read_str()
                vt = struct.unpack("<I", f.read(4))[0]
                skip_val(vt)

            tensors = []
            for _ in range(n_tensors):
                name = read_str()
                n_dims = struct.unpack("<I", f.read(4))[0]
                dims = [struct.unpack("<Q", f.read(8))[0] for _ in range(n_dims)]
                ttype, offset = struct.unpack("<IQ", f.read(12))
                tensors.append((name, dims, ttype, offset))

            tensors_sorted = sorted(tensors, key=lambda x: x[3])
            for i, (name, dims, ttype, offset) in enumerate(tensors_sorted):
                if name.startswith("blk.") and ("_exps.weight" in name):
                    parts = name.split(".")
                    il = int(parts[1])
                    n_exp = dims[2] if len(dims) >= 3 else dims[-1]
                    layer_experts[il] = int(n_exp)
                    if i + 1 < len(tensors_sorted):
                        sz = tensors_sorted[i + 1][3] - offset
                        layer_expert_bytes[il] = layer_expert_bytes.get(il, 0) + sz
    except Exception as e:
        print(f"Warning: could not parse GGUF header: {e}", file=sys.stderr)
    return layer_experts, layer_expert_bytes


def main():
    ap = argparse.ArgumentParser(description="Analyze route_trace JSONL output")
    ap.add_argument("jsonl", help="Path to routes.jsonl from route_trace")
    ap.add_argument("--meta", default=None, help="Optional routes_meta.json path")
    ap.add_argument("--gguf", default=None, help="Optional GGUF model path for exact per-layer expert counts")
    ap.add_argument("--out", default="route_analysis.json", help="Output JSON report path")
    ap.add_argument("--phase", default="all", choices=["all", "decode", "prefill"],
                    help="Which phase to analyze for domain frequency/Jaccard (default: all, plus separate decode)")
    args = ap.parse_args()

    gguf_path = args.gguf
    if not gguf_path and args.meta and os.path.exists(args.meta):
        try:
            gguf_path = json.load(open(args.meta)).get("model")
        except Exception:
            pass

    layer_n_experts, layer_exp_bytes = read_gguf_layer_expert_counts(gguf_path)

    # Load JSONL records
    # Structure: records_by_layer[layer] = list of (prompt_idx, domain, phase, token_idx, experts_list)
    layers_set = set()
    domains_set = set()
    prompts_set = set()
    records = []

    with open(args.jsonl, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            layers_set.add(r["layer"])
            domains_set.add(r["domain"])
            prompts_set.add(r["prompt_idx"])
            records.append(r)

    if not records:
        print("No records found in", args.jsonl, file=sys.stderr)
        sys.exit(1)

    layers = sorted(layers_set)
    domains = sorted(domains_set)
    top_k_used = len(records[0]["experts"])

    # Determine number of experts per layer
    for il in layers:
        if il not in layer_n_experts:
            max_id = max(max(r["experts"]) for r in records if r["layer"] == il)
            # Round up to standard multiple of 16 or power of 2
            layer_n_experts[il] = max(max_id + 1, ((max_id // 16) + 1) * 16)

    fracs = [0.05, 0.10, 0.20, 0.30, 0.50]

    # Organize data: per layer, per domain, per phase
    # counts[il][domain][phase] = np.ndarray(n_exp)
    counts = {
        il: {
            d: {"all": np.zeros(layer_n_experts[il], dtype=np.int64),
                "prefill": np.zeros(layer_n_experts[il], dtype=np.int64),
                "decode": np.zeros(layer_n_experts[il], dtype=np.int64)}
            for d in domains + ["ALL"]
        }
        for il in layers
    }

    # Also keep token sequences per (prompt_idx, layer) for temporal locality & cache simulation
    seq_by_prompt_layer = collections.defaultdict(lambda: {"prefill": [], "decode": [], "domain": ""})

    for r in records:
        il = r["layer"]
        d = r["domain"]
        ph = r["phase"]
        pidx = r["prompt_idx"]
        exps = r["experts"]
        for e in exps:
            if 0 <= e < layer_n_experts[il]:
                counts[il][d][ph][e] += 1
                counts[il][d]["all"][e] += 1
                counts[il]["ALL"][ph][e] += 1
                counts[il]["ALL"]["all"][e] += 1
        entry = seq_by_prompt_layer[(pidx, il)]
        entry["domain"] = d
        entry[ph].append(exps)

    def calc_coverage_and_stats(cnt_arr, n_exp):
        tot = int(cnt_arr.sum())
        if tot == 0:
            return {"total_activations": 0, "coverage": {f"{int(f*100)}%": 0.0 for f in fracs},
                    "effective_experts": 0.0, "max_share": 0.0, "unused_experts": int(n_exp)}
        s = np.sort(cnt_arr)[::-1] / tot
        cov = {}
        for f in fracs:
            k_cnt = max(1, int(round(f * n_exp)))
            cov[f"{int(f*100)}%"] = float(s[:k_cnt].sum())
        p = s[s > 0]
        eff = float(np.exp(-(p * np.log(p)).sum()))
        return {
            "total_activations": tot,
            "coverage": cov,
            "effective_experts": eff,
            "effective_frac": eff / n_exp,
            "max_share": float(s[0]),
            "unused_experts": int((cnt_arr == 0).sum()),
            "top10_experts": [int(x) for x in np.argsort(-cnt_arr)[:10]],
        }

    # 1. Per-domain expert frequency & top-k% coverage (averaged over layers and per-layer)
    domain_summary = {}
    for d in domains + ["ALL"]:
        domain_summary[d] = {}
        for ph in ["all", "decode", "prefill"]:
            layer_stats = [calc_coverage_and_stats(counts[il][d][ph], layer_n_experts[il]) for il in layers]
            valid = [st for st in layer_stats if st["total_activations"] > 0]
            if not valid:
                continue
            domain_summary[d][ph] = {
                "total_activations": int(sum(st["total_activations"] for st in valid)),
                "mean_coverage": {
                    k: float(np.mean([st["coverage"][k] for st in valid]))
                    for k in valid[0]["coverage"]
                },
                "mean_effective_experts": float(np.mean([st["effective_experts"] for st in valid])),
                "mean_effective_frac": float(np.mean([st["effective_frac"] for st in valid])),
                "mean_max_share": float(np.mean([st["max_share"] for st in valid])),
                "mean_unused_experts": float(np.mean([st["unused_experts"] for st in valid])),
            }

    # 2. Jaccard overlap across domains for top-k% experts (for k = 10%, 20%, 30%, 50%)
    jaccard_cross_domain = {}
    for ph in ["all", "decode"]:
        jaccard_cross_domain[ph] = {}
        for f in [0.10, 0.20, 0.30, 0.50]:
            f_key = f"top_{int(f*100)}%"
            jaccard_cross_domain[ph][f_key] = {}
            for i, d1 in enumerate(domains):
                jaccard_cross_domain[ph][f_key][d1] = {}
                for j, d2 in enumerate(domains):
                    if i == j:
                        jaccard_cross_domain[ph][f_key][d1][d2] = 1.0
                        continue
                    j_vals = []
                    for il in layers:
                        c1 = counts[il][d1][ph]
                        c2 = counts[il][d2][ph]
                        if c1.sum() == 0 or c2.sum() == 0:
                            continue
                        k_cnt = max(1, int(round(f * layer_n_experts[il])))
                        top1 = set(np.argsort(-c1)[:k_cnt].tolist())
                        top2 = set(np.argsort(-c2)[:k_cnt].tolist())
                        inter = len(top1 & top2)
                        union = len(top1 | top2)
                        j_vals.append(inter / union if union > 0 else 0.0)
                    jaccard_cross_domain[ph][f_key][d1][d2] = float(np.mean(j_vals)) if j_vals else 0.0

    # Also compute Jensen-Shannon / Cosine similarity of full frequency distributions across domains
    cosine_cross_domain = {}
    for ph in ["all", "decode"]:
        cosine_cross_domain[ph] = {}
        for d1 in domains:
            cosine_cross_domain[ph][d1] = {}
            for d2 in domains:
                cos_vals = []
                for il in layers:
                    c1 = counts[il][d1][ph].astype(np.float64)
                    c2 = counts[il][d2][ph].astype(np.float64)
                    n1, n2 = np.linalg.norm(c1), np.linalg.norm(c2)
                    if n1 > 0 and n2 > 0:
                        cos_vals.append(float(np.dot(c1, c2) / (n1 * n2)))
                cosine_cross_domain[ph][d1][d2] = float(np.mean(cos_vals)) if cos_vals else 0.0

    # 3. Per-layer breakdown and temporal locality / cache simulations
    cache_fracs = [0.10, 0.20, 0.30, 0.50]
    prompt_indices = sorted(prompts_set)
    per_layer_report = []

    for il in layers:
        n_exp = layer_n_experts[il]
        st_all = calc_coverage_and_stats(counts[il]["ALL"]["all"], n_exp)
        st_dec = calc_coverage_and_stats(counts[il]["ALL"]["decode"], n_exp)

        # Temporal locality during decode: t+1 reuse and window-4 reuse
        reuse_t1_num = reuse_t1_den = 0
        reuse_w4_num = reuse_w4_den = 0
        for pidx in prompt_indices:
            dec_seq = seq_by_prompt_layer[(pidx, il)]["decode"]
            for t in range(len(dec_seq) - 1):
                s_cur = set(dec_seq[t])
                s_nxt = set(dec_seq[t + 1])
                reuse_t1_num += len(s_cur & s_nxt)
                reuse_t1_den += len(dec_seq[t + 1])
            for t in range(4, len(dec_seq)):
                prev4 = set(e for step in dec_seq[t - 4:t] for e in step)
                reuse_w4_num += sum(1 for e in dec_seq[t] if e in prev4)
                reuse_w4_den += len(dec_seq[t])

        # Simulate LRU and LFU cache over decode tokens (warmed by prefill)
        lru_hits = {}
        lfu_hits = {}
        for cf in cache_fracs:
            cap = max(top_k_used, int(round(cf * n_exp)))
            # LRU
            lru = collections.OrderedDict()
            hits = tot = 0
            for pidx in prompt_indices:
                for step in seq_by_prompt_layer[(pidx, il)]["prefill"]:
                    for e in step:
                        lru[e] = 1
                        lru.move_to_end(e)
                        while len(lru) > cap:
                            lru.popitem(last=False)
                for step in seq_by_prompt_layer[(pidx, il)]["decode"]:
                    for e in step:
                        tot += 1
                        if e in lru:
                            hits += 1
                            lru.move_to_end(e)
                        else:
                            lru[e] = 1
                            while len(lru) > cap:
                                lru.popitem(last=False)
            lru_hits[f"{int(cf*100)}%"] = hits / tot if tot > 0 else 0.0

            # LFU (with recency tie-breaking)
            lfu_freq = {}
            lfu_time = {}
            step_clock = 0
            hits = tot = 0
            for pidx in prompt_indices:
                for step in seq_by_prompt_layer[(pidx, il)]["prefill"]:
                    for e in step:
                        step_clock += 1
                        if e in lfu_freq:
                            lfu_freq[e] += 1
                            lfu_time[e] = step_clock
                        else:
                            if len(lfu_freq) >= cap:
                                victim = min(lfu_freq.keys(), key=lambda k: (lfu_freq[k], lfu_time[k]))
                                del lfu_freq[victim]
                                del lfu_time[victim]
                            lfu_freq[e] = 1
                            lfu_time[e] = step_clock
                for step in seq_by_prompt_layer[(pidx, il)]["decode"]:
                    for e in step:
                        step_clock += 1
                        tot += 1
                        if e in lfu_freq:
                            hits += 1
                            lfu_freq[e] += 1
                            lfu_time[e] = step_clock
                        else:
                            if len(lfu_freq) >= cap:
                                victim = min(lfu_freq.keys(), key=lambda k: (lfu_freq[k], lfu_time[k]))
                                del lfu_freq[victim]
                                del lfu_time[victim]
                            lfu_freq[e] = 1
                            lfu_time[e] = step_clock
            lfu_hits[f"{int(cf*100)}%"] = hits / tot if tot > 0 else 0.0

        per_layer_report.append({
            "layer": il,
            "n_experts": n_exp,
            "bytes_per_expert": layer_exp_bytes.get(il, 0) / n_exp if il in layer_exp_bytes else 0,
            "all": st_all,
            "decode": st_dec,
            "reuse_t1": reuse_t1_num / reuse_t1_den if reuse_t1_den > 0 else 0.0,
            "reuse_w4": reuse_w4_num / reuse_w4_den if reuse_w4_den > 0 else 0.0,
            "lru_decode_hit": lru_hits,
            "lfu_decode_hit": lfu_hits,
        })

    # Global averages across layers
    summary = {
        "n_moe_layers": len(layers),
        "top_k_used": top_k_used,
        "domains": domains,
        "n_prompts": len(prompt_indices),
        "total_records": len(records),
        "mean_reuse_t1": float(np.mean([x["reuse_t1"] for x in per_layer_report])),
        "mean_reuse_w4": float(np.mean([x["reuse_w4"] for x in per_layer_report])),
        "mean_lru_decode_hit": {
            k: float(np.mean([x["lru_decode_hit"][k] for x in per_layer_report]))
            for k in per_layer_report[0]["lru_decode_hit"]
        },
        "mean_lfu_decode_hit": {
            k: float(np.mean([x["lfu_decode_hit"][k] for x in per_layer_report]))
            for k in per_layer_report[0]["lfu_decode_hit"]
        },
    }

    report = {
        "summary": summary,
        "domain_summary": domain_summary,
        "jaccard_cross_domain": jaccard_cross_domain,
        "cosine_cross_domain": cosine_cross_domain,
        "per_layer": per_layer_report,
    }

    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)

    # Print concise markdown report to stdout
    print(f"=== Route Trace Analysis ({args.jsonl}) ===")
    print(f"MoE Layers: {len(layers)} | Top-K per token: {top_k_used} | Prompts: {len(prompt_indices)} | Domains: {', '.join(domains)}")
    print(f"Temporal Locality (Decode): P(reuse at t+1) = {summary['mean_reuse_t1']*100:.1f}%, P(reuse in window 4) = {summary['mean_reuse_w4']*100:.1f}%")
    print()
    print("--- Per-Domain Expert Frequency & Top-k% Coverage (All Tokens) ---")
    print(f"{'Domain':<8} | {'Acts':>8} | {'Eff#Exp':>8} | {'Eff%':>6} | {'Top10%':>7} | {'Top20%':>7} | {'Top30%':>7} | {'Top50%':>7} | {'Unused':>6}")
    print("-" * 86)
    for d in domains + ["ALL"]:
        st = domain_summary[d].get("all")
        if not st:
            continue
        cov = st["mean_coverage"]
        print(f"{d:<8} | {st['total_activations']:>8d} | {st['mean_effective_experts']:>8.1f} | {st['mean_effective_frac']*100:>5.1f}% | "
              f"{cov['10%']*100:>6.1f}% | {cov['20%']*100:>6.1f}% | {cov['30%']*100:>6.1f}% | {cov['50%']*100:>6.1f}% | {st['mean_unused_experts']:>6.1f}")
    print()
    print("--- Cross-Domain Jaccard Overlap (Top-20% Experts, All Tokens) ---")
    j20 = jaccard_cross_domain["all"]["top_20%"]
    header = f"{'Domain':<8} | " + " | ".join(f"{d:>8}" for d in domains)
    print(header)
    print("-" * len(header))
    for d1 in domains:
        row = f"{d1:<8} | " + " | ".join(f"{j20[d1][d2]:>8.3f}" for d2 in domains)
        print(row)
    print()
    print("--- Simulated Cache Hit Rate on Decode (Warmed by Prefill) ---")
    for pol_name, pol_key in [("LRU", "mean_lru_decode_hit"), ("LFU", "mean_lfu_decode_hit")]:
        hits = summary[pol_key]
        print(f"{pol_name:<4}: " + ", ".join(f"{k} cache -> {v*100:.2f}% hit" for k, v in hits.items()))
    print(f"\nSaved full analysis to {args.out}")


if __name__ == "__main__":
    main()

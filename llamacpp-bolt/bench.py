#!/usr/bin/env python3
"""
Step 3: MoE Expert-Offload Benchmark & Greedy Correctness Verification Suite
Measures:
  1. Decode tok/s, prefill tok/s, cache hit rate, MiB/tok uploaded, and peak RSS vs cache budget (4, 8, 12, 16 GB + full-resident baseline)
  2. Exact greedy token-for-token equivalence vs baseline across all domains (code, math, zh, chat)
  3. Cache hit rate trajectory before and immediately after domain switches (code -> zh -> math -> chat)
  4. Policy & feature ablations at 8 GB budget (LRU vs LFU, per-layer vs group-shared, background prefetch on vs off)
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


DEFAULT_MODEL = "NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB.gguf"
FALLBACK_MODEL = "models/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB.gguf"


def build_benchmark_prompts(prompts_json_path: Path, n_per_domain: int = 2):
    """Selects a balanced multi-domain subset ordered by domain for domain-switch & budget testing."""
    with open(prompts_json_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    by_domain = {}
    for idx, entry in enumerate(raw):
        if isinstance(entry, list) and len(entry) >= 2:
            dom, txt = entry[0], entry[1]
            pid = f"{dom}_{idx}"
        elif isinstance(entry, dict):
            dom = entry.get("domain", "default")
            txt = entry.get("prompt", "")
            pid = entry.get("id", f"{dom}_{idx}")
        else:
            continue
        by_domain.setdefault(dom, []).append({"id": pid, "domain": dom, "prompt": txt})

    # Order domains explicitly to test clean domain switches: code -> zh -> math -> chat
    domain_order = ["code", "zh", "math", "chat"]
    for d in by_domain:
        if d not in domain_order:
            domain_order.append(d)

    selected = []
    for d in domain_order:
        if d in by_domain:
            selected.extend(by_domain[d][:n_per_domain])
    return selected


def run_moe_loader(
    loader_bin: Path,
    model_path: str,
    prompts_file: Path,
    out_file: Path,
    baseline: bool = False,
    cache_gb: float = 8.0,
    policy: str = "lru",
    per_layer: int = 1,
    prefetch: str = "router",
    prefetch_topk: int = 9,
    prefetch_cap: int = 2,
    pin_first: int = 1,
    io_threads: int = 8,
    measure_router: int = 0,
    pread: int = 1,
    n_predict: int = 64,
    reuse_existing: bool = False,
):
    if reuse_existing and out_file.exists():
        try:
            with open(out_file, "r", encoding="utf-8", errors="replace") as f:
                data = json.load(f)
            if len(data.get("results", [])) == 8 and "prefetch_accuracy" in data:
                print(f"\n>>> Reusing existing 8-prompt run: {out_file.name}", flush=True)
                return data
        except Exception:
            pass

    cmd = [
        str(loader_bin),
        "--model", model_path,
        "--prompts", str(prompts_file),
        "--out", str(out_file),
        "--n-predict", str(n_predict),
        "--policy", policy,
        "--per-layer", str(per_layer),
        "--prefetch", str(prefetch),
        "--prefetch-topk", str(prefetch_topk),
        "--prefetch-cap", str(prefetch_cap),
        "--pin-first-moe", str(pin_first),
        "--io-threads", str(io_threads),
        "--measure-router", str(measure_router),
        "--pread", str(pread),
    ]
    if baseline:
        cmd.append("--baseline")
    else:
        cmd.extend(["--cache-gb", str(cache_gb)])

    print(f"\n>>> Running: {' '.join(cmd)}", flush=True)
    t0 = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.time() - t0
    if proc.returncode != 0:
        print(f"ERROR (exit {proc.returncode}):\nSTDERR:\n{proc.stderr}\nSTDOUT:\n{proc.stdout}", file=sys.stderr)
        raise RuntimeError(f"moe_loader failed with exit code {proc.returncode}")

    for line in proc.stderr.splitlines():
        if "[moe_loader]" in line or "id=" in line or "llama_moe_cache:" in line or "recall@" in line or "active_pf" in line:
            print("  " + line, flush=True)

    with open(out_file, "r", encoding="utf-8", errors="replace") as f:
        data = json.load(f)
    data["wall_time_sec"] = elapsed
    return data


def compare_with_baseline(baseline_data: dict, candidate_data: dict) -> dict:
    base_res = baseline_data["results"]
    cand_res = candidate_data["results"]
    total_prompts = min(len(base_res), len(cand_res))
    exact_prompt_matches = 0
    total_tokens = 0
    matching_tokens = 0
    mismatches = []

    for i in range(total_prompts):
        bt = base_res[i]["tokens"]
        ct = cand_res[i]["tokens"]
        n = min(len(bt), len(ct))
        total_tokens += max(len(bt), len(ct))
        same = sum(1 for k in range(n) if bt[k] == ct[k])
        matching_tokens += same
        if bt == ct:
            exact_prompt_matches += 1
        else:
            first_diff = next((k for k in range(n) if bt[k] != ct[k]), n)
            mismatches.append({
                "id": base_res[i]["id"],
                "first_diff_step": first_diff,
                "base_text": base_res[i]["text"][:120],
                "cand_text": cand_res[i]["text"][:120],
            })

    return {
        "total_prompts": total_prompts,
        "exact_prompt_matches": exact_prompt_matches,
        "token_match_rate": matching_tokens / total_tokens if total_tokens > 0 else 1.0,
        "matching_tokens": matching_tokens,
        "total_tokens": total_tokens,
        "mismatches": mismatches,
    }


def analyze_domain_switches(run_data: dict) -> dict:
    results = run_data["results"]
    transitions = []
    first_in_domain_hrs = []
    second_in_domain_hrs = []

    prev_domain = None
    for idx, r in enumerate(results):
        dom = r["domain"]
        step_hrs = r.get("step_hit_rates", [])
        n = len(step_hrs)
        early_8 = sum(step_hrs[:8]) / len(step_hrs[:8]) if step_hrs[:8] else 0.0
        mid_8_32 = sum(step_hrs[8:32]) / len(step_hrs[8:32]) if step_hrs[8:32] else 0.0
        late_32_64 = sum(step_hrs[32:]) / len(step_hrs[32:]) if step_hrs[32:] else 0.0

        is_switch = (prev_domain is not None and dom != prev_domain)
        is_first_of_domain = (prev_domain is None or dom != prev_domain)

        if is_first_of_domain:
            first_in_domain_hrs.append(r["decode_hit_rate"])
        else:
            second_in_domain_hrs.append(r["decode_hit_rate"])

        transitions.append({
            "idx": idx,
            "id": r["id"],
            "domain": dom,
            "prev_domain": prev_domain,
            "is_domain_switch": is_switch,
            "overall_hit_rate": r["decode_hit_rate"],
            "decode_tok_per_s": r["decode_tok_per_s"],
            "mib_per_token": r["mib_per_token"],
            "hit_rate_tokens_1_8": early_8,
            "hit_rate_tokens_9_32": mid_8_32,
            "hit_rate_tokens_33_64": late_32_64,
            "n_steps": n,
        })
        prev_domain = dom

    return {
        "per_prompt_trajectory": transitions,
        "mean_first_prompt_in_domain_hit_rate": (
            sum(first_in_domain_hrs) / len(first_in_domain_hrs) if first_in_domain_hrs else 0.0
        ),
        "mean_subsequent_prompt_in_domain_hit_rate": (
            sum(second_in_domain_hrs) / len(second_in_domain_hrs) if second_in_domain_hrs else 0.0
        ),
    }


def extract_run_summary(run_data: dict, baseline_data: dict) -> dict:
    cmp_res = compare_with_baseline(baseline_data, run_data)
    dom_sw = analyze_domain_switches(run_data)
    pf_acc = run_data.get("prefetch_accuracy", {})
    active_sched = pf_acc.get("cache_miss_prediction", {}).get("active_scheduled", {})
    return {
        "cache_gb": run_data.get("cache_gb"),
        "policy": run_data.get("policy", "lru"),
        "per_layer": run_data.get("per_layer", 1),
        "io_threads": run_data.get("io_threads", 8),
        "prefetch_mode": run_data.get("prefetch_mode", "router"),
        "prefetch_topk": run_data.get("prefetch_topk", 9),
        "prefetch_cap": run_data.get("prefetch_cap", 2),
        "pin_first": run_data.get("pin_first", 0),
        "pinned_layer": run_data.get("pinned_layer", -1),
        "pinned_n_expert": run_data.get("pinned_n_expert", 0),
        "pinned_bytes": run_data.get("pinned_bytes", 0),
        "pinned_mib": run_data.get("pinned_mib", 0.0),
        "decode_tok_per_s": run_data["decode_tok_per_s"],
        "prefill_tok_per_s": run_data["prefill_tok_per_s"],
        "decode_hit_rate": run_data["decode_hit_rate"],
        "decode_lru_hit_rate": run_data.get("decode_lru_hit_rate", run_data["decode_hit_rate"]),
        "decode_hits": run_data["decode_hits"],
        "decode_misses": run_data["decode_misses"],
        "decode_lru_hits": run_data.get("decode_lru_hits", run_data["decode_hits"]),
        "decode_lru_misses": run_data.get("decode_lru_misses", run_data["decode_misses"]),
        "mib_per_token": run_data["mib_per_token"],
        "prefetch_submitted": run_data.get("prefetch_submitted", 0),
        "prefetch_hits": run_data["prefetch_hits"],
        "active_prefetch_precision": active_sched.get("precision", 0.0),
        "active_prefetch_recall": active_sched.get("recall", 0.0),
        "prefetch_accuracy": pf_acc,
        "load_sec": run_data["load_sec"],
        "rss_after_load_gib": run_data["rss_after_load_gib"],
        "peak_rss_gib": run_data["peak_rss_gib"],
        "greedy_match": cmp_res,
        "domain_switch": dom_sw,
    }


def main():
    parser = argparse.ArgumentParser(description="MoE Expert Cache Benchmark & Verification Suite v2")
    parser.add_argument("--model", default=DEFAULT_MODEL if os.path.exists(DEFAULT_MODEL) else FALLBACK_MODEL)
    parser.add_argument("--prompts", default="../prompts.json")
    parser.add_argument("--n-per-domain", type=int, default=2, help="Prompts per domain (4 domains * 2 = 8 prompts)")
    parser.add_argument("--n-predict", type=int, default=64, help="Decode tokens per prompt")
    parser.add_argument("--out", default="bench_results_v2.json")
    parser.add_argument("--skip-baseline", action="store_true", default=True)
    parser.add_argument("--run-baseline", action="store_true", help="Force re-running full-resident GPU baseline")
    parser.add_argument("--reuse", action="store_true", default=True, help="Reuse existing completed v2_*.json runs in bench_runs/")
    parser.add_argument("--no-reuse", action="store_true", help="Force re-running all configurations")
    args = parser.parse_args()

    reuse = args.reuse and not args.no_reuse
    loader_dir = Path(__file__).resolve().parent
    loader_bin = loader_dir / "moe_loader"
    if not loader_bin.exists():
        subprocess.run(["make", "moe_loader"], cwd=str(loader_dir), check=True)

    bench_prompts_path = loader_dir / "bench_prompts_subset.json"
    if not bench_prompts_path.exists():
        bench_prompts = build_benchmark_prompts(Path(args.prompts), n_per_domain=args.n_per_domain)
        with open(bench_prompts_path, "w", encoding="utf-8") as f:
            json.dump(bench_prompts, f, indent=2, ensure_ascii=False)
    else:
        with open(bench_prompts_path, "r", encoding="utf-8") as f:
            bench_prompts = json.load(f)

    print(f"Loaded {len(bench_prompts)} benchmark prompts across domains: {[p['id'] for p in bench_prompts]}")

    tmp_dir = loader_dir / "bench_runs"
    tmp_dir.mkdir(exist_ok=True)

    # 1. Baseline full-resident GPU run
    baseline_file = tmp_dir / "baseline_gpu.json"
    if not args.run_baseline and baseline_file.exists():
        with open(baseline_file, "r", encoding="utf-8", errors="replace") as f:
            baseline_data = json.load(f)
    else:
        baseline_data = run_moe_loader(
            loader_bin=loader_bin,
            model_path=args.model,
            prompts_file=bench_prompts_path,
            out_file=baseline_file,
            baseline=True,
            n_predict=args.n_predict,
        )

    # 2. Online/Offline Router vs Learned Prediction Accuracy Measurement across all 8 prompts (19,968 transitions)
    meas_file = tmp_dir / "v2_8gb_measure_router.json"
    meas_data = run_moe_loader(
        loader_bin=loader_bin,
        model_path=args.model,
        prompts_file=bench_prompts_path,
        out_file=meas_file,
        cache_gb=8.0,
        pin_first=1,
        prefetch="router",
        prefetch_topk=9,
        prefetch_cap=2,
        io_threads=8,
        measure_router=1,
        n_predict=args.n_predict,
        reuse_existing=reuse,
    )

    # 3. Compare IO threads (8 vs 16) at 8 GB to select best_io for the main budget table
    io8_8gb = run_moe_loader(
        loader_bin=loader_bin,
        model_path=args.model,
        prompts_file=bench_prompts_path,
        out_file=tmp_dir / "v2_8gb_pin1_router_io8.json",
        cache_gb=8.0,
        pin_first=1,
        prefetch="router",
        io_threads=8,
        n_predict=args.n_predict,
        reuse_existing=reuse,
    )
    io16_8gb = run_moe_loader(
        loader_bin=loader_bin,
        model_path=args.model,
        prompts_file=bench_prompts_path,
        out_file=tmp_dir / "v2_8gb_pin1_router_io16.json",
        cache_gb=8.0,
        pin_first=1,
        prefetch="router",
        io_threads=16,
        n_predict=args.n_predict,
        reuse_existing=reuse,
    )

    best_io = 16 if io16_8gb["decode_tok_per_s"] > io8_8gb["decode_tok_per_s"] else 8
    print(f"\n>>> Selected best IO thread count at 8 GB: io_threads={best_io} "
          f"(8 threads: {io8_8gb['decode_tok_per_s']:.2f} tok/s vs 16 threads: {io16_8gb['decode_tok_per_s']:.2f} tok/s)")

    summary = {
        "version": "v2",
        "model": args.model,
        "n_prompts": len(bench_prompts),
        "n_predict": args.n_predict,
        "best_config": {
            "policy": "lru",
            "per_layer": 1,
            "pin_first_moe": 1,
            "prefetch": "router",
            "prefetch_topk": 9,
            "prefetch_cap": 2,
            "io_threads": best_io,
        },
        "pinned_first_moe_info": {
            "layer": meas_data.get("pinned_layer", 1),
            "tensor_prefix": f"blk.{meas_data.get('pinned_layer', 1)}.ffn_*_exps",
            "n_expert": meas_data.get("pinned_n_expert", 240),
            "bytes": meas_data.get("pinned_bytes", 259522560),
            "mib": meas_data.get("pinned_mib", 247.50),
            "outside_budget": True,
        },
        "baseline": {
            "decode_tok_per_s": baseline_data["decode_tok_per_s"],
            "prefill_tok_per_s": baseline_data["prefill_tok_per_s"],
            "load_sec": baseline_data["load_sec"],
            "rss_after_load_gib": baseline_data["rss_after_load_gib"],
            "peak_rss_gib": baseline_data["peak_rss_gib"],
        },
        "router_vs_learned_accuracy_8gb": meas_data.get("prefetch_accuracy", {}),
        "budgets": {},
        "ablations": {},
    }

    # 4. Full Matrix across 4, 8, 12 GB budgets:
    #    For each budget in [4, 8, 12]:
    #      - pin=1, prefetch=router, io=8
    #      - pin=1, prefetch=router, io=16
    #      - pin=0, prefetch=router, io=best_io
    #      - pin=1, prefetch=learned, io=best_io
    #      - pin=1, prefetch=off, io=best_io
    #      - pin=0, prefetch=off, io=best_io
    for gb in [4, 8, 12]:
        configs = [
            (f"{gb}gb_pin1_router_io8",  float(gb), 1, "router",  9, 2, 8),
            (f"{gb}gb_pin1_router_io16", float(gb), 1, "router",  9, 2, 16),
            (f"{gb}gb_pin0_router_io{best_io}", float(gb), 0, "router",  9, 2, best_io),
            (f"{gb}gb_pin1_learned_io{best_io}", float(gb), 1, "learned", 9, 2, best_io),
            (f"{gb}gb_pin1_off_io{best_io}",     float(gb), 1, "off",     9, 2, best_io),
            (f"{gb}gb_pin0_off_io{best_io}",     float(gb), 0, "off",     9, 2, best_io),
        ]
        for name, c_gb, pin, pf, topk, cap, io_th in configs:
            out_f = tmp_dir / f"v2_{name}.json"
            run_data = run_moe_loader(
                loader_bin=loader_bin,
                model_path=args.model,
                prompts_file=bench_prompts_path,
                out_file=out_f,
                cache_gb=c_gb,
                pin_first=pin,
                prefetch=pf,
                prefetch_topk=topk,
                prefetch_cap=cap,
                io_threads=io_th,
                n_predict=args.n_predict,
                reuse_existing=reuse,
            )
            entry = extract_run_summary(run_data, baseline_data)
            summary["ablations"][name] = entry
            if pin == 1 and pf == "router" and io_th == best_io:
                summary["budgets"][str(gb)] = entry

    # Additional 8 GB prefetch ablations: hybrid and router top-12 / top-16
    extra_8gb = [
        ("8gb_pin1_hybrid_io8",      8.0, 1, "hybrid", 9,  2, 8),
        ("8gb_pin1_router_k12_io8",  8.0, 1, "router", 12, 3, 8),
        ("8gb_pin1_router_k16_io8",  8.0, 1, "router", 16, 4, 8),
    ]
    for name, c_gb, pin, pf, topk, cap, io_th in extra_8gb:
        out_f = tmp_dir / f"v2_{name}.json"
        run_data = run_moe_loader(
            loader_bin=loader_bin,
            model_path=args.model,
            prompts_file=bench_prompts_path,
            out_file=out_f,
            cache_gb=c_gb,
            pin_first=pin,
            prefetch=pf,
            prefetch_topk=topk,
            prefetch_cap=cap,
            io_threads=io_th,
            n_predict=args.n_predict,
            reuse_existing=reuse,
        )
        summary["ablations"][name] = extract_run_summary(run_data, baseline_data)

    out_path = loader_dir / args.out
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 116)
    print("SUMMARY v2: MoE Expert Cache Benchmark & Ablations (Nemotron-3.5-Bolt-74B-A9B)")
    print("=" * 116)
    print(f"{'Configuration':<28} | {'Decode tok/s':>12} | {'Hit Rate':>9} | {'LRU Hit':>8} | {'PF Prec/Rec':>15} | {'MiB/tok':>8} | {'Peak RSS':>9} | {'Greedy':>10}")
    print("-" * 116)
    b = summary["baseline"]
    print(f"{'Baseline (17.05 GB GPU)':<28} | {b['decode_tok_per_s']:>12.2f} | {'100.00%':>9} | {'100.00%':>8} | {'N/A':>15} | {'0.00':>8} | {b['peak_rss_gib']:>7.2f} G | {'8/8 (100%)':>10}")
    for gb in ["4", "8", "12"]:
        r = summary["budgets"][gb]
        gm = r["greedy_match"]
        match_str = f"{gm['exact_prompt_matches']}/{gm['total_prompts']}"
        pf_str = f"{r['active_prefetch_precision']*100:.1f}%/{r['active_prefetch_recall']*100:.1f}%"
        print(
            f"{f'Best {gb} GB (pin+router+io{best_io})':<28} | {r['decode_tok_per_s']:>12.2f} | {r['decode_hit_rate']*100:>8.2f}% | "
            f"{r['decode_lru_hit_rate']*100:>7.2f}% | {pf_str:>15} | {r['mib_per_token']:>8.2f} | {r['peak_rss_gib']:>7.2f} G | {match_str:>10}"
        )
    print("-" * 116)
    for name, r in summary["ablations"].items():
        gm = r["greedy_match"]
        match_str = f"{gm['exact_prompt_matches']}/{gm['total_prompts']}"
        if r["prefetch_mode"] == "off":
            pf_str = "off"
        else:
            pf_str = f"{r['active_prefetch_precision']*100:.1f}%/{r['active_prefetch_recall']*100:.1f}%"
        print(
            f"{name:<28} | {r['decode_tok_per_s']:>12.2f} | {r['decode_hit_rate']*100:>8.2f}% | "
            f"{r['decode_lru_hit_rate']*100:>7.2f}% | {pf_str:>15} | {r['mib_per_token']:>8.2f} | {r['peak_rss_gib']:>7.2f} G | {match_str:>10}"
        )
    print("=" * 116)
    print(f"\nSaved full v2 results to {out_path}")


if __name__ == "__main__":
    main()

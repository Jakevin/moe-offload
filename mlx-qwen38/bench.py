#!/usr/bin/env python3
"""Benchmark suite for Qwen3.8-Flash-Next MLX Expert-Offload Loader.

Runs:
1. Step 1 Correctness check (Prompts 0 & 1, 128 greedy tokens vs tracer.py reference).
2. Step 2 Multi-domain 8-prompt benchmark (code, zh_chat, math, en_chat with continuous
   domain switches) across budgets 20% (~7.5 GB) and 30% (~11 GB), prefetch on/off,
   and even vs proportional slot allocation.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

_BASE_SITE = "/Applications/oMLX.app/Contents/Resources/Python/framework-mlx-base/lib/python3.11/site-packages"
_OMLX_RES = "/Applications/oMLX.app/Contents/Resources"
for _p in (_OMLX_RES, _BASE_SITE):
    if _p in sys.path:
        sys.path.remove(_p)
    sys.path.insert(0, _p)

import mlx.core as mx
import numpy as np

from loader import (
    LOADER_DIR,
    MODEL_PATH,
    PROJECT_DIR,
    ROUTES_REF_PATH,
    TRACE_META_PATH,
    OffloadEngine,
    get_rss_kb,
    get_swap_used_mb,
)

PYTHON_BIN = "/Applications/oMLX.app/Contents/Resources/Python/cpython-3.11/bin/python3.11"
RUN_SH = LOADER_DIR / "run.sh"
BENCH_RUNS_DIR = LOADER_DIR / "bench_runs"
BENCH_RESULTS_PATH = LOADER_DIR / "bench_results.json"
CORRECTNESS_REPORT_PATH = LOADER_DIR / "correctness_report.json"

# 8 prompts across 4 domains (code, zh_chat, math, en_chat) ordered so every transition
# is a cross-domain switch. Prompts 0 and 1 match tracer.py for continuous exact-match verification.
BENCH_PROMPTS: list[dict[str, Any]] = [
    {
        "prompt_id": 0,
        "domain": "code",
        "user_content": (
            "Write a complete Python implementation of a thread-safe LRU cache with TTL "
            "expiration and O(1) get/put operations using a doubly linked list, including "
            "type annotations and unit tests."
        ),
    },
    {
        "prompt_id": 1,
        "domain": "zh_chat",
        "user_content": (
            "请详细介绍中国杭州西湖四季的自然风光、历史典故以及当地特色美食，"
            "并为第一次去杭州旅行的游客提供一份三日游行程建议。"
        ),
    },
    {
        "prompt_id": 2,
        "domain": "math",
        "user_content": (
            "Prove that the square root of 2 is irrational, then derive the closed-form "
            "solution for the linear recurrence a_n = 3*a_{n-1} - 2*a_{n-2} with a_0 = 1 "
            "and a_1 = 2, and compute the exact sum of k^2 for k from 1 to 100."
        ),
    },
    {
        "prompt_id": 3,
        "domain": "en_chat",
        "user_content": (
            "Explain the architectural differences between TCP, UDP, and QUIC, how QUIC "
            "eliminates head-of-line blocking and reduces handshake latency, and when a "
            "distributed systems engineer should choose each protocol."
        ),
    },
    {
        "prompt_id": 4,
        "domain": "code",
        "user_content": (
            "Implement an iterative quicksort and merge sort in C with in-place Lomuto and "
            "Hoare partitioning schemes, analyze their best/average/worst-case time and "
            "space complexity, and provide a self-contained test harness."
        ),
    },
    {
        "prompt_id": 5,
        "domain": "zh_chat",
        "user_content": (
            "请深入浅出地解释机器学习与深度学习中的过拟合（Overfitting）与欠拟合（Underfitting），"
            "并结合现代大模型训练实践列出五种有效的正则化与泛化方法。"
        ),
    },
    {
        "prompt_id": 6,
        "domain": "math",
        "user_content": (
            "Find all eigenvalues and orthonormal eigenvectors of the real symmetric matrix "
            "A = [[2, 1, 1], [1, 2, 1], [1, 1, 2]], verify its spectral decomposition, "
            "and derive the closed-form matrix exponential exp(t*A)."
        ),
    },
    {
        "prompt_id": 7,
        "domain": "en_chat",
        "user_content": (
            "I am planning a 5-day autumn cultural and culinary itinerary in Kyoto and Osaka. "
            "Provide a detailed day-by-day schedule covering historic temples, local dishes, "
            "walking routes, and transit recommendations."
        ),
    },
]


def load_reference_tokens(max_tokens: int) -> dict[int, list[int]]:
    ref_records = [
        json.loads(line)
        for line in ROUTES_REF_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    out: dict[int, list[int]] = {}
    for pid in (0, 1):
        out[pid] = [
            int(r["token_id"])
            for r in ref_records
            if int(r["prompt_id"]) == pid and r["phase"] == "decode"
        ][:max_tokens]
    return out


BENCH_RESULTS_V2_PATH = LOADER_DIR / "bench_results_v2.json"
CORRECTNESS_REPORT_V2_PATH = LOADER_DIR / "correctness_report_v2.json"


def run_single_benchmark(
    tag: str,
    budget_pct: float | None,
    budget_gb: float | None,
    alloc_mode: str,
    prefetch: bool,
    decode_tokens: int,
    out_path: Path,
    prefetch_top_m: int = 12,
    prefetch_cap: int = 3,
) -> dict[str, Any]:
    start_swap_mb = get_swap_used_mb()
    peak_swap_mb = start_swap_mb
    peak_rss_kb = get_rss_kb()

    def update_mem() -> tuple[int, float]:
        nonlocal peak_swap_mb, peak_rss_kb
        r = get_rss_kb()
        s = get_swap_used_mb()
        if r > peak_rss_kb:
            peak_rss_kb = r
        if s > peak_swap_mb:
            peak_swap_mb = s
        return r, s

    ref_tokens = load_reference_tokens(decode_tokens)

    print(
        f"\n[bench:{tag}] Loading OffloadEngine (budget_pct={budget_pct}, budget_gb={budget_gb}, "
        f"alloc_mode={alloc_mode}, prefetch={prefetch}, top_m={prefetch_top_m}, cap={prefetch_cap})...",
        flush=True,
    )
    engine = OffloadEngine(
        model_path=MODEL_PATH,
        budget_gb=budget_gb,
        budget_pct=budget_pct,
        alloc_mode=alloc_mode,
        prefetch=prefetch,
        prefetch_top_m=prefetch_top_m,
        prefetch_cap=prefetch_cap,
    )
    rss_kb, swap_mb = update_mem()
    print(
        f"[bench:{tag}] Loaded in {engine.load_seconds:.2f}s | "
        f"L0 pinned={engine.layer0_pinned_bytes / (1024**2):.1f} MiB ({engine.layer0_pinned_bytes / 1e9:.3f} GB) | "
        f"L1..47 cache={engine.cache_budget_bytes / (1024**3):.2f} GiB ({engine.cache_budget_bytes / 1e9:.3f} GB, "
        f"{engine.total_cached_slots} slots) | "
        f"RSS={rss_kb / (1024**2):.2f} GiB | MLX active={mx.get_active_memory() / (1024**3):.2f} GiB",
        flush=True,
    )

    prompt_results: list[dict[str, Any]] = []
    tot_prefill_tokens = 0
    tot_prefill_sec = 0.0
    tot_decode_tokens = 0
    tot_decode_sec = 0.0

    for idx, pinfo in enumerate(BENCH_PROMPTS):
        pid = int(pinfo["prompt_id"])
        domain = pinfo["domain"]
        prev_domain = BENCH_PROMPTS[idx - 1]["domain"] if idx > 0 else "cold_start"

        res = engine.generate_greedy(
            user_content=pinfo["user_content"],
            max_new_tokens=decode_tokens,
            reset_cache_before=False,
            reset_stats_before=False,
        )
        rss_kb, swap_mb = update_mem()

        tot_prefill_tokens += res["prompt_tokens"]
        tot_prefill_sec += res["prefill_seconds"]
        tot_decode_tokens += res["decode_tokens"]
        tot_decode_sec += res["decode_seconds"]

        exact_match = None
        if pid in ref_tokens:
            exact_match = res["generated_ids"] == ref_tokens[pid]

        print(
            f"[bench:{tag}] P{pid} ({prev_domain}->{domain}): "
            f"prefill={res['prefill_tok_per_s']:.2f} tok/s ({res['prompt_tokens']} tok), "
            f"decode={res['decode_tok_per_s']:.2f} tok/s (first32={res['first_window_tok_per_s']:.2f}, "
            f"steady={res['steady_window_tok_per_s']:.2f}) | "
            f"hit={res['prompt_pure_lru_hit_rate']*100:.1f}% (eff={res['prompt_effective_hit_rate']*100:.1f}%, "
            f"first32={res['first_window_pure_hit_rate']*100:.1f}%->steady={res['steady_window_pure_hit_rate']*100:.1f}%) | "
            f"read={res['prompt_mb_per_token']:.1f} MB/tok | "
            f"RSS={rss_kb / (1024**2):.2f} GiB | SwapD={swap_mb - start_swap_mb:+.1f} MiB"
            + (f" | ref_match={exact_match}" if exact_match is not None else ""),
            flush=True,
        )

        prompt_results.append(
            {
                "prompt_id": pid,
                "domain": domain,
                "transition": f"{prev_domain}->{domain}",
                "exact_ref_match": exact_match,
                "user_content": pinfo["user_content"],
                **res,
            }
        )

    stats = engine.collect_stats(
        total_decode_tokens=tot_decode_tokens, total_prefill_tokens=tot_prefill_tokens
    )
    end_rss_kb, end_swap_mb = update_mem()

    # Aggregate domain-switch metrics across prompts 1..7 (where cache was warmed by previous domain)
    switch_prompts = prompt_results[1:]
    mean_switch_first_pure_hit = float(
        np.mean([p["first_window_pure_hit_rate"] for p in switch_prompts])
    )
    mean_switch_steady_pure_hit = float(
        np.mean([p["steady_window_pure_hit_rate"] for p in switch_prompts])
    )
    mean_switch_first_eff_hit = float(
        np.mean([p["first_window_effective_hit_rate"] for p in switch_prompts])
    )
    mean_switch_steady_eff_hit = float(
        np.mean([p["steady_window_effective_hit_rate"] for p in switch_prompts])
    )
    mean_switch_first_tok_s = float(np.mean([p["first_window_tok_per_s"] for p in switch_prompts]))
    mean_switch_steady_tok_s = float(
        np.mean([p["steady_window_tok_per_s"] for p in switch_prompts])
    )

    # Per-domain breakdown
    by_domain: dict[str, dict[str, Any]] = {}
    for dom in ("code", "zh_chat", "math", "en_chat"):
        dom_ps = [p for p in prompt_results if p["domain"] == dom]
        d_pre_tok = sum(p["prompt_tokens"] for p in dom_ps)
        d_pre_sec = sum(p["prefill_seconds"] for p in dom_ps)
        d_dec_tok = sum(p["decode_tokens"] for p in dom_ps)
        d_dec_sec = sum(p["decode_seconds"] for p in dom_ps)
        by_domain[dom] = {
            "num_prompts": len(dom_ps),
            "prefill_tok_per_s": round(d_pre_tok / max(1e-6, d_pre_sec), 3),
            "decode_tok_per_s": round(d_dec_tok / max(1e-6, d_dec_sec), 3),
            "mean_pure_lru_hit_rate": round(
                float(np.mean([p["prompt_pure_lru_hit_rate"] for p in dom_ps])), 5
            ),
            "mean_effective_hit_rate": round(
                float(np.mean([p["prompt_effective_hit_rate"] for p in dom_ps])), 5
            ),
            "mean_mb_per_token": round(
                float(np.mean([p["prompt_mb_per_token"] for p in dom_ps])), 3
            ),
        }

    summary = {
        "tag": tag,
        "load_seconds": round(engine.load_seconds, 4),
        "total_prefill_tokens": tot_prefill_tokens,
        "total_prefill_seconds": round(tot_prefill_sec, 4),
        "overall_prefill_tok_per_s": round(tot_prefill_tokens / max(1e-6, tot_prefill_sec), 3),
        "total_decode_tokens": tot_decode_tokens,
        "total_decode_seconds": round(tot_decode_sec, 4),
        "overall_decode_tok_per_s": round(tot_decode_tokens / max(1e-6, tot_decode_sec), 3),
        "steady_state_mean_decode_tok_per_s": round(mean_switch_steady_tok_s, 3),
        "domain_switch_summary": {
            "first_window_pure_hit_rate": round(mean_switch_first_pure_hit, 5),
            "steady_window_pure_hit_rate": round(mean_switch_steady_pure_hit, 5),
            "pure_hit_rate_delta": round(
                mean_switch_first_pure_hit - mean_switch_steady_pure_hit, 5
            ),
            "first_window_effective_hit_rate": round(mean_switch_first_eff_hit, 5),
            "steady_window_effective_hit_rate": round(mean_switch_steady_eff_hit, 5),
            "first_window_tok_per_s": round(mean_switch_first_tok_s, 3),
            "steady_window_tok_per_s": round(mean_switch_steady_tok_s, 3),
        },
        "by_domain": by_domain,
        "peak_rss_kb": peak_rss_kb,
        "peak_rss_gib": round(peak_rss_kb / (1024**2), 3),
        "peak_rss_gb": round((peak_rss_kb * 1024.0) / 1e9, 3),
        "mlx_active_gib": round(mx.get_active_memory() / (1024**3), 3),
        "mlx_peak_gib": round(mx.get_peak_memory() / (1024**3), 3),
        "start_swap_mb": round(start_swap_mb, 2),
        "end_swap_mb": round(end_swap_mb, 2),
        "peak_swap_mb": round(peak_swap_mb, 2),
        "end_swap_delta_mb": round(end_swap_mb - start_swap_mb, 2),
        "peak_swap_delta_mb": round(peak_swap_mb - start_swap_mb, 2),
        "p0_p1_exact_ref_match": all(
            p["exact_ref_match"] is True for p in prompt_results if p["prompt_id"] in (0, 1)
        ),
        "stats": stats,
        "prompts": prompt_results,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[bench:{tag}] Saved {out_path}", flush=True)
    engine.close()
    return summary


def run_watchdog_cmd(cmd: list[str], wd_log: Path, wd_stats: Path) -> dict[str, Any]:
    env = os.environ.copy()
    env["WATCHDOG_LOG"] = str(wd_log)
    env["WATCHDOG_STATS"] = str(wd_stats)
    full_cmd = [str(RUN_SH)] + cmd
    print(f"\n[suite] Executing via watchdog: {' '.join(full_cmd)}", flush=True)
    subprocess.run(full_cmd, cwd=str(LOADER_DIR), env=env, check=True)
    return json.loads(wd_stats.read_text(encoding="utf-8"))


def run_v2_suite(decode_tokens: int = 64, skip_step3: bool = False) -> None:
    BENCH_RUNS_DIR.mkdir(parents=True, exist_ok=True)

    # Step 3: 128-token greedy correctness check on Prompts 0 & 1 (both prefetch=off and prefetch=on)
    if not skip_step3 or not CORRECTNESS_REPORT_V2_PATH.exists():
        print("\n=== STEP 3 (v2): 128-Token Greedy Correctness Verification ===", flush=True)
        wd_step3 = run_watchdog_cmd(
            [
                PYTHON_BIN,
                str(LOADER_DIR / "loader.py"),
                "--mode",
                "verify",
                "--budget-pct",
                "20",
                "--alloc-mode",
                "proportional",
                "--prefetch",
                "off",
                "--verify-both-prefetch",
                "--prefetch-top-m",
                "12",
                "--prefetch-cap",
                "3",
                "--decode-tokens",
                "128",
                "--out",
                str(CORRECTNESS_REPORT_V2_PATH),
            ],
            BENCH_RUNS_DIR / "watchdog_step3_v2.log",
            BENCH_RUNS_DIR / "watchdog_step3_v2.json",
        )
        step3_data = json.loads(CORRECTNESS_REPORT_V2_PATH.read_text(encoding="utf-8"))
        step3_data["watchdog"] = wd_step3
        CORRECTNESS_REPORT_V2_PATH.write_text(
            json.dumps(step3_data, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    # Step 4: Re-benchmark 20% proportional (prefetch off + best prefetch variants) on the 8 prompts
    configs = [
        {
            "tag": "v2_budget20_proportional_prefetch_off",
            "budget_pct": 20.0,
            "alloc_mode": "proportional",
            "prefetch": "off",
            "prefetch_top_m": 12,
            "prefetch_cap": 0,
        },
        {
            "tag": "v2_budget20_proportional_prefetch_top10_cap2",
            "budget_pct": 20.0,
            "alloc_mode": "proportional",
            "prefetch": "on",
            "prefetch_top_m": 10,
            "prefetch_cap": 2,
        },
        {
            "tag": "v2_budget20_proportional_prefetch_top12_cap3",
            "budget_pct": 20.0,
            "alloc_mode": "proportional",
            "prefetch": "on",
            "prefetch_top_m": 12,
            "prefetch_cap": 3,
        },
    ]

    suite_results: dict[str, Any] = {}
    for cfg in configs:
        tag = cfg["tag"]
        out_json = BENCH_RUNS_DIR / f"{tag}.json"
        wd_log = BENCH_RUNS_DIR / f"watchdog_{tag}.log"
        wd_stats = BENCH_RUNS_DIR / f"watchdog_{tag}.json"

        wd = run_watchdog_cmd(
            [
                PYTHON_BIN,
                str(LOADER_DIR / "bench.py"),
                "--single-config",
                "--tag",
                tag,
                "--budget-pct",
                str(cfg["budget_pct"]),
                "--alloc-mode",
                cfg["alloc_mode"],
                "--prefetch",
                cfg["prefetch"],
                "--prefetch-top-m",
                str(cfg["prefetch_top_m"]),
                "--prefetch-cap",
                str(cfg["prefetch_cap"]),
                "--decode-tokens",
                str(decode_tokens),
                "--out",
                str(out_json),
            ],
            wd_log,
            wd_stats,
        )
        run_data = json.loads(out_json.read_text(encoding="utf-8"))
        run_data["watchdog"] = wd
        out_json.write_text(json.dumps(run_data, indent=2, ensure_ascii=False), encoding="utf-8")
        suite_results[tag] = run_data

    trace_meta = json.loads(TRACE_META_PATH.read_text(encoding="utf-8"))
    v1_results = (
        json.loads(BENCH_RESULTS_PATH.read_text(encoding="utf-8"))
        if BENCH_RESULTS_PATH.exists()
        else {}
    )
    v1_prop_off = v1_results.get("runs", {}).get("budget20_proportional_prefetch_off", {})
    v1_even_on = v1_results.get("runs", {}).get("budget20_even_prefetch_on", {})

    v2_off = suite_results["v2_budget20_proportional_prefetch_off"]
    v2_top10 = suite_results["v2_budget20_proportional_prefetch_top10_cap2"]
    v2_top12 = suite_results["v2_budget20_proportional_prefetch_top12_cap3"]

    v1_comparison = {
        "v1_budget20_proportional_prefetch_off": {
            "overall_prefill_tok_per_s": v1_prop_off.get("overall_prefill_tok_per_s", 14.82),
            "overall_decode_tok_per_s": v1_prop_off.get("overall_decode_tok_per_s", 8.511),
            "steady_state_mean_decode_tok_per_s": v1_prop_off.get("steady_state_mean_decode_tok_per_s", 8.508),
            "pure_lru_hit_rate": v1_prop_off.get("stats", {}).get("decode_aggregate", {}).get("pure_lru_hit_rate", 0.84299),
            "effective_hit_rate": v1_prop_off.get("stats", {}).get("decode_aggregate", {}).get("effective_hit_rate", 0.84299),
            "total_mb_per_token": v1_prop_off.get("stats", {}).get("decode_aggregate", {}).get("total_mb_per_token", 113.352),
            "peak_rss_gib": v1_prop_off.get("peak_rss_gib", 11.617),
            "peak_swap_delta_mb": v1_prop_off.get("peak_swap_delta_mb", 0.0),
        },
        "v1_budget20_even_prefetch_on_top16_uncapped": {
            "overall_prefill_tok_per_s": v1_even_on.get("overall_prefill_tok_per_s", 15.881),
            "overall_decode_tok_per_s": v1_even_on.get("overall_decode_tok_per_s", 5.059),
            "steady_state_mean_decode_tok_per_s": v1_even_on.get("steady_state_mean_decode_tok_per_s", 5.266),
            "pure_lru_hit_rate": v1_even_on.get("stats", {}).get("decode_aggregate", {}).get("pure_lru_hit_rate", 0.83112),
            "effective_hit_rate": v1_even_on.get("stats", {}).get("decode_aggregate", {}).get("effective_hit_rate", 0.98275),
            "total_mb_per_token": v1_even_on.get("stats", {}).get("decode_aggregate", {}).get("total_mb_per_token", 270.032),
            "peak_rss_gib": v1_even_on.get("peak_rss_gib", 9.01),
            "peak_swap_delta_mb": v1_even_on.get("peak_swap_delta_mb", 0.0),
        },
        "speedup_vs_v1_best_8_51_tok_s": {
            "v2_prefetch_off_decode_speedup": round(v2_off["overall_decode_tok_per_s"] / 8.511, 4),
            "v2_prefetch_off_prefill_speedup": round(v2_off["overall_prefill_tok_per_s"] / 14.82, 4),
            "v2_prefetch_top10_cap2_decode_speedup": round(v2_top10["overall_decode_tok_per_s"] / 8.511, 4),
            "v2_prefetch_top10_cap2_prefill_speedup": round(v2_top10["overall_prefill_tok_per_s"] / 14.82, 4),
            "v2_prefetch_top12_cap3_decode_speedup": round(v2_top12["overall_decode_tok_per_s"] / 8.511, 4),
            "v2_prefetch_top12_cap3_prefill_speedup": round(v2_top12["overall_prefill_tok_per_s"] / 14.82, 4),
            "v2_prefetch_top12_cap3_vs_v1_prefetch_on_speedup": round(v2_top12["overall_decode_tok_per_s"] / 5.059, 4),
        },
    }

    combined = {
        "version": "v2",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "decode_tokens_per_prompt": decode_tokens,
        "num_prompts": len(BENCH_PROMPTS),
        "baselines": {
            "tracer_hybrid_28resident_20pread": {
                "overall_prefill_tok_per_s": trace_meta["overall_prefill_tok_per_s"],
                "overall_decode_tok_per_s": trace_meta["overall_decode_tok_per_s"],
                "mlx_active_gib": trace_meta["mlx_active_gib"],
                "mlx_peak_gib": trace_meta["mlx_peak_gib"],
                "peak_rss_gib": trace_meta["peak_rss_gib"],
                "peak_swap_delta_mb": trace_meta["peak_swap_delta_mb"],
            },
            "llamacpp_q2_0_full_resident_gguf": {
                "decode_tok_per_s": 24.6,
                "date": "2026-10-05",
                "setup": "Full resident GGUF Q2_0 in llama.cpp (different setup: all weights resident in RAM/VRAM, no PLE 32GB table)",
            },
        },
        "v1_comparison": v1_comparison,
        "correctness_v2": json.loads(CORRECTNESS_REPORT_V2_PATH.read_text(encoding="utf-8")),
        "runs": suite_results,
    }
    BENCH_RESULTS_V2_PATH.write_text(
        json.dumps(combined, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\n[suite:v2] Saved complete v2 benchmark results to {BENCH_RESULTS_V2_PATH}", flush=True)


def run_full_suite(decode_tokens: int = 64, skip_step1: bool = False) -> None:
    BENCH_RUNS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Step 1: 128-token greedy correctness check on Prompts 0 & 1
    if not skip_step1 or not CORRECTNESS_REPORT_PATH.exists():
        print("\n=== STEP 1: 128-Token Greedy Correctness Verification ===", flush=True)
        wd_step1 = run_watchdog_cmd(
            [
                PYTHON_BIN,
                str(LOADER_DIR / "loader.py"),
                "--mode",
                "verify",
                "--budget-pct",
                "20",
                "--alloc-mode",
                "even",
                "--prefetch",
                "on",
                "--decode-tokens",
                "128",
                "--out",
                str(CORRECTNESS_REPORT_PATH),
            ],
            BENCH_RUNS_DIR / "watchdog_step1.log",
            BENCH_RUNS_DIR / "watchdog_step1.json",
        )
        step1_data = json.loads(CORRECTNESS_REPORT_PATH.read_text(encoding="utf-8"))
        step1_data["watchdog"] = wd_step1
        CORRECTNESS_REPORT_PATH.write_text(
            json.dumps(step1_data, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    # 2. Step 2: Benchmark 8 prompts across budgets 20% (~7.5 GB) and 30% (~11 GB), prefetch off/on
    configs = [
        {
            "tag": "budget20_even_prefetch_off",
            "budget_pct": 20.0,
            "alloc_mode": "even",
            "prefetch": "off",
        },
        {
            "tag": "budget20_even_prefetch_on",
            "budget_pct": 20.0,
            "alloc_mode": "even",
            "prefetch": "on",
        },
        {
            "tag": "budget30_even_prefetch_off",
            "budget_pct": 30.0,
            "alloc_mode": "even",
            "prefetch": "off",
        },
        {
            "tag": "budget30_even_prefetch_on",
            "budget_pct": 30.0,
            "alloc_mode": "even",
            "prefetch": "on",
        },
        {
            "tag": "budget20_proportional_prefetch_off",
            "budget_pct": 20.0,
            "alloc_mode": "proportional",
            "prefetch": "off",
        },
    ]

    suite_results: dict[str, Any] = {}
    for cfg in configs:
        tag = cfg["tag"]
        out_json = BENCH_RUNS_DIR / f"{tag}.json"
        wd_log = BENCH_RUNS_DIR / f"watchdog_{tag}.log"
        wd_stats = BENCH_RUNS_DIR / f"watchdog_{tag}.json"

        wd = run_watchdog_cmd(
            [
                PYTHON_BIN,
                str(LOADER_DIR / "bench.py"),
                "--single-config",
                "--tag",
                tag,
                "--budget-pct",
                str(cfg["budget_pct"]),
                "--alloc-mode",
                cfg["alloc_mode"],
                "--prefetch",
                cfg["prefetch"],
                "--decode-tokens",
                str(decode_tokens),
                "--out",
                str(out_json),
            ],
            wd_log,
            wd_stats,
        )
        run_data = json.loads(out_json.read_text(encoding="utf-8"))
        run_data["watchdog"] = wd
        out_json.write_text(json.dumps(run_data, indent=2, ensure_ascii=False), encoding="utf-8")
        suite_results[tag] = run_data

    trace_meta = json.loads(TRACE_META_PATH.read_text(encoding="utf-8"))
    combined = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "decode_tokens_per_prompt": decode_tokens,
        "num_prompts": len(BENCH_PROMPTS),
        "baselines": {
            "tracer_hybrid_28resident_20pread": {
                "overall_prefill_tok_per_s": trace_meta["overall_prefill_tok_per_s"],
                "overall_decode_tok_per_s": trace_meta["overall_decode_tok_per_s"],
                "mlx_active_gib": trace_meta["mlx_active_gib"],
                "mlx_peak_gib": trace_meta["mlx_peak_gib"],
                "peak_rss_gib": trace_meta["peak_rss_gib"],
                "peak_swap_delta_mb": trace_meta["peak_swap_delta_mb"],
            },
            "llamacpp_q2_0_full_resident_gguf": {
                "decode_tok_per_s": 24.6,
                "date": "2026-10-05",
                "setup": "Full resident GGUF Q2_0 in llama.cpp (different setup: all weights resident in RAM/VRAM, no PLE 32GB table)",
            },
        },
        "correctness_step1": json.loads(CORRECTNESS_REPORT_PATH.read_text(encoding="utf-8")),
        "runs": suite_results,
    }
    BENCH_RESULTS_PATH.write_text(
        json.dumps(combined, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\n[suite] Saved complete benchmark results to {BENCH_RESULTS_PATH}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark Qwen3.8-Flash-Next Expert-Offload Loader")
    parser.add_argument("--single-config", action="store_true", help="Run a single config")
    parser.add_argument("--v1-suite", action="store_true", help="Run full v1 5-config suite")
    parser.add_argument("--tag", type=str, default="single", help="Run tag")
    parser.add_argument("--budget-pct", type=float, default=None, help="Budget %% of 512 experts")
    parser.add_argument("--budget-gb", type=float, default=None, help="Budget GB for L1..47")
    parser.add_argument("--alloc-mode", choices=["even", "proportional"], default="proportional")
    parser.add_argument("--prefetch", choices=["on", "off"], default="off")
    parser.add_argument("--prefetch-top-m", type=int, default=12)
    parser.add_argument("--prefetch-cap", type=int, default=3)
    parser.add_argument("--decode-tokens", type=int, default=64)
    parser.add_argument("--skip-step1", action="store_true")
    parser.add_argument("--skip-step3", action="store_true")
    parser.add_argument(
        "--out",
        type=str,
        default=str(BENCH_RUNS_DIR / "single.json"),
    )
    args = parser.parse_args()

    if args.single_config:
        run_single_benchmark(
            tag=args.tag,
            budget_pct=args.budget_pct if (args.budget_pct is not None or args.budget_gb is None) else None,
            budget_gb=args.budget_gb,
            alloc_mode=args.alloc_mode,
            prefetch=(args.prefetch == "on"),
            decode_tokens=args.decode_tokens,
            out_path=Path(args.out),
            prefetch_top_m=args.prefetch_top_m,
            prefetch_cap=args.prefetch_cap,
        )
    elif args.v1_suite:
        run_full_suite(decode_tokens=args.decode_tokens, skip_step1=args.skip_step1)
    else:
        run_v2_suite(decode_tokens=args.decode_tokens, skip_step3=args.skip_step3)


if __name__ == "__main__":
    main()


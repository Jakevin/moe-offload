#!/usr/bin/env python3
"""A2 trial harness v2 (design-review fixes). One model process; loader = mlx-qwen38/loader.py of this repo, unmodified.
stage exact : prompts x N tokens; arms off, off(repeat), seed arms. Compares tokens, logits of every forward (sha256 of
              raw bits, incl. prefill last position), full model state after prefill and at the end
              (prompt_cache.snapshot_state arrays, raw bits) plus the CPU-PLE history, against the first `off`.
              Exit code 4 if anything differs.
stage speed : requires --exact-ok <exact jsonl> whose every row passed. Warm-up (each arm once on `short`), then R
              rounds alternating arm order (AB, BA, ...). Expert LRU reset before every run; OS page cache is NOT
              cleared (warm condition; residency fraction recorded with mincore before each run).
Timing: t0 = prefill start; token 1 is produced by prefill (TTFT); tokens 2..N by N-1 decode forwards;
decode tok/s = (N-1)/(t_N - t_1); request_s = t_N - t0. No resource polling inside the timed loop
(external runner monitors RSS/swap every 5 s); per-run swap start/end and MLX peak are recorded.
Only a single full prefill per prompt is exercised (not the server's chunked path)."""
import argparse, ctypes, gc, hashlib, json, mmap, os, sys, time
from pathlib import Path
WT = Path(os.environ.get("MLX_QWEN38_DIR", Path(__file__).resolve().parents[2]))  # mlx-qwen38/
sys.path.insert(0, str(WT)); sys.path.insert(0, str(Path(__file__).resolve().parent))
assert "QWEN38_MODEL" in os.environ, "set QWEN38_MODEL to the model directory"
import numpy as np
import mlx.core as mx
import loader as Ld
import prompt_cache as PC
import a2_seed as S
mx.set_cache_limit(2 * 1024**3)
HERE = Path(__file__).resolve().parent
def log(m): print(f"{time.strftime('%F %T')} {m}", flush=True)
_U = {1: mx.uint8, 2: mx.uint16, 4: mx.uint32, 8: mx.uint64}
def abits(a):
    return np.array(mx.view(mx.contiguous(a), _U[a.dtype.size])).tobytes() if a.size else b""
def state_hash(cache, lm, ple_hist):
    st, nb = PC.snapshot_state(cache, lm); h = hashlib.sha256()
    def walk(v, path):
        if isinstance(v, mx.array):
            h.update(path.encode()); h.update(str(v.dtype).encode()); h.update(str(v.shape).encode()); h.update(abits(v))
        elif isinstance(v, dict):
            for k in sorted(v, key=str): walk(v[k], f"{path}.{k}")
        elif isinstance(v, (list, tuple)):
            for i, x in enumerate(v): walk(x, f"{path}[{i}]")
        elif isinstance(v, (bool, int, float, str, type(None))):
            h.update(f"{path}={v!r}".encode())
    for i, (cls, s) in enumerate(st["layers"]): walk(s, f"L{i}:{cls.__name__}")
    walk(st["lm"], "lm"); walk(list(ple_hist) if ple_hist is not None else None, "ple_cpu_history")
    del st; return h.hexdigest()
_libc = ctypes.CDLL(None)
def page_resident_frac(model_dir):
    tot = res = 0; ps = mmap.PAGESIZE
    for f in sorted(Path(model_dir).glob("experts-*.safetensors")):
        if f.name == "experts-00.safetensors": continue
        n = f.stat().st_size
        with open(f, "rb") as fh:
            mm = mmap.mmap(fh.fileno(), n, prot=mmap.PROT_READ)
            a = np.frombuffer(mm, dtype=np.uint8); npg = (n + ps - 1) // ps; vec = (ctypes.c_ubyte * npg)()
            if _libc.mincore(ctypes.c_void_p(a.ctypes.data), ctypes.c_size_t(n), vec) == 0:
                res += int((np.frombuffer(vec, dtype=np.uint8) & 1).sum()); tot += npg
            del a; mm.close()
    return round(res / tot, 4) if tot else None

def run(eng, ids, n_new, mode, frac, record, max_rss_kb):
    eng.reset_caches(); S.A2.mode, S.A2.frac = mode, frac; S.begin_prompt()
    lm = eng.model.language_model; cache = lm.make_cache(); m = eng.model
    swap0 = Ld.get_swap_used_mb(); mx.reset_peak_memory()
    m._ple_cpu_active = False
    t0 = time.perf_counter()
    out = lm(mx.array([ids], dtype=mx.int32), cache=cache)
    lg = out.logits[:, -1:, :]; cur = mx.argmax(lg, axis=-1); mx.eval(cur)
    t_tok = [time.perf_counter()]
    seed_stats = dict(S.A2.stats); S.A2.mode = "off"
    rec = {"logit_sha": [], "tokens": []}
    if record: rec["logit_sha"].append(hashlib.sha256(abits(lg)).hexdigest())
    eos = getattr(m, "_ple_eos_id", 248044); pt = [eos, eos] + list(ids)
    m._ple_cpu_history = [pt[-2], pt[-1], eos]; m._ple_cpu_active = True; hist = m._ple_cpu_history
    if record:
        rec["state_prefill"] = state_hash(cache, lm, hist)
        t_tok = [None]   # exact stage: timing not used
    s0 = eng._snapshot_decode_totals()
    for i in range(n_new - 1):
        tid = int(cur.item()); rec["tokens"].append(tid); hist[2] = tid
        out = lm(cur, cache=cache); lg = out.logits[:, -1:, :]; cur = mx.argmax(lg, axis=-1); mx.eval(cur)
        if record: rec["logit_sha"].append(hashlib.sha256(abits(lg)).hexdigest())
        else: t_tok.append(time.perf_counter())
        hist[0] = hist[1]; hist[1] = tid
        if (i + 1) % 32 == 0: mx.clear_cache()
    rec["tokens"].append(int(cur.item()))
    s1 = eng._snapshot_decode_totals(); m._ple_cpu_active = False
    if record: rec["state_end"] = state_hash(cache, lm, hist)
    peak = mx.get_peak_memory() / 1e9; swap1 = Ld.get_swap_used_mb()
    del cache; gc.collect(); mx.clear_cache()
    rss = Ld.get_rss_kb()
    if rss > max_rss_kb or swap1 - swap0 > 8192:
        log(f"ABORT rss_kb={rss} swap_delta_mb={swap1 - swap0:.0f}"); sys.exit(3)
    req = max(1, s1[3] - s0[3])
    rec.update({"mode": mode, "frac": frac, "n_prompt": len(ids), "n_new": n_new,
                "hit_rate": (s1[0] - s0[0] + s1[1] - s0[1]) / req, "lru_hit_rate": (s1[0] - s0[0]) / req,
                "demand_read_MB": (s1[4] - s0[4]) / 1e6, "prefetch_read_MB": (s1[5] - s0[5]) / 1e6,
                "seed": seed_stats, "mlx_peak_GB": peak, "rss_gib_after": rss / 1024**2,
                "swap_start_mb": swap0, "swap_end_mb": swap1})
    if not record:
        rec.update({"ttft_s": t_tok[0] - t0, "decode_s": t_tok[-1] - t_tok[0],
                    "decode_tok_s": (n_new - 1) / (t_tok[-1] - t_tok[0]), "request_s": t_tok[-1] - t0})
    return rec

def parse_arm(a):  # off | full | topk:0.25
    k, _, v = a.partition(":"); return (k, float(v) if k == "topk" else 1.0)
def tag(mode, frac): return mode if mode != "topk" else f"topk:{frac}"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["exact", "speed"]); ap.add_argument("--prompts", default="story,doc6k")
    ap.add_argument("--arms", default="off,off,topk:0.25,topk:0.5,full"); ap.add_argument("--n-new", type=int, default=32)
    ap.add_argument("--rounds", type=int, default=6); ap.add_argument("--max-rss-gb", type=float, required=True)
    ap.add_argument("--exact-ok", default=None); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    arms = [parse_arm(x) for x in a.arms.split(",")]
    if a.stage == "speed":
        assert a.exact_ok, "speed requires --exact-ok"
        rows = [json.loads(l) for l in open(a.exact_ok)]
        need = {tag(md, fr) for md, fr in arms}
        okarms = {r["arm"].split("#")[0] for r in rows if all(v for k, v in r["cmp_vs_first_off"].items() if k != "n_logit_steps")}
        bad = {r["arm"].split("#")[0] for r in rows if not all(v for k, v in r["cmp_vs_first_off"].items() if k != "n_logit_steps")}
        if bad & need or not need <= okarms:
            log(f"REFUSE speed: exact not passed for {sorted(need - okarms | (bad & need))}"); sys.exit(5)
    P = json.load(open(HERE / "out/prompts_a2.json"))["ids"]; max_kb = a.max_rss_gb * 1e6
    S.install(Ld)
    eng = Ld.OffloadEngine(model_path=Path(os.environ["QWEN38_MODEL"]), budget_pct=20.0, alloc_mode="even",
                           prefetch=True, prefetch_top_m=12, prefetch_cap=3)
    log(f"loaded in {eng.load_seconds:.1f}s slots={eng.total_cached_slots} cap_layer1={eng.capacities[1]} mlx={mx.__version__}")
    outf = open(a.out, "a"); MD = os.environ["QWEN38_MODEL"]
    if a.stage == "exact":
        allok = True
        for pn in a.prompts.split(","):
            ref = None
            for j, (mode, frac) in enumerate(arms):
                r = run(eng, P[pn], a.n_new, mode, frac, True, max_kb)
                if ref is None: ref = r
                cmp = {"tokens": r["tokens"] == ref["tokens"], "logits_all_steps": r["logit_sha"] == ref["logit_sha"],
                       "n_logit_steps": len(r["logit_sha"]), "state_prefill": r["state_prefill"] == ref["state_prefill"],
                       "state_end": r["state_end"] == ref["state_end"]}
                ok = all(v for k, v in cmp.items() if k != "n_logit_steps"); allok &= ok
                row = {"prompt": pn, "arm": f"{tag(mode, frac)}#{j}", "ok": ok, "cmp_vs_first_off": cmp,
                       **{k: v for k, v in r.items() if k not in ("logit_sha", "tokens")}}
                outf.write(json.dumps(row) + "\n"); outf.flush()
                log(json.dumps({k: row[k] for k in ("prompt", "arm", "ok", "cmp_vs_first_off", "seed", "swap_start_mb", "swap_end_mb")}))
        log(f"exact DONE all_ok={allok}"); sys.exit(0 if allok else 4)
    for mode, frac in arms:   # identical warm-up for each arm, then the LRU reset inside run()
        run(eng, P["short"], 8, mode, frac, False, max_kb)
    log("warm-up done")
    for rd in range(a.rounds):
        order = arms if rd % 2 == 0 else arms[::-1]
        for pn in a.prompts.split(","):
            for mode, frac in order:
                prf = page_resident_frac(MD)
                r = run(eng, P[pn], a.n_new, mode, frac, False, max_kb)
                row = {"round": rd + 1, "prompt": pn, "arm": tag(mode, frac), "page_resident_before": prf,
                       "wall_start": time.time() - r["request_s"],
                       **{k: v for k, v in r.items() if k not in ("logit_sha",)}}
                row["tokens_sha"] = hashlib.sha256(json.dumps(r["tokens"]).encode()).hexdigest()[:16]; row.pop("tokens")
                outf.write(json.dumps(row) + "\n"); outf.flush()
                log(json.dumps({k: row[k] for k in ("round", "prompt", "arm", "page_resident_before", "ttft_s", "decode_tok_s", "request_s", "hit_rate", "seed", "swap_start_mb", "swap_end_mb")}))
    log("speed DONE")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Strict bit-exact comparison of C++ outputs against a reference. Never prints token ids.
usage: cmp_exact.py --cand-prefix out/c5_smoke_p --prompts 0 --n-expect 8 --ref-tag ssmoke [--state --state-tag ssmoke]
       cmp_exact.py --cand-prefix out/c5_p1_p --prompts 0 --n-expect 128 --ref-prefix out/c5_exact_p   (C++ vs C++ serial path)
Writes <cand-prefix>cmp.json. Exit 0 only if EVERY prompt passes: ids count == ids_ref count == n-expect and equal,
logits count == n-expect on both sides, all finite and raw-bit equal, and (with --state) identical key set with
equal shape/dtype/raw bytes and finite floats. Any missing file / key / shape / dtype problem is a failure."""
import argparse, json, hashlib, sys, numpy as np, mlx.core as mx
ap = argparse.ArgumentParser(); ap.add_argument("--cand-prefix", required=True); ap.add_argument("--prompts", type=int, nargs="+", default=[0])
ap.add_argument("--n-expect", type=int, required=True); ap.add_argument("--ref-tag", default="refst"); ap.add_argument("--state-tag", default=None)
ap.add_argument("--ref-prefix", default=None, help="use another candidate's outputs (prefix) as the reference instead of out/py_*")
ap.add_argument("--state", action="store_true"); a = ap.parse_args(); V = 248320; res = {}; all_ok = True
st_tag = a.state_tag or a.ref_tag
def ref_path(kind, p):
    if a.ref_prefix: return {"ids": f"{a.ref_prefix}{p}.json", "logits": f"{a.ref_prefix}{p}.logits.npy", "state": f"{a.ref_prefix}{p}.state.safetensors"}[kind]
    return {"ids": f"out/py_ids_{a.ref_tag}_p{p}.json", "logits": f"out/py_logits_{a.ref_tag}_p{p}.npy", "state": f"out/py_state_{st_tag}_p{p}.safetensors"}[kind]
def raw(x):
    return np.array(x.view(mx.uint8)).tobytes() if x.size else b""
def finite(x):
    return bool(mx.all(mx.isfinite(x)).item()) if mx.issubdtype(x.dtype, mx.floating) and x.size else True
def load_logits(path):
    if path.endswith(".npy") and not a.ref_prefix and path.startswith("out/py_"): return np.load(path, mmap_mode="r")
    return np.array(mx.load(path).astype(mx.float32)).reshape(-1, V)
for p in a.prompts:
    r = {}; fails = []
    try:
        ref = json.load(open(ref_path("ids", p)))["ids"]; ci = json.load(open(f"{a.cand_prefix}{p}.json"))["ids"]
        n = min(len(ref), len(ci)); fm = next((i for i in range(n) if ref[i] != ci[i]), -1)
        r.update(n_ref=len(ref), n_cand=len(ci), first_mismatch=fm, prefix_equal=(fm < 0), cand_sha16=hashlib.sha256(json.dumps(ci).encode()).hexdigest()[:16])
        if len(ci) != a.n_expect: fails.append(f"n_cand {len(ci)}!={a.n_expect}")
        if len(ref) < a.n_expect: fails.append(f"n_ref {len(ref)}<{a.n_expect}")
        if ci != ref[:a.n_expect]: fails.append("ids differ")
    except Exception as e: fails.append("ids_err " + repr(e)[:160])
    try:
        cl = load_logits(f"{a.cand_prefix}{p}.logits.npy"); rl = load_logits(ref_path("logits", p))
        if cl.shape[-1] != V or rl.shape[-1] != V: fails.append(f"logit vocab {cl.shape}/{rl.shape}")
        if len(cl) != a.n_expect: fails.append(f"cand logit steps {len(cl)}!={a.n_expect}")
        if len(rl) < a.n_expect: fails.append(f"ref logit steps {len(rl)}<{a.n_expect}")
        k = min(len(cl), len(rl), a.n_expect); d = []; nbits = 0; nonfin = 0; margins = []
        for i in range(k):
            c, q = np.asarray(cl[i], np.float32), np.asarray(rl[i], np.float32)
            nonfin += int(not np.all(np.isfinite(c))) + int(not np.all(np.isfinite(q)))
            d.append(float(np.max(np.abs(c - q)))); nbits += int(np.count_nonzero(c.view(np.uint32) != q.view(np.uint32)))
            t2 = np.partition(q, -2)[-2:]; margins.append(float(t2[1] - t2[0]))
        am = int(sum(int(np.argmax(cl[i]) == np.argmax(rl[i])) for i in range(k)))
        r.update(logit_steps=k, logit_maxdiff=max(d) if d else None, logit_first_nonzero=next((i for i, x in enumerate(d) if x > 0), -1),
                 logit_diff_first8=[round(x, 5) for x in d[:8]], logit_raw_bits_neq=nbits, logit_nonfinite_rows=nonfin, argmax_equal=am,
                 ref_min_margin=round(min(margins), 5) if margins else None)
        if nbits: fails.append(f"logits raw bits differ ({nbits} elems, max {max(d):.4g})")
        if nonfin: fails.append(f"non-finite logits rows {nonfin}")
    except Exception as e: fails.append("logits_err " + repr(e)[:160])
    if a.state:
        try:
            rs = mx.load(ref_path("state", p)); cs = mx.load(f"{a.cand_prefix}{p}.state.safetensors"); bad = []
            for key in sorted(set(rs) | set(cs)):
                if key not in cs: bad.append(f"{key}:missing_in_cand"); continue
                if key not in rs: bad.append(f"{key}:extra_in_cand"); continue
                x, y = rs[key], cs[key]
                if x.shape != y.shape: bad.append(f"{key}:shape{tuple(x.shape)}vs{tuple(y.shape)}"); continue
                if x.dtype != y.dtype: bad.append(f"{key}:dtype {x.dtype}vs{y.dtype}"); continue
                if not finite(y): bad.append(f"{key}:nonfinite_cand"); continue
                if not finite(x): bad.append(f"{key}:nonfinite_ref"); continue
                if raw(x) != raw(y):
                    dd = float(mx.max(mx.abs(x.astype(mx.float32) - y.astype(mx.float32))).item()) if x.size else 0.0
                    bad.append(f"{key}:bits_differ maxabs={dd:.3g}")
            r.update(state_keys_ref=len(rs), state_keys_cand=len(cs), state_not_equal=len(bad), state_issues=bad[:40])
            if bad: fails.append(f"state {len(bad)} keys not bit-identical")
        except Exception as e: fails.append("state_err " + repr(e)[:160])
    r["fails"] = fails; r["pass"] = not fails; all_ok &= not fails
    res[p] = r; print(f"[cmp p{p}] {'PASS' if not fails else 'FAIL'}", json.dumps(r))
res["all_pass"] = all_ok; json.dump(res, open(f"{a.cand_prefix}cmp.json", "w"), indent=1)
print("[cmp] ALL PASS" if all_ok else "[cmp] FAIL"); sys.exit(0 if all_ok else 1)

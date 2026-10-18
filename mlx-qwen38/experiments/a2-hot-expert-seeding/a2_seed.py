"""A2: after a prefill forward of a cached MoE layer, make that layer's expert cache hold the experts this prompt
routed to most (instead of whatever the last prefill chunk left behind). Wrapper only: loader.py is not modified.

Compute order, chunk grouping and kernels of prefill are untouched: the original CachedSwitchGLU.__call__ runs as is,
its output is evaluated, and only then are slots re-filled (so no pending computation can read a re-filled slot).
Modes:
  off     : original behaviour (arm A)
  full    : resident set = top-`capacity` experts by this prompt's routed-token count in this layer
  topk    : make sure the top-K (K = int(frac * capacity)) are resident; other residents stay
  thresh  : like full, but only experts with count >= T
Ordering after seeding: experts not selected keep their old relative order at the LRU end; selected experts follow,
ascending by count (hottest = most recently used). Counts are accumulated over the prefill calls of one prompt
(reset with begin_prompt()).
"""
from __future__ import annotations
import ctypes, time
import numpy as np
import mlx.core as mx

class A2:
    mode = "off"; frac = 1.0; thresh = 1
    stats: dict = {}
    counts: dict = {}

def begin_prompt():
    A2.counts = {}; A2.stats = {"layers": 0, "loaded_experts": 0, "seed_s": 0.0, "eval_s": 0.0}

def install(loader_mod):
    C = loader_mod.CachedSwitchGLU
    if getattr(C, "_a2_installed", False):
        return
    orig = C.__call__
    NE = loader_mod.NUM_EXPERTS

    def call(self, x, indices):
        y = orig(self, x, indices)
        if A2.mode == "off" or indices.shape[1] == 1:
            return y
        t0 = time.perf_counter(); mx.eval(y); t1 = time.perf_counter()
        cnt = A2.counts.get(self.layer_idx)
        c_now = np.bincount(np.asarray(indices).reshape(-1).astype(np.int64), minlength=NE)
        cnt = c_now if cnt is None else cnt + c_now
        A2.counts[self.layer_idx] = cnt
        n = _seed(self, cnt, loader_mod)
        st = A2.stats; st["layers"] += 1; st["loaded_experts"] += n
        st["eval_s"] += t1 - t0; st["seed_s"] += time.perf_counter() - t1
        return y

    C.__call__ = call
    C._a2_installed = True

def _seed(self, cnt, L):
    cap = self.capacity
    used = [int(e) for e in np.lexsort((np.arange(len(cnt)), -cnt)) if cnt[e] > 0]   # count desc, eid asc
    if A2.mode == "full":
        want = used[:cap]
    elif A2.mode == "topk":
        want = used[:max(0, min(cap, int(A2.frac * cap)))]
    elif A2.mode == "thresh":
        want = [e for e in used if cnt[e] >= A2.thresh][:cap]
    else:
        raise ValueError(A2.mode)
    wset = set(want)
    lru = list(self.lru.items())                       # LRU first
    resident = dict(lru)
    missing = [e for e in want if e not in resident]
    free = list(self.free_slots)
    victims = sorted((e for e, _ in lru if e not in wset), key=lambda e: (cnt[e], [k for k, _ in lru].index(e)))
    load_e, load_s = [], []
    for e in missing:
        if free:
            s = free.pop()
        else:
            v = victims.pop(0); s = resident.pop(v)
        load_e.append(e); load_s.append(s); resident[e] = s
    if load_e:
        self._load_coalesced_sync(load_e, load_s)
    keep_rest = [e for e, _ in lru if e in resident and e not in wset]
    hot = sorted((e for e in want), key=lambda e: (cnt[e], -e))   # ascending count -> hottest last (MRU)
    order = keep_rest + hot
    self.lru.clear()
    for e in order:
        self.lru[e] = resident[e]
    used_s = set(self.lru.values())
    self.free_slots = [s for s in range(cap - 1, -1, -1) if s not in used_s]
    n = len(order)
    L.C_LIB.c_layer_set_lru_state(self.layer_idx, (ctypes.c_int32 * n)(*order),
                                  (ctypes.c_int32 * n)(*[self.lru[e] for e in order]), n, 0, 0, 0)
    return len(load_e)

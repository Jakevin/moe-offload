"""Minimal single-session, in-memory prompt cache for the Qwen3.8 offload server.

Design (PROFESSOR_CPU_SPEEDUP.md, P0 "prompt cache + linear-attention state snapshots"):
  * One session, in memory only, at most `max_snapshots` snapshots and `max_bytes` total (LRU eviction).
  * Key = exact prompt token IDs (prefix match on the list of ints, never text) + a settings key built from
    model path, chat-template hash, control-vector/ablation config and the chunking policy.
  * A snapshot is a bit-exact deep copy of every per-layer cache object (GDN conv/ssm state, layer-1 PLE window
    and PLE conv state, QSA KV + indexer + pooled-index state, offsets/counters) plus LanguageModel
    _position_ids/_rope_deltas, plus the last-position logits.
  * Reusable prefix points: A1 = end of the system message (system prompt + tool definitions, i.e. right after the
    first "<|im_end|>\\n"), and END = end of the full prompt = end of the last confirmed turn incl. the generation
    prompt.  (With this chat template a prior prompt, including "<|im_start|>assistant\\n<think>...", is a token
    prefix of the next turn's prompt.)
  * Off-by-one: snapshots are taken only at prefill boundaries, where n_consumed == len(tokens) exactly (every
    token in the key has been fed through the model) and there is no pending token; the next-token logits are
    stored so a FULL-prefix hit (identical prompt) needs no forward pass.  Post-decode states are never
    snapshotted: the server yields a token before feeding it, so the last generated token is not in the cache,
    and decode-path (L=1) numerics differ from prefill anyway.
  * Fixed chunking: the prompt is prefilled in chunks whose boundaries are the deterministic set
    {restore point H (if hit), A1 (only if that system+tools prefix was seen before), n}, with every span
    further cut into `chunk_tokens` pieces from its start (0 = no cap). Boundaries depend only on the token
    IDs, the restore point and the A1-seen bit; logits differ slightly between different chunkings (disclosed
    in the release notes), but a cached run and an uncached run with the same boundaries are bit-identical.
  * Snapshots are never shared with live caches: saving clones, restoring clones again, so in-place KV writes
    during decode can never corrupt a stored snapshot (MLX buffer donation / custom kernels).
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from typing import Any

import mlx.core as mx

SCHEMA_VERSION = 1
_UINT = {1: mx.uint8, 2: mx.uint16, 4: mx.uint32, 8: mx.uint64}


def copy_array(a: mx.array) -> mx.array:
    """Bit-exact copy into a fresh buffer (view as unsigned int, +0, view back: preserves -0.0 / NaN bits)."""
    if a.size == 0:
        return mx.zeros(a.shape, dtype=a.dtype)
    ut = _UINT[a.dtype.size]
    b = mx.view(mx.contiguous(a), ut) + mx.array(0, dtype=ut)
    return mx.view(b, a.dtype) if a.dtype != ut else b


def _clone(v: Any, arrays: list) -> Any:
    if isinstance(v, mx.array):
        c = copy_array(v); arrays.append(c); return c
    if v is None or isinstance(v, (bool, int, float, str, bytes)):
        return v
    if isinstance(v, list):
        return [_clone(x, arrays) for x in v]
    if isinstance(v, tuple):
        return tuple(_clone(x, arrays) for x in v)
    if isinstance(v, dict):
        return {k: _clone(x, arrays) for k, x in v.items()}
    if callable(v) and not hasattr(v, "__dict__") or type(v).__name__ in ("gc_func", "function", "builtin_function_or_method"):
        # compiled-kernel / function memo entries hold no per-sequence data: shared by reference
        return v
    raise TypeError(f"prompt_cache: unsupported cache attribute type {type(v).__name__}")


def _obj_state(c: Any) -> dict:
    if hasattr(c, "__slots__") and not hasattr(c, "__dict__"):
        raise TypeError(f"prompt_cache: __slots__ cache {type(c).__name__} unsupported")
    st = dict(vars(c))
    for k in ("_speculation", "_qwen4_exp_ple_speculative_state"):
        if st.get(k) is not None:
            raise RuntimeError(f"prompt_cache: refusing to snapshot cache with live {k}")
    return st


def _nbytes(arrays: list) -> int:
    return int(sum(a.nbytes for a in arrays))


def snapshot_state(cache: list, lm: Any) -> tuple[dict, int]:
    arrays: list = []
    layers = [(type(c), _clone(_obj_state(c), arrays)) for c in cache]
    lm_state = {"_position_ids": _clone(getattr(lm, "_position_ids", None), arrays),
                "_rope_deltas": _clone(getattr(lm, "_rope_deltas", None), arrays)}
    if arrays:
        mx.eval(*arrays)
    return {"layers": layers, "lm": lm_state}, _nbytes(arrays)


def restore_state(state: dict, lm: Any) -> list:
    arrays: list = []
    cache = []
    for cls, st in state["layers"]:
        obj = cls.__new__(cls)
        obj.__dict__.update(_clone(st, arrays))
        cache.append(obj)
    lm._position_ids = _clone(state["lm"]["_position_ids"], arrays)
    lm._rope_deltas = _clone(state["lm"]["_rope_deltas"], arrays)
    if arrays:
        mx.eval(*arrays)
    return cache


@dataclass
class Snapshot:
    tokens: tuple
    settings: str
    kind: str
    state: dict
    logits: mx.array  # [1, 1, V] logits at the last consumed position
    nbytes: int
    created: float = field(default_factory=time.time)
    last_used: float = field(default_factory=time.time)
    hits: int = 0

    @property
    def n_consumed(self) -> int:
        return len(self.tokens)


def find_anchor_a1(tokens: list[int], im_start: int, im_end: int, nl: int, system_tok: list[int]) -> int | None:
    """End of the first message if it is a system message: index right after the first <|im_end|>\\n."""
    if len(tokens) < 4 or tokens[0] != im_start or tokens[1:1 + len(system_tok)] != system_tok:
        return None
    for i in range(1, len(tokens) - 1):
        if tokens[i] == im_end:
            return i + 2 if tokens[i + 1] == nl else i + 1
    return None


def plan_chunks(n: int, start: int, anchors: list[int], chunk_tokens: int) -> list[tuple[int, int]]:
    """Boundaries = start, anchors in (start, n), n; then every span longer than chunk_tokens is cut into
    chunk_tokens pieces counted from the span start (0 = no cap)."""
    pts = sorted({start, n} | {b for b in anchors if start < b < n})
    out = []
    for a, b in zip(pts[:-1], pts[1:]):
        if chunk_tokens and chunk_tokens > 0:
            for x in range(a, b, chunk_tokens):
                out.append((x, min(b, x + chunk_tokens)))
        else:
            out.append((a, b))
    return out


class _LastPositionHead:
    """Context manager: make lm.lm_head project only the last position during prefill.

    Avoids materialising [1, L, 248320] prefill logits (~7.4 GB at L=15k) when only the last row is used.
    Note: an M=1 projection may pick a different Metal kernel than the full [L, V] projection, so the
    last-row logits are not guaranteed bit-identical to the original one-shot server path; with/without
    prompt cache both use this same path, so cache on/off comparisons stay exact.
    """

    def __init__(self, lm: Any, enabled: bool = True):
        self.head = getattr(lm, "lm_head", None) if enabled else None
        self.orig_cls = None

    def __enter__(self):
        if self.head is not None:
            self.orig_cls = type(self.head)
            base = self.orig_cls

            def _call(this, x, *a, **k):
                return base.__call__(this, x[:, -1:, :], *a, **k)

            self.head.__class__ = type("LastPos" + base.__name__, (base,), {"__call__": _call})
        return self

    def __exit__(self, *exc):
        if self.orig_cls is not None:
            self.head.__class__ = self.orig_cls
        return False


class PromptCache:
    def __init__(self, engine: Any, max_snapshots: int = 4, max_gb: float = 3.0, chunk_tokens: int = 0,
                 min_prefix: int = 64, last_logits_only: bool = True):
        self.engine = engine
        self.max_snapshots = int(max_snapshots)
        self.max_bytes = int(max_gb * 2**30)
        self.chunk_tokens = int(chunk_tokens)
        self.min_prefix = int(min_prefix)
        self.last_logits_only = bool(last_logits_only)
        self.debug_keep_cache = False
        self.debug_last_cache = None
        self.snaps: list[Snapshot] = []
        tok = engine.tokenizer
        self.im_start = int(tok.convert_tokens_to_ids("<|im_start|>"))
        self.im_end = int(tok.convert_tokens_to_ids("<|im_end|>"))
        self.nl = tok.encode("\n", add_special_tokens=False)
        self.nl = int(self.nl[0]) if len(self.nl) == 1 else -1
        self.system_tok = tok.encode("system", add_special_tokens=False)
        self.seen_a1: dict[str, float] = {}  # sha of system+tools token prefixes seen in earlier requests
        self.stats = {"lookups": 0, "hits": 0, "full_hits": 0, "misses": 0, "saved": 0, "evicted": 0,
                      "reused_tokens": 0, "prefilled_tokens": 0}
        self.last_info: dict = {}

    # ---------------------------------------------------------------- keys
    def settings_key(self) -> str:
        e = self.engine
        tmpl = getattr(e.tokenizer, "chat_template", None) or ""
        cv = getattr(e, "cv_config", None)
        d = {"v": SCHEMA_VERSION, "model": str(getattr(e, "model_path", "")),
             "template_sha": hashlib.sha256(str(tmpl).encode()).hexdigest()[:16],
             "cv": json.dumps(cv, sort_keys=True, default=str),
             "cv_active": bool(getattr(e.model, "_cv_has_active", False)),
             "chunk_tokens": self.chunk_tokens, "last_logits_only": self.last_logits_only}
        return hashlib.sha256(json.dumps(d, sort_keys=True).encode()).hexdigest()

    @property
    def total_bytes(self) -> int:
        return sum(s.nbytes for s in self.snaps)

    def clear(self) -> None:
        self.snaps.clear(); self.seen_a1.clear()

    def lookup(self, tokens: list[int], settings: str) -> Snapshot | None:
        best = None
        n = len(tokens)
        for s in self.snaps:
            k = s.n_consumed
            if s.settings != settings or k > n or k < self.min_prefix:
                continue
            if best is not None and k <= best.n_consumed:
                continue
            if list(s.tokens) == tokens[:k]:
                best = s
        return best

    def _save(self, tokens: list[int], settings: str, kind: str, cache: list, lm: Any, logits: mx.array) -> None:
        key = tuple(tokens)
        for s in self.snaps:  # already have exactly this prefix
            if s.settings == settings and s.tokens == key:
                s.last_used = time.time(); return
        state, nb = snapshot_state(cache, lm)
        lg = copy_array(logits); mx.eval(lg)
        nb += int(lg.nbytes)
        if nb > self.max_bytes:
            return
        self.snaps.append(Snapshot(key, settings, kind, state, lg, nb))
        self.stats["saved"] += 1
        while len(self.snaps) > self.max_snapshots or self.total_bytes > self.max_bytes:
            victim = min(self.snaps[:-1], key=lambda s: s.last_used)
            self.snaps.remove(victim); self.stats["evicted"] += 1

    # ---------------------------------------------------------------- prefill
    def prefill(self, lm: Any, prompt_ids: list[int], *, boundaries: list[int] | None = None,
                use_cache: bool = True, save: bool = True, chunk_tokens: int | None = None) -> tuple[list, mx.array, dict]:
        """Chunked prefill with snapshot reuse. Returns (cache, last_logits[1,1,V], info).

        `boundaries` overrides the anchor set (used by the verifier to force identical chunking in A/B runs).
        """
        t0 = time.perf_counter()
        n = len(prompt_ids)
        settings = self.settings_key()
        self.stats["lookups"] += 1
        hit = self.lookup(prompt_ids, settings) if use_cache else None
        info: dict = {"prompt_tokens": n, "hit": hit is not None, "hit_kind": hit.kind if hit else None}
        if hit is not None:
            hit.last_used = time.time(); hit.hits += 1
            cache = restore_state(hit.state, lm)
            start = hit.n_consumed
            self.stats["hits"] += 1; self.stats["reused_tokens"] += start
            info["restore_s"] = round(time.perf_counter() - t0, 4)
        else:
            cache = lm.make_cache()
            start = 0
            self.stats["misses"] += 1
        info["reused_tokens"] = start
        a1 = find_anchor_a1(prompt_ids, self.im_start, self.im_end, self.nl, self.system_tok)
        # A1 split policy: every extra prefill pass re-streams the routed experts (~a full expert sweep), so a
        # first-time miss is prefilled without the A1 cut (miss overhead = snapshot copy only). The A1 cut (and
        # an A1 snapshot) is made only when this system+tools prefix was already seen in an earlier request,
        # i.e. when it is demonstrably shared across conversations.
        a1_key = None
        if a1 is not None:
            a1_key = hashlib.sha256(json.dumps(prompt_ids[:a1]).encode()).hexdigest() + settings[:16]
        split_a1 = a1 is not None and use_cache and start < a1 and a1_key in self.seen_a1
        if boundaries is not None:
            anchors = list(boundaries)
        else:
            anchors = [a1] if split_a1 else []
        info["a1"] = a1; info["a1_split"] = bool(boundaries is None and split_a1)
        if a1_key is not None and use_cache:
            self.seen_a1[a1_key] = time.time()
            while len(self.seen_a1) > 64:
                self.seen_a1.pop(min(self.seen_a1, key=self.seen_a1.get))
        if start == n:  # full-prefix hit: nothing to feed; next token comes from the stored logits
            self.stats["full_hits"] += 1
            info.update({"chunks": [], "full_hit": True, "prefill_s": round(time.perf_counter() - t0, 4)})
            self.last_info = info
            if self.debug_keep_cache:
                self.debug_last_cache = cache
            return cache, copy_array(hit.logits), info
        chunks = plan_chunks(n, start, anchors, self.chunk_tokens if chunk_tokens is None else int(chunk_tokens))
        info["chunks"] = chunks
        self.engine.model._ple_cpu_active = False
        logits = None
        for (a, b) in chunks:
            with _LastPositionHead(lm, self.last_logits_only):
                out = lm(mx.array([prompt_ids[a:b]], dtype=mx.int32), cache=cache)
            logits = out.logits[:, -1:, :]
            mx.eval(logits)
            del out
            self.stats["prefilled_tokens"] += b - a
            if save and use_cache and b < n and a1 is not None and b == a1 and b >= self.min_prefix and a1 in anchors:
                self._save(prompt_ids[:b], settings, "A1_system", cache, lm, logits)
            mx.clear_cache()
        if save and use_cache and n >= self.min_prefix:
            self._save(prompt_ids, settings, "END_prompt", cache, lm, logits)
        info["prefill_s"] = round(time.perf_counter() - t0, 4)
        if self.debug_keep_cache:
            self.debug_last_cache = cache
        info["snapshots"] = len(self.snaps); info["snapshot_gib"] = round(self.total_bytes / 2**30, 3)
        self.last_info = info
        return cache, logits, info

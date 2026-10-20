#!/usr/bin/env python3
"""MLX expert-offload loader for Qwen3.8-Flash-Next (GSQ-RCO Q2 MLX).

Features:
- All non-routed-expert weights resident (embeddings, norms, full/linear attention,
  hyper-connections, routers, shared experts, lm_head); PLE n-gram table stays SSD mmap.
- Layer 0 routed experts pinned fully resident outside the cache budget (750.00 MiB / 0.786 GB).
- Layers 1..47 routed experts managed by per-layer LRU caches with configurable total budget
  (split evenly or proportionally to empirical active-expert concentration).
- Preallocated per-layer MLX/unified-memory slot buffers updated in-place via coalesced
  parallel os.preadv scatter reads from experts-*.safetensors (zero per-token mx.array
  weight allocation churn and zero copy on cache hits).
- Cross-layer speculative prefetch ('hc_post_moe'): after layer il MoE output, mixes the
  4 residual streams using layer il+1's mlp_hyper_connection + norm, applies layer il+1's
  router, selects top-16 experts, and prefetches missing experts asynchronously on a thread
  pool while layer il+1's attention + router + shared expert execute on the Metal GPU.
"""

from __future__ import annotations

import argparse
import ctypes
import gc
import json
import os
import re
import struct
import subprocess
import sys
import time
from collections import OrderedDict
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any

# Ensure framework-mlx-base is before vendor so mlx_vlm top-level package imports cleanly,
# then apply_mlx_vlm_qwen4_exp_compat_patch() prepends vendor/mlx_vlm/models.
_BASE_SITE = "/Applications/oMLX.app/Contents/Resources/Python/framework-mlx-base/lib/python3.11/site-packages"
_OMLX_RES = "/Applications/oMLX.app/Contents/Resources"
for _p in (_OMLX_RES, _BASE_SITE):
    if _p in sys.path:
        sys.path.remove(_p)
    sys.path.insert(0, _p)

import mlx.core as mx
import mlx.nn as nn
import numpy as np

# Hard memory limits required by task specification
mx.set_cache_limit(2 * 1024**3)
if hasattr(mx, "set_memory_limit"):
    mx.set_memory_limit(36 * 1024**3)

from omlx.patches.mlx_vlm_qwen4_exp_compat import (
    apply_mlx_vlm_qwen4_exp_compat_patch,
    configure_qwen4_exp_runtime,
)

apply_mlx_vlm_qwen4_exp_compat_patch()
import mlx_vlm.models.qwen4_exp.language as q4_lang
from mlx_vlm.models.qwen4_exp.language import _hc_write
from mlx_vlm.models.switch_layers import QuantizedSwitchLinear, SwiGLU, SwitchGLU
from mlx_lm.models.switch_layers import _gather_sort, _scatter_unsort
from mlx_vlm.utils import (
    _quantization_for_module_path,
    apply_generation_config_defaults,
    get_model_and_args,
    load_config,
    update_module_configs,
)
from transformers import AutoTokenizer

from prefill_qproj import prefill_qproj

MODEL_PATH = Path(os.environ.get("QWEN38_MODEL", "models/Qwen3.8-Flash-Next-GSQ-RCO-Q2-mlx"))
PROJECT_DIR = Path(os.environ.get("MOE_WORKDIR", "."))
LOADER_DIR = PROJECT_DIR / "loader"
ANALYSIS_SUMMARY_PATH = PROJECT_DIR / "analysis_summary.json"
ROUTES_REF_PATH = PROJECT_DIR / "routes.jsonl"
TRACE_META_PATH = PROJECT_DIR / "trace_meta.json"

NUM_LAYERS = 48
NUM_EXPERTS = 512
TOP_K = 10
PREFETCH_TOP_M = 12
PREFETCH_MAX_CAP = 3

# Exact byte size of 1 routed expert across all 9 tensors (gate_proj, up_proj, down_proj x weight, scales, biases)
# gate_proj: weight 409,600 B + scales 51,200 B + biases 51,200 B = 512,000 B
# up_proj:   weight 409,600 B + scales 51,200 B + biases 51,200 B = 512,000 B
# down_proj: weight 409,600 B + scales 51,200 B + biases 51,200 B = 512,000 B
EXPERT_BYTES = 1_536_000
LAYER_EXPERT_BYTES = NUM_EXPERTS * EXPERT_BYTES  # 786,432,000 B (750.00 MiB / 0.786432 GB)
ALL_48_EXPERT_BYTES = NUM_LAYERS * LAYER_EXPERT_BYTES  # 37,748,736,000 B (35.15625 GiB / 37.748736 GB)
CACHED_47_EXPERT_BYTES = (NUM_LAYERS - 1) * LAYER_EXPERT_BYTES  # 36,962,304,000 B

IO_POOL = ThreadPoolExecutor(max_workers=8, thread_name_prefix="expert-preadv")


def _load_c_expert_io() -> ctypes.CDLL:
    dylib_path = LOADER_DIR / "libexpert_io.dylib"
    c_src_path = LOADER_DIR / "expert_io.c"
    if not dylib_path.exists() and c_src_path.exists():
        subprocess.run(
            [
                "clang",
                "-O3",
                "-shared",
                "-fPIC",
                "-lpthread",
                str(c_src_path),
                "-o",
                str(dylib_path),
            ],
            check=True,
        )
    lib = ctypes.CDLL(str(dylib_path))
    lib.c_io_init.argtypes = [ctypes.c_int]
    lib.c_io_init.restype = None

    lib.c_layer_init.argtypes = [
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.c_uint64,
    ]
    lib.c_layer_init.restype = None

    lib.c_layer_reset_cache.argtypes = [ctypes.c_int]
    lib.c_layer_reset_cache.restype = None

    lib.c_layer_reset_stats.argtypes = [ctypes.c_int]
    lib.c_layer_reset_stats.restype = None

    lib.c_layer_wait_prefetch.argtypes = [ctypes.c_int]
    lib.c_layer_wait_prefetch.restype = None

    lib.c_layer_start_prefetch.argtypes = [
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_int32),
        ctypes.c_int,
        ctypes.c_int,
    ]
    lib.c_layer_start_prefetch.restype = None

    lib.c_layer_start_prefetch10.argtypes = [
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32,
        ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32,
    ]
    lib.c_layer_start_prefetch10.restype = None

    lib.c_layer_start_prefetch12.argtypes = [
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32,
        ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32,
    ]
    lib.c_layer_start_prefetch12.restype = None

    lib.c_layer_step_decode10.argtypes = [
        ctypes.c_int,
        ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32,
        ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32,
    ]
    lib.c_layer_step_decode10.restype = ctypes.c_int

    lib.c_layer_step_decode10_async.argtypes = [
        ctypes.c_int,
        ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32,
        ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32,
    ]
    lib.c_layer_step_decode10_async.restype = ctypes.c_int

    lib.c_layer_load_sync.argtypes = [
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_int32),
        ctypes.POINTER(ctypes.c_int32),
        ctypes.c_int,
        ctypes.c_int,
    ]
    lib.c_layer_load_sync.restype = None

    lib.c_layer_get_lru_state.argtypes = [
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_int32),
        ctypes.POINTER(ctypes.c_int32),
        ctypes.POINTER(ctypes.c_int),
    ]
    lib.c_layer_get_lru_state.restype = None

    lib.c_layer_set_lru_state.argtypes = [
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_int32),
        ctypes.POINTER(ctypes.c_int32),
        ctypes.c_int,
        ctypes.c_uint64,
        ctypes.c_uint64,
        ctypes.c_uint64,
    ]
    lib.c_layer_set_lru_state.restype = None

    lib.c_layer_get_stats.argtypes = [
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_uint64),
    ]
    lib.c_layer_get_stats.restype = None

    lib.c_get_decode_totals.argtypes = [
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_uint64),
    ]
    lib.c_get_decode_totals.restype = None

    lib.c_io_init(8)
    return lib


C_LIB = _load_c_expert_io()


def get_rss_kb() -> int:
    out = subprocess.check_output(["ps", "-o", "rss=", "-p", str(os.getpid())]).decode().strip()
    return int(out)


def get_swap_used_mb() -> float:
    out = subprocess.check_output(["sysctl", "vm.swapusage"]).decode()
    m = re.search(r"used\s*=\s*([0-9.]+)M", out)
    return float(m.group(1)) if m else 0.0


def topk_sorted_indices(scores: mx.array, k: int) -> mx.array:
    """Return top-k indices along last axis sorted by descending score."""
    inds = mx.argpartition(scores, kth=-k, axis=-1)[..., -k:]
    vals = mx.take_along_axis(scores, inds, axis=-1)
    order = mx.argsort(-vals, axis=-1)
    return mx.take_along_axis(inds, order, axis=-1)


def compute_layer_capacities(
    budget_gb: float | None = None,
    budget_pct: float | None = None,
    alloc_mode: str = "even",
) -> dict[int, int]:
    """Compute per-layer LRU cache slot capacities for layers 1..47 (layer 0 is pinned resident).

    Args:
        budget_gb: Total routed-expert cache budget in decimal GB across layers 1..47.
        budget_pct: Cache capacity as a percentage of 512 experts per layer (e.g., 20 or 30).
        alloc_mode: 'even' (equal slots per layer) or 'proportional' (weighted by empirical
            active_experts_decode from ../analysis_summary.json).
    """
    num_cached_layers = NUM_LAYERS - 1  # Layers 1..47
    if budget_pct is not None:
        frac = budget_pct / 100.0 if budget_pct > 1.0 else budget_pct
        per_layer_even = max(16, min(NUM_EXPERTS, round(frac * NUM_EXPERTS)))
        total_slots = num_cached_layers * per_layer_even
    else:
        if budget_gb is None:
            budget_gb = 7.5
        total_slots = int(round((budget_gb * 1e9) / EXPERT_BYTES))
        total_slots = max(num_cached_layers * 16, min(num_cached_layers * NUM_EXPERTS, total_slots))

    if alloc_mode == "proportional" and ANALYSIS_SUMMARY_PATH.exists():
        summary = json.loads(ANALYSIS_SUMMARY_PATH.read_text(encoding="utf-8"))
        per_layer_info = summary["expert_concentration"]["per_layer"]
        weights = np.array(
            [float(per_layer_info[il]["active_experts_decode"]) for il in range(1, NUM_LAYERS)],
            dtype=np.float64,
        )
        raw = weights / weights.sum() * total_slots
        caps = np.clip(np.floor(raw).astype(int), 16, NUM_EXPERTS)
        rem = total_slots - int(caps.sum())
        if rem > 0:
            fracs = raw - np.floor(raw)
            order = np.argsort(-fracs)
            for idx in order:
                if rem == 0:
                    break
                if caps[idx] < NUM_EXPERTS:
                    caps[idx] += 1
                    rem -= 1
        elif rem < 0:
            fracs = raw - np.floor(raw)
            order = np.argsort(fracs)
            for idx in order:
                if rem == 0:
                    break
                if caps[idx] > 16:
                    caps[idx] -= 1
                    rem += 1
        return {il: int(caps[il - 1]) for il in range(1, NUM_LAYERS)}

    base, rem = divmod(total_slots, num_cached_layers)
    return {
        il: min(NUM_EXPERTS, max(16, base + (1 if (il - 1) < rem else 0)))
        for il in range(1, NUM_LAYERS)
    }


class CachedSwitchGLU(nn.Module):
    """Per-layer LRU-cached SwitchGLU backed by preallocated unified-memory slot buffers
    and coalesced parallel os.preadv reads from experts-XX.safetensors.
    """

    def __init__(self, shard_path: Path, layer_idx: int, capacity: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.shard_path = shard_path
        self.capacity = capacity
        self.fd = os.open(str(shard_path), os.O_RDONLY)

        hsize = struct.unpack("<Q", os.pread(self.fd, 8, 0))[0]
        hdr = json.loads(os.pread(self.fd, hsize, 8))
        data_start = 8 + hsize

        # Build persistent SwitchGLU with preallocated C-slot QuantizedSwitchLinear modules.
        # Use object.__setattr__ so nn.Module does not register slot buffers as missing checkpoint keys.
        slot_glu = SwitchGLU.__new__(SwitchGLU)
        nn.Module.__init__(slot_glu)
        slot_glu.activation = SwiGLU()

        raw_specs: list[tuple[int, int, list[memoryview]]] = []
        raw_backing: list[mx.array] = []
        eval_list: list[mx.array] = []

        for proj in ("gate_proj", "up_proj", "down_proj"):
            ql = QuantizedSwitchLinear.__new__(QuantizedSwitchLinear)
            nn.Module.__init__(ql)
            ql.group_size = 64
            ql.bits = 2
            ql.mode = "affine"
            prefix = f"language_model.model.layers.{layer_idx}.mlp.switch_mlp.{proj}"
            for suffix in ("weight", "scales", "biases"):
                info = hdr[f"{prefix}.{suffix}"]
                shape = tuple(info["shape"])
                start, end = info["data_offsets"]
                row_bytes = (end - start) // shape[0]
                abs_offset = data_start + start

                if info["dtype"] == "U32":
                    arr = mx.zeros((capacity, shape[1], shape[2]), dtype=mx.uint32)
                    mx.eval(arr)
                    raw_backing.append(arr)
                    eval_list.append(arr)
                    np_2d = np.frombuffer(memoryview(arr), dtype=np.uint8).reshape(capacity, row_bytes)
                    slot_views = [memoryview(np_2d[s]) for s in range(capacity)]
                    setattr(ql, suffix, arr)
                else:
                    u16 = mx.zeros((capacity, shape[1], shape[2]), dtype=mx.uint16)
                    mx.eval(u16)
                    bf16 = u16.view(mx.bfloat16)
                    mx.eval(bf16)
                    raw_backing.append(u16)
                    raw_backing.append(bf16)
                    eval_list.extend([u16, bf16])
                    np_2d = np.frombuffer(memoryview(u16), dtype=np.uint8).reshape(capacity, row_bytes)
                    slot_views = [memoryview(np_2d[s]) for s in range(capacity)]
                    setattr(ql, suffix, bf16)

                raw_specs.append((abs_offset, row_bytes, slot_views))

            setattr(slot_glu, proj, ql)

        # Persistent (1, 1, TOP_K) uint32 buffer mutated in-place by C during decode
        slot_inds_buf = mx.zeros((1, 1, TOP_K), dtype=mx.uint32)
        mx.eval(slot_inds_buf)
        slot_inds_np = np.frombuffer(memoryview(slot_inds_buf), dtype=np.uint32)

        mx.eval(*eval_list)
        object.__setattr__(self, "_slot_glu", slot_glu)
        object.__setattr__(self, "_raw_backing", raw_backing)
        object.__setattr__(self, "_slot_inds_buf", slot_inds_buf)
        object.__setattr__(self, "_slot_inds_np", slot_inds_np)

        # Sort the 9 tensor specs by ascending file offset so each worker reads sequentially
        raw_specs.sort(key=lambda item: item[0])
        self.tensor_specs = tuple(raw_specs)
        # 4 balanced groups across the 9 tensors (3 x 400 KiB weights + 6 x 50 KiB scales/biases)
        self.tensor_groups = ((0, 1), (2, 3), (4, 5), (6, 7, 8))

        # Register layer slot buffers and file offsets with native C pthread cache
        u64_9 = ctypes.c_uint64 * 9
        base_offsets = u64_9()
        row_bytes_arr = u64_9()
        base_ptrs = u64_9()
        for t_idx, (boff, rbytes, views) in enumerate(self.tensor_specs):
            base_offsets[t_idx] = boff
            row_bytes_arr[t_idx] = rbytes
            arr0 = np.frombuffer(views[0], dtype=np.uint8)
            base_ptrs[t_idx] = arr0.ctypes.data

        C_LIB.c_layer_init(
            layer_idx,
            self.fd,
            capacity,
            base_offsets,
            row_bytes_arr,
            base_ptrs,
            slot_inds_np.ctypes.data,
        )

        # LRU state: maps expert_id -> slot_idx (0..capacity-1), ordered LRU (first) to MRU (last)
        self.lru: OrderedDict[int, int] = OrderedDict()
        self.free_slots: list[int] = list(range(capacity - 1, -1, -1))

        # Async prefetch state (Python fallback)
        self._prefetch_futures: list[Future] = []
        self._pre_prefetch_cache: set[int] | None = None
        self._last_pred16: list[int] = []
        self._last_missing_pred: list[int] = []
        self._c_stats_buf = (ctypes.c_uint64 * 18)()

        self.reset_stats()

    def reset_cache(self) -> None:
        self.wait_prefetch()
        C_LIB.c_layer_reset_cache(self.layer_idx)
        self.lru.clear()
        self.free_slots = list(range(self.capacity - 1, -1, -1))
        self._pre_prefetch_cache = None
        self._last_pred16 = []
        self._last_missing_pred = []

    def reset_stats(self) -> None:
        C_LIB.c_layer_reset_stats(self.layer_idx)
        # Decode stats
        self.decode_lru_hits = 0
        self.decode_prefetch_hits = 0
        self.decode_demand_misses = 0
        self.decode_requests = 0
        self.decode_demand_bytes = 0
        self.decode_prefetch_bytes = 0

        # Prefill stats
        self.prefill_lru_hits = 0
        self.prefill_demand_misses = 0
        self.prefill_requests = 0
        self.prefill_demand_bytes = 0

        # Router-level top-M vs top-10 prediction stats (decode)
        self.pred_top16_hits = 0
        self.pred_top16_total_pred = 0
        self.pred_top16_total_actual = 0

        # Miss-level prefetch stats (among actual top-10 misses not in cache before prefetch)
        self.miss_pref_hits = 0
        self.miss_pref_total_fetched = 0
        self.miss_pref_total_actual_misses = 0

        # Coalescing stats
        self.uncoalesced_tensor_reads = 0
        self.coalesced_preadv_calls = 0

    def sync_stats_from_c(self) -> None:
        st = self._c_stats_buf
        C_LIB.c_layer_get_stats(self.layer_idx, st)
        self.decode_requests = int(st[0])
        self.decode_lru_hits = int(st[1])
        self.decode_prefetch_hits = int(st[2])
        self.decode_demand_misses = int(st[3])
        self.decode_demand_bytes = int(st[4])
        self.decode_prefetch_bytes = int(st[5])
        self.prefill_requests = int(st[6])
        self.prefill_lru_hits = int(st[7])
        self.prefill_demand_misses = int(st[8])
        self.prefill_demand_bytes = int(st[9])
        self.pred_top16_hits = int(st[10])
        self.pred_top16_total_actual = int(st[11])
        self.pred_top16_total_pred = int(st[12])
        self.miss_pref_hits = int(st[13])
        self.miss_pref_total_actual_misses = int(st[14])
        self.miss_pref_total_fetched = int(st[15])
        self.uncoalesced_tensor_reads = int(st[16])
        self.coalesced_preadv_calls = int(st[17])

    def _alloc_slot(self, protected: set[int]) -> int:
        if self.free_slots:
            return self.free_slots.pop()
        for k in list(self.lru.keys()):
            if k not in protected:
                return self.lru.pop(k)
        # Fallback if capacity <= len(protected)
        _, slot = self.lru.popitem(last=False)
        return slot

    @staticmethod
    def _coalesce_runs(eids: list[int], slots: list[int]) -> list[tuple[int, list[int]]]:
        """Sort (eid, slot) by eid and coalesce contiguous expert IDs into single preadv runs."""
        pairs = sorted(zip(eids, slots), key=lambda x: x[0])
        runs: list[tuple[int, list[int]]] = []
        cur_start = pairs[0][0]
        cur_prev = cur_start
        cur_slots = [pairs[0][1]]
        for eid, sl in pairs[1:]:
            if eid == cur_prev + 1:
                cur_slots.append(sl)
                cur_prev = eid
            else:
                runs.append((cur_start, cur_slots))
                cur_start = eid
                cur_prev = eid
                cur_slots = [sl]
        runs.append((cur_start, cur_slots))
        return runs

    def _submit_coalesced_reads(self, eids: list[int], slots: list[int]) -> list[Future]:
        runs = self._coalesce_runs(eids, slots)
        self.uncoalesced_tensor_reads += len(eids) * 9
        self.coalesced_preadv_calls += len(runs) * 9
        fd = self.fd
        specs = self.tensor_specs

        def _read_group(group_indices: tuple[int, ...]) -> None:
            for t_idx in group_indices:
                base, row_bytes, views = specs[t_idx]
                for start_eid, run_slots in runs:
                    os.preadv(
                        fd,
                        [views[s] for s in run_slots],
                        base + start_eid * row_bytes,
                    )

        return [IO_POOL.submit(_read_group, g) for g in self.tensor_groups]

    def _load_coalesced_sync(self, eids: list[int], slots: list[int]) -> None:
        n = len(eids)
        if n == 0:
            return
        c_eids = (ctypes.c_int32 * n)(*eids)
        c_slots = (ctypes.c_int32 * n)(*slots)
        C_LIB.c_layer_load_sync(self.layer_idx, c_eids, c_slots, n, 1)

    def start_prefetch(self, pred_eids: list[int], max_fetch: int = PREFETCH_MAX_CAP) -> None:
        """Asynchronously prefetch missing experts from pred_eids into LRU slots via C pthread pool."""
        n = len(pred_eids)
        if n == 10:
            C_LIB.c_layer_start_prefetch10(self.layer_idx, max_fetch, *pred_eids)
        elif n == 12:
            C_LIB.c_layer_start_prefetch12(self.layer_idx, max_fetch, *pred_eids)
        else:
            c_preds = (ctypes.c_int32 * n)(*pred_eids)
            C_LIB.c_layer_start_prefetch(self.layer_idx, c_preds, n, max_fetch)

    def wait_prefetch(self) -> None:
        C_LIB.c_layer_wait_prefetch(self.layer_idx)
        if self._prefetch_futures:
            for f in self._prefetch_futures:
                f.result()
            self._prefetch_futures.clear()

    def __call__(self, x: mx.array, indices: mx.array) -> mx.array:
        B, L, K = indices.shape

        if L == 1:
            mx.eval(indices)
            eids = indices.tolist()[0][0]
            C_LIB.c_layer_step_decode10(self.layer_idx, *eids)
            return self._slot_glu(x, self._slot_inds_buf)

        # Prefill (L > 1): sync C LRU state into Python, run chunked sorted gather_qmm,
        # load missing experts via C pthread pool, and push final LRU state back to C.
        mx.eval(indices)
        out_eids = (ctypes.c_int32 * NUM_EXPERTS)()
        out_slots = (ctypes.c_int32 * NUM_EXPERTS)()
        out_cnt = ctypes.c_int(0)
        C_LIB.c_layer_get_lru_state(self.layer_idx, out_eids, out_slots, ctypes.byref(out_cnt))
        self.lru.clear()
        used_s = set()
        for i in range(out_cnt.value):
            e = int(out_eids[i])
            s = int(out_slots[i])
            self.lru[e] = s
            used_s.add(s)
        self.free_slots = [
            s for s in range(self.capacity - 1, -1, -1) if s not in used_s
        ]

        x_exp = mx.expand_dims(x, (-2, -3))
        inds_stop = mx.stop_gradient(indices)
        x_s, idx_s, inv_order = _gather_sort(x_exp, inds_stop)
        mx.eval(x_s, idx_s, inv_order)
        idx_np = np.asarray(idx_s)
        uniq_eids, first_pos, counts = np.unique(
            idx_np, return_index=True, return_counts=True
        )
        n_uniq = len(uniq_eids)
        M = indices.size
        n_qmm_experts = NUM_EXPERTS if self.layer_idx < 28 else n_uniq
        use_sorted = (M >= 64) and ((M // n_qmm_experts) >= 4)

        prior_cached = set(self.lru.keys())
        add_pre_req = M
        add_pre_hits = 0
        add_pre_misses = 0
        for eid_raw, cnt_raw in zip(uniq_eids, counts):
            eid = int(eid_raw)
            cnt = int(cnt_raw)
            if eid in prior_cached:
                add_pre_hits += cnt
            else:
                add_pre_misses += 1
                add_pre_hits += cnt - 1

        slot_to_eid: dict[int, int] = {sl: eid for eid, sl in self.lru.items()}
        eid_to_slot: dict[int, int] = dict(self.lru)

        def _run_prefill_qproj(
            ql: QuantizedSwitchLinear,
            x_in: mx.array,
            idx_in: mx.array,
            n_chunk: int,
        ) -> mx.array:
            return prefill_qproj(
                ql["weight"], ql["scales"], ql.get("biases"), x_in, idx_in,
                use_sorted, n_chunk, ql.group_size, ql.bits, ql.mode,
            )

        multi_chunk = n_uniq > self.capacity
        y_parts: list[mx.array] = []
        last_chunk_len = 0
        for u0 in range(0, n_uniq, self.capacity):
            u1 = min(n_uniq, u0 + self.capacity)
            chunk_eids_arr = uniq_eids[u0:u1]
            chunk_eids = [int(e) for e in chunk_eids_arr]
            last_chunk_len = len(chunk_eids)
            pos_start = int(first_pos[u0])
            pos_end = int(first_pos[u1 - 1] + counts[u1 - 1])

            load_eids: list[int] = []
            load_slots: list[int] = []
            for s_idx, eid in enumerate(chunk_eids):
                if slot_to_eid.get(s_idx) != eid:
                    old_eid = slot_to_eid.get(s_idx)
                    if old_eid is not None and eid_to_slot.get(old_eid) == s_idx:
                        del eid_to_slot[old_eid]
                    prev_s = eid_to_slot.get(eid)
                    if prev_s is not None and prev_s != s_idx:
                        slot_to_eid.pop(prev_s, None)
                    slot_to_eid[s_idx] = eid
                    eid_to_slot[eid] = s_idx
                    load_eids.append(eid)
                    load_slots.append(s_idx)

            if load_eids:
                self._load_coalesced_sync(load_eids, load_slots)

            local_idx = mx.array(
                np.searchsorted(chunk_eids_arr, idx_np[pos_start:pos_end]).astype(
                    np.uint32
                )
            )
            xc = x_s[pos_start:pos_end]
            x_up = _run_prefill_qproj(self._slot_glu.up_proj, xc, local_idx, last_chunk_len)
            x_gate = _run_prefill_qproj(self._slot_glu.gate_proj, xc, local_idx, last_chunk_len)
            xo = _run_prefill_qproj(
                self._slot_glu.down_proj,
                self._slot_glu.activation(x_up, x_gate),
                local_idx,
                last_chunk_len,
            )
            if multi_chunk:
                mx.eval(xo)
            y_parts.append(xo)

        # Rebuild LRU from final resident slot bijection (putting most recent chunk at MRU end)
        self.lru.clear()
        for s_idx in range(last_chunk_len, self.capacity):
            if s_idx in slot_to_eid:
                self.lru[slot_to_eid[s_idx]] = s_idx
        for s_idx in range(last_chunk_len):
            if s_idx in slot_to_eid:
                self.lru[slot_to_eid[s_idx]] = s_idx
        self.free_slots = [
            s_idx
            for s_idx in range(self.capacity - 1, -1, -1)
            if s_idx not in slot_to_eid
        ]

        n_lru = len(self.lru)
        in_eids = (ctypes.c_int32 * n_lru)(*self.lru.keys())
        in_slots = (ctypes.c_int32 * n_lru)(*self.lru.values())
        C_LIB.c_layer_set_lru_state(
            self.layer_idx,
            in_eids,
            in_slots,
            n_lru,
            add_pre_req,
            add_pre_hits,
            add_pre_misses,
        )

        y_all = y_parts[0] if len(y_parts) == 1 else mx.concatenate(y_parts, axis=0)
        return _scatter_unsort(y_all, inv_order, indices.shape).squeeze(-2)

    def get_layer_stats(self) -> dict[str, Any]:
        self.sync_stats_from_c()
        dec_req = max(1, self.decode_requests)
        pre_req = max(1, self.prefill_requests)
        return {
            "layer": self.layer_idx,
            "capacity_experts": self.capacity,
            "capacity_mb": round(self.capacity * EXPERT_BYTES / 1e6, 3),
            "decode_requests": self.decode_requests,
            "decode_lru_hits": self.decode_lru_hits,
            "decode_prefetch_hits": self.decode_prefetch_hits,
            "decode_demand_misses": self.decode_demand_misses,
            "decode_pure_hit_rate": round(self.decode_lru_hits / dec_req, 5),
            "decode_effective_hit_rate": round(
                (self.decode_lru_hits + self.decode_prefetch_hits) / dec_req, 5
            ),
            "decode_demand_bytes": self.decode_demand_bytes,
            "decode_prefetch_bytes": self.decode_prefetch_bytes,
            "decode_total_bytes": self.decode_demand_bytes + self.decode_prefetch_bytes,
            "prefill_requests": self.prefill_requests,
            "prefill_lru_hits": self.prefill_lru_hits,
            "prefill_demand_misses": self.prefill_demand_misses,
            "prefill_hit_rate": round(self.prefill_lru_hits / pre_req, 5),
            "prefill_demand_bytes": self.prefill_demand_bytes,
            "top16_router_recall": round(
                self.pred_top16_hits / max(1, self.pred_top16_total_actual), 5
            ),
            "top16_router_precision": round(
                self.pred_top16_hits / max(1, self.pred_top16_total_pred), 5
            ),
            "miss_prefetch_recall": round(
                self.miss_pref_hits / max(1, self.miss_pref_total_actual_misses), 5
            ),
            "miss_prefetch_precision": round(
                self.miss_pref_hits / max(1, self.miss_pref_total_fetched), 5
            ),
            "uncoalesced_tensor_reads": self.uncoalesced_tensor_reads,
            "coalesced_preadv_calls": self.coalesced_preadv_calls,
        }

    def close(self) -> None:
        self.wait_prefetch()
        if getattr(self, "fd", None) is not None:
            try:
                os.close(self.fd)
            except OSError:
                pass
            self.fd = None


def _install_cpu_ple_hook(model: nn.Module) -> None:
    """Eliminate Layer 1 PLE mx.eval(flat) GPU sync during decode by computing n-gram indices on CPU."""
    layers = model.language_model.model.layers
    if len(layers) <= 1 or "ple" not in layers[1]:
        return

    ple_emb = layers[1].ple.ple_embedding
    ngram_emb = ple_emb.ngram_embedding
    cpu_multipliers = [
        int(np.asarray(m, dtype=np.int64).item()) for m in ple_emb.layer_multipliers
    ]
    cpu_vocab_sizes = np.asarray(ple_emb.ngram_heads_vocab_sizes, dtype=np.int64)
    cpu_offsets = np.asarray(ple_emb.ngram_heads_offsets, dtype=np.int64)
    eos_id = int(ple_emb.eos_token_id)
    ngram_size = int(ple_emb.ngram_size)
    heads_per_ngram = int(ple_emb.heads_per_ngram)
    ngram_heads = int(ple_emb.ngram_heads)

    model._ple_eos_id = eos_id
    model._ple_cpu_history = [eos_id, eos_id, eos_id]
    model._ple_cpu_active = False

    def fast_ple_emb_call(input_ids: mx.array, cache):
        if input_ids.shape[1] > 1 or not getattr(model, "_ple_cpu_active", False):
            input_ids_i64 = input_ids.astype(mx.int64)
            previous_context = ple_emb._previous_context(input_ids_i64, cache)
            token_history = mx.concatenate([previous_context, input_ids_i64], axis=-1)
            if cache is not None:
                cache.update_window(3, token_history, ple_emb.context_len)
            ngram_ids = ple_emb._ngram_indices(token_history, input_ids.shape[1])
            embeddings = ngram_emb(ngram_ids)
            return embeddings.reshape(*embeddings.shape[:-2], -1)

        hist = model._ple_cpu_history
        t0 = np.int64(hist[2])
        t1 = np.int64(eos_id if hist[1] == eos_id else hist[1])
        t2 = np.int64(
            eos_id if (hist[1] == eos_id or hist[0] == eos_id) else hist[0]
        )
        shifted = (t0, t1, t2)

        host = np.empty(ngram_heads, dtype=np.int64)
        for ngram in range(2, ngram_size + 1):
            start = (ngram - 2) * heads_per_ngram
            end = start + heads_per_ngram
            mixed = shifted[0] * cpu_multipliers[0]
            for pos in range(1, ngram):
                mixed = np.bitwise_xor(mixed, shifted[pos] * cpu_multipliers[pos])
            host[start:end] = (mixed % cpu_vocab_sizes[start:end]) + cpu_offsets[start:end]

        plan = ngram_emb._plan(host)
        buffers = ngram_emb._assemble(host, plan)
        _, _, touched, _, families, dtypes, bits, group_size = plan
        ngram_emb.last_touched_shards = tuple(touched)
        ngram_emb.rows_read = int(host.size)
        arrays = [
            q4_lang._SafeTensorMMap.to_mx(buffers[family], dtypes[family])
            for family in families
        ]
        ngram_emb.last_uploads = len(arrays)
        values = arrays[0]
        if bits is not None:
            values = mx.dequantize(
                values,
                arrays[1],
                arrays[2],
                group_size=group_size,
                bits=bits,
                mode="affine",
            )
        values = values.astype(mx.bfloat16) * ngram_emb.weight_scale
        return values.reshape(1, 1, -1)

    ple_emb.__call__ = fast_ple_emb_call


def install_offload_hooks(
    model: nn.Module,
    prefetch_enabled: bool = True,
    prefetch_top_m: int = PREFETCH_TOP_M,
    prefetch_cap: int = PREFETCH_MAX_CAP,
) -> None:
    """Install v2 layer and MoE hooks:
    - Eliminates periodic Qwen4ExpModel _EAGER_DISPATCH async_eval calls.
    - Eliminates Layer 1 PLE mx.eval(flat) sync via CPU n-gram index hashing.
    - Minimizes mx.eval syncs to 1 per cached layer (48 total/token including final argmax).
    - Overlaps demand C pthread preadv with GPU execution of scores + shared_expert and Python _slot_glu construction.
    - When prefetch_enabled=True, evaluates next-layer router prediction (hc_pre_moe) in the SAME mx.eval(inds, pred_next)
      sync (0 extra syncs/layer) and dispatches capped top-10/12 C pthread prefetch concurrently with current-layer MoE.
    """
    q4_lang._EAGER_DISPATCH = False
    _install_cpu_ple_hook(model)

    layers = model.language_model.model.layers
    num_layers = len(layers)
    model._prefetch_enabled = prefetch_enabled
    model._prefetch_top_m = prefetch_top_m
    model._prefetch_cap = prefetch_cap
    c_pred_buf = (ctypes.c_int32 * 32)()

    for il, layer in enumerate(layers):
        def make_layer_call(layer_idx: int):
            moe_block = layer.mlp

            def hooked_layer_call(
                self_layer,
                hidden_states: mx.array,
                input_ids: mx.array,
                mask=None,
                cache=None,
                position_ids=None,
                gdn_sink=None,
                target_verify: bool = False,
                write=None,
                defer_write: bool = False,
            ):
                is_decode = hidden_states.shape[1] == 1

                if "ple" in self_layer:
                    if write is not None:
                        hidden_states = _hc_write(hidden_states, *write)
                        write = None
                    hidden_states = hidden_states + self_layer["ple"](
                        hidden_states,
                        input_ids,
                        cache,
                        mask,
                        target_verify=target_verify,
                    )

                mixed_attn, hyper_input, injection_weights = self_layer["attn_hyper_connection"](
                    hidden_states,
                    target_verify=target_verify,
                    write=write,
                )
                if self_layer.is_linear:
                    branch_attn = self_layer["linear_attn"](mixed_attn, mask=mask, cache=cache)
                else:
                    branch_attn = self_layer["self_attn"](
                        mixed_attn,
                        mask=mask,
                        cache=cache,
                        position_ids=position_ids,
                        target_verify=target_verify,
                    )

                mixed_mlp, h_pre_moe, inj_mlp = self_layer["mlp_hyper_connection"](
                    hyper_input,
                    target_verify=target_verify,
                    write=(branch_attn, injection_weights),
                )

                gates = moe_block.gate(mixed_mlp)
                gates = mx.softmax(gates, axis=-1, precise=True)
                k = moe_block.top_k
                inds = mx.argpartition(gates, kth=-k, axis=-1)[..., -k:]
                scores = mx.take_along_axis(gates, inds, axis=-1)
                scores = scores / scores.sum(axis=-1, keepdims=True)
                shared_y = moe_block.shared_expert(mixed_mlp)
                shared_y = moe_block._shared_expert_scale(mixed_mlp) * shared_y

                if not is_decode:
                    y = moe_block.switch_mlp(mixed_mlp, inds)
                    branch_mlp = (y * scores[..., None]).sum(axis=-2) + shared_y
                    if defer_write:
                        return h_pre_moe, branch_mlp, inj_mlp
                    return _hc_write(h_pre_moe, branch_mlp, inj_mlp)

                # Decode path (L == 1)
                if layer_idx == 0:
                    y = moe_block.switch_mlp(mixed_mlp, inds)
                    branch_mlp = (y * scores[..., None]).sum(axis=-2) + shared_y
                    if model._prefetch_enabled:
                        top_m = model._prefetch_top_m
                        max_cap = model._prefetch_cap
                        h_post_0 = _hc_write(h_pre_moe, branch_mlp, inj_mlp)
                        l_next = layers[1]
                        m_next, _, _ = l_next["mlp_hyper_connection"](h_post_0)
                        lg_next = l_next["mlp"].gate(m_next)
                        pred_next = topk_sorted_indices(lg_next, top_m)
                        mx.eval(h_post_0, pred_next)
                        p_list = pred_next.tolist()[0][0]
                        if top_m == 10:
                            C_LIB.c_layer_start_prefetch10(1, max_cap, *p_list)
                        elif top_m == 12:
                            C_LIB.c_layer_start_prefetch12(1, max_cap, *p_list)
                        else:
                            for idx_p, val_p in enumerate(p_list):
                                c_pred_buf[idx_p] = val_p
                            C_LIB.c_layer_start_prefetch(1, c_pred_buf, top_m, max_cap)
                    else:
                        mx.async_eval(branch_mlp)

                    if defer_write:
                        return h_pre_moe, branch_mlp, inj_mlp
                    return _hc_write(h_pre_moe, branch_mlp, inj_mlp)

                # Cached Layers 1..47 decode
                cmod: CachedSwitchGLU = moe_block.switch_mlp
                if model._prefetch_enabled and layer_idx < num_layers - 1:
                    top_m = model._prefetch_top_m
                    max_cap = model._prefetch_cap
                    l_next = layers[layer_idx + 1]
                    m_next, _, _ = l_next["mlp_hyper_connection"](h_pre_moe)
                    lg_next = l_next["mlp"].gate(m_next)
                    pred_next = topk_sorted_indices(lg_next, top_m)
                    # Single mx.eval sync evaluates both current-layer router (inds) and next-layer prediction (pred_next)
                    mx.eval(inds, pred_next)
                    eids = inds.tolist()[0][0]
                    p_list = pred_next.tolist()[0][0]
                    n_miss = C_LIB.c_layer_step_decode10_async(layer_idx, *eids)
                    if n_miss > 0:
                        mx.async_eval(scores, shared_y)
                    y = cmod._slot_glu(mixed_mlp, cmod._slot_inds_buf)
                    branch_mlp = (y * scores[..., None]).sum(axis=-2) + shared_y
                    if n_miss > 0:
                        C_LIB.c_layer_wait_prefetch(layer_idx)
                    # Dispatch current-layer MoE to Metal GPU BEFORE starting next-layer C pthread prefetch
                    mx.async_eval(branch_mlp)
                    if top_m == 10:
                        C_LIB.c_layer_start_prefetch10(layer_idx + 1, max_cap, *p_list)
                    elif top_m == 12:
                        C_LIB.c_layer_start_prefetch12(layer_idx + 1, max_cap, *p_list)
                    else:
                        for idx_p, val_p in enumerate(p_list):
                            c_pred_buf[idx_p] = val_p
                        C_LIB.c_layer_start_prefetch(layer_idx + 1, c_pred_buf, top_m, max_cap)
                else:
                    mx.eval(inds)
                    eids = inds.tolist()[0][0]
                    n_miss = C_LIB.c_layer_step_decode10_async(layer_idx, *eids)
                    if n_miss > 0:
                        mx.async_eval(scores, shared_y)
                    y = cmod._slot_glu(mixed_mlp, cmod._slot_inds_buf)
                    branch_mlp = (y * scores[..., None]).sum(axis=-2) + shared_y
                    if n_miss > 0:
                        C_LIB.c_layer_wait_prefetch(layer_idx)
                    mx.async_eval(branch_mlp)

                if defer_write:
                    return h_pre_moe, branch_mlp, inj_mlp
                return _hc_write(h_pre_moe, branch_mlp, inj_mlp)

            return hooked_layer_call

        hooked_cls = type(
            f"OffloadQwen4ExpDecoderLayer_{il}",
            (type(layer),),
            {"__call__": make_layer_call(il)},
        )
        layer.__class__ = hooked_cls


class OffloadEngine:
    """High-level Qwen3.8-Flash-Next MLX expert-offload inference engine."""

    def __init__(
        self,
        model_path: Path = MODEL_PATH,
        budget_gb: float | None = None,
        budget_pct: float | None = 20.0,
        alloc_mode: str = "even",
        prefetch: bool = True,
        prefetch_top_m: int = PREFETCH_TOP_M,
        prefetch_cap: int = PREFETCH_MAX_CAP,
    ):
        self.model_path = model_path
        self.alloc_mode = alloc_mode
        self.prefetch = prefetch
        self.prefetch_top_m = prefetch_top_m
        self.prefetch_cap = prefetch_cap
        self.capacities = compute_layer_capacities(
            budget_gb=budget_gb, budget_pct=budget_pct, alloc_mode=alloc_mode
        )
        self.total_cached_slots = sum(self.capacities.values())
        self.cache_budget_bytes = self.total_cached_slots * EXPERT_BYTES
        self.layer0_pinned_bytes = LAYER_EXPERT_BYTES
        self._decode_totals_buf = (ctypes.c_uint64 * 6)()

        t0 = time.perf_counter()
        self.ple_mode = configure_qwen4_exp_runtime(model_path, mode="mmap", mtp_enabled=False)
        config = load_config(model_path)
        model_class, _ = get_model_and_args(config=config, model_path=model_path)
        model_config = model_class.ModelConfig.from_dict(config)
        model_config = update_module_configs(
            model_config, model_class, config, ["text", "vision", "perceiver", "projector", "audio"]
        )
        model_config = apply_generation_config_defaults(model_config, config)

        self.model = model_class.Model(model_config)
        self.model.vision_tower = None

        self.cached_modules: dict[int, CachedSwitchGLU] = {}
        for il in range(1, NUM_LAYERS):
            cmod = CachedSwitchGLU(
                model_path / f"experts-{il:02d}.safetensors",
                il,
                capacity=self.capacities[il],
            )
            self.model.language_model.model.layers[il].mlp.switch_mlp = cmod
            self.cached_modules[il] = cmod

        weights: dict[str, mx.array] = {}
        for bf in sorted(model_path.glob("base-*.safetensors")):
            for k, v in mx.load(str(bf)).items():
                if not k.startswith("vision_tower."):
                    weights[k] = v

        # Layer 0 routed experts are pinned fully resident (outside the cache budget)
        weights.update(mx.load(str(model_path / "experts-00.safetensors")))

        weights = self.model.sanitize(weights)
        quantization = config["quantization"]

        def class_pred(p, m):
            per_mod = _quantization_for_module_path(quantization, p, self.model)
            if per_mod is not None:
                return per_mod
            if not hasattr(m, "to_quantized"):
                return False
            if hasattr(m, "weight") and m.weight.size % 64 != 0:
                return False
            return f"{p}.scales" in weights

        nn.quantize(
            self.model,
            group_size=quantization["group_size"],
            bits=quantization["bits"],
            mode=quantization.get("mode", "affine"),
            class_predicate=class_pred,
        )
        self.model.load_weights(list(weights.items()), strict=True)
        mx.eval(self.model.parameters())
        del weights
        gc.collect()
        mx.clear_cache()

        install_offload_hooks(
            self.model,
            prefetch_enabled=prefetch,
            prefetch_top_m=prefetch_top_m,
            prefetch_cap=prefetch_cap,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(str(model_path))
        self.load_seconds = time.perf_counter() - t0

    def set_prefetch(
        self,
        enabled: bool,
        top_m: int | None = None,
        cap: int | None = None,
    ) -> None:
        self.prefetch = enabled
        self.model._prefetch_enabled = enabled
        if top_m is not None:
            self.prefetch_top_m = top_m
            self.model._prefetch_top_m = top_m
        if cap is not None:
            self.prefetch_cap = cap
            self.model._prefetch_cap = cap

    def reset_caches(self) -> None:
        for cmod in self.cached_modules.values():
            cmod.reset_cache()

    def reset_stats(self) -> None:
        for cmod in self.cached_modules.values():
            cmod.reset_stats()

    def _snapshot_decode_totals(self) -> tuple[int, int, int, int, int, int]:
        C_LIB.c_get_decode_totals(NUM_LAYERS, self._decode_totals_buf)
        b = self._decode_totals_buf
        return int(b[0]), int(b[1]), int(b[2]), int(b[3]), int(b[4]), int(b[5])

    def generate_greedy(
        self,
        user_content: str,
        max_new_tokens: int = 128,
        reset_cache_before: bool = False,
        reset_stats_before: bool = False,
    ) -> dict[str, Any]:
        if reset_cache_before:
            self.reset_caches()
        if reset_stats_before:
            self.reset_stats()

        formatted = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": user_content}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        prompt_ids = self.tokenizer.encode(formatted, add_special_tokens=False)
        n_prefill = len(prompt_ids)

        lm = self.model.language_model
        cache = lm.make_cache()

        # 1. Prefill
        self.model._ple_cpu_active = False
        inp = mx.array([prompt_ids], dtype=mx.int32)
        t_pre0 = time.perf_counter()
        out = lm(inp, cache=cache)
        next_tok_arr = mx.argmax(out.logits[:, -1:, :], axis=-1)
        mx.eval(next_tok_arr)
        prefill_sec = time.perf_counter() - t_pre0
        mx.clear_cache()

        # Initialize CPU history for Layer 1 PLE n-gram hashing during decode
        eos_id = getattr(self.model, "_ple_eos_id", 248044)
        p_tail = [eos_id, eos_id] + prompt_ids
        self.model._ple_cpu_history = [p_tail[-2], p_tail[-1], eos_id]
        self.model._ple_cpu_active = True
        ple_hist = self.model._ple_cpu_history

        # Snapshot decode counters before decode starts
        s_start = self._snapshot_decode_totals()
        s_mid = s_start
        mid_tok = min(32, max_new_tokens // 2) if max_new_tokens >= 16 else max_new_tokens // 2

        # 2. Greedy Decode
        generated_ids: list[int] = []
        step_latencies_ms: list[float] = []
        cur_tok = next_tok_arr
        t_dec0 = time.perf_counter()
        for d_idx in range(max_new_tokens):
            tok_id = int(cur_tok.item())
            generated_ids.append(tok_id)
            ple_hist[2] = tok_id

            t_step0 = time.perf_counter()
            out = lm(cur_tok, cache=cache)
            next_tok_arr = mx.argmax(out.logits[:, -1:, :], axis=-1)
            mx.eval(next_tok_arr)
            step_sec = time.perf_counter() - t_step0
            step_latencies_ms.append(round(step_sec * 1000.0, 3))
            cur_tok = next_tok_arr

            ple_hist[0] = ple_hist[1]
            ple_hist[1] = tok_id

            if (d_idx + 1) == mid_tok:
                s_mid = self._snapshot_decode_totals()
            if (d_idx + 1) % 32 == 0:
                mx.clear_cache()

        decode_sec = time.perf_counter() - t_dec0
        self.model._ple_cpu_active = False
        s_end = self._snapshot_decode_totals()
        gen_text = self.tokenizer.decode(generated_ids)

        del cache
        gc.collect()
        mx.clear_cache()

        tot_req = max(1, s_end[3] - s_start[3])
        tot_lru = s_end[0] - s_start[0]
        tot_pref = s_end[1] - s_start[1]
        tot_dem_b = s_end[4] - s_start[4]
        tot_pref_b = s_end[5] - s_start[5]

        w1_req = max(1, s_mid[3] - s_start[3])
        w1_lru = s_mid[0] - s_start[0]
        w1_pref = s_mid[1] - s_start[1]

        w2_req = max(1, s_end[3] - s_mid[3])
        w2_lru = s_end[0] - s_mid[0]
        w2_pref = s_end[1] - s_mid[1]

        w1_sec = sum(step_latencies_ms[:mid_tok]) / 1000.0 if mid_tok > 0 else 0.0
        w2_sec = sum(step_latencies_ms[mid_tok:]) / 1000.0 if max_new_tokens > mid_tok else 0.0

        return {
            "prompt_tokens": n_prefill,
            "prefill_seconds": round(prefill_sec, 4),
            "prefill_tok_per_s": round(n_prefill / prefill_sec, 3),
            "decode_tokens": max_new_tokens,
            "decode_seconds": round(decode_sec, 4),
            "decode_tok_per_s": round(max_new_tokens / decode_sec, 3),
            "prompt_pure_lru_hit_rate": round(tot_lru / tot_req, 5),
            "prompt_effective_hit_rate": round((tot_lru + tot_pref) / tot_req, 5),
            "prompt_mb_per_token": round((tot_dem_b + tot_pref_b) / max(1, max_new_tokens) / 1e6, 3),
            "first_window_tokens": mid_tok,
            "first_window_pure_hit_rate": round(w1_lru / w1_req, 5),
            "first_window_effective_hit_rate": round((w1_lru + w1_pref) / w1_req, 5),
            "first_window_tok_per_s": round(mid_tok / max(1e-6, w1_sec), 3),
            "steady_window_tokens": max_new_tokens - mid_tok,
            "steady_window_pure_hit_rate": round(w2_lru / w2_req, 5),
            "steady_window_effective_hit_rate": round((w2_lru + w2_pref) / w2_req, 5),
            "steady_window_tok_per_s": round((max_new_tokens - mid_tok) / max(1e-6, w2_sec), 3),
            "generated_ids": generated_ids,
            "generated_text": gen_text,
            "step_latencies_ms": step_latencies_ms,
        }

    def collect_stats(self, total_decode_tokens: int, total_prefill_tokens: int) -> dict[str, Any]:
        per_layer = [self.cached_modules[il].get_layer_stats() for il in range(1, NUM_LAYERS)]
        tot_dec_req = sum(d["decode_requests"] for d in per_layer)
        tot_dec_lru_hits = sum(d["decode_lru_hits"] for d in per_layer)
        tot_dec_pref_hits = sum(d["decode_prefetch_hits"] for d in per_layer)
        tot_dec_misses = sum(d["decode_demand_misses"] for d in per_layer)
        tot_dec_demand_bytes = sum(d["decode_demand_bytes"] for d in per_layer)
        tot_dec_pref_bytes = sum(d["decode_prefetch_bytes"] for d in per_layer)
        tot_dec_bytes = tot_dec_demand_bytes + tot_dec_pref_bytes

        tot_pre_req = sum(d["prefill_requests"] for d in per_layer)
        tot_pre_hits = sum(d["prefill_lru_hits"] for d in per_layer)
        tot_pre_misses = sum(d["prefill_demand_misses"] for d in per_layer)
        tot_pre_bytes = sum(d["prefill_demand_bytes"] for d in per_layer)

        tot_pred16_hits = sum(c.pred_top16_hits for c in self.cached_modules.values())
        tot_pred16_actual = sum(c.pred_top16_total_actual for c in self.cached_modules.values())
        tot_pred16_pred = sum(c.pred_top16_total_pred for c in self.cached_modules.values())

        tot_miss_pref_hits = sum(c.miss_pref_hits for c in self.cached_modules.values())
        tot_miss_pref_actual = sum(
            c.miss_pref_total_actual_misses for c in self.cached_modules.values()
        )
        tot_miss_pref_fetched = sum(
            c.miss_pref_total_fetched for c in self.cached_modules.values()
        )

        tot_uncoal = sum(d["uncoalesced_tensor_reads"] for d in per_layer)
        tot_coal = sum(d["coalesced_preadv_calls"] for d in per_layer)

        dec_tok = max(1, total_decode_tokens)
        pre_tok = max(1, total_prefill_tokens)

        return {
            "budget_summary": {
                "alloc_mode": self.alloc_mode,
                "prefetch_enabled": self.prefetch,
                "prefetch_top_m": self.prefetch_top_m if self.prefetch else 0,
                "prefetch_cap": self.prefetch_cap if self.prefetch else 0,
                "layer0_pinned_experts": NUM_EXPERTS,
                "layer0_pinned_bytes": self.layer0_pinned_bytes,
                "layer0_pinned_mib": round(self.layer0_pinned_bytes / (1024**2), 2),
                "layer0_pinned_gb": round(self.layer0_pinned_bytes / 1e9, 4),
                "cached_layers_count": NUM_LAYERS - 1,
                "total_cached_slots": self.total_cached_slots,
                "mean_slots_per_cached_layer": round(self.total_cached_slots / (NUM_LAYERS - 1), 2),
                "cache_budget_bytes": self.cache_budget_bytes,
                "cache_budget_gib": round(self.cache_budget_bytes / (1024**3), 3),
                "cache_budget_gb": round(self.cache_budget_bytes / 1e9, 3),
                "total_routed_expert_resident_gb": round(
                    (self.cache_budget_bytes + self.layer0_pinned_bytes) / 1e9, 3
                ),
            },
            "decode_aggregate": {
                "total_decode_tokens": total_decode_tokens,
                "total_requests_layers_1_47": tot_dec_req,
                "lru_hits": tot_dec_lru_hits,
                "prefetch_hits": tot_dec_pref_hits,
                "demand_misses": tot_dec_misses,
                "pure_lru_hit_rate": round(tot_dec_lru_hits / max(1, tot_dec_req), 5),
                "effective_hit_rate": round(
                    (tot_dec_lru_hits + tot_dec_pref_hits) / max(1, tot_dec_req), 5
                ),
                "top16_router_recall": round(tot_pred16_hits / max(1, tot_pred16_actual), 5),
                "top16_router_precision": round(tot_pred16_hits / max(1, tot_pred16_pred), 5),
                "miss_prefetch_recall": round(
                    tot_miss_pref_hits / max(1, tot_miss_pref_actual), 5
                ),
                "miss_prefetch_precision": round(
                    tot_miss_pref_hits / max(1, tot_miss_pref_fetched), 5
                ),
                "demand_bytes_read": tot_dec_demand_bytes,
                "prefetch_bytes_read": tot_dec_pref_bytes,
                "total_bytes_read": tot_dec_bytes,
                "demand_mb_per_token": round(tot_dec_demand_bytes / dec_tok / 1e6, 3),
                "prefetch_mb_per_token": round(tot_dec_pref_bytes / dec_tok / 1e6, 3),
                "total_mb_per_token": round(tot_dec_bytes / dec_tok / 1e6, 3),
                "total_mib_per_token": round(tot_dec_bytes / dec_tok / (1024**2), 3),
                "uncoalesced_tensor_reads": tot_uncoal,
                "coalesced_preadv_calls": tot_coal,
                "coalesce_reduction_ratio": round(tot_uncoal / max(1, tot_coal), 4),
            },
            "prefill_aggregate": {
                "total_prefill_tokens": total_prefill_tokens,
                "total_requests_layers_1_47": tot_pre_req,
                "lru_hits": tot_pre_hits,
                "demand_misses": tot_pre_misses,
                "hit_rate": round(tot_pre_hits / max(1, tot_pre_req), 5),
                "demand_bytes_read": tot_pre_bytes,
                "mb_per_token": round(tot_pre_bytes / pre_tok / 1e6, 3),
            },
            "per_layer": per_layer,
        }

    def close(self) -> None:
        for cmod in self.cached_modules.values():
            cmod.close()
        self.model.close()


def run_correctness_check(
    budget_pct: float = 20.0,
    alloc_mode: str = "proportional",
    prefetch: bool = False,
    prefetch_top_m: int = PREFETCH_TOP_M,
    prefetch_cap: int = PREFETCH_MAX_CAP,
    decode_tokens: int = 128,
    out_path: Path | None = None,
    verify_both_prefetch: bool = False,
) -> dict[str, Any]:
    """Verify token-for-token exact match against the tracer.py reference run."""
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

    ref_records = [
        json.loads(line)
        for line in ROUTES_REF_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    ref_tokens_by_prompt: dict[int, list[int]] = {}
    for pid in (0, 1):
        ref_tokens_by_prompt[pid] = [
            int(r["token_id"])
            for r in ref_records
            if int(r["prompt_id"]) == pid and r["phase"] == "decode"
        ][:decode_tokens]

    trace_meta = json.loads(TRACE_META_PATH.read_text(encoding="utf-8"))
    ref_prompts = trace_meta["prompts"]

    print(
        f"[loader] Initializing OffloadEngine (budget_pct={budget_pct}%, "
        f"alloc_mode={alloc_mode}, prefetch={prefetch}, top_m={prefetch_top_m}, cap={prefetch_cap})...",
        flush=True,
    )
    engine = OffloadEngine(
        model_path=MODEL_PATH,
        budget_pct=budget_pct,
        alloc_mode=alloc_mode,
        prefetch=prefetch,
        prefetch_top_m=prefetch_top_m,
        prefetch_cap=prefetch_cap,
    )
    rss_kb, swap_mb = update_mem()
    print(
        f"[loader] Ready in {engine.load_seconds:.2f}s | "
        f"Layer 0 pinned={engine.layer0_pinned_bytes / (1024**2):.1f} MiB ({engine.layer0_pinned_bytes / 1e9:.3f} GB) | "
        f"L1..47 cache={engine.cache_budget_bytes / (1024**3):.2f} GiB ({engine.cache_budget_bytes / 1e9:.3f} GB, "
        f"{engine.total_cached_slots} slots) | "
        f"RSS={rss_kb / (1024**2):.2f} GiB | MLX active={mx.get_active_memory() / (1024**3):.2f} GiB",
        flush=True,
    )

    def _run_two_prompts(tag_label: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        engine.reset_caches()
        engine.reset_stats()
        results_per_prompt = []
        tot_pre_tok = 0
        tot_dec_tok = 0
        for pinfo in ref_prompts:
            pid = int(pinfo["prompt_id"])
            domain = pinfo["domain"]
            user_content = pinfo["user_content"]

            res = engine.generate_greedy(
                user_content=user_content,
                max_new_tokens=decode_tokens,
                reset_cache_before=False,
                reset_stats_before=False,
            )
            r_kb, s_mb = update_mem()
            tot_pre_tok += res["prompt_tokens"]
            tot_dec_tok += res["decode_tokens"]

            ref_ids = ref_tokens_by_prompt[pid]
            gen_ids = res["generated_ids"]
            exact_match = gen_ids == ref_ids
            first_mismatch = None
            for idx, (a, b) in enumerate(zip(gen_ids, ref_ids)):
                if a != b:
                    first_mismatch = {"token_idx": idx, "generated": a, "reference": b}
                    break

            matched_tokens = sum(1 for a, b in zip(gen_ids, ref_ids) if a == b)
            print(
                f"[loader:{tag_label}] Prompt {pid} ({domain}): prefill={res['prefill_tok_per_s']:.2f} tok/s, "
                f"decode={res['decode_tok_per_s']:.2f} tok/s | "
                f"match={matched_tokens}/{len(ref_ids)} exact={exact_match} | "
                f"RSS={r_kb / (1024**2):.2f} GiB | Swap delta={s_mb - start_swap_mb:+.1f} MiB",
                flush=True,
            )
            results_per_prompt.append(
                {
                    "prompt_id": pid,
                    "domain": domain,
                    "exact_token_match": exact_match,
                    "matched_tokens": matched_tokens,
                    "total_compared_tokens": len(ref_ids),
                    "first_mismatch": first_mismatch,
                    **res,
                }
            )
        st = engine.collect_stats(total_decode_tokens=tot_dec_tok, total_prefill_tokens=tot_pre_tok)
        return results_per_prompt, st

    results_per_prompt, stats = _run_two_prompts("prefetch_on" if prefetch else "prefetch_off")
    extra_runs: dict[str, Any] = {}
    if verify_both_prefetch:
        alt_prefetch = not prefetch
        engine.set_prefetch(alt_prefetch, top_m=prefetch_top_m, cap=prefetch_cap)
        alt_label = "prefetch_on" if alt_prefetch else "prefetch_off"
        alt_prompts, alt_stats = _run_two_prompts(alt_label)
        extra_runs = {
            ("prefetch_on" if prefetch else "prefetch_off"): {
                "all_prompts_exact_match": all(p["exact_token_match"] for p in results_per_prompt),
                "prompts": results_per_prompt,
                "stats": stats,
            },
            alt_label: {
                "all_prompts_exact_match": all(p["exact_token_match"] for p in alt_prompts),
                "prompts": alt_prompts,
                "stats": alt_stats,
            },
        }

    end_rss_kb, end_swap_mb = update_mem()

    all_exact = all(p["exact_token_match"] for p in results_per_prompt)
    if extra_runs:
        all_exact = all_exact and all(r["all_prompts_exact_match"] for r in extra_runs.values())

    report: dict[str, Any] = {
        "all_prompts_exact_match": all_exact,
        "load_seconds": round(engine.load_seconds, 4),
        "peak_rss_kb": peak_rss_kb,
        "peak_rss_gib": round(peak_rss_kb / (1024**2), 3),
        "peak_rss_gb": round((peak_rss_kb * 1024.0) / 1e9, 3),
        "mlx_active_gib": round(mx.get_active_memory() / (1024**3), 3),
        "mlx_peak_gib": round(mx.get_peak_memory() / (1024**3), 3),
        "start_swap_mb": round(start_swap_mb, 2),
        "end_swap_mb": round(end_swap_mb, 2),
        "peak_swap_mb": round(peak_swap_mb, 2),
        "peak_swap_delta_mb": round(peak_swap_mb - start_swap_mb, 2),
        "prompts": results_per_prompt,
        "stats": stats,
    }
    if extra_runs:
        report["variants"] = extra_runs

    if out_path is not None:
        out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[loader] Saved correctness report to {out_path}", flush=True)

    engine.close()
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Qwen3.8-Flash-Next MLX Expert-Offload Loader")
    parser.add_argument("--budget-gb", type=float, default=None, help="Total L1..47 cache budget in GB")
    parser.add_argument(
        "--budget-pct",
        type=float,
        default=None,
        help="Per-layer cache budget as %% of 512 experts (e.g. 20 or 30)",
    )
    parser.add_argument(
        "--alloc-mode",
        choices=["even", "proportional"],
        default="proportional",
        help="Slot distribution across layers 1..47",
    )
    parser.add_argument(
        "--prefetch",
        choices=["on", "off"],
        default="off",
        help="Enable cross-layer hc_pre_moe capped top-M async prefetch",
    )
    parser.add_argument(
        "--prefetch-top-m",
        type=int,
        default=PREFETCH_TOP_M,
        help="Number of candidate experts to rank in cross-layer prefetch (e.g. 10 or 12)",
    )
    parser.add_argument(
        "--prefetch-cap",
        type=int,
        default=PREFETCH_MAX_CAP,
        help="Max missing experts to prefetch per layer (e.g. 2 or 3)",
    )
    parser.add_argument(
        "--verify-both-prefetch",
        action="store_true",
        help="Verify both prefetch=off and prefetch=on in a single correctness run",
    )
    parser.add_argument("--decode-tokens", type=int, default=128, help="Greedy decode tokens per prompt")
    parser.add_argument(
        "--mode",
        choices=["verify", "single"],
        default="verify",
        help="Run Step 1 correctness verification or single prompt",
    )
    parser.add_argument("--prompt", type=str, default=None, help="Custom prompt for --mode single")
    parser.add_argument(
        "--out",
        type=str,
        default=str(LOADER_DIR / "correctness_report_v2.json"),
        help="Output JSON path",
    )
    args = parser.parse_args()

    budget_pct = args.budget_pct if (args.budget_pct is not None or args.budget_gb is None) else None
    if budget_pct is None and args.budget_gb is None:
        budget_pct = 20.0

    if args.mode == "verify":
        run_correctness_check(
            budget_pct=budget_pct if budget_pct is not None else 20.0,
            alloc_mode=args.alloc_mode,
            prefetch=(args.prefetch == "on"),
            prefetch_top_m=args.prefetch_top_m,
            prefetch_cap=args.prefetch_cap,
            decode_tokens=args.decode_tokens,
            out_path=Path(args.out),
            verify_both_prefetch=args.verify_both_prefetch,
        )
    else:
        engine = OffloadEngine(
            model_path=MODEL_PATH,
            budget_gb=args.budget_gb,
            budget_pct=budget_pct,
            alloc_mode=args.alloc_mode,
            prefetch=(args.prefetch == "on"),
            prefetch_top_m=args.prefetch_top_m,
            prefetch_cap=args.prefetch_cap,
        )
        prompt = args.prompt or "Write a Python function to compute the nth Fibonacci number iteratively."
        res = engine.generate_greedy(prompt, max_new_tokens=args.decode_tokens)
        stats = engine.collect_stats(res["decode_tokens"], res["prompt_tokens"])
        print(json.dumps({"result": res, "stats": stats["decode_aggregate"]}, indent=2, ensure_ascii=False))
        engine.close()


if __name__ == "__main__":
    main()


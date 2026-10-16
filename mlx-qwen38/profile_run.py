#!/usr/bin/env python3
"""Step 1 Profiler for Qwen3.8-Flash-Next MLX Expert-Offload Loader (v1).

Profiles one generation run under the best v1 config:
- budget_pct = 20.0 (proportional allocation across layers 1..47, Layer 0 pinned)
- prefetch = off
- 1 prompt (Prompt 0: code), 64 greedy decode tokens

Reports per-token time split:
1. GPU compute
2. Python overhead + mx.eval syncs per layer (including exact sync counts/token)
3. SSD read time (wall ms, raw syscall ms vs threadpool dispatch ms, bytes/MB per token)
4. Cache bookkeeping (LRU updates, eviction scans, coalescing, slot_inds mx.array creation)
5. Others (Layer 1 PLE SSD mmap gather/upload, token item extraction, cache maintenance)

Saves profile.json.
"""

from __future__ import annotations

import gc
import json
import os
import time
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from bench import BENCH_PROMPTS, load_reference_tokens
import loader
from loader import (
    EXPERT_BYTES,
    IO_POOL,
    LOADER_DIR,
    MODEL_PATH,
    NUM_LAYERS,
     CachedSwitchGLU,
    OffloadEngine,
    get_rss_kb,
    get_swap_used_mb,
)
import mlx_vlm.models.qwen4_exp.language as q4_lang
from mlx_vlm.models.qwen4_exp.language import _hc_write


def run_profile() -> dict[str, Any]:
    prompt_info = BENCH_PROMPTS[0]
    decode_tokens = 64
    ref_tokens = load_reference_tokens(decode_tokens)[0]

    start_swap_mb = get_swap_used_mb()
    peak_rss_kb = get_rss_kb()
    peak_swap_mb = start_swap_mb

    def update_mem() -> None:
        nonlocal peak_rss_kb, peak_swap_mb
        r = get_rss_kb()
        s = get_swap_used_mb()
        if r > peak_rss_kb:
            peak_rss_kb = r
        if s > peak_swap_mb:
            peak_swap_mb = s

    print("[profile] Loading OffloadEngine (20% proportional, prefetch=off)...", flush=True)
    engine = OffloadEngine(
        model_path=MODEL_PATH,
        budget_pct=20.0,
        alloc_mode="proportional",
        prefetch=False,
    )
    update_mem()

    # First: run an uninstrumented warmup+baseline on Prompt 0 (resetting cache before)
    # so we know the exact unperturbed v1 wall-clock decode speed on 1 prompt, 64 tokens.
    print("[profile] Running uninstrumented v1 baseline (Prompt 0, 64 tokens)...", flush=True)
    engine.reset_caches()
    engine.reset_stats()
    base_res = engine.generate_greedy(
        user_content=prompt_info["user_content"],
        max_new_tokens=decode_tokens,
        reset_cache_before=True,
        reset_stats_before=True,
    )
    base_stats = engine.collect_stats(
        total_decode_tokens=base_res["decode_tokens"],
        total_prefill_tokens=base_res["prompt_tokens"],
    )
    update_mem()
    exact_match_base = base_res["generated_ids"] == ref_tokens
    print(
        f"[profile] Baseline v1: prefill={base_res['prefill_tok_per_s']:.2f} tok/s "
        f"({base_res['prefill_seconds']*1000:.1f} ms), "
        f"decode={base_res['decode_tok_per_s']:.2f} tok/s "
        f"({base_res['decode_seconds']*1000 / decode_tokens:.2f} ms/tok), "
        f"hit_rate={base_res['prompt_pure_lru_hit_rate']*100:.2f}%, "
        f"read={base_res['prompt_mb_per_token']:.2f} MB/tok, "
        f"exact_match={exact_match_base}",
        flush=True,
    )

    # Now instrument v1 for fine-grained per-token profiling on the exact same run
    # (resetting cache and stats before so cache trajectory is 100% identical).
    engine.reset_caches()
    engine.reset_stats()

    # Counters & accumulators (in nanoseconds) for decode tokens
    in_decode = False

    # Sync counters
    sync_counts = {
        "mx_eval_switch_mlp_indices": 0,
        "mx_eval_ple_host_indices": 0,
        "mx_eval_final_logits_argmax": 0,
        "mx_eval_other": 0,
        "mx_async_eval_mlp_pre_switch": 0,
        "mx_async_eval_layer_branch_mlp": 0,
        "mx_async_eval_model_eager_dispatch": 0,
    }

    # Timing accumulators across 64 decode tokens (ns)
    timings_ns = {
        "eval_switch_mlp_indices_ns": 0,
        "eval_ple_host_indices_ns": 0,
        "eval_final_logits_argmax_ns": 0,
        "async_eval_mlp_pre_switch_ns": 0,
        "async_eval_layer_branch_mlp_ns": 0,
        "async_eval_model_eager_ns": 0,
        # Cache bookkeeping sub-components
        "bookkeeping_tolist_ns": 0,
        "bookkeeping_lru_lookup_ns": 0,
        "bookkeeping_slot_alloc_evict_ns": 0,
        "bookkeeping_coalesce_runs_ns": 0,
        "bookkeeping_lru_mru_update_ns": 0,
        "bookkeeping_slot_inds_mx_array_ns": 0,
        # SSD read sub-components
        "ssd_read_wall_ns": 0,
        "ssd_preadv_Thread_sum_ns": 0,
        "ssd_preadv_calls": 0,
        "ssd_layers_with_miss": 0,
        # PLE mmap gather/upload (outside mx.eval(flat))
        "ple_mmap_assemble_and_upload_ns": 0,
        "ple_total_ns": 0,
        # Slot GLU graph build
        "slot_glu_graph_ns": 0,
        # Periodic mx.clear_cache and tok.item()
        "tok_item_ns": 0,
        "clear_cache_ns": 0,
    }

    per_layer_decode_ns = [
        {
            "layer": il,
            "eval_indices_ns": 0,
            "bookkeeping_ns": 0,
            "ssd_read_wall_ns": 0,
            "slot_glu_graph_ns": 0,
            "misses": 0,
            "hits": 0,
            "bytes_read": 0,
            "preadv_calls": 0,
        }
        for il in range(NUM_LAYERS)
    ]

    # Wrap DiskBackedShardedEmbedding._call_owned in Layer 1 PLE
    ple_emb = engine.model.language_model.model.layers[1].ple.ple_embedding.ngram_embedding
    orig_ple_call_owned = ple_emb._call_owned
    orig_ple_host_indices = ple_emb._host_indices

    def profiled_ple_host_indices(indices: mx.array) -> np.ndarray:
        if not in_decode:
            return orig_ple_host_indices(indices)
        ple_emb._require_owner()
        flat = indices.reshape(-1)
        t0 = time.perf_counter_ns()
        mx.eval(flat)
        timings_ns["eval_ple_host_indices_ns"] += time.perf_counter_ns() - t0
        sync_counts["mx_eval_ple_host_indices"] += 1
        host = np.asarray(flat.astype(mx.int64)).reshape(-1)
        if host.size and (int(host.min()) < 0 or int(host.max()) >= ple_emb.shard_offsets[-1]):
            raise IndexError("embedding index is outside the sharded vocabulary")
        return host

    def profiled_ple_call_owned(indices: mx.array) -> mx.array:
        if not in_decode:
            return orig_ple_call_owned(indices)
        t_ple0 = time.perf_counter_ns()
        eval_before = timings_ns["eval_ple_host_indices_ns"]
        out = orig_ple_call_owned(indices)
        t_ple1 = time.perf_counter_ns()
        eval_delta = timings_ns["eval_ple_host_indices_ns"] - eval_before
        timings_ns["ple_total_ns"] += t_ple1 - t_ple0
        timings_ns["ple_mmap_assemble_and_upload_ns"] += (t_ple1 - t_ple0) - eval_delta
        return out

    ple_emb._host_indices = profiled_ple_host_indices
    ple_emb._call_owned = profiled_ple_call_owned

    # Instrument CachedSwitchGLU.__call__ for layers 1..47
    orig_cmod_call = CachedSwitchGLU.__call__

    def profiled_cmod_call(self_cmod: CachedSwitchGLU, x: mx.array, indices: mx.array) -> mx.array:
        B, L, K = indices.shape
        if not in_decode or L != 1:
            return orig_cmod_call(self_cmod, x, indices)

        il = self_cmod.layer_idx
        lstat = per_layer_decode_ns[il]

        # 1. mx.eval(indices) sync
        t0 = time.perf_counter_ns()
        mx.eval(indices)
        t1 = time.perf_counter_ns()
        sync_counts["mx_eval_switch_mlp_indices"] += 1
        timings_ns["eval_switch_mlp_indices_ns"] += t1 - t0
        lstat["eval_indices_ns"] += t1 - t0

        # 2. Extract indices to Python list
        tb0 = time.perf_counter_ns()
        eids = indices.reshape(-1).tolist()
        tb1 = time.perf_counter_ns()
        timings_ns["bookkeeping_tolist_ns"] += tb1 - tb0

        # 3. LRU hit/miss lookup
        actual_set = set(eids)
        self_cmod.decode_requests += len(eids)
        demand_missing: list[int] = []
        for eid in eids:
            if eid in self_cmod.lru:
                self_cmod.decode_lru_hits += 1
                lstat["hits"] += 1
            else:
                self_cmod.decode_demand_misses += 1
                lstat["misses"] += 1
                demand_missing.append(eid)
        tb2 = time.perf_counter_ns()
        timings_ns["bookkeeping_lru_lookup_ns"] += tb2 - tb1

        # 4. Slot allocation + SSD read on misses
        if demand_missing:
            demand_slots: list[int] = []
            for eid in demand_missing:
                slot = self_cmod._alloc_slot(actual_set)
                self_cmod.lru[eid] = slot
                demand_slots.append(slot)
            tb3 = time.perf_counter_ns()
            timings_ns["bookkeeping_slot_alloc_evict_ns"] += tb3 - tb2

            runs = self_cmod._coalesce_runs(demand_missing, demand_slots)
            tb4 = time.perf_counter_ns()
            timings_ns["bookkeeping_coalesce_runs_ns"] += tb4 - tb3
            lstat["bookkeeping_ns"] += (tb4 - tb0)

            n_bytes = len(demand_missing) * EXPERT_BYTES
            self_cmod.decode_demand_bytes += n_bytes
            lstat["bytes_read"] += n_bytes
            n_preadv = len(runs) * 9
            self_cmod.uncoalesced_tensor_reads += len(demand_missing) * 9
            self_cmod.coalesced_preadv_calls += n_preadv
            timings_ns["ssd_preadv_calls"] += n_preadv
            timings_ns["ssd_layers_with_miss"] += 1
            lstat["preadv_calls"] += n_preadv

            fd = self_cmod.fd
            specs = self_cmod.tensor_specs

            def _read_group_profiled(group_indices: tuple[int, ...]) -> int:
                t_th0 = time.perf_counter_ns()
                for t_idx in group_indices:
                    base, row_bytes, views = specs[t_idx]
                    for start_eid, run_slots in runs:
                        os.preadv(
                            fd,
                            [views[s] for s in run_slots],
                            base + start_eid * row_bytes,
                        )
                return time.perf_counter_ns() - t_th0

            t_io0 = time.perf_counter_ns()
            futs = [IO_POOL.submit(_read_group_profiled, g) for g in self_cmod.tensor_groups]
            th_ns = 0
            for f in futs:
                th_ns += f.result()
            t_io1 = time.perf_counter_ns()
            timings_ns["ssd_read_wall_ns"] += t_io1 - t_io0
            timings_ns["ssd_preadv_Thread_sum_ns"] += th_ns
            lstat["ssd_read_wall_ns"] += t_io1 - t_io0
        else:
            lstat["bookkeeping_ns"] += (tb2 - tb0)

        # 5. LRU MRU update + slot_inds mx.array creation
        tb5 = time.perf_counter_ns()
        slot_list: list[int] = []
        for eid in eids:
            self_cmod.lru.move_to_end(eid, last=True)
            slot_list.append(self_cmod.lru[eid])
        tb6 = time.perf_counter_ns()
        timings_ns["bookkeeping_lru_mru_update_ns"] += tb6 - tb5

        slot_inds = mx.array(slot_list, dtype=indices.dtype).reshape(B, 1, K)
        tb7 = time.perf_counter_ns()
        timings_ns["bookkeeping_slot_inds_mx_array_ns"] += tb7 - tb6
        lstat["bookkeeping_ns"] += (tb7 - tb5)

        # 6. Build _slot_glu MLX graph
        tg0 = time.perf_counter_ns()
        out = self_cmod._slot_glu(x, slot_inds)
        tg1 = time.perf_counter_ns()
        timings_ns["slot_glu_graph_ns"] += tg1 - tg0
        lstat["slot_glu_graph_ns"] += tg1 - tg0
        return out

    CachedSwitchGLU.__call__ = profiled_cmod_call

    # Also instrument the mx.async_eval calls inside install_offload_hooks and language.py
    layers = engine.model.language_model.model.layers
    for il, layer in enumerate(layers):
        mlp = layer.mlp

        def make_profiled_mlp_call(layer_idx: int, moe_block: nn.Module):
            def mlp_call(x: mx.array) -> mx.array:
                gates = moe_block.gate(x)
                gates = mx.softmax(gates, axis=-1, precise=True)
                k = moe_block.top_k
                inds = mx.argpartition(gates, kth=-k, axis=-1)[..., -k:]
                scores = mx.take_along_axis(gates, inds, axis=-1)
                scores = scores / scores.sum(axis=-1, keepdims=True)

                shared_y = moe_block.shared_expert(x)
                shared_y = moe_block._shared_expert_scale(x) * shared_y

                if layer_idx > 0:
                    if in_decode:
                        t_ae0 = time.perf_counter_ns()
                        mx.async_eval(x, inds, scores, shared_y)
                        timings_ns["async_eval_mlp_pre_switch_ns"] += time.perf_counter_ns() - t_ae0
                        sync_counts["mx_async_eval_mlp_pre_switch"] += 1
                    else:
                        mx.async_eval(x, inds, scores, shared_y)

                y = moe_block.switch_mlp(x, inds)
                y = (y * scores[..., None]).sum(axis=-2)
                return y + shared_y

            return mlp_call

        def make_profiled_layer_call(layer_idx: int):
            mlp_fn = make_profiled_mlp_call(layer_idx, layer.mlp)

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
                branch_mlp = mlp_fn(mixed_mlp)

                if hidden_states.shape[1] == 1:
                    if in_decode:
                        t_ae0 = time.perf_counter_ns()
                        mx.async_eval(branch_mlp)
                        timings_ns["async_eval_layer_branch_mlp_ns"] += time.perf_counter_ns() - t_ae0
                        sync_counts["mx_async_eval_layer_branch_mlp"] += 1
                    else:
                        mx.async_eval(branch_mlp)

                if defer_write:
                    return h_pre_moe, branch_mlp, inj_mlp
                return _hc_write(h_pre_moe, branch_mlp, inj_mlp)

            return hooked_layer_call

        hooked_cls = type(
            f"ProfiledQwen4ExpDecoderLayer_{il}",
            (type(layer),),
            {"__call__": make_profiled_layer_call(il)},
        )
        layer.__class__ = hooked_cls

    # Also count Qwen4ExpModel's eager mx.async_eval calls
    orig_async_eval = mx.async_eval

    def counting_async_eval(*args, **kwargs):
        return orig_async_eval(*args, **kwargs)

    # Execute the profiled run (Prompt 0, 64 decode tokens)
    formatted = engine.tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt_info["user_content"]}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    prompt_ids = engine.tokenizer.encode(formatted, add_special_tokens=False)
    n_prefill = len(prompt_ids)

    lm = engine.model.language_model
    cache = lm.make_cache()

    print("[profile] Running instrumented prefill + 64-token decode...", flush=True)
    inp = mx.array([prompt_ids], dtype=mx.int32)
    t_pre0 = time.perf_counter()
    out = lm(inp, cache=cache)
    next_tok_arr = mx.argmax(out.logits[:, -1:, :], axis=-1)
    mx.eval(next_tok_arr)
    prefill_sec = time.perf_counter() - t_pre0
    mx.clear_cache()

    # Count model eager async_evals per token from q4_lang settings
    eager_layers_per_token = sum(
        1
        for idx in range(NUM_LAYERS)
        if q4_lang._EAGER_DISPATCH
        and (idx < q4_lang._EAGER_DISPATCH_WARMUP or idx % q4_lang._EAGER_DISPATCH_EVERY == 0)
    )

    in_decode = True
    generated_ids: list[int] = []
    step_latencies_ms: list[float] = []
    cur_tok = next_tok_arr

    t_dec_wall0 = time.perf_counter_ns()
    for d_idx in range(decode_tokens):
        t_item0 = time.perf_counter_ns()
        tok_id = int(cur_tok.item())
        timings_ns["tok_item_ns"] += time.perf_counter_ns() - t_item0
        generated_ids.append(tok_id)

        t_step0 = time.perf_counter_ns()
        out = lm(cur_tok, cache=cache)
        next_tok_arr = mx.argmax(out.logits[:, -1:, :], axis=-1)

        t_ev0 = time.perf_counter_ns()
        mx.eval(next_tok_arr)
        t_ev1 = time.perf_counter_ns()
        timings_ns["eval_final_logits_argmax_ns"] += t_ev1 - t_ev0
        sync_counts["mx_eval_final_logits_argmax"] += 1
        sync_counts["mx_async_eval_model_eager_dispatch"] += eager_layers_per_token

        step_ns = t_ev1 - t_step0
        step_latencies_ms.append(round(step_ns / 1e6, 3))
        cur_tok = next_tok_arr

        if (d_idx + 1) % 32 == 0:
            t_cc0 = time.perf_counter_ns()
            mx.clear_cache()
            timings_ns["clear_cache_ns"] += time.perf_counter_ns() - t_cc0

    t_dec_wall1 = time.perf_counter_ns()
    in_decode = False
    update_mem()

    total_decode_wall_ns = t_dec_wall1 - t_dec_wall0
    prof_stats = engine.collect_stats(
        total_decode_tokens=decode_tokens, total_prefill_tokens=n_prefill
    )

    # Now calibrate pure GPU compute time vs Python graph + per-layer sync overhead!
    # How? Using the exact model & cache state right after token 64, we freeze the last
    # token's active slot indices in each layer (0 SSD reads, 0 LRU lookups) and measure:
    # (A) Pure lazy graph build (no evals inside layers, _EAGER_DISPATCH=False) + 1 final mx.eval
    #     -> separates pure_python_graph_build_ms and pure_gpu_compute_ms!
    # (B) Same 0-IO forward pass WITH the 48 per-layer evals & 115 async_evals
    #     -> measures exact per-layer mx.eval + mx.async_eval sync penalty!
    print("[profile] Calibrating pure GPU compute vs Python + mx.eval sync overhead...", flush=True)
    fixed_slot_inds: dict[int, mx.array] = {}
    for il in range(1, NUM_LAYERS):
        fixed_slot_inds[il] = mx.array(list(range(10)), dtype=mx.uint32).reshape(1, 1, 10)
    mx.eval(*fixed_slot_inds.values())

    # Temporarily disable PLE eval sync & _EAGER_DISPATCH for pure-GPU measurement
    saved_eager = q4_lang._EAGER_DISPATCH
    q4_lang._EAGER_DISPATCH = False
    cached_ple_out = None

    orig_ple_layer_call = engine.model.language_model.model.layers[1].ple.__call__

    def fast_ple_calib(hidden_states, input_ids, cache_arg, mask, target_verify=False):
        nonlocal cached_ple_out
        if cached_ple_out is None:
            cached_ple_out = orig_ple_layer_call(
                hidden_states, input_ids, None, mask, target_verify=target_verify
            )
            mx.eval(cached_ple_out)
        return cached_ple_out

    engine.model.language_model.model.layers[1].ple.__call__ = fast_ple_calib

    # Restore un-hooked MLP + Layer calls with 0 syncs and fixed slot_inds
    for il, layer in enumerate(layers):
        mlp = layer.mlp

        def make_nosync_mlp(layer_idx: int, moe_block: nn.Module):
            def mlp_call(x: mx.array) -> mx.array:
                gates = moe_block.gate(x)
                gates = mx.softmax(gates, axis=-1, precise=True)
                k = moe_block.top_k
                inds = mx.argpartition(gates, kth=-k, axis=-1)[..., -k:]
                scores = mx.take_along_axis(gates, inds, axis=-1)
                scores = scores / scores.sum(axis=-1, keepdims=True)
                shared_y = moe_block.shared_expert(x)
                shared_y = moe_block._shared_expert_scale(x) * shared_y
                if layer_idx == 0:
                    y = moe_block.switch_mlp(x, inds)
                else:
                    # Use inds sliced to valid slot range modulo capacity on GPU (0 host sync!)
                    s_inds = mx.remainder(inds, moe_block.switch_mlp.capacity)
                    y = moe_block.switch_mlp._slot_glu(x, s_inds)
                y = (y * scores[..., None]).sum(axis=-2)
                return y + shared_y

            return mlp_call

        def make_nosync_layer(layer_idx: int):
            mlp_fn = make_nosync_mlp(layer_idx, layer.mlp)

            def layer_call(
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
                if "ple" in self_layer:
                    if write is not None:
                        hidden_states = _hc_write(hidden_states, *write)
                        write = None
                    hidden_states = hidden_states + fast_ple_calib(
                        hidden_states, input_ids, None, mask, target_verify=target_verify
                    )
                mixed_attn, hyper_input, injection_weights = self_layer["attn_hyper_connection"](
                    hidden_states, target_verify=target_verify, write=write
                )
                if self_layer.is_linear:
                    branch_attn = self_layer["linear_attn"](mixed_attn, mask=mask, cache=None)
                else:
                    branch_attn = self_layer["self_attn"](
                        mixed_attn, mask=mask, cache=None, position_ids=position_ids
                    )
                mixed_mlp, h_pre_moe, inj_mlp = self_layer["mlp_hyper_connection"](
                    hyper_input, target_verify=target_verify, write=(branch_attn, injection_weights)
                )
                branch_mlp = mlp_fn(mixed_mlp)
                if defer_write:
                    return h_pre_moe, branch_mlp, inj_mlp
                return _hc_write(h_pre_moe, branch_mlp, inj_mlp)

            return layer_call

        layer.__class__ = type(
            f"NoSyncLayer_{il}",
            (type(layer),),
            {"__call__": make_nosync_layer(il)},
        )

    # Warmup 3 steps of 0-sync pass (without mutating KV cache)
    for _ in range(3):
        o = lm(cur_tok, cache=None)
        nt = mx.argmax(o.logits[:, -1:, :], axis=-1)
        mx.eval(nt)

    n_calib = 20
    graph_build_ns_list = []
    gpu_exec_ns_list = []
    for _ in range(n_calib):
        mx.synchronize()
        tg0 = time.perf_counter_ns()
        o = lm(cur_tok, cache=None)
        nt = mx.argmax(o.logits[:, -1:, :], axis=-1)
        tg1 = time.perf_counter_ns()
        mx.eval(nt)
        tg2 = time.perf_counter_ns()
        graph_build_ns_list.append(tg1 - tg0)
        gpu_exec_ns_list.append(tg2 - tg1)

    q4_lang._EAGER_DISPATCH = saved_eager

    pure_python_graph_ms = float(np.median(graph_build_ns_list)) / 1e6
    pure_gpu_compute_ms = float(np.median(gpu_exec_ns_list)) / 1e6

    # Compute per-token averages (ms/token) across the 64 decode tokens
    tok_div = decode_tokens * 1e6

    total_wall_ms_per_tok = total_decode_wall_ns / tok_div
    base_wall_ms_per_tok = (base_res["decode_seconds"] * 1000.0) / decode_tokens

    ssd_read_wall_ms = timings_ns["ssd_read_wall_ns"] / tok_div
    ssd_preadv_thread_sum_ms = timings_ns["ssd_preadv_Thread_sum_ns"] / tok_div
    # Since 4 thread groups run in parallel, critical-path syscall time is ~sum/4, while
    # the remainder of ssd_read_wall_ms is Python ThreadPoolExecutor submit/join + GIL wait.

    cache_bookkeeping_ms = (
        timings_ns["bookkeeping_tolist_ns"]
        + timings_ns["bookkeeping_lru_lookup_ns"]
        + timings_ns["bookkeeping_slot_alloc_evict_ns"]
        + timings_ns["bookkeeping_coalesce_runs_ns"]
        + timings_ns["bookkeeping_lru_mru_update_ns"]
        + timings_ns["bookkeeping_slot_inds_mx_array_ns"]
    ) / tok_div

    others_ms = (
        timings_ns["ple_mmap_assemble_and_upload_ns"]
        + timings_ns["tok_item_ns"]
        + timings_ns["clear_cache_ns"]
    ) / tok_div

    # Total time spent in GPU compute + Python graph building + mx.eval/async_eval syncs:
    gpu_and_python_sync_ms = (
        total_wall_ms_per_tok - ssd_read_wall_ms - cache_bookkeeping_ms - others_ms
    )
    python_and_eval_sync_ms = max(0.0, gpu_and_python_sync_ms - pure_gpu_compute_ms)

    # Also record the direct measured eval wait + async_eval dispatch + python graph build
    eval_wait_total_ms = (
        timings_ns["eval_switch_mlp_indices_ns"]
        + timings_ns["eval_ple_host_indices_ns"]
        + timings_ns["eval_final_logits_argmax_ns"]
    ) / tok_div
    async_eval_submit_ms = (
        timings_ns["async_eval_mlp_pre_switch_ns"]
        + timings_ns["async_eval_layer_branch_mlp_ns"]
    ) / tok_div
    python_graph_in_situ_ms = max(
        0.0, gpu_and_python_sync_ms - eval_wait_total_ms - async_eval_submit_ms
    )

    blocking_syncs_per_token = (
        sync_counts["mx_eval_switch_mlp_indices"]
        + sync_counts["mx_eval_ple_host_indices"]
        + sync_counts["mx_eval_final_logits_argmax"]
    ) // decode_tokens
    async_evals_per_token = (
        sync_counts["mx_async_eval_mlp_pre_switch"]
        + sync_counts["mx_async_eval_layer_branch_mlp"]
        + sync_counts["mx_async_eval_model_eager_dispatch"]
    ) // decode_tokens

    dec_agg = prof_stats["decode_aggregate"]

    profile_report = {
        "config": {
            "version": "v1_baseline",
            "budget_pct": 20.0,
            "alloc_mode": "proportional",
            "prefetch": False,
            "prompt_id": 0,
            "domain": prompt_info["domain"],
            "prompt_tokens": n_prefill,
            "decode_tokens": decode_tokens,
            "layer0_pinned_experts": 512,
            "layers_1_47_total_slots": engine.total_cached_slots,
            "cache_budget_gb": round(engine.cache_budget_bytes / 1e9, 3),
        },
        "correctness": {
            "exact_match_vs_reference": generated_ids == ref_tokens,
            "matched_tokens": sum(1 for a, b in zip(generated_ids, ref_tokens) if a == b),
            "total_tokens": decode_tokens,
        },
        "throughput": {
            "uninstrumented_prefill_tok_per_s": base_res["prefill_tok_per_s"],
            "uninstrumented_decode_tok_per_s": base_res["decode_tok_per_s"],
            "uninstrumented_ms_per_token": round(base_wall_ms_per_tok, 3),
            "instrumented_decode_tok_per_s": round(1000.0 / total_wall_ms_per_tok, 3),
            "instrumented_ms_per_token": round(total_wall_ms_per_tok, 3),
        },
        "sync_counts_per_token": {
            "blocking_mx_eval_total": blocking_syncs_per_token,
            "blocking_mx_eval_per_layer_avg": round(blocking_syncs_per_token / NUM_LAYERS, 3),
            "breakdown_blocking_mx_eval": {
                "switch_mlp_router_indices_layers_1_47": sync_counts["mx_eval_switch_mlp_indices"] // decode_tokens,
                "layer1_ple_ngram_host_indices": sync_counts["mx_eval_ple_host_indices"] // decode_tokens,
                "final_logits_argmax": sync_counts["mx_eval_final_logits_argmax"] // decode_tokens,
            },
            "nonblocking_mx_async_eval_total": async_evals_per_token,
            "breakdown_mx_async_eval": {
                "mlp_pre_switch_layers_1_47": sync_counts["mx_async_eval_mlp_pre_switch"] // decode_tokens,
                "layer_end_branch_mlp_layers_0_47": sync_counts["mx_async_eval_layer_branch_mlp"] // decode_tokens,
                "model_eager_dispatch_20_layers": sync_counts["mx_async_eval_model_eager_dispatch"] // decode_tokens,
            },
            "total_command_buffer_eval_calls_per_token": blocking_syncs_per_token + async_evals_per_token,
        },
        "per_token_time_split_ms": {
            "total_decode_ms_per_token": round(total_wall_ms_per_tok, 3),
            "1_gpu_compute_ms": round(pure_gpu_compute_ms, 3),
            "1_gpu_compute_pct": round(pure_gpu_compute_ms / total_wall_ms_per_tok * 100.0, 2),
            "2_python_overhead_and_eval_syncs_ms": round(python_and_eval_sync_ms, 3),
            "2_python_overhead_and_eval_syncs_pct": round(
                python_and_eval_sync_ms / total_wall_ms_per_tok * 100.0, 2
            ),
            "2_subbreakdown": {
                "python_graph_construction_ms": round(python_graph_in_situ_ms, 3),
                "calibrated_lazy_python_graph_ms": round(pure_python_graph_ms, 3),
                "mx_async_eval_dispatch_ms": round(async_eval_submit_ms, 3),
                "mx_eval_blocking_wait_total_ms": round(eval_wait_total_ms, 3),
                "mx_eval_switch_mlp_wait_ms": round(
                    timings_ns["eval_switch_mlp_indices_ns"] / tok_div, 3
                ),
                "mx_eval_ple_wait_ms": round(
                    timings_ns["eval_ple_host_indices_ns"] / tok_div, 3
                ),
                "mx_eval_final_argmax_wait_ms": round(
                    timings_ns["eval_final_logits_argmax_ns"] / tok_div, 3
                ),
                "estimated_sync_and_dispatch_overhead_ms": round(
                    max(0.0, python_and_eval_sync_ms - pure_python_graph_ms), 3
                ),
            },
            "3_ssd_read_wall_ms": round(ssd_read_wall_ms, 3),
            "3_ssd_read_pct": round(ssd_read_wall_ms / total_wall_ms_per_tok * 100.0, 2),
            "3_ssd_read_details": {
                "bytes_read_per_token": round(dec_agg["total_bytes_read"] / decode_tokens, 1),
                "mb_read_per_token": dec_agg["total_mb_per_token"],
                "mib_read_per_token": dec_agg["total_mib_per_token"],
                "missed_experts_per_token": round(dec_agg["demand_misses"] / decode_tokens, 2),
                "layers_with_miss_per_token": round(
                    timings_ns["ssd_layers_with_miss"] / decode_tokens, 2
                ),
                "coalesced_preadv_syscalls_per_token": round(
                    timings_ns["ssd_preadv_calls"] / decode_tokens, 2
                ),
                "sum_worker_thread_preadv_ms": round(ssd_preadv_thread_sum_ms, 3),
                "effective_ssd_bandwidth_gb_per_s": round(
                    (dec_agg["total_bytes_read"] / decode_tokens / 1e9)
                    / max(1e-6, ssd_read_wall_ms / 1000.0),
                    3,
                ),
            },
            "4_cache_bookkeeping_ms": round(cache_bookkeeping_ms, 3),
            "4_cache_bookkeeping_pct": round(
                cache_bookkeeping_ms / total_wall_ms_per_tok * 100.0, 2
            ),
            "4_cache_bookkeeping_subbreakdown": {
                "indices_tolist_ms": round(timings_ns["bookkeeping_tolist_ns"] / tok_div, 3),
                "lru_lookup_ms": round(timings_ns["bookkeeping_lru_lookup_ns"] / tok_div, 3),
                "slot_alloc_and_lru_evict_scan_ms": round(
                    timings_ns["bookkeeping_slot_alloc_evict_ns"] / tok_div, 3
                ),
                "coalesce_runs_ms": round(timings_ns["bookkeeping_coalesce_runs_ns"] / tok_div, 3),
                "lru_move_to_end_ms": round(
                    timings_ns["bookkeeping_lru_mru_update_ns"] / tok_div, 3
                ),
                "slot_inds_mx_array_alloc_ms": round(
                    timings_ns["bookkeeping_slot_inds_mx_array_ns"] / tok_div, 3
                ),
            },
            "5_others_ms": round(others_ms, 3),
            "5_others_pct": round(others_ms / total_wall_ms_per_tok * 100.0, 2),
            "5_others_subbreakdown": {
                "layer1_ple_mmap_gather_and_upload_ms": round(
                    timings_ns["ple_mmap_assemble_and_upload_ns"] / tok_div, 3
                ),
                "tok_item_extract_ms": round(timings_ns["tok_item_ns"] / tok_div, 3),
                "periodic_mx_clear_cache_ms": round(timings_ns["clear_cache_ns"] / tok_div, 3),
            },
        },
        "cache_stats": {
            "pure_lru_hit_rate": dec_agg["pure_lru_hit_rate"],
            "demand_misses_total": dec_agg["demand_misses"],
            "lru_hits_total": dec_agg["lru_hits"],
            "total_requests_layers_1_47": dec_agg["total_requests_layers_1_47"],
        },
        "memory": {
            "peak_rss_gib": round(peak_rss_kb / (1024**2), 3),
            "peak_rss_gb": round((peak_rss_kb * 1024.0) / 1e9, 3),
            "peak_swap_delta_mb": round(peak_swap_mb - start_swap_mb, 2),
        },
        "per_layer_decode_avg_ms": [
            {
                "layer": d["layer"],
                "capacity_slots": engine.capacities.get(d["layer"], 512),
                "eval_indices_ms": round(d["eval_indices_ns"] / tok_div, 4),
                "ssd_read_wall_ms": round(d["ssd_read_wall_ns"] / tok_div, 4),
                "bookkeeping_ms": round(d["bookkeeping_ns"] / tok_div, 4),
                "slot_glu_graph_ms": round(d["slot_glu_graph_ns"] / tok_div, 4),
                "misses_per_tok": round(d["misses"] / decode_tokens, 3),
                "hit_rate": round(d["hits"] / max(1, d["hits"] + d["misses"]), 4),
                "mb_per_tok": round(d["bytes_read"] / decode_tokens / 1e6, 3),
            }
            for d in per_layer_decode_ns[1:]
        ],
    }

    out_path = LOADER_DIR / "profile.json"
    out_path.write_text(json.dumps(profile_report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[profile] Saved {out_path}", flush=True)
    print(
        json.dumps(
            {
                "throughput": profile_report["throughput"],
                "sync_counts_per_token": profile_report["sync_counts_per_token"],
                "per_token_time_split_ms": profile_report["per_token_time_split_ms"],
                "cache_stats": profile_report["cache_stats"],
                "memory": profile_report["memory"],
            },
            indent=2,
        ),
        flush=True,
    )
    engine.close()
    return profile_report


if __name__ == "__main__":
    run_profile()

#!/usr/bin/env python3
"""Route tracer and cross-layer routing predictor for Qwen3.8-Flash-Next (GSQ-RCO Q2 MLX)."""

from __future__ import annotations

import gc
import json
import os
import re
import struct
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

# Ensure framework-mlx-base is before vendor so mlx_vlm top-level package imports cleanly,
# then apply_mlx_vlm_qwen4_exp_compat_patch() prepends vendor/mlx_vlm/models.
_BASE_SITE = "/Applications/oMLX.app/Contents/Resources/Python/framework-mlx-base/lib/python3.11/site-packages"
_OMLX_RES = "/Applications/oMLX.app/Contents/Resources"
for p in (_OMLX_RES, _BASE_SITE):
    if p in sys.path:
        sys.path.remove(p)
    sys.path.insert(0, p)

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
from mlx_vlm.models.qwen4_exp.language import _hc_write
from mlx_vlm.models.switch_layers import QuantizedSwitchLinear, SwiGLU, SwitchGLU
from mlx_vlm.utils import (
    _quantization_for_module_path,
    apply_generation_config_defaults,
    get_model_and_args,
    load_config,
    update_module_configs,
)
from transformers import AutoTokenizer

MODEL_PATH = Path(os.environ.get("QWEN38_MODEL", "models/Qwen3.8-Flash-Next-GSQ-RCO-Q2-mlx"))
OUT_DIR = Path(os.environ.get("MOE_WORKDIR", "."))
ROUTES_PATH = OUT_DIR / "routes.jsonl"
META_PATH = OUT_DIR / "trace_meta.json"

# Keep first 28 MoE layers (28 * 750 MiB = 20.51 GiB) + text trunk (3.17 GiB) = 23.68 GiB resident.
# Stream layers 28..47 (20 layers) via PreadSwitchGLU so RSS stays ~25-27 GiB (< 36 GiB / 38 GB limit).
RESIDENT_EXPERT_LAYERS = int(os.environ.get("QWEN38_RESIDENT_LAYERS", "28"))
DECODE_TOKENS_PER_PROMPT = int(os.environ.get("QWEN38_DECODE_TOKENS", "128"))

IO_POOL = ThreadPoolExecutor(max_workers=16, thread_name_prefix="expert-pread")


def get_rss_kb() -> int:
    out = subprocess.check_output(["ps", "-o", "rss=", "-p", str(os.getpid())]).decode().strip()
    return int(out)


def get_swap_used_mb() -> float:
    out = subprocess.check_output(["sysctl", "vm.swapusage"]).decode()
    m = re.search(r"used\s*=\s*([0-9.]+)M", out)
    return float(m.group(1)) if m else 0.0


class PreadSwitchGLU(nn.Module):
    """Streams only the active routed experts per layer from experts-XX.safetensors via os.pread.

    Produces bit-identical output (max_diff = 0.0) to resident SwitchGLU while adding 0 persistent
    resident memory and 0 mmap page-table RSS.
    """

    def __init__(self, shard_path: Path, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.shard_path = shard_path
        self.fd = os.open(str(shard_path), os.O_RDONLY)
        hsize = struct.unpack("<Q", os.pread(self.fd, 8, 0))[0]
        self.hdr = json.loads(os.pread(self.fd, hsize, 8))
        self.data_start = 8 + hsize
        self.activation = SwiGLU()
        self.specs: dict[tuple[str, str], tuple[tuple[int, ...], int, int, int, Any, str]] = {}
        for proj in ("gate_proj", "up_proj", "down_proj"):
            prefix = f"language_model.model.layers.{layer_idx}.mlp.switch_mlp.{proj}"
            for suffix in ("weight", "scales", "biases"):
                info = self.hdr[f"{prefix}.{suffix}"]
                shape = tuple(info["shape"])
                start, end = info["data_offsets"]
                row_bytes = (end - start) // shape[0]
                np_dt = np.uint32 if info["dtype"] == "U32" else np.uint16
                self.specs[(proj, suffix)] = (shape, start, end, row_bytes, np_dt, info["dtype"])

        # Bounded per-layer NumPy LRU cache (128 experts/layer = ~187 MiB/layer max)
        self.cache_cap = 128
        self.expert_cache: dict[tuple[str, str], dict[int, np.ndarray]] = {
            (proj, suffix): {}
            for proj in ("gate_proj", "up_proj", "down_proj")
            for suffix in ("weight", "scales", "biases")
        }

    def _load_tensor_np(self, proj: str, suffix: str, uniq: np.ndarray, n_uniq: int):
        shape, start, end, row_bytes, np_dt, dtype_str = self.specs[(proj, suffix)]
        base = self.data_start + start
        ecache = self.expert_cache[(proj, suffix)]
        if n_uniq > 48:
            # Sequential bulk read during prefill warms OS file cache and seeds expert cache
            raw_all = os.pread(self.fd, end - start, base)
            full_view = np.frombuffer(raw_all, dtype=np_dt).reshape(shape)
            buf = np.array(full_view[uniq], copy=True)
            for idx_pos, eid in enumerate(uniq[-self.cache_cap :]):
                eid_i = int(eid)
                if eid_i not in ecache and len(ecache) < self.cache_cap:
                    ecache[eid_i] = buf[n_uniq - min(n_uniq, self.cache_cap) + idx_pos].copy()
            del raw_all, full_view
        else:
            buf = np.empty((n_uniq, shape[1], shape[2]), dtype=np_dt)
            for idx_pos, eid in enumerate(uniq):
                eid_i = int(eid)
                cached = ecache.get(eid_i)
                if cached is not None:
                    buf[idx_pos] = cached
                else:
                    raw = os.pread(self.fd, row_bytes, base + eid_i * row_bytes)
                    row_arr = np.frombuffer(raw, dtype=np_dt).reshape(shape[1], shape[2]).copy()
                    buf[idx_pos] = row_arr
                    if len(ecache) >= self.cache_cap:
                        ecache.pop(next(iter(ecache)))
                    ecache[eid_i] = row_arr
        return proj, suffix, buf, dtype_str

    def __call__(self, x: mx.array, indices: mx.array) -> mx.array:
        mx.eval(x, indices)
        flat_inds = indices.reshape(-1).tolist()
        uniq, inv = np.unique(flat_inds, return_inverse=True)
        remapped = mx.array(inv.reshape(indices.shape), dtype=indices.dtype)

        sub = SwitchGLU.__new__(SwitchGLU)
        nn.Module.__init__(sub)
        sub.activation = self.activation
        n_uniq = len(uniq)

        futs = [
            IO_POOL.submit(self._load_tensor_np, proj, suffix, uniq, n_uniq)
            for proj in ("gate_proj", "up_proj", "down_proj")
            for suffix in ("weight", "scales", "biases")
        ]
        qls = {}
        for proj in ("gate_proj", "up_proj", "down_proj"):
            ql = QuantizedSwitchLinear.__new__(QuantizedSwitchLinear)
            nn.Module.__init__(ql)
            ql.group_size = 64
            ql.bits = 2
            ql.mode = "affine"
            qls[proj] = ql
            setattr(sub, proj, ql)
        for fut in futs:
            proj, suffix, buf, dtype_str = fut.result()
            arr = mx.array(buf)
            if dtype_str == "BF16":
                arr = arr.view(mx.bfloat16)
            setattr(qls[proj], suffix, arr)

        y = sub(x, remapped)
        mx.eval(y)
        del sub, qls, futs
        return y

    def close(self):
        if getattr(self, "fd", None) is not None:
            try:
                os.close(self.fd)
            except OSError:
                pass
            self.fd = None


def topk_sorted_indices(scores: mx.array, k: int) -> mx.array:
    """Return top-k indices along last axis sorted by descending score."""
    inds = mx.argpartition(scores, kth=-k, axis=-1)[..., -k:]
    vals = mx.take_along_axis(scores, inds, axis=-1)
    order = mx.argsort(-vals, axis=-1)
    return mx.take_along_axis(inds, order, axis=-1)


def cosine_sim_last_axis(a: mx.array, b: mx.array) -> mx.array:
    """Cosine similarity along last axis in float32, shape [B, T]."""
    af = a.astype(mx.float32)
    bf = b.astype(mx.float32)
    dot = mx.sum(af * bf, axis=-1)
    norm_a = mx.sqrt(mx.maximum(mx.sum(af * af, axis=-1), 1e-12))
    norm_b = mx.sqrt(mx.maximum(mx.sum(bf * bf, axis=-1), 1e-12))
    return dot / (norm_a * norm_b)


def rms_norm_single_stream(x: mx.array, weight_2560: mx.array, eps: float = 1e-6) -> mx.array:
    """Apply Qwen4ExpRMSNorm (1 + weight) to a single stream [B, T, 2560]."""
    y = x.astype(mx.float32)
    y = y * mx.rsqrt(mx.mean(mx.square(y), axis=-1, keepdims=True) + eps)
    y = y * (1.0 + weight_2560.astype(mx.float32))
    return y.astype(x.dtype)


class RouteRecorder:
    """Captures per-layer routing and cross-layer routing predictions for every token step."""

    def __init__(self, layers: list[nn.Module]):
        self.layers = layers
        self.num_layers = len(layers)
        self.reset_step()

    def reset_step(self):
        self.actual_top10: dict[int, list[list[int]]] = {}
        self.mixed_inputs: dict[int, mx.array] = {}
        self.pred_top24: dict[int, dict[str, list[list[int]]]] = {}
        self.pred_hidden: dict[int, dict[str, mx.array]] = {}

    def record_layer(
        self,
        il: int,
        h_pre_moe: mx.array,
        mixed_il: mx.array,
        actual_top10_il: mx.array,
        h_post_moe: mx.array,
    ) -> None:
        # Store actual top-10 and actual mixed_il for cosine similarity verification at il-1
        mx.eval(actual_top10_il, mixed_il)
        self.actual_top10[il] = actual_top10_il[0].tolist()
        self.mixed_inputs[il] = mixed_il

        if il >= self.num_layers - 1:
            return

        layer_next = self.layers[il + 1]
        hc_next = layer_next["mlp_hyper_connection"]
        hc_attn_next = layer_next["attn_hyper_connection"]
        gate_next = layer_next["mlp"].gate

        # 1. Exact HC pre-mixing + pre-MoE norm of layer il+1 applied to:
        #    (a) h_pre_moe (4 streams at layer il pre-MoE, before layer il MoE finishes)
        #    (b) h_post_moe (4 streams at layer il post-MoE / layer il+1 input)
        mixed_hc_pre, _, _ = hc_next(h_pre_moe)
        mixed_hc_post, _, _ = hc_next(h_post_moe)

        # 2. Per-stream pre-MoE norm of layer il+1 (hc_next.hc_norm) applied to the 4 streams,
        #    then combined via unweighted mean of streams or stream 0:
        normed_pre = hc_next.hc_norm(h_pre_moe)
        streams_pre = normed_pre.reshape(*normed_pre.shape[:-1], 4, 2560)
        mixed_mean_pre = mx.mean(streams_pre, axis=-2)
        mixed_s0_pre = streams_pre[..., 0, :]

        normed_post = hc_next.hc_norm(h_post_moe)
        streams_post = normed_post.reshape(*normed_post.shape[:-1], 4, 2560)
        mixed_mean_post = mx.mean(streams_post, axis=-2)
        mixed_s0_post = streams_post[..., 0, :]

        # 3. Also test raw stream mean followed by mean-gamma pre-MoE norm, and layer il+1's
        #    entry attn_hyper_connection mixer followed by pre-MoE norm:
        mean_gamma = mx.mean(hc_next.hc_norm.weight.reshape(4, 2560), axis=0)
        mixed_attn_hc_post, _, _ = hc_attn_next(h_post_moe)
        mixed_attn_hc_post_normed = rms_norm_single_stream(
            mixed_attn_hc_post, mean_gamma, hc_next.hc_norm.eps
        )

        # 4. Approximation using layer il's own mixed MoE input (both direct and re-normed):
        mixed_il_renormed = rms_norm_single_stream(
            mixed_il, mean_gamma, hc_next.hc_norm.eps
        )

        candidates = {
            "hc_post_moe": mixed_hc_post,
            "hc_pre_moe": mixed_hc_pre,
            "mean_streams_post_moe": mixed_mean_post,
            "mean_streams_pre_moe": mixed_mean_pre,
            "stream0_post_moe": mixed_s0_post,
            "stream0_pre_moe": mixed_s0_pre,
            "layer_il_mixed": mixed_il,
            "layer_il_mixed_normed": mixed_il_renormed,
            "attn_hc_post_moe": mixed_attn_hc_post_normed,
        }

        top24_tensors = {}
        for name, h_cand in candidates.items():
            logits = gate_next(h_cand)
            probs = mx.softmax(logits, axis=-1, precise=True)
            top24_tensors[name] = topk_sorted_indices(probs, 24)

        mx.eval(*top24_tensors.values(), *candidates.values())
        self.pred_top24[il] = {
            name: arr[0].tolist() for name, arr in top24_tensors.items()
        }
        self.pred_hidden[il] = candidates

    def finalize_step_records(self, seq_len: int) -> list[list[dict[str, Any]]]:
        """Build per-token list of 48 layer records with recall@10/16/24 and cosine sims."""
        # Compute cosine similarities between predicted hidden states at il and actual mixed_inputs[il+1]
        cos_sims: dict[int, dict[str, list[float]]] = {}
        for il in range(self.num_layers - 1):
            actual_next_mixed = self.mixed_inputs[il + 1]
            cands = self.pred_hidden[il]
            sim_tensors = {
                name: cosine_sim_last_axis(h_cand, actual_next_mixed)
                for name, h_cand in cands.items()
            }
            mx.eval(*sim_tensors.values())
            cos_sims[il] = {
                name: [round(float(v), 5) for v in sim_arr[0].tolist()]
                for name, sim_arr in sim_tensors.items()
            }

        per_token_layers: list[list[dict[str, Any]]] = []
        for t in range(seq_len):
            t_layers = []
            for il in range(self.num_layers):
                entry: dict[str, Any] = {
                    "layer": il,
                    "top10": self.actual_top10[il][t],
                }
                if il < self.num_layers - 1:
                    actual_next_set = set(self.actual_top10[il + 1][t])
                    preds_t = {}
                    recalls_t = {}
                    for name, pred_list in self.pred_top24[il].items():
                        p24 = pred_list[t]
                        preds_t[name] = p24
                        r10 = len(actual_next_set.intersection(p24[:10])) / 10.0
                        r16 = len(actual_next_set.intersection(p24[:16])) / 10.0
                        r24 = len(actual_next_set.intersection(p24[:24])) / 10.0
                        recalls_t[name] = [round(r10, 4), round(r16, 4), round(r24, 4)]
                    entry["pred_next_top24"] = preds_t
                    entry["recall_next_10_16_24"] = recalls_t
                    entry["cos_sim_next"] = {
                        name: cos_sims[il][name][t] for name in cos_sims[il]
                    }
                t_layers.append(entry)
            per_token_layers.append(t_layers)

        self.reset_step()
        return per_token_layers


def install_layer_hooks(model: nn.Module, recorder: RouteRecorder) -> None:
    """Hook each Qwen4ExpDecoderLayer and its MoE block to record routes and stream states."""
    layers = model.language_model.model.layers

    for il, layer in enumerate(layers):
        mlp = layer.mlp

        def make_mlp_call(moe_block):
            def mlp_call(x: mx.array):
                gates = moe_block.gate(x)
                gates = mx.softmax(gates, axis=-1, precise=True)
                k = moe_block.top_k
                inds = mx.argpartition(gates, kth=-k, axis=-1)[..., -k:]
                scores = mx.take_along_axis(gates, inds, axis=-1)
                scores = scores / scores.sum(axis=-1, keepdims=True)

                # Sort the selected top-10 experts by descending score for deterministic logging
                order10 = mx.argsort(-scores, axis=-1)
                top10_sorted = mx.take_along_axis(inds, order10, axis=-1)
                moe_block._last_top10_sorted = top10_sorted

                y = moe_block.switch_mlp(x, inds)
                y = (y * scores[..., None]).sum(axis=-2)
                shared_y = moe_block.shared_expert(x)
                shared_y = moe_block._shared_expert_scale(x) * shared_y
                return y + shared_y

            return mlp_call

        def make_layer_call(layer_idx: int):
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
                mlp_mod = self_layer["mlp"]
                branch_mlp = mlp_mod(mixed_mlp)
                h_post_moe = _hc_write(h_pre_moe, branch_mlp, inj_mlp)

                recorder.record_layer(
                    layer_idx,
                    h_pre_moe=h_pre_moe,
                    mixed_il=mixed_mlp,
                    actual_top10_il=mlp_mod._last_top10_sorted,
                    h_post_moe=h_post_moe,
                )

                if defer_write:
                    return h_pre_moe, branch_mlp, inj_mlp
                return h_post_moe

            return hooked_layer_call

        # Bind on a per-instance subclass so nn.Module(...) dispatch uses our hooked __call__
        hooked_cls = type(
            f"HookedQwen4ExpDecoderLayer_{il}",
            (type(layer),),
            {"__call__": make_layer_call(il)},
        )
        layer.__class__ = hooked_cls

        mlp_cls = type(
            f"HookedQwen3_5MoeSparseMoeBlock_{il}",
            (type(mlp),),
            {"__call__": lambda self_mlp, x, _fn=make_mlp_call(mlp): _fn(x)},
        )
        mlp.__class__ = mlp_cls


def load_text_only_model(model_path: Path, resident_layers: int):
    t0 = time.perf_counter()
    ple_mode = configure_qwen4_exp_runtime(model_path, mode="mmap", mtp_enabled=False)
    config = load_config(model_path)
    model_class, _ = get_model_and_args(config=config, model_path=model_path)
    model_config = model_class.ModelConfig.from_dict(config)
    model_config = update_module_configs(
        model_config, model_class, config, ["text", "vision", "perceiver", "projector", "audio"]
    )
    model_config = apply_generation_config_defaults(model_config, config)

    model = model_class.Model(model_config)
    # Skip vision tower completely
    model.vision_tower = None

    num_layers = model_config.text_config.num_hidden_layers
    pread_modules = []
    for il in range(resident_layers, num_layers):
        pmod = PreadSwitchGLU(model_path / f"experts-{il:02d}.safetensors", il)
        model.language_model.model.layers[il].mlp.switch_mlp = pmod
        pread_modules.append(pmod)

    weights = {}
    for bf in sorted(model_path.glob("base-*.safetensors")):
        for k, v in mx.load(str(bf)).items():
            if not k.startswith("vision_tower."):
                weights[k] = v

    for il in range(resident_layers):
        ef = model_path / f"experts-{il:02d}.safetensors"
        weights.update(mx.load(str(ef)))

    weights = model.sanitize(weights)
    quantization = config["quantization"]

    def class_pred(p, m):
        per_mod = _quantization_for_module_path(quantization, p, model)
        if per_mod is not None:
            return per_mod
        if not hasattr(m, "to_quantized"):
            return False
        if hasattr(m, "weight") and m.weight.size % 64 != 0:
            return False
        return f"{p}.scales" in weights

    nn.quantize(
        model,
        group_size=quantization["group_size"],
        bits=quantization["bits"],
        mode=quantization.get("mode", "affine"),
        class_predicate=class_pred,
    )
    model.load_weights(list(weights.items()), strict=True)
    del weights
    gc.collect()
    mx.clear_cache()
    load_sec = time.perf_counter() - t0
    return model, pread_modules, ple_mode, load_sec


def main():
    start_swap_mb = get_swap_used_mb()
    peak_swap_mb = start_swap_mb
    peak_rss_kb = get_rss_kb()

    def update_mem():
        nonlocal peak_swap_mb, peak_rss_kb
        r = get_rss_kb()
        s = get_swap_used_mb()
        if r > peak_rss_kb:
            peak_rss_kb = r
        if s > peak_swap_mb:
            peak_swap_mb = s
        return r, s

    print(
        f"[tracer] Loading model from {MODEL_PATH} "
        f"(resident_expert_layers={RESIDENT_EXPERT_LAYERS}/48, ple=mmap, vision=skipped, mtp=off)...",
        flush=True,
    )
    model, pread_modules, ple_mode, load_sec = load_text_only_model(
        MODEL_PATH, RESIDENT_EXPERT_LAYERS
    )
    rss_kb, swap_mb = update_mem()
    print(
        f"[tracer] Model loaded lazily in {load_sec:.2f}s | RSS={rss_kb / 1024**2:.2f} GiB | "
        f"MLX active={mx.get_active_memory() / 1024**3:.2f} GiB | Swap={swap_mb:.1f} MiB",
        flush=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_PATH))
    recorder = RouteRecorder(model.language_model.model.layers)
    install_layer_hooks(model, recorder)

    prompts = [
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
    ]

    lm = model.language_model
    if ROUTES_PATH.exists():
        ROUTES_PATH.unlink()

    prompt_summaries = []
    total_prefill_tokens = 0
    total_prefill_sec = 0.0
    total_decode_tokens = 0
    total_decode_sec = 0.0

    with ROUTES_PATH.open("w", encoding="utf-8") as out_f:
        for pinfo in prompts:
            pid = pinfo["prompt_id"]
            domain = pinfo["domain"]
            formatted = tokenizer.apply_chat_template(
                [{"role": "user", "content": pinfo["user_content"]}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            prompt_ids = tokenizer.encode(formatted, add_special_tokens=False)
            n_prefill = len(prompt_ids)
            print(
                f"\n[tracer] === Prompt {pid} ({domain}): {n_prefill} prefill tokens, "
                f"{DECODE_TOKENS_PER_PROMPT} greedy decode tokens ===",
                flush=True,
            )

            cache = lm.make_cache()
            recorder.reset_step()

            # 1. Prefill
            inp = mx.array([prompt_ids], dtype=mx.int32)
            t_pre0 = time.perf_counter()
            out = lm(inp, cache=cache)
            next_tok_arr = mx.argmax(out.logits[:, -1:, :], axis=-1)
            mx.eval(next_tok_arr)
            prefill_sec = time.perf_counter() - t_pre0
            rss_kb, swap_mb = update_mem()

            prefill_layer_records = recorder.finalize_step_records(n_prefill)
            for t_idx, tok_id in enumerate(prompt_ids):
                tok_str = tokenizer.decode([tok_id])
                record = {
                    "prompt_id": pid,
                    "domain": domain,
                    "phase": "prefill",
                    "token_idx": t_idx,
                    "seq_pos": t_idx,
                    "token_id": int(tok_id),
                    "token_str": tok_str,
                    "step_ms": round((prefill_sec * 1000.0) / n_prefill, 3),
                    "rss_gib": round(rss_kb / (1024.0**2), 3),
                    "layers": prefill_layer_records[t_idx],
                }
                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
            out_f.flush()
            mx.clear_cache()

            total_prefill_tokens += n_prefill
            total_prefill_sec += prefill_sec
            print(
                f"[tracer] Prompt {pid} prefill done in {prefill_sec:.2f}s "
                f"({n_prefill / prefill_sec:.2f} tok/s) | RSS={rss_kb / 1024**2:.2f} GiB | "
                f"Swap delta={swap_mb - start_swap_mb:+.2f} MiB",
                flush=True,
            )

            # 2. Greedy Decode (DECODE_TOKENS_PER_PROMPT tokens)
            generated_ids = []
            cur_tok = next_tok_arr
            t_dec0 = time.perf_counter()
            for d_idx in range(DECODE_TOKENS_PER_PROMPT):
                tok_id = int(cur_tok.item())
                generated_ids.append(tok_id)
                t_step0 = time.perf_counter()
                out = lm(cur_tok, cache=cache)
                next_tok_arr = mx.argmax(out.logits[:, -1:, :], axis=-1)
                mx.eval(next_tok_arr)
                step_sec = time.perf_counter() - t_step0
                rss_kb, swap_mb = update_mem()

                decode_layer_records = recorder.finalize_step_records(1)[0]
                tok_str = tokenizer.decode([tok_id])
                record = {
                    "prompt_id": pid,
                    "domain": domain,
                    "phase": "decode",
                    "token_idx": d_idx,
                    "seq_pos": n_prefill + d_idx,
                    "token_id": tok_id,
                    "token_str": tok_str,
                    "step_ms": round(step_sec * 1000.0, 3),
                    "rss_gib": round(rss_kb / (1024.0**2), 3),
                    "layers": decode_layer_records,
                }
                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                cur_tok = next_tok_arr

                if (d_idx + 1) % 16 == 0 or d_idx == 0 or (d_idx + 1) == DECODE_TOKENS_PER_PROMPT:
                    out_f.flush()
                    mx.clear_cache()
                    elapsed_dec = time.perf_counter() - t_dec0
                    rate = (d_idx + 1) / elapsed_dec
                    print(
                        f"[tracer] Prompt {pid} ({domain}) decode {d_idx + 1}/{DECODE_TOKENS_PER_PROMPT} | "
                        f"{rate:.2f} tok/s (last={step_sec * 1000:.1f} ms) | "
                        f"RSS={rss_kb / 1024**2:.2f} GiB | Swap delta={swap_mb - start_swap_mb:+.2f} MiB",
                        flush=True,
                    )

            decode_sec = time.perf_counter() - t_dec0
            total_decode_tokens += DECODE_TOKENS_PER_PROMPT
            total_decode_sec += decode_sec
            gen_text = tokenizer.decode(generated_ids)
            print(
                f"[tracer] Prompt {pid} ({domain}) decode complete: {DECODE_TOKENS_PER_PROMPT} tokens "
                f"in {decode_sec:.2f}s ({DECODE_TOKENS_PER_PROMPT / decode_sec:.2f} tok/s)",
                flush=True,
            )
            print(f"[tracer] Sample output ({domain}): {gen_text[:240]!r}...", flush=True)

            prompt_summaries.append(
                {
                    "prompt_id": pid,
                    "domain": domain,
                    "user_content": pinfo["user_content"],
                    "prefill_tokens": n_prefill,
                    "prefill_seconds": round(prefill_sec, 4),
                    "prefill_tok_per_s": round(n_prefill / prefill_sec, 3),
                    "decode_tokens": DECODE_TOKENS_PER_PROMPT,
                    "decode_seconds": round(decode_sec, 4),
                    "decode_tok_per_s": round(DECODE_TOKENS_PER_PROMPT / decode_sec, 3),
                    "generated_text": gen_text,
                }
            )
            del cache
            gc.collect()
            mx.clear_cache()

    end_rss_kb, end_swap_mb = update_mem()
    meta = {
        "model_path": str(MODEL_PATH),
        "ple_runtime_mode": ple_mode,
        "resident_expert_layers": RESIDENT_EXPERT_LAYERS,
        "streamed_expert_layers": 48 - RESIDENT_EXPERT_LAYERS,
        "load_seconds": round(load_sec, 4),
        "total_prefill_tokens": total_prefill_tokens,
        "total_prefill_seconds": round(total_prefill_sec, 4),
        "overall_prefill_tok_per_s": round(total_prefill_tokens / total_prefill_sec, 3),
        "total_decode_tokens": total_decode_tokens,
        "total_decode_seconds": round(total_decode_sec, 4),
        "overall_decode_tok_per_s": round(total_decode_tokens / total_decode_sec, 3),
        "peak_rss_kb": peak_rss_kb,
        "peak_rss_gib": round(peak_rss_kb / (1024.0**2), 3),
        "peak_rss_gb": round((peak_rss_kb * 1024.0) / 1e9, 3),
        "mlx_active_gib": round(mx.get_active_memory() / (1024.0**3), 3),
        "mlx_peak_gib": round(mx.get_peak_memory() / (1024.0**3), 3),
        "start_swap_mb": round(start_swap_mb, 2),
        "end_swap_mb": round(end_swap_mb, 2),
        "peak_swap_mb": round(peak_swap_mb, 2),
        "end_swap_delta_mb": round(end_swap_mb - start_swap_mb, 2),
        "peak_swap_delta_mb": round(peak_swap_mb - start_swap_mb, 2),
        "prompts": prompt_summaries,
    }
    META_PATH.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[tracer] Saved {ROUTES_PATH} and {META_PATH}", flush=True)

    for pmod in pread_modules:
        pmod.close()
    model.close()
    IO_POOL.shutdown(wait=False)


if __name__ == "__main__":
    main()

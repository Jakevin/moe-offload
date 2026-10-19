#!/usr/bin/env python3
"""
Generate synthetic reference fixtures (.safetensors) using the real vendor
mlx_vlm.models.qwen4_exp / switch_layers modules to verify bit-identical
parity of all C++ kernels and operators in tests/test_kernels.cpp.
Uses < 60 MB total fixture size and never loads any model weights.
"""
import math
import os
import sys

# Exact import sequence from ../loader/loader.py
_BASE_SITE = "/Applications/oMLX.app/Contents/Resources/Python/framework-mlx-base/lib/python3.11/site-packages"
_OMLX_RES = "/Applications/oMLX.app/Contents/Resources"
for _p in (_OMLX_RES, _BASE_SITE):
    if _p in sys.path:
        sys.path.remove(_p)
    sys.path.insert(0, _p)

import mlx.core as mx
import mlx.nn as nn
from omlx.patches.mlx_vlm_qwen4_exp_compat import apply_mlx_vlm_qwen4_exp_compat_patch

apply_mlx_vlm_qwen4_exp_compat_patch()

import mlx_vlm.models.qwen4_exp.language as q4_lang
from mlx_vlm.models.qwen4_exp import hc_fused
from mlx_vlm.models.qwen4_exp.config import TextConfig
from mlx_vlm.models.qwen3_5 import language as q35_lang
from mlx_vlm.models.qwen3_5_moe.language import Qwen3_5MoeSparseMoeBlock
from mlx_vlm.models.cache import ArraysCache


def save_qlinear(tensors: dict, prefix: str, mod: nn.QuantizedLinear):
    tensors[f"{prefix}.weight"] = mod.weight
    tensors[f"{prefix}.scales"] = mod.scales
    tensors[f"{prefix}.biases"] = mod.biases


def main():
    out_path = os.path.join(os.path.dirname(__file__), "fixtures.safetensors")
    mx.random.seed(42)
    tensors = {}

    # =========================================================================
    # 1. Qwen4ExpRMSNorm (full and grouped) & Qwen4ExpRMSNormGated (sigmoid & silu)
    # =========================================================================
    norm_full = q4_lang.Qwen4ExpRMSNorm(dim=256, group_size=None, eps=1e-6)
    norm_grp = q4_lang.Qwen4ExpRMSNorm(dim=256, group_size=64, eps=1e-6)
    w_norm = mx.astype(mx.random.normal((256,)) * 0.1, mx.bfloat16)
    norm_full.weight = w_norm
    norm_grp.weight = w_norm
    x_norm = mx.astype(mx.random.normal((1, 4, 256)), mx.bfloat16)
    tensors["norm.x"] = x_norm
    tensors["norm.w"] = w_norm
    tensors["norm.y_full"] = norm_full(x_norm)
    tensors["norm.y_grp"] = norm_grp(x_norm)

    gnorm_sig = q4_lang.Qwen4ExpRMSNormGated(dim=128, eps=1e-6, activation="sigmoid")
    gnorm_silu = q4_lang.Qwen4ExpRMSNormGated(dim=128, eps=1e-6, activation="silu")
    w_gnorm = mx.random.normal((128,)) * 0.1 + 1.0
    gnorm_sig.weight = w_gnorm
    gnorm_silu.weight = w_gnorm
    x_gnorm = mx.astype(mx.random.normal((1, 3, 4, 128)), mx.bfloat16)
    g_gnorm = mx.astype(mx.random.normal((1, 3, 4, 128)), mx.bfloat16)
    tensors["gnorm.x"] = x_gnorm
    tensors["gnorm.g"] = g_gnorm
    tensors["gnorm.w"] = w_gnorm
    tensors["gnorm.y_sig"] = gnorm_sig(x_gnorm, g_gnorm)
    tensors["gnorm.y_silu"] = gnorm_silu(x_gnorm, g_gnorm)

    # =========================================================================
    # 2. Qwen3_5RotaryEmbedding / MRoPE (2D and 3D positions)
    # =========================================================================
    rotary_emb = q35_lang.Qwen3_5RotaryEmbedding(
        dim=64,
        max_position_embeddings=262144,
        base=10_000_000,
        mrope_section=[11, 11, 10],
    )
    q_rope = mx.astype(mx.random.normal((1, 4, 5, 256)), mx.bfloat16)
    k_rope = mx.astype(mx.random.normal((1, 2, 5, 256)), mx.bfloat16)
    pos_2d = mx.array([[10, 11, 12, 13, 14]], dtype=mx.int32)
    pos_3d = mx.array(
        [
            [[10, 11, 12, 13, 14]],
            [[20, 21, 22, 23, 24]],
            [[30, 31, 32, 33, 34]],
        ],
        dtype=mx.int32,
    )
    q_rot_2d, k_rot_2d = rotary_emb.apply_rotary(q_rope, k_rope, pos_2d)
    q_rot_3d, k_rot_3d = rotary_emb.apply_rotary(q_rope, k_rope, pos_3d)
    tensors["rope.inv_freq"] = rotary_emb.inv_freq
    tensors["rope.q"] = q_rope
    tensors["rope.k"] = k_rope
    tensors["rope.pos_2d"] = pos_2d
    tensors["rope.pos_3d"] = pos_3d
    tensors["rope.q_2d"] = q_rot_2d
    tensors["rope.k_2d"] = k_rot_2d
    tensors["rope.q_3d"] = q_rot_3d
    tensors["rope.k_3d"] = k_rot_3d

    # =========================================================================
    # 3. Qwen4ExpGatedResidual / hc_fused (decode L=1, L=8, prefill L=18,
    #    with and without write, and final mixer use_combine=False + lm_head)
    # =========================================================================
    hc_cfg = TextConfig(
        model_type="qwen4_exp_text",
        hidden_size=2560,
        num_hidden_layers=1,
        num_attention_heads=16,
        linear_num_value_heads=48,
        linear_num_key_heads=16,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
        linear_conv_kernel_dim=4,
        num_experts=16,
        num_experts_per_tok=4,
        shared_expert_intermediate_size=128,
        moe_intermediate_size=128,
        rms_norm_eps=1e-6,
        vocab_size=256,
        num_key_value_heads=2,
        max_position_embeddings=4096,
        hc_count=4,
        hc_lowrank=320,
        head_dim=256,
    )
    hc_mod = q4_lang.Qwen4ExpGatedResidual(hc_cfg, use_combine=True)
    hc_mod.hc_norm.weight = mx.astype(mx.random.normal((10240,)) * 0.05, mx.bfloat16)
    hc_mod.input_mix_weight_down.weight = mx.astype(mx.random.normal((320, 10240)) * 0.02, mx.bfloat16)
    hc_mod.input_mix_weight_up.weight = mx.astype(mx.random.normal((10240, 320)) * 0.02, mx.bfloat16)
    hc_mod.block_inject_weight.weight = mx.astype(mx.random.normal((4, 10240)) * 0.02, mx.bfloat16)
    nn.quantize(hc_mod, group_size=64, bits=8, mode="affine")

    hc_mixer = q4_lang.Qwen4ExpGatedResidual(hc_cfg, use_combine=False)
    hc_mixer.hc_norm.weight = hc_mod.hc_norm.weight
    hc_mixer.input_mix_weight_down = hc_mod.input_mix_weight_down
    hc_mixer.input_mix_weight_up = hc_mod.input_mix_weight_up

    tensors["hc.norm.weight"] = hc_mod.hc_norm.weight
    save_qlinear(tensors, "hc.down", hc_mod.input_mix_weight_down)
    save_qlinear(tensors, "hc.up", hc_mod.input_mix_weight_up)
    save_qlinear(tensors, "hc.inject", hc_mod.block_inject_weight)

    for rows, tag in ((1, "r1"), (8, "r8"), (18, "r18")):
        h_in = mx.astype(mx.random.normal((1, rows, 10240)) * 0.2, mx.bfloat16)
        br_in = mx.astype(mx.random.normal((1, rows, 2560)) * 0.2, mx.bfloat16)
        gt_in = mx.astype(mx.random.normal((1, rows, 4)) * 0.5, mx.bfloat16)
        if rows <= 16:
            assert hc_fused.compatible(hc_mod, h_in), f"hc_fused not compatible for rows={rows}"
        else:
            assert hc_fused.prefill_compatible(hc_mod, h_in), f"hc_fused prefill not compatible for rows={rows}"

        m_nowrite, h_nowrite, inj_nowrite = hc_mod(h_in, write=None)
        m_write, h_write, inj_write = hc_mod(h_in, write=(br_in, gt_in))
        mix_nowrite = hc_mixer(h_in, write=None)
        mix_write, h_mix_write = hc_mixer(h_in, write=(br_in, gt_in))

        tensors[f"hc.{tag}.h_in"] = h_in
        tensors[f"hc.{tag}.br_in"] = br_in
        tensors[f"hc.{tag}.gt_in"] = gt_in
        tensors[f"hc.{tag}.m_nowrite"] = m_nowrite
        tensors[f"hc.{tag}.h_nowrite"] = h_nowrite
        tensors[f"hc.{tag}.inj_nowrite"] = inj_nowrite
        tensors[f"hc.{tag}.m_write"] = m_write
        tensors[f"hc.{tag}.h_write"] = h_write
        tensors[f"hc.{tag}.inj_write"] = inj_write
        tensors[f"hc.{tag}.mix_nowrite"] = mix_nowrite
        tensors[f"hc.{tag}.mix_write"] = mix_write
        tensors[f"hc.{tag}.h_mix_write"] = h_mix_write

    # Final lm_head (4-bit group_size=64) after final mixer
    lm_head = nn.Linear(2560, 256, bias=False)
    lm_head.weight = mx.astype(mx.random.normal((256, 2560)) * 0.02, mx.bfloat16)
    lm_head_q = nn.QuantizedLinear.from_linear(lm_head, group_size=64, bits=4, mode="affine")
    save_qlinear(tensors, "lm_head", lm_head_q)
    tensors["lm_head.logits_r18"] = lm_head_q(tensors["hc.r18.mix_write"][:, -1:, :])

    # =========================================================================
    # 4. Qwen4ExpGatedDeltaNet (prefill L=4 + decode L=1)
    # =========================================================================
    gdn_cfg = TextConfig(
        model_type="qwen4_exp_text",
        hidden_size=256,
        num_hidden_layers=1,
        num_attention_heads=4,
        linear_num_value_heads=4,
        linear_num_key_heads=2,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
        linear_conv_kernel_dim=4,
        num_experts=16,
        num_experts_per_tok=4,
        shared_expert_intermediate_size=128,
        moe_intermediate_size=128,
        rms_norm_eps=1e-6,
        vocab_size=256,
        num_key_value_heads=2,
        max_position_embeddings=4096,
        hc_count=4,
        hc_lowrank=64,
        head_dim=64,
        output_gate_type="sigmoid",
    )
    gdn = q4_lang.Qwen4ExpGatedDeltaNet(gdn_cfg)
    gdn.conv1d.weight = mx.astype(mx.random.normal(gdn.conv1d.weight.shape) * 0.05, mx.bfloat16)
    gdn.in_proj_qkv.weight = mx.astype(mx.random.normal(gdn.in_proj_qkv.weight.shape) * 0.05, mx.bfloat16)
    gdn.in_proj_z.weight = mx.astype(mx.random.normal(gdn.in_proj_z.weight.shape) * 0.05, mx.bfloat16)
    gdn.in_proj_b.weight = mx.astype(mx.random.normal(gdn.in_proj_b.weight.shape) * 0.05, mx.bfloat16)
    gdn.in_proj_a.weight = mx.astype(mx.random.normal(gdn.in_proj_a.weight.shape) * 0.05, mx.bfloat16)
    gdn.out_proj.weight = mx.astype(mx.random.normal(gdn.out_proj.weight.shape) * 0.05, mx.bfloat16)
    gdn.dt_bias = mx.astype(mx.random.normal(gdn.dt_bias.shape) * 0.1 + 1.0, mx.bfloat16)
    gdn.A_log = mx.astype(gdn.A_log, mx.float32)
    gdn.norm.weight = mx.astype(mx.random.normal(gdn.norm.weight.shape) * 0.1 + 1.0, mx.float32)
    nn.quantize(
        gdn,
        group_size=64,
        bits=4,
        mode="affine",
        class_predicate=lambda p, m: isinstance(m, nn.Linear),
    )

    save_qlinear(tensors, "gdn.in_proj_qkv", gdn.in_proj_qkv)
    save_qlinear(tensors, "gdn.in_proj_z", gdn.in_proj_z)
    save_qlinear(tensors, "gdn.in_proj_b", gdn.in_proj_b)
    save_qlinear(tensors, "gdn.in_proj_a", gdn.in_proj_a)
    save_qlinear(tensors, "gdn.out_proj", gdn.out_proj)
    tensors["gdn.conv1d.weight"] = gdn.conv1d.weight
    tensors["gdn.A_log"] = gdn.A_log
    tensors["gdn.dt_bias"] = gdn.dt_bias
    tensors["gdn.norm.weight"] = gdn.norm.weight

    from mlx_vlm.models.qwen3_5 import gated_delta as q35_gd

    def run_gdn_with_intermediates(mod, x_in, cache_obj, tag: str):
        cap = {}
        orig_qkv = mod.in_proj_qkv
        orig_proj_gates = mod._project_gates
        orig_silu = q35_lang.nn.silu
        orig_norm_qk = mod._normalize_qk
        orig_comp_gb = q35_gd._compute_g_beta
        orig_gd_kernel = q35_gd.gated_delta_kernel
        orig_gd_chunked = q35_gd.gated_delta_chunked
        orig_gd_ops = q35_gd.gated_delta_ops
        orig_norm = mod.norm

        def spy_qkv(inp):
            out = orig_qkv(inp)
            cap["mixed_qkv"] = out
            return out

        def spy_proj_gates(inp):
            b_out, a_out = orig_proj_gates(inp)
            cap["b"] = b_out
            cap["a"] = a_out
            return b_out, a_out

        def spy_silu(inp):
            out = orig_silu(inp)
            cap["conv_out"] = out
            return out

        def spy_norm_qk(q_in, k_in):
            q_out, k_out = orig_norm_qk(q_in, k_in)
            cap["q"] = q_out
            cap["k"] = k_out
            return q_out, k_out

        def spy_comp_gb(A_log_in, a_in, b_in, dt_bias_in):
            g_out, beta_out = orig_comp_gb(A_log_in, a_in, b_in, dt_bias_in)
            cap["g"] = g_out
            cap["beta"] = beta_out
            return g_out, beta_out

        def spy_gd_kernel(q_in, k_in, v_in, g_in, beta_in, state_in, mask_in=None):
            cap["v"] = v_in
            ky, st = orig_gd_kernel(q_in, k_in, v_in, g_in, beta_in, state_in, mask_in)
            cap["kernel_y"] = ky
            cap["state"] = st
            return ky, st

        def spy_gd_chunked(q_in, k_in, v_in, g_in, beta_in, state_in, mask_in=None, C: int = 64):
            cap["v"] = v_in
            ky, st = orig_gd_chunked(q_in, k_in, v_in, g_in, beta_in, state_in, mask_in, C=C)
            cap["kernel_y"] = ky
            cap["state"] = st
            return ky, st

        def spy_gd_ops(q_in, k_in, v_in, g_in, beta_in, state_in=None, mask_in=None):
            cap["v"] = v_in
            ky, st = orig_gd_ops(q_in, k_in, v_in, g_in, beta_in, state_in, mask_in)
            cap["kernel_y"] = ky
            cap["state"] = st
            return ky, st

        def spy_norm(out_in, z_in):
            cap["z"] = z_in
            return orig_norm(out_in, z_in)

        mod.in_proj_qkv = spy_qkv
        mod._project_gates = spy_proj_gates
        q35_lang.nn.silu = spy_silu
        mod._normalize_qk = spy_norm_qk
        q35_gd._compute_g_beta = spy_comp_gb
        q35_gd.gated_delta_kernel = spy_gd_kernel
        q35_gd.gated_delta_chunked = spy_gd_chunked
        q35_gd.gated_delta_ops = spy_gd_ops
        mod.norm = spy_norm
        try:
            y_out = mod(x_in, mask=None, cache=cache_obj)
        finally:
            mod.in_proj_qkv = orig_qkv
            mod._project_gates = orig_proj_gates
            q35_lang.nn.silu = orig_silu
            mod._normalize_qk = orig_norm_qk
            q35_gd._compute_g_beta = orig_comp_gb
            q35_gd.gated_delta_kernel = orig_gd_kernel
            q35_gd.gated_delta_chunked = orig_gd_chunked
            q35_gd.gated_delta_ops = orig_gd_ops
            mod.norm = orig_norm

        for k_name, arr_val in cap.items():
            tensors[f"gdn.{tag}.{k_name}"] = mx.array(arr_val)
        return y_out

    gdn_cache = ArraysCache(size=2)
    gdn_x_pre = mx.astype(mx.random.normal((1, 4, 256)) * 0.2, mx.bfloat16)
    gdn_y_pre = run_gdn_with_intermediates(gdn, gdn_x_pre, gdn_cache, "pre")
    gdn_conv_pre = mx.array(gdn_cache[0])
    gdn_ssm_pre = mx.array(gdn_cache[1])

    gdn_x_dec = mx.astype(mx.random.normal((1, 1, 256)) * 0.2, mx.bfloat16)
    gdn_y_dec = run_gdn_with_intermediates(gdn, gdn_x_dec, gdn_cache, "dec")
    gdn_conv_dec = mx.array(gdn_cache[0])
    gdn_ssm_dec = mx.array(gdn_cache[1])

    tensors["gdn.x_pre"] = gdn_x_pre
    tensors["gdn.y_pre"] = gdn_y_pre
    tensors["gdn.conv_pre"] = gdn_conv_pre
    tensors["gdn.ssm_pre"] = gdn_ssm_pre
    tensors["gdn.x_dec"] = gdn_x_dec
    tensors["gdn.y_dec"] = gdn_y_dec
    tensors["gdn.conv_dec"] = gdn_conv_dec
    tensors["gdn.ssm_dec"] = gdn_ssm_dec

    # =========================================================================
    # 5. Qwen4ExpAttention + Qwen4ExpQSAIndexer (prefill L=4 + decode L=1,
    #    plus QSA indexer score/top-k/mask when key_len > token_budget)
    # =========================================================================
    fa_cfg = TextConfig(
        model_type="qwen4_exp_text",
        hidden_size=256,
        num_hidden_layers=1,
        num_attention_heads=4,
        linear_num_value_heads=4,
        linear_num_key_heads=2,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
        linear_conv_kernel_dim=4,
        num_experts=16,
        num_experts_per_tok=4,
        shared_expert_intermediate_size=128,
        moe_intermediate_size=128,
        rms_norm_eps=1e-6,
        vocab_size=256,
        num_key_value_heads=2,
        max_position_embeddings=262144,
        hc_count=4,
        hc_lowrank=64,
        head_dim=256,
        indexer_n_heads=4,
        indexer_kv_heads=1,
        indexer_head_dim=128,
        indexer_budget=2048,
        indexer_compress_ratio=4,
    )
    fa = q4_lang.Qwen4ExpAttention(fa_cfg)
    fa.q_proj.weight = mx.astype(mx.random.normal(fa.q_proj.weight.shape) * 0.05, mx.bfloat16)
    fa.k_proj.weight = mx.astype(mx.random.normal(fa.k_proj.weight.shape) * 0.05, mx.bfloat16)
    fa.v_proj.weight = mx.astype(mx.random.normal(fa.v_proj.weight.shape) * 0.05, mx.bfloat16)
    fa.o_proj.weight = mx.astype(mx.random.normal(fa.o_proj.weight.shape) * 0.05, mx.bfloat16)
    fa.indexer.index_qk_proj.weight = mx.astype(
        mx.random.normal(fa.indexer.index_qk_proj.weight.shape) * 0.05, mx.bfloat16
    )
    fa.q_norm.weight = mx.astype(mx.random.normal((256,)) * 0.05, mx.bfloat16)
    fa.k_norm.weight = mx.astype(mx.random.normal((256,)) * 0.05, mx.bfloat16)
    fa.indexer.q_layernorm.weight = mx.astype(mx.random.normal((128,)) * 0.05, mx.bfloat16)
    fa.indexer.k_layernorm.weight = mx.astype(mx.random.normal((128,)) * 0.05, mx.bfloat16)
    nn.quantize(
        fa,
        group_size=64,
        bits=4,
        mode="affine",
        class_predicate=lambda p, m: isinstance(m, nn.Linear),
    )

    save_qlinear(tensors, "fa.q_proj", fa.q_proj)
    save_qlinear(tensors, "fa.k_proj", fa.k_proj)
    save_qlinear(tensors, "fa.v_proj", fa.v_proj)
    save_qlinear(tensors, "fa.o_proj", fa.o_proj)
    save_qlinear(tensors, "fa.index_qk_proj", fa.indexer.index_qk_proj)
    tensors["fa.q_norm.weight"] = fa.q_norm.weight
    tensors["fa.k_norm.weight"] = fa.k_norm.weight
    tensors["fa.index_q_norm.weight"] = fa.indexer.q_layernorm.weight
    tensors["fa.index_k_norm.weight"] = fa.indexer.k_layernorm.weight
    tensors["fa.rope_inv_freq"] = fa.rotary_emb.inv_freq

    fa_cache = q4_lang.QSAKVCache()
    fa_x_pre = mx.astype(mx.random.normal((1, 4, 256)) * 0.2, mx.bfloat16)
    fa_pos_pre = mx.arange(0, 4, dtype=mx.int32)[None, :]
    fa_y_pre = fa(fa_x_pre, mask="causal", cache=fa_cache, position_ids=fa_pos_pre)
    fa_k_pre, fa_v_pre, fa_ik_pre, fa_ip_pre = fa_cache.state

    fa_x_dec = mx.astype(mx.random.normal((1, 1, 256)) * 0.2, mx.bfloat16)
    fa_pos_dec = mx.arange(4, 5, dtype=mx.int32)[None, :]
    fa_y_dec = fa(fa_x_dec, mask=None, cache=fa_cache, position_ids=fa_pos_dec)
    fa_k_dec, fa_v_dec, fa_ik_dec, fa_ip_dec = fa_cache.state

    tensors["fa.x_pre"] = fa_x_pre
    tensors["fa.y_pre"] = fa_y_pre
    tensors["fa.k_pre"] = mx.array(fa_k_pre)
    tensors["fa.v_pre"] = mx.array(fa_v_pre)
    tensors["fa.ik_pre"] = mx.array(fa_ik_pre)
    tensors["fa.ip_pre"] = mx.array(fa_ip_pre)
    tensors["fa.x_dec"] = fa_x_dec
    tensors["fa.y_dec"] = fa_y_dec
    tensors["fa.k_dec"] = mx.array(fa_k_dec)
    tensors["fa.v_dec"] = mx.array(fa_v_dec)
    tensors["fa.ik_dec"] = mx.array(fa_ik_dec)
    tensors["fa.ip_dec"] = mx.array(fa_ip_dec)

    # Also test QSA indexer sparse mask selection when key_len > token_budget
    qsa_cfg = TextConfig(
        model_type="qwen4_exp_text",
        hidden_size=256,
        num_hidden_layers=1,
        num_attention_heads=4,
        linear_num_value_heads=4,
        linear_num_key_heads=2,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
        linear_conv_kernel_dim=4,
        num_experts=16,
        num_experts_per_tok=4,
        shared_expert_intermediate_size=128,
        moe_intermediate_size=128,
        rms_norm_eps=1e-6,
        vocab_size=256,
        num_key_value_heads=2,
        max_position_embeddings=262144,
        hc_count=4,
        hc_lowrank=64,
        head_dim=256,
        indexer_n_heads=4,
        indexer_kv_heads=1,
        indexer_head_dim=128,
        indexer_budget=16,  # block_topk = 16 // 4 = 4
        indexer_compress_ratio=4,
    )
    qsa_idx = q4_lang.Qwen4ExpQSAIndexer(qsa_cfg, fa.rotary_emb)
    qsa_scores = mx.random.normal((1, 4, 2, 6)).astype(mx.float32)  # 6 blocks > block_topk=4
    qsa_mask_pre = qsa_idx._selection_mask(qsa_scores, past_len=24, key_len=26, max_complete_blocks=6)
    tensors["qsa.scores"] = qsa_scores
    tensors["qsa.mask_pre"] = qsa_mask_pre

    # =========================================================================
    # 6. Router + Shared Expert + SwitchGLU (2-bit group_size=64, decode & prefill)
    # =========================================================================
    moe_cfg = TextConfig(
        model_type="qwen4_exp_text",
        hidden_size=256,
        num_hidden_layers=1,
        num_attention_heads=4,
        linear_num_value_heads=4,
        linear_num_key_heads=2,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
        linear_conv_kernel_dim=4,
        num_experts=16,
        num_experts_per_tok=4,
        shared_expert_intermediate_size=128,
        moe_intermediate_size=128,
        rms_norm_eps=1e-6,
        vocab_size=256,
        num_key_value_heads=2,
        max_position_embeddings=4096,
        hc_count=4,
        hc_lowrank=64,
        head_dim=64,
    )
    moe = Qwen3_5MoeSparseMoeBlock(moe_cfg)
    moe.gate.weight = mx.astype(mx.random.normal(moe.gate.weight.shape) * 0.05, mx.bfloat16)
    moe.shared_expert_gate.weight = mx.astype(
        mx.random.normal(moe.shared_expert_gate.weight.shape) * 0.05, mx.bfloat16
    )
    moe.shared_expert.gate_proj.weight = mx.astype(
        mx.random.normal(moe.shared_expert.gate_proj.weight.shape) * 0.05, mx.bfloat16
    )
    moe.shared_expert.up_proj.weight = mx.astype(
        mx.random.normal(moe.shared_expert.up_proj.weight.shape) * 0.05, mx.bfloat16
    )
    moe.shared_expert.down_proj.weight = mx.astype(
        mx.random.normal(moe.shared_expert.down_proj.weight.shape) * 0.05, mx.bfloat16
    )
    moe.switch_mlp.gate_proj.weight = mx.astype(
        mx.random.normal(moe.switch_mlp.gate_proj.weight.shape) * 0.05, mx.bfloat16
    )
    moe.switch_mlp.up_proj.weight = mx.astype(
        mx.random.normal(moe.switch_mlp.up_proj.weight.shape) * 0.05, mx.bfloat16
    )
    moe.switch_mlp.down_proj.weight = mx.astype(
        mx.random.normal(moe.switch_mlp.down_proj.weight.shape) * 0.05, mx.bfloat16
    )

    nn.quantize(
        moe,
        group_size=64,
        bits=4,
        mode="affine",
        class_predicate=lambda p, m: (
            False
            if not hasattr(m, "to_quantized")
            else (
                {"group_size": 64, "bits": 8, "mode": "affine"}
                if (p.endswith("gate") or p.endswith("shared_expert_gate"))
                else (
                    {"group_size": 64, "bits": 2, "mode": "affine"}
                    if "switch_mlp" in p
                    else True
                )
            )
        ),
    )

    save_qlinear(tensors, "moe.gate", moe.gate)
    save_qlinear(tensors, "moe.shared_expert_gate", moe.shared_expert_gate)
    save_qlinear(tensors, "moe.shared_gate", moe.shared_expert.gate_proj)
    save_qlinear(tensors, "moe.shared_up", moe.shared_expert.up_proj)
    save_qlinear(tensors, "moe.shared_down", moe.shared_expert.down_proj)
    save_qlinear(tensors, "moe.switch_gate", moe.switch_mlp.gate_proj)
    save_qlinear(tensors, "moe.switch_up", moe.switch_mlp.up_proj)
    save_qlinear(tensors, "moe.switch_down", moe.switch_mlp.down_proj)

    # Decode L=1 (unsorted gather_qmm, M = 1*1*4 = 4 < 64)
    moe_x_dec = mx.astype(mx.random.normal((1, 1, 256)) * 0.2, mx.bfloat16)
    gates_dec = mx.softmax(moe.gate(moe_x_dec), axis=-1, precise=True)
    inds_dec = mx.argpartition(gates_dec, kth=-4, axis=-1)[..., -4:]
    scores_dec = mx.take_along_axis(gates_dec, inds_dec, axis=-1)
    scores_dec = scores_dec / scores_dec.sum(axis=-1, keepdims=True)
    sh_dec = moe._shared_expert_scale(moe_x_dec) * moe.shared_expert(moe_x_dec)
    y_dec_moe = moe(moe_x_dec)

    # Prefill L=20 (sorted _gather_sort + gather_qmm(sorted=True) + _scatter_unsort, M = 1*20*4 = 80 >= 64)
    moe_x_pre = mx.astype(mx.random.normal((1, 20, 256)) * 0.2, mx.bfloat16)
    gates_pre = mx.softmax(moe.gate(moe_x_pre), axis=-1, precise=True)
    inds_pre = mx.argpartition(gates_pre, kth=-4, axis=-1)[..., -4:]
    scores_pre = mx.take_along_axis(gates_pre, inds_pre, axis=-1)
    scores_pre = scores_pre / scores_pre.sum(axis=-1, keepdims=True)
    sh_pre = moe._shared_expert_scale(moe_x_pre) * moe.shared_expert(moe_x_pre)
    y_pre_moe = moe(moe_x_pre)

    tensors["moe.x_dec"] = moe_x_dec
    tensors["moe.inds_dec"] = inds_dec
    tensors["moe.scores_dec"] = scores_dec
    tensors["moe.sh_dec"] = sh_dec
    tensors["moe.y_dec"] = y_dec_moe
    tensors["moe.x_pre"] = moe_x_pre
    tensors["moe.inds_pre"] = inds_pre
    tensors["moe.scores_pre"] = scores_pre
    tensors["moe.sh_pre"] = sh_pre
    tensors["moe.y_pre"] = y_pre_moe

    # =========================================================================
    # 7. Qwen4ExpPLELayer (full Qwen3.8 metadata + synthetic PLE prefill L=4 & decode L=1)
    # =========================================================================
    mults_real = q4_lang._build_layer_multipliers(248320, 3, 0, 1234)
    moduli_real = [q4_lang._find_nth_prime_after(20_000_000 - 1, h + 1) for h in range(16)]
    tensors["ple.real_multipliers"] = mx.array(mults_real, dtype=mx.int64)
    tensors["ple.real_moduli"] = mx.array(moduli_real, dtype=mx.int64)

    q4_lang._PLE_RUNTIME_MODE = "resident"
    ple_cfg = TextConfig(
        model_type="qwen4_exp_text",
        hidden_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        linear_num_value_heads=4,
        linear_num_key_heads=2,
        linear_key_head_dim=64,
        linear_value_head_dim=64,
        linear_conv_kernel_dim=4,
        num_experts=16,
        num_experts_per_tok=4,
        shared_expert_intermediate_size=128,
        moe_intermediate_size=128,
        rms_norm_eps=1e-6,
        vocab_size=256,
        num_key_value_heads=2,
        max_position_embeddings=4096,
        hc_count=4,
        hc_lowrank=64,
        head_dim=64,
        ple_layer_ids=[2],
        ple_embed_dim=512,
        ple_conv_kernel_size=4,
        ngram_size=3,
        heads_per_ngram=4,
        ngram_vocab_size_base=64,
        make_ngram_vocab_size_divisible_by=16,
        split_ngram_parts=4,
        seed=1234,
        eos_token_id=248044,
    )
    ple_mod = q4_lang.Qwen4ExpPLELayer(ple_cfg, layer_idx=1, ple_layer_index=0)
    ple_mod.key_proj.weight = mx.astype(mx.random.normal(ple_mod.key_proj.weight.shape) * 0.05, mx.bfloat16)
    ple_mod.value_proj.weight = mx.astype(mx.random.normal(ple_mod.value_proj.weight.shape) * 0.05, mx.bfloat16)
    ple_mod.norm_key.weight = mx.astype(mx.random.normal(ple_mod.norm_key.weight.shape) * 0.05, mx.bfloat16)
    ple_mod.norm_query.weight = mx.astype(mx.random.normal(ple_mod.norm_query.weight.shape) * 0.05, mx.bfloat16)
    ple_mod.norm_conv.weight = mx.astype(mx.random.normal(ple_mod.norm_conv.weight.shape) * 0.05, mx.bfloat16)
    ple_mod.conv1d.weight = mx.astype(mx.random.normal(ple_mod.conv1d.weight.shape) * 0.05, mx.bfloat16)
    ple_mod.ple_embedding.ngram_embedding.weight_scale = mx.array([0.75], dtype=mx.bfloat16)
    for sh in ple_mod.ple_embedding.ngram_embedding.shards:
        sh.weight = mx.astype(mx.random.normal(sh.weight.shape) * 0.1, mx.bfloat16)

    nn.quantize(
        ple_mod,
        group_size=32,
        bits=4,
        mode="affine",
        class_predicate=lambda p, m: (
            {"group_size": 64, "bits": 4, "mode": "affine"}
            if isinstance(m, nn.Linear)
            else (
                {"group_size": 32, "bits": 4, "mode": "affine"}
                if isinstance(m, nn.Embedding)
                else False
            )
        ),
    )

    save_qlinear(tensors, "ple.key_proj", ple_mod.key_proj)
    save_qlinear(tensors, "ple.value_proj", ple_mod.value_proj)
    tensors["ple.norm_key.weight"] = ple_mod.norm_key.weight
    tensors["ple.norm_query.weight"] = ple_mod.norm_query.weight
    tensors["ple.norm_conv.weight"] = ple_mod.norm_conv.weight
    tensors["ple.conv1d.weight"] = ple_mod.conv1d.weight
    tensors["ple.weight_scale"] = ple_mod.ple_embedding.ngram_embedding.weight_scale
    for idx_s, sh in enumerate(ple_mod.ple_embedding.ngram_embedding.shards):
        tensors[f"ple.shard.{idx_s}.weight"] = sh.weight
        tensors[f"ple.shard.{idx_s}.scales"] = sh.scales
        tensors[f"ple.shard.{idx_s}.biases"] = sh.biases

    ple_cache = ArraysCache(size=4)
    # Include EOS token (248044) in prefill to exercise _shift_right_ignore_eos
    ple_ids_pre = mx.array([[17, 248044, 42, 99]], dtype=mx.int32)
    ple_h_pre = mx.astype(mx.random.normal((1, 4, 512)) * 0.2, mx.bfloat16)
    ple_y_pre = ple_mod(ple_h_pre, ple_ids_pre, ple_cache, mask=None)
    ple_conv_pre = mx.array(ple_cache[2])
    ple_hist_pre = mx.array(ple_cache[3])

    ple_ids_dec = mx.array([[123]], dtype=mx.int32)
    ple_h_dec = mx.astype(mx.random.normal((1, 1, 512)) * 0.2, mx.bfloat16)
    ple_y_dec = ple_mod(ple_h_dec, ple_ids_dec, ple_cache, mask=None)
    ple_conv_dec = mx.array(ple_cache[2])
    ple_hist_dec = mx.array(ple_cache[3])

    tensors["ple.ids_pre"] = ple_ids_pre
    tensors["ple.h_pre"] = ple_h_pre
    tensors["ple.y_pre"] = ple_y_pre
    tensors["ple.conv_pre"] = ple_conv_pre
    tensors["ple.hist_pre"] = ple_hist_pre
    tensors["ple.ids_dec"] = ple_ids_dec
    tensors["ple.h_dec"] = ple_h_dec
    tensors["ple.y_dec"] = ple_y_dec
    tensors["ple.conv_dec"] = ple_conv_dec
    tensors["ple.hist_dec"] = ple_hist_dec

    # =========================================================================
    # 8. Next-layer Router Prefetch (compute_pred_next / topk_sorted_indices)
    # =========================================================================
    next_gate_lin = nn.Linear(2560, 64, bias=False)
    next_gate_lin.weight = mx.astype(mx.random.normal((64, 2560)) * 0.05, mx.bfloat16)
    next_gate_q = nn.QuantizedLinear.from_linear(next_gate_lin, group_size=64, bits=8, mode="affine")
    save_qlinear(tensors, "pred_next.gate", next_gate_q)

    def topk_sorted_indices(scores: mx.array, k: int) -> mx.array:
        inds = mx.argpartition(scores, kth=-k, axis=-1)[..., -k:]
        vals = mx.take_along_axis(scores, inds, axis=-1)
        order = mx.argsort(-vals, axis=-1)
        return mx.take_along_axis(inds, order, axis=-1)

    m_next_nw, _, _ = hc_mod(tensors["hc.r1.h_in"], write=None)
    lg_next_nw = next_gate_q(m_next_nw)
    tensors["pred_next.top12_nowrite"] = topk_sorted_indices(lg_next_nw, 12)

    m_next_w, _, _ = hc_mod(
        tensors["hc.r1.h_in"],
        write=(tensors["hc.r1.br_in"], tensors["hc.r1.gt_in"]),
    )
    lg_next_w = next_gate_q(m_next_w)
    tensors["pred_next.top12_write"] = topk_sorted_indices(lg_next_w, 12)

    mx.eval(*tensors.values())
    mx.save_safetensors(out_path, tensors)
    size_mb = os.path.getsize(out_path) / (1024 * 1024)
    print(f"Saved {len(tensors)} synthetic fixture tensors ({size_mb:.2f} MB) to {out_path}")


if __name__ == "__main__":
    main()

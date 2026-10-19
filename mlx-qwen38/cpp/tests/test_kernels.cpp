#include "model.hpp"
#include "expert_io.h"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

namespace qwen38 {
void init_ple_metadata(PleLayer& ple, int unigram_vocab_size, int ngram_vocab_base, int divisor, int seed);
}

using namespace qwen38;

static std::string dtype_str(mx::Dtype dt) {
    std::ostringstream oss;
    oss << dt;
    return oss.str();
}

static std::string shape_str(const mx::Shape& s) {
    std::ostringstream oss;
    oss << "[";
    for (size_t i = 0; i < s.size(); ++i) {
        if (i > 0) oss << ",";
        oss << s[i];
    }
    oss << "]";
    return oss.str();
}

static float max_abs_diff(const mx::array& a, const mx::array& b) {
    if (a.shape() != b.shape() || a.size() == 0) return INFINITY;
    auto diff = mx::abs(mx::astype(a, mx::float32) - mx::astype(b, mx::float32));
    auto m = mx::max(diff);
    mx::eval({m});
    return m.item<float>();
}

static bool bit_equal(const mx::array& a, const mx::array& b) {
    if (a.dtype() != b.dtype() || a.shape() != b.shape()) return false;
    auto eq = mx::array_equal(a, b);
    mx::eval({eq});
    return eq.item<bool>();
}

struct TestItem {
    std::string name;
    mx::array cand;
    mx::array ref;
};

static bool check_group_strict(const std::string& group_name, const std::vector<TestItem>& items, bool report_first_mismatch = false) {
    bool all_ok = true;
    float max_d = 0.0f;
    std::string first_bad;
    for (const auto& it : items) {
        bool eq = bit_equal(it.cand, it.ref);
        float d = max_abs_diff(it.cand, it.ref);
        if (d > max_d) max_d = d;
        if (!eq) {
            all_ok = false;
            char diff_buf[64];
            std::snprintf(diff_buf, sizeof(diff_buf), "%.6g", d);
            std::string detail = it.name + " (max_diff=" + diff_buf +
                                 " cand=" + dtype_str(it.cand.dtype()) + shape_str(it.cand.shape()) +
                                 " vs ref=" + dtype_str(it.ref.dtype()) + shape_str(it.ref.shape()) + ")";
            if (first_bad.empty()) {
                first_bad = detail;
            }
            std::printf("  [MISMATCH] %s: %s (max_diff=%.6g, cand=%s%s, ref=%s%s)\n",
                        group_name.c_str(), it.name.c_str(), d,
                        dtype_str(it.cand.dtype()).c_str(), shape_str(it.cand.shape()).c_str(),
                        dtype_str(it.ref.dtype()).c_str(), shape_str(it.ref.shape()).c_str());
        }
    }
    if (all_ok) {
        std::printf("[PASS] %s: all %zu tensors bit-equal (max_diff=0)\n", group_name.c_str(), items.size());
    } else {
        if (report_first_mismatch) {
            std::printf("[FAIL] %s: FIRST non-bit-equal intermediate: %s\n", group_name.c_str(), first_bad.c_str());
        } else {
            std::printf("[FAIL] %s: NOT bit-equal (max_diff=%.6g, first_bad=%s)\n",
                        group_name.c_str(), max_d, first_bad.c_str());
        }
    }
    return all_ok;
}

static QLinear load_qlinear_fix(
    const std::unordered_map<std::string, mx::array>& fix,
    const std::string& prefix,
    int bits,
    int group_size) {
    QLinear ql;
    ql.weight = fix.at(prefix + ".weight");
    ql.scales = fix.at(prefix + ".scales");
    ql.biases = fix.at(prefix + ".biases");
    ql.is_quantized = true;
    ql.bits = bits;
    ql.group_size = group_size;
    return ql;
}

static QSwitchLinear load_qswitch_fix(
    const std::unordered_map<std::string, mx::array>& fix,
    const std::string& prefix,
    int bits,
    int group_size) {
    QSwitchLinear qsl;
    qsl.weight = fix.at(prefix + ".weight");
    qsl.scales = fix.at(prefix + ".scales");
    qsl.biases = fix.at(prefix + ".biases");
    qsl.bits = bits;
    qsl.group_size = group_size;
    return qsl;
}

int main(int argc, char** argv) {
    std::string fix_path = (argc > 1) ? argv[1] : "tests/fixtures.safetensors";
    int failures = 0;

    // 1. Test full Qwen3.8 PLE metadata math (pure CPU)
    PleLayer ple_real;
    init_ple_metadata(ple_real, 248320, 20000000, 128, 1234);
    if (ple_real.multipliers_cpu.size() != 3 || ple_real.moduli_cpu.size() != 16 || ple_real.shard_offsets.size() != 129) {
        std::printf("[FAIL] init_ple_metadata sizes mismatch\n");
        return 1;
    }
    std::printf("[PASS] init_ple_metadata (real Qwen3.8): mults=[%lld, %lld, %lld] mod0=%lld mod15=%lld total_padded=%lld\n",
                static_cast<long long>(ple_real.multipliers_cpu[0]),
                static_cast<long long>(ple_real.multipliers_cpu[1]),
                static_cast<long long>(ple_real.multipliers_cpu[2]),
                static_cast<long long>(ple_real.moduli_cpu[0]),
                static_cast<long long>(ple_real.moduli_cpu[15]),
                static_cast<long long>(ple_real.shard_offsets[128]));

    auto [fix, _meta] = mx::load_safetensors(fix_path);

    // Verify PLE multipliers & moduli match Python exactly
    {
        auto ref_m = fix.at("ple.real_multipliers");
        auto ref_mod = fix.at("ple.real_moduli");
        mx::eval({ref_m, ref_mod});
        const int64_t* mp = ref_m.data<int64_t>();
        const int64_t* modp = ref_mod.data<int64_t>();
        bool ok = true;
        for (int i = 0; i < 3; ++i) {
            if (ple_real.multipliers_cpu[i] != mp[i]) {
                std::printf("[FAIL] ple.real_multipliers[%d] C++=%lld != Py=%lld\n",
                            i, static_cast<long long>(ple_real.multipliers_cpu[i]), static_cast<long long>(mp[i]));
                ok = false;
            }
        }
        for (int i = 0; i < 16; ++i) {
            if (ple_real.moduli_cpu[i] != modp[i]) {
                std::printf("[FAIL] ple.real_moduli[%d] C++=%lld != Py=%lld\n",
                            i, static_cast<long long>(ple_real.moduli_cpu[i]), static_cast<long long>(modp[i]));
                ok = false;
            }
        }
        if (ok) {
            std::printf("[PASS] PLE multipliers & 16 prime moduli bit-identical to vendor Python\n");
        } else {
            failures++;
        }
    }

    EngineOptions opts;
    Qwen38Engine eng(opts);

    // 2. Verify Qwen4ExpRMSNorm & Qwen4ExpRMSNormGated
    {
        auto x = fix.at("norm.x");
        auto w = fix.at("norm.w");
        auto y_full = Qwen38Engine::qwen4_rms_norm(x, w, 1e-6f, -1);
        auto y_grp = Qwen38Engine::qwen4_rms_norm(x, w, 1e-6f, 64);
        if (!check_group_strict("Qwen4ExpRMSNorm", {
            {"y_full", y_full, fix.at("norm.y_full")},
            {"y_grp", y_grp, fix.at("norm.y_grp")},
        })) failures++;

        auto gx = fix.at("gnorm.x");
        auto gg = fix.at("gnorm.g");
        auto gw = fix.at("gnorm.w");
        auto y_sig = Qwen38Engine::qwen4_rms_norm_gated(gx, gg, gw, 1e-6f, true);
        auto y_silu = Qwen38Engine::qwen4_rms_norm_gated(gx, gg, gw, 1e-6f, false);
        if (!check_group_strict("Qwen4ExpRMSNormGated", {
            {"y_sig", y_sig, fix.at("gnorm.y_sig")},
            {"y_silu", y_silu, fix.at("gnorm.y_silu")},
        })) failures++;
    }

    // 3. Verify Qwen3_5RotaryEmbedding / apply_mrope_one (2D and 3D)
    {
        auto q = fix.at("rope.q");
        auto k = fix.at("rope.k");
        auto inv_freq = fix.at("rope.inv_freq");
        auto pos_2d = fix.at("rope.pos_2d");
        auto pos_3d = fix.at("rope.pos_3d");
        auto q_2d = eng.apply_mrope_one(q, inv_freq, pos_2d);
        auto k_2d = eng.apply_mrope_one(k, inv_freq, pos_2d);
        auto q_3d = eng.apply_mrope_one(q, inv_freq, pos_3d);
        auto k_3d = eng.apply_mrope_one(k, inv_freq, pos_3d);
        if (!check_group_strict("apply_mrope_one (2D & 3D Q/K)", {
            {"q_2d", q_2d, fix.at("rope.q_2d")},
            {"k_2d", k_2d, fix.at("rope.k_2d")},
            {"q_3d", q_3d, fix.at("rope.q_3d")},
            {"k_3d", k_3d, fix.at("rope.k_3d")},
        })) failures++;
    }

    // 4. Verify Qwen4ExpGatedResidual / hc_fused across rows=1 (decode v2), rows=8 (decode v1),
    //    rows=18 (prefill), with & without write, final mixer (use_combine=False) + lm_head,
    //    and compute_pred_next (next-layer router prefetch)
    {
        HyperConnection hc;
        hc.use_combine = true;
        hc.hc_count = 4;
        hc.hidden_size = 2560;
        hc.hc_lowrank = 320;
        hc.eps_arr = mx::array({1e-6f}, mx::float32);
        hc.norm_weight = fix.at("hc.norm.weight");
        hc.down = load_qlinear_fix(fix, "hc.down", 8, 64);
        hc.up = load_qlinear_fix(fix, "hc.up", 8, 64);
        hc.inject = load_qlinear_fix(fix, "hc.inject", 8, 64);
        mx::eval({hc.eps_arr});

        HyperConnection hc_mixer = hc;
        hc_mixer.use_combine = false;

        for (const char* tag : {"r1", "r8", "r18"}) {
            std::string p = std::string("hc.") + tag + ".";
            auto h_in = fix.at(p + "h_in");
            auto br_in = fix.at(p + "br_in");
            auto gt_in = fix.at(p + "gt_in");

            auto [m_nw, h_nw, inj_nw] = eng.run_hc(hc, h_in, std::nullopt);
            auto [m_w, h_w, inj_w] = eng.run_hc(hc, h_in, std::make_pair(br_in, gt_in));
            auto [mix_nw, _hnw, _inw] = eng.run_hc(hc_mixer, h_in, std::nullopt);
            auto [mix_w, h_mw, _iw] = eng.run_hc(hc_mixer, h_in, std::make_pair(br_in, gt_in));

            if (!check_group_strict(std::string("hc_fused (") + tag + ")", {
                {"m_nowrite", m_nw, fix.at(p + "m_nowrite")},
                {"h_nowrite", h_nw, fix.at(p + "h_nowrite")},
                {"inj_nowrite", inj_nw, fix.at(p + "inj_nowrite")},
                {"m_write", m_w, fix.at(p + "m_write")},
                {"h_write", h_w, fix.at(p + "h_write")},
                {"inj_write", inj_w, fix.at(p + "inj_write")},
                {"mix_nowrite", mix_nw, fix.at(p + "mix_nowrite")},
                {"mix_write", mix_w, fix.at(p + "mix_write")},
                {"h_mix_write", h_mw, fix.at(p + "h_mix_write")},
            })) failures++;

            if (std::string(tag) == "r18") {
                QLinear lm_head = load_qlinear_fix(fix, "lm_head", 4, 64);
                auto last_h = mx::slice(mix_w, {0, 17, 0}, {1, 18, 2560});
                auto logits = lm_head.forward(last_h);
                if (!check_group_strict("final_mixer + lm_head (no extra RMSNorm)", {
                    {"logits_r18", logits, fix.at("lm_head.logits_r18")},
                })) failures++;
            }
        }

        // Next-layer router prefetch (compute_pred_next / topk_sorted_indices)
        DecoderLayer next_layer;
        next_layer.mlp_hc = hc;
        next_layer.gate = load_qlinear_fix(fix, "pred_next.gate", 8, 64);
        auto pred_nw = eng.compute_pred_next(next_layer, fix.at("hc.r1.h_in"), std::nullopt, 12);
        auto pred_w = eng.compute_pred_next(
            next_layer,
            fix.at("hc.r1.h_in"),
            std::make_pair(fix.at("hc.r1.br_in"), fix.at("hc.r1.gt_in")),
            12);
        if (!check_group_strict("Router prefetch compute_pred_next (top_m=12)", {
            {"top12_nowrite", pred_nw, fix.at("pred_next.top12_nowrite")},
            {"top12_write", pred_w, fix.at("pred_next.top12_write")},
        })) failures++;
    }

    // 5. Verify Qwen4ExpGatedDeltaNet (prefill L=4 + decode L=1) with all intermediates
    {
        LinearAttnLayer la;
        la.num_k_heads = 2;
        la.num_v_heads = 4;
        la.head_k_dim = 128;
        la.head_v_dim = 128;
        la.key_dim = 2 * 128;
        la.value_dim = 4 * 128;
        la.conv_dim = la.key_dim * 2 + la.value_dim;
        la.conv_kernel_size = 4;
        la.rms_norm_eps = 1e-6f;
        la.in_proj_qkv = load_qlinear_fix(fix, "gdn.in_proj_qkv", 4, 64);
        la.in_proj_z = load_qlinear_fix(fix, "gdn.in_proj_z", 4, 64);
        la.in_proj_b = load_qlinear_fix(fix, "gdn.in_proj_b", 4, 64);
        la.in_proj_a = load_qlinear_fix(fix, "gdn.in_proj_a", 4, 64);
        la.out_proj = load_qlinear_fix(fix, "gdn.out_proj", 4, 64);
        la.conv1d_weight = fix.at("gdn.conv1d.weight");
        la.A_log = fix.at("gdn.A_log");
        la.dt_bias = fix.at("gdn.dt_bias");
        la.norm_weight = fix.at("gdn.norm.weight");

        LayerExecCache cache;
        cache.is_linear = true;
        Qwen38Engine::GdnIntermediates dbg_pre;
        auto y_pre = eng.run_linear_attn(la, fix.at("gdn.x_pre"), cache, &dbg_pre);
        if (!check_group_strict("Qwen4ExpGatedDeltaNet prefill (L=4)", {
            {"mixed_qkv", dbg_pre.mixed_qkv, fix.at("gdn.pre.mixed_qkv")},
            {"z", dbg_pre.z, fix.at("gdn.pre.z")},
            {"a", dbg_pre.a, fix.at("gdn.pre.a")},
            {"b", dbg_pre.b, fix.at("gdn.pre.b")},
            {"conv_state", cache.conv_state, fix.at("gdn.conv_pre")},
            {"conv_out", dbg_pre.conv_out, fix.at("gdn.pre.conv_out")},
            {"q", dbg_pre.q, fix.at("gdn.pre.q")},
            {"k", dbg_pre.k, fix.at("gdn.pre.k")},
            {"v", dbg_pre.v, fix.at("gdn.pre.v")},
            {"g", dbg_pre.g, fix.at("gdn.pre.g")},
            {"beta", dbg_pre.beta, fix.at("gdn.pre.beta")},
            {"kernel_y", dbg_pre.kernel_y, fix.at("gdn.pre.kernel_y")},
            {"ssm_state", cache.ssm_state, fix.at("gdn.ssm_pre")},
            {"y", y_pre, fix.at("gdn.y_pre")},
        }, true)) failures++;

        Qwen38Engine::GdnIntermediates dbg_dec;
        auto y_dec = eng.run_linear_attn(la, fix.at("gdn.x_dec"), cache, &dbg_dec);
        if (!check_group_strict("Qwen4ExpGatedDeltaNet decode (L=1)", {
            {"mixed_qkv", dbg_dec.mixed_qkv, fix.at("gdn.dec.mixed_qkv")},
            {"z", dbg_dec.z, fix.at("gdn.dec.z")},
            {"a", dbg_dec.a, fix.at("gdn.dec.a")},
            {"b", dbg_dec.b, fix.at("gdn.dec.b")},
            {"conv_state", cache.conv_state, fix.at("gdn.conv_dec")},
            {"conv_out", dbg_dec.conv_out, fix.at("gdn.dec.conv_out")},
            {"q", dbg_dec.q, fix.at("gdn.dec.q")},
            {"k", dbg_dec.k, fix.at("gdn.dec.k")},
            {"v", dbg_dec.v, fix.at("gdn.dec.v")},
            {"g", dbg_dec.g, fix.at("gdn.dec.g")},
            {"beta", dbg_dec.beta, fix.at("gdn.dec.beta")},
            {"kernel_y", dbg_dec.kernel_y, fix.at("gdn.dec.kernel_y")},
            {"ssm_state", cache.ssm_state, fix.at("gdn.ssm_dec")},
            {"y", y_dec, fix.at("gdn.y_dec")},
        }, true)) failures++;
    }

    // 6. Verify Qwen4ExpAttention + Qwen4ExpQSAIndexer (prefill L=4 + decode L=1)
    {
        FullAttnLayer fa;
        fa.num_heads = 4;
        fa.num_kv_heads = 2;
        fa.head_dim = 256;
        fa.scale = 1.0f / std::sqrt(256.0f);
        fa.rms_norm_eps = 1e-6f;
        fa.indexer_n_heads = 4;
        fa.indexer_kv_heads = 1;
        fa.indexer_head_dim = 128;
        fa.indexer_budget = 2048;
        fa.indexer_compress_ratio = 4;
        fa.q_proj = load_qlinear_fix(fix, "fa.q_proj", 4, 64);
        fa.k_proj = load_qlinear_fix(fix, "fa.k_proj", 4, 64);
        fa.v_proj = load_qlinear_fix(fix, "fa.v_proj", 4, 64);
        fa.o_proj = load_qlinear_fix(fix, "fa.o_proj", 4, 64);
        fa.index_qk_proj = load_qlinear_fix(fix, "fa.index_qk_proj", 4, 64);
        fa.q_norm_scale = mx::array(1.0f, mx::float32) + mx::astype(fix.at("fa.q_norm.weight"), mx::float32);
        fa.k_norm_scale = mx::array(1.0f, mx::float32) + mx::astype(fix.at("fa.k_norm.weight"), mx::float32);
        fa.index_q_norm_scale = mx::array(1.0f, mx::float32) + mx::astype(fix.at("fa.index_q_norm.weight"), mx::float32);
        fa.index_k_norm_scale = mx::array(1.0f, mx::float32) + mx::astype(fix.at("fa.index_k_norm.weight"), mx::float32);
        fa.rope_inv_freq = fix.at("fa.rope_inv_freq");
        mx::eval({fa.q_norm_scale, fa.k_norm_scale, fa.index_q_norm_scale, fa.index_k_norm_scale});

        LayerExecCache cache;
        cache.is_linear = false;
        auto y_pre = eng.run_full_attn(fa, fix.at("fa.x_pre"), cache, std::nullopt);
        auto k_pre = mx::slice(cache.keys, {0, 0, 0, 0}, {1, 2, cache.offset, 256});
        auto v_pre = mx::slice(cache.values, {0, 0, 0, 0}, {1, 2, cache.offset, 256});
        auto ik_pre = mx::slice(cache.index_keys, {0, 0, 0}, {1, cache.index_offset, 128});
        auto ip_pre = mx::slice(cache.index_position_ids, {0, 0}, {1, cache.index_offset});

        if (!check_group_strict("Qwen4ExpAttention + QSAIndexer prefill (L=4)", {
            {"y", y_pre, fix.at("fa.y_pre")},
            {"keys", k_pre, fix.at("fa.k_pre")},
            {"values", v_pre, fix.at("fa.v_pre")},
            {"index_keys", ik_pre, fix.at("fa.ik_pre")},
            {"index_position_ids", ip_pre, fix.at("fa.ip_pre")},
        })) failures++;

        auto y_dec = eng.run_full_attn(fa, fix.at("fa.x_dec"), cache, std::nullopt);
        auto k_dec = mx::slice(cache.keys, {0, 0, 0, 0}, {1, 2, cache.offset, 256});
        auto v_dec = mx::slice(cache.values, {0, 0, 0, 0}, {1, 2, cache.offset, 256});
        auto ik_dec = mx::slice(cache.index_keys, {0, 0, 0}, {1, cache.index_offset, 128});
        auto ip_dec = mx::slice(cache.index_position_ids, {0, 0}, {1, cache.index_offset});

        if (!check_group_strict("Qwen4ExpAttention + QSAIndexer decode (L=1)", {
            {"y", y_dec, fix.at("fa.y_dec")},
            {"keys", k_dec, fix.at("fa.k_dec")},
            {"values", v_dec, fix.at("fa.v_dec")},
            {"index_keys", ik_dec, fix.at("fa.ik_dec")},
            {"index_position_ids", ip_dec, fix.at("fa.ip_dec")},
        })) failures++;

        // Also test QSA indexer sparse block selection mask (scores -> ReLU sum / sqrt(D) -> argpartition -> mask)
        auto raw_sc = fix.at("qsa.scores"); // [1, 4, 2, 6]
        int batch = 1, seq_len = 2, past_len = 24, key_len = 26, max_complete_blocks = 6;
        int compress_ratio = 4, block_topk = 4;
        int complete_key_len = max_complete_blocks * compress_ratio;
        auto sc = mx::divide(mx::sum(mx::maximum(raw_sc, mx::array(0.0f)), 1), mx::array(std::sqrt(128.0f)));
        auto query_ends = mx::array(past_len, mx::int32) + mx::arange(seq_len, mx::int32) + mx::array(1, mx::int32);
        auto complete_counts = mx::floor_divide(query_ends, mx::array(compress_ratio, mx::int32));
        auto valid_blocks = mx::less(
            mx::reshape(mx::arange(max_complete_blocks, mx::int32), {1, 1, max_complete_blocks}),
            mx::reshape(complete_counts, {1, seq_len, 1}));
        sc = mx::where(valid_blocks, sc, mx::array(-INFINITY, mx::float32));
        auto part = mx::argpartition(sc, -block_topk, -1);
        auto sel_blocks = mx::slice(part, {0, 0, max_complete_blocks - block_topk}, {batch, seq_len, max_complete_blocks});
        auto block_hits = mx::put_along_axis(
            mx::zeros({batch, seq_len, max_complete_blocks}, mx::bool_),
            sel_blocks,
            mx::array(true, mx::bool_),
            -1);
        auto sel_tokens = mx::repeat(block_hits, compress_ratio, -1);
        if (complete_key_len < key_len) {
            sel_tokens = mx::concatenate(
                {sel_tokens, mx::zeros({batch, seq_len, key_len - complete_key_len}, mx::bool_)}, -1);
        }
        auto tok_idx = mx::reshape(mx::arange(key_len, mx::int32), {1, 1, key_len});
        auto tail_starts = mx::reshape(complete_counts * mx::array(compress_ratio, mx::int32), {1, seq_len, 1});
        auto q_ends_3d = mx::reshape(query_ends, {1, seq_len, 1});
        auto tail = mx::logical_and(mx::greater_equal(tok_idx, tail_starts), mx::less(tok_idx, q_ends_3d));
        auto causal = mx::less(tok_idx, q_ends_3d);
        auto use_sparse = mx::reshape(mx::greater(complete_counts, mx::array(block_topk, mx::int32)), {1, seq_len, 1});
        auto qsa_mask = mx::expand_dims(mx::where(use_sparse, mx::logical_or(sel_tokens, tail), causal), 1);
        if (!check_group_strict("QSAIndexer sparse _selection_mask", {
            {"mask_pre", qsa_mask, fix.at("qsa.mask_pre")},
        })) failures++;
    }

    // 7. Verify Router (precise softmax + argpartition + renorm), Shared Expert, and SwitchGLU (2-bit g64)
    {
        eng.layers().resize(1);
        auto& layer = eng.layers()[0];
        layer.layer_idx = 0;
        layer.resident_moe = true;
        layer.gate = load_qlinear_fix(fix, "moe.gate", 8, 64);
        layer.shared_expert_gate = load_qlinear_fix(fix, "moe.shared_expert_gate", 8, 64);
        layer.shared_gate_proj = load_qlinear_fix(fix, "moe.shared_gate", 4, 64);
        layer.shared_up_proj = load_qlinear_fix(fix, "moe.shared_up", 4, 64);
        layer.shared_down_proj = load_qlinear_fix(fix, "moe.shared_down", 4, 64);
        layer.res_gate_proj = load_qswitch_fix(fix, "moe.switch_gate", 2, 64);
        layer.res_up_proj = load_qswitch_fix(fix, "moe.switch_up", 2, 64);
        layer.res_down_proj = load_qswitch_fix(fix, "moe.switch_down", 2, 64);

        // Decode L=1
        auto x_dec = fix.at("moe.x_dec");
        auto [inds_dec, scores_dec] = eng.route_topk(layer.gate, x_dec, 4);
        auto sh_g_dec = layer.shared_gate_proj.forward(x_dec);
        auto sh_u_dec = layer.shared_up_proj.forward(x_dec);
        auto sh_act_dec = eng.run_swiglu(sh_g_dec, sh_u_dec);
        auto sh_out_dec = layer.shared_down_proj.forward(sh_act_dec);
        auto sh_dec = mx::sigmoid(layer.shared_expert_gate.forward(x_dec)) * sh_out_dec;
        auto y_dec = eng.run_moe_prefill(0, x_dec, inds_dec, scores_dec, sh_dec);

        if (!check_group_strict("MoE Router + Shared + SwitchGLU (2-bit g64) decode (L=1)", {
            {"inds", inds_dec, fix.at("moe.inds_dec")},
            {"scores", scores_dec, fix.at("moe.scores_dec")},
            {"shared_y", sh_dec, fix.at("moe.sh_dec")},
            {"y", y_dec, fix.at("moe.y_dec")},
        })) failures++;

        // Prefill L=20 (sorted path M=80 >= 64)
        auto x_pre = fix.at("moe.x_pre");
        auto [inds_pre, scores_pre] = eng.route_topk(layer.gate, x_pre, 4);
        auto sh_g_pre = layer.shared_gate_proj.forward(x_pre);
        auto sh_u_pre = layer.shared_up_proj.forward(x_pre);
        auto sh_act_pre = eng.run_swiglu(sh_g_pre, sh_u_pre);
        auto sh_out_pre = layer.shared_down_proj.forward(sh_act_pre);
        auto sh_pre = mx::sigmoid(layer.shared_expert_gate.forward(x_pre)) * sh_out_pre;
        auto y_pre = eng.run_moe_prefill(0, x_pre, inds_pre, scores_pre, sh_pre);

        if (!check_group_strict("MoE Router + Shared + SwitchGLU (2-bit g64) prefill (L=20)", {
            {"inds", inds_pre, fix.at("moe.inds_pre")},
            {"scores", scores_pre, fix.at("moe.scores_pre")},
            {"shared_y", sh_pre, fix.at("moe.sh_pre")},
            {"y", y_pre, fix.at("moe.y_pre")},
        })) failures++;

        // Also test offloaded slot path (run_slot_glu + offloaded run_moe_prefill on layer 28)
        eng.layers().resize(29);
        auto& off_layer = eng.layers()[28];
        off_layer.layer_idx = 28;
        off_layer.resident_moe = false;
        off_layer.offload.capacity = 16;
        off_layer.offload.bits = 2;
        off_layer.offload.group_size = 64;
        off_layer.offload.gate_w = fix.at("moe.switch_gate.weight");
        off_layer.offload.gate_s = fix.at("moe.switch_gate.scales");
        off_layer.offload.gate_b = fix.at("moe.switch_gate.biases");
        off_layer.offload.up_w = fix.at("moe.switch_up.weight");
        off_layer.offload.up_s = fix.at("moe.switch_up.scales");
        off_layer.offload.up_b = fix.at("moe.switch_up.biases");
        off_layer.offload.down_w = fix.at("moe.switch_down.weight");
        off_layer.offload.down_s = fix.at("moe.switch_down.scales");
        off_layer.offload.down_b = fix.at("moe.switch_down.biases");
        off_layer.offload.slot_inds_buf = mx::zeros({1, 1, 4}, mx::uint32);
        mx::eval({off_layer.offload.slot_inds_buf});

        uint64_t dummy_offsets[9] = {0};
        uint64_t dummy_row_bytes[9] = {0};
        uint64_t dummy_ptrs[9] = {0};
        c_layer_init(28, -1, 16, dummy_offsets, dummy_row_bytes, dummy_ptrs,
                     reinterpret_cast<uint64_t>(off_layer.offload.slot_inds_buf.data<uint32_t>()));
        int32_t init_eids[16], init_slots[16];
        for (int i = 0; i < 16; ++i) { init_eids[i] = i; init_slots[i] = i; }
        c_layer_set_lru_state(28, init_eids, init_slots, 16, 0, 0, 0);

        auto y_slot_dec = mx::sum(eng.run_slot_glu(28, x_dec, inds_dec, false) * mx::expand_dims(scores_dec, -1), -2) + sh_dec;
        auto y_off_pre = eng.run_moe_prefill(28, x_pre, inds_pre, scores_pre, sh_pre);
        if (!check_group_strict("Offloaded run_slot_glu (L=1) & run_moe_prefill (L=20 sorted)", {
            {"slot_dec", y_slot_dec, fix.at("moe.y_dec")},
            {"off_pre", y_off_pre, fix.at("moe.y_pre")},
        })) failures++;
    }

    // 8. Verify Qwen4ExpPLELayer (hashing, EOS boundary reset, shard gather + dequant, dilated conv1d, gating)
    {
        auto& ple = eng.ple();
        ple = PleLayer{};
        ple.active = true;
        ple.ngram_size = 3;
        ple.context_len = 2;
        ple.heads_per_ngram = 4;
        ple.ngram_heads = 8;
        ple.ple_embed_dim = 512;
        ple.hidden_size = 128;
        ple.hc_count = 4;
        ple.hc_hidden_size = 512;
        ple.conv_kernel_size = 4;
        ple.conv_dilation = 3;
        ple.short_conv_state_len = 9;
        ple.split_ngram_parts = 4;
        ple.eos_token_id = 248044;
        ple.rms_norm_eps = 1e-6f;
        init_ple_metadata(ple, 256, 64, 16, 1234);

        ple.key_proj = load_qlinear_fix(fix, "ple.key_proj", 4, 64);
        ple.value_proj = load_qlinear_fix(fix, "ple.value_proj", 4, 64);
        ple.norm_key_weight = fix.at("ple.norm_key.weight");
        ple.norm_query_weight = fix.at("ple.norm_query.weight");
        ple.norm_conv_weight = fix.at("ple.norm_conv.weight");
        ple.short_conv_weight = fix.at("ple.conv1d.weight");
        ple.weight_scale = fix.at("ple.weight_scale");

        std::vector<mx::array> keep_alive;
        ple.shards.resize(4);
        for (int s = 0; s < 4; ++s) {
            std::string sp = "ple.shard." + std::to_string(s);
            auto w = fix.at(sp + ".weight");
            auto sc = fix.at(sp + ".scales");
            auto bi = fix.at(sp + ".biases");
            mx::eval({w, sc, bi});
            keep_alive.push_back(w);
            keep_alive.push_back(sc);
            keep_alive.push_back(bi);

            ple.shards[s].bits = 4;
            ple.shards[s].group_size = 32;
            ple.shards[s].weight.data_ptr = reinterpret_cast<const uint8_t*>(w.data<uint32_t>());
            ple.shards[s].weight.rows = w.shape(0);
            ple.shards[s].weight.cols = w.shape(1);
            ple.shards[s].weight.row_bytes = w.shape(1) * sizeof(uint32_t);

            ple.shards[s].scales.data_ptr = reinterpret_cast<const uint8_t*>(sc.data<mx::bfloat16_t>());
            ple.shards[s].scales.rows = sc.shape(0);
            ple.shards[s].scales.cols = sc.shape(1);
            ple.shards[s].scales.row_bytes = sc.shape(1) * sizeof(uint16_t);

            ple.shards[s].biases.data_ptr = reinterpret_cast<const uint8_t*>(bi.data<mx::bfloat16_t>());
            ple.shards[s].biases.rows = bi.shape(0);
            ple.shards[s].biases.cols = bi.shape(1);
            ple.shards[s].biases.row_bytes = bi.shape(1) * sizeof(uint16_t);
        }

        std::vector<int32_t> ids_pre = {17, 248044, 42, 99};
        auto y_pre = eng.run_ple(fix.at("ple.h_pre"), ids_pre);
        auto hist_pre_arr = mx::array(ple.history_cpu.data(), {1, 2}, mx::int64);
        if (!check_group_strict("Qwen4ExpPLELayer prefill (L=4 w/ EOS)", {
            {"y", y_pre, fix.at("ple.y_pre")},
            {"conv_state", ple.conv_state, fix.at("ple.conv_pre")},
            {"history", hist_pre_arr, fix.at("ple.hist_pre")},
        })) failures++;

        std::vector<int32_t> ids_dec = {123};
        auto y_dec = eng.run_ple(fix.at("ple.h_dec"), ids_dec);
        auto hist_dec_arr = mx::array(ple.history_cpu.data(), {1, 2}, mx::int64);
        if (!check_group_strict("Qwen4ExpPLELayer decode (L=1)", {
            {"y", y_dec, fix.at("ple.y_dec")},
            {"conv_state", ple.conv_state, fix.at("ple.conv_dec")},
            {"history", hist_dec_arr, fix.at("ple.hist_dec")},
        })) failures++;
    }

    // 9. v2 parity tests: official RoPE inv_freq helper,
    //    real-size MRoPE (q 24 heads / k 2 heads, positions up to 2047, 2D + 3D) and real-shape lm_head.
    {
        std::string v2_path = (argc > 2) ? argv[2] : "tests/fixtures_v2.safetensors";
        auto [v2, _m2] = mx::load_safetensors(v2_path);
        auto inv = make_rope_inv_freq(64, 10000000.0f);  // the helper the engine loader really uses
        if (!check_group_strict("make_rope_inv_freq (official helper) vs Python compute_inv_freq", {
            {"inv_freq_vs_v2", inv, v2.at("rope2.inv_freq")},
            {"inv_freq_vs_fa_fixture", inv, fix.at("fa.rope_inv_freq")},
        })) failures++;
        auto k = v2.at("rope2.k");
        auto q = v2.at("rope2.q");
        if (!check_group_strict("apply_mrope_one real size (k2 x 2048 pos, q24 x 512 pos; 2D/3D)", {
            {"k_2d", eng.apply_mrope_one(k, inv, v2.at("rope2.k_pos2d")), v2.at("rope2.k_2d")},
            {"k_3d_text", eng.apply_mrope_one(k, inv, v2.at("rope2.k_pos3d_text")), v2.at("rope2.k_3d_text")},
            {"k_3d_mm", eng.apply_mrope_one(k, inv, v2.at("rope2.k_pos3d_mm")), v2.at("rope2.k_3d_mm")},
            {"q_2d", eng.apply_mrope_one(q, inv, v2.at("rope2.q_pos2d")), v2.at("rope2.q_2d")},
            {"q_3d_text", eng.apply_mrope_one(q, inv, v2.at("rope2.q_pos3d_text")), v2.at("rope2.q_3d_text")},
        })) failures++;
    }
    {
        std::string lmh_path = (argc > 3) ? argv[3] : "tests/tmp/lmhead_real.safetensors";
        std::ifstream probe(lmh_path);
        if (!probe.good()) {
            std::printf("[FAIL] real-shape lm_head fixture missing: %s (run tests/gen_fixtures_v2.py)\n", lmh_path.c_str());
            failures++;
        } else {
            auto [lf, _m3] = mx::load_safetensors(lmh_path);
            QLinear head = load_qlinear_fix(lf, "lmh", 4, 32);  // real lm_head: 4-bit, group 32, 2560 -> 248320
            auto x = lf.at("lmh.x");
            int L = x.shape(1);
            auto prefill_row = lm_head_last_row(head, x);  // engine prefill path
            auto x_last = mx::slice(x, {0, L - 1, 0}, {1, L, x.shape(2)});
            auto decode_row = head.forward(x_last);        // engine decode path (M = 1)
            bool py_m49_ne_m1 = !bit_equal(lf.at("lmh.ref_last_M49"), lf.at("lmh.ref_M1"));
            std::printf("  [info] python lm_head M=%d-then-slice vs M=1 differ: %s (why prefill must not slice first)\n",
                        L, py_m49_ne_m1 ? "yes" : "no");
            if (!check_group_strict("lm_head real shape: prefill M=L then last row / decode M=1", {
                {"prefill_last_row_M49", prefill_row, lf.at("lmh.ref_last_M49")},
                {"decode_M1", decode_row, lf.at("lmh.ref_M1")},
            })) failures++;
        }
    }

    if (failures > 0) {
        std::printf("[FAIL] %d test groups failed strict bit-equality.\n", failures);
        return 1;
    }
    std::printf("ALL KERNEL PARITY TESTS PASSED BIT-IDENTICALLY.\n");
    return 0;
}

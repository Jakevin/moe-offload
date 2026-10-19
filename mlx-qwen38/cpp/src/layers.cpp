#include "model.hpp"
#include "expert_io.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <numeric>
#include <stdexcept>

namespace qwen38 {

// Exact Qwen4ExpGatedDeltaNet._normalize_qk (qwen4_exp/language.py:1199-1205)
// Note: round4/cpp4 mistakenly used Qwen3_5GatedDeltaNet._normalize_qk (mx::fast::rms_norm),
// which caused 1-ULP drift in linear attention that snowballed at token 45/74!
static std::pair<mx::array, mx::array> normalize_qwen4_gdn_qk(const mx::array& q, const mx::array& k) {
    float scale_f = std::pow(static_cast<float>(q.shape(-1)), -0.5f);
    auto eps = mx::array(1e-6f, q.dtype());
    auto scale = mx::array(scale_f, q.dtype());
    auto q_norm = (q * mx::rsqrt(mx::sum(mx::square(q), -1, true) + eps)) * scale;
    auto k_norm = k * mx::rsqrt(mx::sum(mx::square(k), -1, true) + eps);
    return {q_norm, k_norm};
}

mx::array Qwen38Engine::run_linear_attn(
    const LinearAttnLayer& la,
    const mx::array& x,
    LayerExecCache& cache,
    GdnIntermediates* dbg) {
    init_kernels();
    int B = x.shape(0);
    int T = x.shape(1);
    int Hk = la.num_k_heads;
    int Hv = la.num_v_heads;
    int Dk = la.head_k_dim;
    int Dv = la.head_v_dim;
    int key_dim = la.key_dim;
    int conv_dim = la.conv_dim;

    auto qkv = la.in_proj_qkv.forward(x);
    auto z = mx::reshape(la.in_proj_z.forward(x), {B, T, Hv, Dv});
    auto b = la.in_proj_b.forward(x);
    auto a = la.in_proj_a.forward(x);

    if (!cache.gdn_initialized) {
        cache.conv_state = mx::zeros({B, la.conv_kernel_size - 1, conv_dim}, x.dtype());
        cache.ssm_state = mx::zeros({B, Hv, Dv, Dk}, mx::float32);
        cache.gdn_initialized = true;
    }

    auto conv_input = mx::concatenate({cache.conv_state, qkv}, 1);
    int total_len = conv_input.shape(1);
    cache.conv_state = mx::contiguous(mx::slice(
        conv_input,
        {0, total_len - (la.conv_kernel_size - 1), 0},
        {B, total_len, conv_dim}));

    mx::array conv_out = qkv;
    if (T == 1 && total_len == la.conv_kernel_size) {
        auto w_t_f32 = mx::astype(mx::transpose(mx::squeeze(la.conv1d_weight, -1)), mx::float32);
        auto c_dec = depthwise_conv_compiled_({conv_input, w_t_f32})[0];
        conv_out = silu_compiled_({c_dec})[0];
    } else {
        auto conv_raw = mx::conv1d(conv_input, la.conv1d_weight, 1, 0, 1, conv_dim);
        conv_out = silu_compiled_({conv_raw})[0];
    }

    auto parts = mx::split(conv_out, {key_dim, 2 * key_dim}, -1);
    auto q = mx::reshape(parts[0], {B, T, Hk, Dk});
    auto k = mx::reshape(parts[1], {B, T, Hk, Dk});
    auto v = mx::reshape(parts[2], {B, T, Hv, Dv});

    auto [q_norm, k_norm] = normalize_qwen4_gdn_qk(q, k);
    auto gb = compute_g_beta_compiled_({la.A_log, a, b, la.dt_bias});
    auto g = gb[0];
    auto beta = gb[1];

    // Qwen4ExpGatedDeltaNet inherits Qwen3_5GatedDeltaNet.__call__, which passes
    // use_kernel = not self.training. Because OffloadEngine / sup_ref never calls
    // model.eval(), self.training is True (use_kernel=False), so gated_delta_update
    // runs gated_delta_chunked for prefill (T > 1) and gated_delta_ops for decode (T == 1).
    auto [y, new_state] = (T > 1 && g.ndim() == 3)
        ? gated_delta_chunked(q_norm, k_norm, v, g, beta, cache.ssm_state)
        : gated_delta_ops(q_norm, k_norm, v, g, beta, cache.ssm_state);
    cache.ssm_state = new_state;

    if (dbg != nullptr) {
        dbg->mixed_qkv = qkv;
        dbg->z = z;
        dbg->a = a;
        dbg->b = b;
        dbg->conv_out = conv_out;
        dbg->q = q_norm;
        dbg->k = k_norm;
        dbg->v = v;
        dbg->g = g;
        dbg->beta = beta;
        dbg->kernel_y = y;
        dbg->state = cache.ssm_state;
    }

    auto out = qwen4_rms_norm_gated(y, z, la.norm_weight, la.rms_norm_eps, true);
    return la.out_proj.forward(mx::reshape(out, {B, T, Hv * Dv}));
}

static void ensure_kv_capacity(LayerExecCache& cache, int B, int kv_heads, int head_dim, int needed, mx::Dtype dtype) {
    constexpr int step = 8192;
    int cur = (cache.keys.size() == 0) ? 0 : cache.keys.shape(2);
    if (needed <= cur) return;
    int cap = ((needed + step - 1) / step) * step;
    if (cur > 0 && cap < 2 * cur) cap = 2 * cur;
    auto new_k = mx::zeros({B, kv_heads, cap, head_dim}, dtype);
    auto new_v = mx::zeros({B, kv_heads, cap, head_dim}, dtype);
    if (cache.offset > 0 && cache.keys.size() > 0) {
        new_k = mx::slice_update(
            new_k,
            mx::slice(cache.keys, {0, 0, 0, 0}, {B, kv_heads, cache.offset, head_dim}),
            {0, 0, 0, 0},
            {B, kv_heads, cache.offset, head_dim});
        new_v = mx::slice_update(
            new_v,
            mx::slice(cache.values, {0, 0, 0, 0}, {B, kv_heads, cache.offset, head_dim}),
            {0, 0, 0, 0},
            {B, kv_heads, cache.offset, head_dim});
    }
    cache.keys = new_k;
    cache.values = new_v;
}

static void update_indexer_cache(
    LayerExecCache& cache,
    const mx::array& raw_keys,
    const mx::array& position_ids) {
    constexpr int step = 8192;
    int B = raw_keys.shape(0);
    int S = raw_keys.shape(1);
    int D = raw_keys.shape(2);
    int end = cache.index_offset + S;

    mx::array pos = position_ids;
    if (cache.index_position_ids.size() > 0) {
        if (cache.index_position_ids.ndim() == 3 && pos.ndim() == 2) {
            pos = mx::broadcast_to(
                mx::expand_dims(pos, 0),
                {cache.index_position_ids.shape(0), pos.shape(0), pos.shape(1)});
        } else if (cache.index_position_ids.ndim() == 2 && pos.ndim() == 3) {
            auto old_pos = mx::slice(cache.index_position_ids, {0, 0}, {B, cache.index_offset});
            cache.index_position_ids = mx::broadcast_to(
                mx::expand_dims(old_pos, 0),
                {pos.shape(0), B, cache.index_offset});
        }
    }

    int cur = (cache.index_keys.size() == 0) ? 0
        : std::min(cache.index_keys.shape(1), cache.index_position_ids.shape(-1));
    if (end > cur) {
        int cap = ((end + step - 1) / step) * step;
        if (cur > 0 && cap < 2 * cur) cap = 2 * cur;
        auto new_ik = mx::zeros({B, cap, D}, raw_keys.dtype());
        mx::Shape pos_shape = (pos.ndim() == 3)
            ? mx::Shape{pos.shape(0), B, cap}
            : mx::Shape{B, cap};
        auto new_ip = mx::zeros(pos_shape, pos.dtype());
        if (cache.index_offset > 0 && cache.index_keys.size() > 0) {
            new_ik = mx::slice_update(
                new_ik,
                mx::slice(cache.index_keys, {0, 0, 0}, {B, cache.index_offset, D}),
                {0, 0, 0},
                {B, cache.index_offset, D});
            if (pos.ndim() == 3) {
                new_ip = mx::slice_update(
                    new_ip,
                    mx::slice(cache.index_position_ids, {0, 0, 0}, {pos.shape(0), B, cache.index_offset}),
                    {0, 0, 0},
                    {pos.shape(0), B, cache.index_offset});
            } else {
                new_ip = mx::slice_update(
                    new_ip,
                    mx::slice(cache.index_position_ids, {0, 0}, {B, cache.index_offset}),
                    {0, 0},
                    {B, cache.index_offset});
            }
        }
        cache.index_keys = new_ik;
        cache.index_position_ids = new_ip;
    }

    cache.index_keys = mx::slice_update(
        cache.index_keys, raw_keys, {0, cache.index_offset, 0}, {B, end, D});
    if (cache.index_position_ids.ndim() == 3) {
        cache.index_position_ids = mx::slice_update(
            cache.index_position_ids, pos,
            {0, 0, cache.index_offset},
            {cache.index_position_ids.shape(0), B, end});
    } else {
        cache.index_position_ids = mx::slice_update(
            cache.index_position_ids, pos,
            {0, cache.index_offset},
            {B, end});
    }
    cache.index_offset = end;
}

mx::array Qwen38Engine::run_full_attn(
    const FullAttnLayer& fa,
    const mx::array& x,
    LayerExecCache& cache,
    const std::optional<mx::array>& position_ids_opt) {
    init_kernels();
    int B = x.shape(0);
    int L = x.shape(1);
    int past_len = cache.offset;
    int end_len = past_len + L;

    mx::array text_pos_2d = position_ids_opt.has_value()
        ? (position_ids_opt->ndim() == 3
            ? mx::slice(*position_ids_opt, {0, 0, 0}, {1, B, L})
            : *position_ids_opt)
        : mx::expand_dims(mx::arange(past_len, end_len, mx::int32), 0);
    if (text_pos_2d.ndim() == 3) {
        text_pos_2d = mx::squeeze(text_pos_2d, 0);
    }

    // QSA Indexer
    int idx_n_heads = fa.indexer_n_heads;
    int idx_kv_heads = fa.indexer_kv_heads;
    int idx_dim = fa.indexer_head_dim;
    auto qk_idx = mx::reshape(
        fa.index_qk_proj.forward(x),
        {B, L, idx_n_heads + idx_kv_heads, idx_dim});
    auto idx_query = mx::slice(qk_idx, {0, 0, 0, 0}, {B, L, idx_n_heads, idx_dim});
    auto raw_keys = mx::squeeze(
        mx::slice(qk_idx, {0, 0, idx_n_heads, 0}, {B, L, idx_n_heads + idx_kv_heads, idx_dim}),
        2);
    idx_query = mx::transpose(
        qwen4_rms_norm_scale(idx_query, fa.index_q_norm_scale, fa.rms_norm_eps),
        {0, 2, 1, 3});
    update_indexer_cache(cache, raw_keys, text_pos_2d);

    // Main Q, K, V projections
    auto q_proj_out = mx::reshape(fa.q_proj.forward(x), {B, L, fa.num_heads, fa.head_dim * 2});
    auto q_parts = mx::split(q_proj_out, 2, -1);
    auto queries = q_parts[0];
    auto gate = mx::reshape(q_parts[1], {B, L, fa.num_heads * fa.head_dim});
    auto keys = mx::reshape(fa.k_proj.forward(x), {B, L, fa.num_kv_heads, fa.head_dim});
    auto values = mx::reshape(fa.v_proj.forward(x), {B, L, fa.num_kv_heads, fa.head_dim});

    queries = mx::transpose(
        qwen4_rms_norm_scale(queries, fa.q_norm_scale, fa.rms_norm_eps),
        {0, 2, 1, 3});
    keys = mx::transpose(
        qwen4_rms_norm_scale(keys, fa.k_norm_scale, fa.rms_norm_eps),
        {0, 2, 1, 3});
    values = mx::transpose(values, {0, 2, 1, 3});

    // Apply MRoPE: Qwen3_5LanguageModel.__call__ passes 2D (1, L) position_ids for both prefill and decode
    mx::array rotary_pos = position_ids_opt.has_value() ? *position_ids_opt : text_pos_2d;
    queries = apply_mrope_one(queries, fa.rope_inv_freq, rotary_pos);
    keys = apply_mrope_one(keys, fa.rope_inv_freq, rotary_pos);

    ensure_kv_capacity(cache, B, fa.num_kv_heads, fa.head_dim, end_len, keys.dtype());
    cache.keys = mx::slice_update(
        cache.keys, keys, {0, 0, past_len, 0}, {B, fa.num_kv_heads, end_len, fa.head_dim});
    cache.values = mx::slice_update(
        cache.values, values, {0, 0, past_len, 0}, {B, fa.num_kv_heads, end_len, fa.head_dim});
    cache.offset = end_len;

    auto k_valid = mx::slice(cache.keys, {0, 0, 0, 0}, {B, fa.num_kv_heads, end_len, fa.head_dim});
    auto v_valid = mx::slice(cache.values, {0, 0, 0, 0}, {B, fa.num_kv_heads, end_len, fa.head_dim});

    // For prefill when end_len <= 2048 + 15, Qwen4ExpAttention._gathered_text_prefill
    // also eagerly pools completed indexer blocks into cache.pooled_index_keys.
    int complete_blocks = cache.index_offset / fa.indexer_compress_ratio;
    if (L >= 16 && end_len > fa.indexer_budget && cache.pooled_index_offset < complete_blocks) {
        int start_b = cache.pooled_index_offset;
        int raw_start = start_b * fa.indexer_compress_ratio;
        int raw_stop = complete_blocks * fa.indexer_compress_ratio;
        int b_cnt = complete_blocks - start_b;
        auto ik_slice = mx::slice(cache.index_keys, {0, raw_start, 0}, {B, raw_stop, idx_dim});
        auto pooled = mx::reshape(ik_slice, {B, b_cnt, fa.indexer_compress_ratio, idx_dim});
        pooled = mx::astype(mx::mean(mx::astype(pooled, mx::float32), -2), ik_slice.dtype());
        pooled = qwen4_rms_norm_scale(pooled, fa.index_k_norm_scale, fa.rms_norm_eps);
        cache.pooled_index_offset = complete_blocks;
    }

    std::string mask_mode = (L > 1) ? "causal" : "";
    auto attn_out = mx::fast::scaled_dot_product_attention(
        queries, k_valid, v_valid, fa.scale, mask_mode);
    attn_out = mx::reshape(
        mx::transpose(attn_out, {0, 2, 1, 3}),
        {B, L, fa.num_heads * fa.head_dim});
    return fa.o_proj.forward(attn_out * mx::sigmoid(gate));
}

std::pair<mx::array, mx::array> Qwen38Engine::route_topk(
    const QLinear& gate,
    const mx::array& x,
    int top_k) {
    auto router_logits = gate.forward(x);
    auto gates = mx::softmax(router_logits, -1, true);
    auto all_inds = mx::argpartition(gates, -top_k, -1);
    mx::Shape start(all_inds.ndim(), 0);
    start.back() = all_inds.shape(-1) - top_k;
    auto inds = mx::slice(all_inds, start, all_inds.shape());
    auto scores = mx::take_along_axis(gates, inds, -1);
    scores = scores / mx::sum(scores, -1, true);
    scores = mx::astype(scores, x.dtype());
    return {inds, scores};
}

mx::array Qwen38Engine::compute_pred_next(
    const DecoderLayer& next_layer,
    const mx::array& h,
    const std::optional<std::pair<mx::array, mx::array>>& write,
    int top_m) {
    auto [m_next, _h, _inj] = run_hc(next_layer.mlp_hc, h, write);
    auto g_next = next_layer.gate.forward(m_next);
    auto part = mx::argpartition(g_next, -top_m, -1);
    mx::Shape start(part.ndim(), 0);
    start.back() = part.shape(-1) - top_m;
    auto topm_idx = mx::slice(part, start, part.shape());
    auto topm_vals = mx::take_along_axis(g_next, topm_idx, -1);
    auto order = mx::argsort(mx::negative(topm_vals), -1);
    return mx::take_along_axis(topm_idx, order, -1);
}

mx::array Qwen38Engine::run_slot_glu(
    int layer_idx,
    const mx::array& x,
    const mx::array& slot_inds,
    bool sorted_indices) {
    init_kernels();
    const auto& c = layers_[layer_idx].offload;
    auto x_exp = (x.ndim() == 5) ? x : mx::expand_dims(mx::expand_dims(x, -2), -2);
    auto x_up = mx::gather_qmm(
        x_exp, c.up_w, c.up_s, c.up_b, std::nullopt, slot_inds,
        true, c.group_size, c.bits, "affine", sorted_indices);
    auto x_gate = mx::gather_qmm(
        x_exp, c.gate_w, c.gate_s, c.gate_b, std::nullopt, slot_inds,
        true, c.group_size, c.bits, "affine", sorted_indices);
    auto act = swiglu_compiled_({x_gate, x_up})[0];
    auto out = mx::gather_qmm(
        act, c.down_w, c.down_s, c.down_b, std::nullopt, slot_inds,
        true, c.group_size, c.bits, "affine", sorted_indices);
    return mx::squeeze(out, -2);
}

mx::array Qwen38Engine::run_moe_prefill(
    int layer_idx,
    const mx::array& x,
    const mx::array& topk_inds,
    const mx::array& scores,
    const mx::array& shared_y) {
    init_kernels();
    auto& layer = layers_[layer_idx];
    if (layer.resident_moe) {
        int B = x.shape(0);
        int L = x.shape(1);
        int K = topk_inds.shape(-1);
        int D = x.shape(-1);
        auto x_exp = mx::expand_dims(mx::expand_dims(x, -2), -2);
        bool do_sort = (B * L * K) >= 64;
        mx::array y = x;
        if (do_sort) {
            // Exact match to _gather_sort / _scatter_unsort in switch_layers.py:55-76
            auto flat_idx = mx::reshape(topk_inds, {-1});
            auto order = mx::argsort(flat_idx);
            auto inv_order = mx::argsort(order);
            auto idx_sorted = mx::take(flat_idx, order, 0);
            auto x_flat = mx::reshape(x_exp, {B * L, 1, D});
            auto x_sorted = mx::take(x_flat, mx::floor_divide(order, mx::array(K, order.dtype())), 0);
            auto u = layer.res_up_proj.forward(x_sorted, idx_sorted, true);
            auto g = layer.res_gate_proj.forward(x_sorted, idx_sorted, true);
            auto a = swiglu_compiled_({g, u})[0];
            auto d = layer.res_down_proj.forward(a, idx_sorted, true);
            auto uns = mx::take(d, inv_order, 0);
            y = mx::squeeze(mx::reshape(uns, {B, L, K, 1, D}), -2);
        } else {
            auto u = layer.res_up_proj.forward(x_exp, topk_inds, false);
            auto g = layer.res_gate_proj.forward(x_exp, topk_inds, false);
            auto a = swiglu_compiled_({g, u})[0];
            y = mx::squeeze(layer.res_down_proj.forward(a, topk_inds, false), -2);
        }
        return mx::sum(y * mx::expand_dims(scores, -1), -2) + shared_y;
    }

    // Offloaded layer 1..47 prefill (exact match to CachedSwitchGLU.__call__ L > 1 in loader.py:680-843)
    int B = x.shape(0);
    int L = x.shape(1);
    int K = topk_inds.shape(-1);
    int D = x.shape(-1);
    int M = B * L * K;
    int cap = layer.offload.capacity;
    const auto& c = layer.offload;

    // Fetch current LRU state from C or P1IoScheduler (loader.py:706-718)
    std::vector<int32_t> lru_eids(512);
    std::vector<int32_t> lru_slots(512);
    int lru_cnt = 0;
    if (opts_.io_mode == "p1") {
        p1_sched_.get_lru_state(layer_idx, lru_eids.data(), lru_slots.data(), &lru_cnt);
    } else {
        c_layer_get_lru_state(layer_idx, lru_eids.data(), lru_slots.data(), &lru_cnt);
    }

    std::vector<int32_t> slot_to_eid(cap, -1);
    std::vector<int32_t> eid_to_slot(512, -1);
    std::vector<uint8_t> prior_cached(512, 0);
    for (int i = 0; i < lru_cnt; ++i) {
        int32_t e = lru_eids[i];
        int32_t s = lru_slots[i];
        if (s >= 0 && s < cap && e >= 0 && e < 512) {
            slot_to_eid[s] = e;
            eid_to_slot[e] = s;
            prior_cached[e] = 1;
        }
    }

    // Sort all M assignments by expert ID once on device (loader.py:722-727)
    auto flat_idx = mx::reshape(topk_inds, {-1});
    auto order = mx::argsort(flat_idx);
    auto inv_order = mx::argsort(order);
    auto idx_sorted = mx::take(flat_idx, order, 0);
    auto x_flat = mx::reshape(x, {B * L, 1, D});
    auto x_s = mx::take(x_flat, mx::floor_divide(order, mx::array(K, order.dtype())), 0);

    mx::eval({x_s, idx_sorted, inv_order});

    const uint32_t* idx_ptr = idx_sorted.data<uint32_t>();
    std::vector<int32_t> uniq_eids;
    std::vector<int32_t> first_pos;
    std::vector<int32_t> counts;
    for (int i = 0; i < M; ++i) {
        int32_t eid = static_cast<int32_t>(idx_ptr[i]);
        if (uniq_eids.empty() || uniq_eids.back() != eid) {
            uniq_eids.push_back(eid);
            first_pos.push_back(i);
            counts.push_back(1);
        } else {
            counts.back()++;
        }
    }
    int n_uniq = static_cast<int>(uniq_eids.size());

    // Heuristic matching loader.py:740-741
    int n_qmm_experts = (layer_idx < 28) ? 512 : n_uniq;
    bool use_sorted = (M >= 64) && ((M / std::max(1, n_qmm_experts)) >= 4);

    // Prefill hit/miss accounting (loader.py:744-750)
    uint64_t add_req = static_cast<uint64_t>(M);
    uint64_t add_hits = 0;
    uint64_t add_misses = 0;
    for (int u = 0; u < n_uniq; ++u) {
        int32_t eid = uniq_eids[u];
        int32_t cnt = counts[u];
        if (eid >= 0 && eid < 512 && prior_cached[eid]) {
            add_hits += static_cast<uint64_t>(cnt);
        } else {
            add_misses += 1;
            add_hits += static_cast<uint64_t>(cnt - 1);
        }
    }

    auto run_prefill_qproj = [&](const mx::array& w,
                                 const mx::array& s,
                                 const mx::array& b,
                                 const mx::array& x_in,
                                 const mx::array& idx_in) -> mx::array {
        if (use_sorted) {
            int orig_n = x_in.shape(0);
            mx::array x_use = x_in;
            mx::array idx_use = idx_in;
            if (orig_n < 16) {
                int pad_n = 16 - orig_n;
                mx::Shape x_start(x_in.ndim(), 0);
                x_start[0] = orig_n - 1;
                mx::Shape x_stop = x_in.shape();
                auto last_x = mx::slice(x_in, x_start, x_stop);
                std::vector<mx::array> x_cat = {x_in};
                for (int p = 0; p < pad_n; ++p) x_cat.push_back(last_x);
                x_use = mx::concatenate(x_cat, 0);

                auto last_i = mx::slice(idx_in, {orig_n - 1}, {orig_n});
                std::vector<mx::array> i_cat = {idx_in};
                for (int p = 0; p < pad_n; ++p) i_cat.push_back(last_i);
                idx_use = mx::concatenate(i_cat, 0);
            }
            int k_hdr = std::max(1, std::min(static_cast<int>(w.shape(0)), static_cast<int>(x_use.shape(0)) / 4));
            auto slice_hdr = [&](const mx::array& arr) {
                mx::Shape stop = arr.shape();
                stop[0] = k_hdr;
                return mx::slice(arr, mx::Shape(arr.ndim(), 0), stop);
            };
            auto out = mx::gather_qmm(
                x_use, slice_hdr(w), slice_hdr(s), slice_hdr(b),
                std::nullopt, idx_use, true, c.group_size, c.bits, "affine", true);
            if (orig_n < 16) {
                mx::Shape o_stop = out.shape();
                o_stop[0] = orig_n;
                out = mx::slice(out, mx::Shape(out.ndim(), 0), o_stop);
            }
            return out;
        }
        return mx::gather_qmm(
            x_in, w, s, b, std::nullopt, idx_in,
            true, c.group_size, c.bits, "affine", false);
    };

    std::vector<mx::array> y_parts;
    int last_chunk_len = 0;

    for (int u0 = 0; u0 < n_uniq; u0 += cap) {
        int u1 = std::min(n_uniq, u0 + cap);
        last_chunk_len = u1 - u0;
        int pos_start = first_pos[u0];
        int pos_end = first_pos[u1 - 1] + counts[u1 - 1];

        std::vector<int32_t> load_eids;
        std::vector<int32_t> load_slots;
        for (int s_idx = 0; s_idx < last_chunk_len; ++s_idx) {
            int32_t eid = uniq_eids[u0 + s_idx];
            if (slot_to_eid[s_idx] != eid) {
                int32_t old_eid = slot_to_eid[s_idx];
                if (old_eid >= 0 && old_eid < 512 && eid_to_slot[old_eid] == s_idx) {
                    eid_to_slot[old_eid] = -1;
                }
                int32_t prev_s = (eid >= 0 && eid < 512) ? eid_to_slot[eid] : -1;
                if (prev_s >= 0 && prev_s != s_idx) {
                    slot_to_eid[prev_s] = -1;
                }
                slot_to_eid[s_idx] = eid;
                if (eid >= 0 && eid < 512) {
                    eid_to_slot[eid] = s_idx;
                }
                load_eids.push_back(eid);
                load_slots.push_back(s_idx);
            }
        }

        if (!load_eids.empty()) {
            wait_slot_reader(layer_idx);
            if (opts_.io_mode == "p1") {
                p1_sched_.load_sync(
                    layer_idx,
                    load_eids.data(),
                    load_slots.data(),
                    static_cast<int>(load_eids.size()),
                    true);
            } else {
                c_layer_load_sync(
                    layer_idx,
                    load_eids.data(),
                    load_slots.data(),
                    static_cast<int>(load_eids.size()),
                    1);
            }
        }

        int m_chunk = pos_end - pos_start;
        std::vector<uint32_t> local_idx_host;
        local_idx_host.reserve(m_chunk);
        for (int u = u0; u < u1; ++u) {
            uint32_t s_local = static_cast<uint32_t>(u - u0);
            int32_t cnt = counts[u];
            for (int c_i = 0; c_i < cnt; ++c_i) {
                local_idx_host.push_back(s_local);
            }
        }
        auto local_idx = mx::array(local_idx_host.data(), {m_chunk}, mx::uint32);
        auto xc = mx::slice(x_s, {pos_start, 0, 0}, {pos_end, 1, D});

        auto x_up = run_prefill_qproj(c.up_w, c.up_s, c.up_b, xc, local_idx);
        auto x_gate = run_prefill_qproj(c.gate_w, c.gate_s, c.gate_b, xc, local_idx);
        auto act = swiglu_compiled_({x_gate, x_up})[0];
        auto xo = run_prefill_qproj(c.down_w, c.down_s, c.down_b, act, local_idx);

        if (n_uniq > cap) {
            mx::eval({xo});
        }
        y_parts.push_back(xo);
    }

    // Rebuild LRU order after prefill: untouched surviving slots (oldest) first,
    // then the last chunk's active slots [0 .. last_chunk_len - 1] (most recent)
    std::vector<int32_t> new_lru_eids;
    std::vector<int32_t> new_lru_slots;
    for (int s_idx = last_chunk_len; s_idx < cap; ++s_idx) {
        if (slot_to_eid[s_idx] >= 0) {
            new_lru_eids.push_back(slot_to_eid[s_idx]);
            new_lru_slots.push_back(s_idx);
        }
    }
    for (int s_idx = 0; s_idx < last_chunk_len; ++s_idx) {
        if (slot_to_eid[s_idx] >= 0) {
            new_lru_eids.push_back(slot_to_eid[s_idx]);
            new_lru_slots.push_back(s_idx);
        }
    }
    if (opts_.io_mode == "p1") {
        p1_sched_.set_lru_state(
            layer_idx,
            new_lru_eids.data(),
            new_lru_slots.data(),
            static_cast<int>(new_lru_eids.size()),
            add_req, add_hits, add_misses);
    } else {
        c_layer_set_lru_state(
            layer_idx,
            new_lru_eids.data(),
            new_lru_slots.data(),
            static_cast<int>(new_lru_eids.size()),
            add_req, add_hits, add_misses);
    }

    auto y_all = (y_parts.size() == 1) ? y_parts[0] : mx::concatenate(y_parts, 0);
    last_slot_reader_[layer_idx] = y_all;
    auto y_unsorted = mx::take(y_all, inv_order, 0);
    auto y = mx::squeeze(mx::reshape(y_unsorted, {B, L, K, 1, D}), -2);
    return mx::sum(y * mx::expand_dims(scores, -1), -2) + shared_y;
}

} // namespace qwen38



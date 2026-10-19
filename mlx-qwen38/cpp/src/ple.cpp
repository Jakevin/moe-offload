#include "model.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <sys/mman.h>
#include <unistd.h>

namespace qwen38 {

MmapFile::~MmapFile() {
    if (addr != nullptr && addr != MAP_FAILED && size > 0) {
        munmap(addr, size);
    }
    if (fd >= 0) {
        close(fd);
    }
}

static uint64_t splitmix64(uint64_t value) {
    constexpr uint64_t kGamma = 0x9E3779B97F4A7C15ULL;
    constexpr uint64_t kM1 = 0xBF58476D1CE4E5B9ULL;
    constexpr uint64_t kM2 = 0x94D049BB133111EBULL;
    value += kGamma;
    value = (value ^ (value >> 30)) * kM1;
    value = (value ^ (value >> 27)) * kM2;
    return value ^ (value >> 31);
}

static std::vector<int64_t> build_layer_multipliers(
    int64_t unigram_vocab_size,
    int ngram_size,
    int ple_layer_index,
    int64_t seed) {
    constexpr uint64_t kGamma = 0x9E3779B97F4A7C15ULL;
    constexpr int64_t kPrime1 = 10007;
    int64_t max_long = (1ULL << 63) - 1;
    int64_t multiplier_max = max_long / std::max<int64_t>(unigram_vocab_size, 1);
    uint64_t half_bound = static_cast<uint64_t>(std::max<int64_t>(1, multiplier_max / 2));
    uint64_t base_seed = static_cast<uint64_t>(seed + kPrime1 * ple_layer_index);
    std::vector<int64_t> mults(ngram_size);
    for (int idx = 0; idx < ngram_size; ++idx) {
        uint64_t val = base_seed + kGamma * static_cast<uint64_t>(idx + 1);
        mults[idx] = static_cast<int64_t>(2ULL * (splitmix64(val) % half_bound) + 1ULL);
    }
    return mults;
}

static bool is_prime_int64(int64_t v) {
    if (v < 2) return false;
    if (v % 2 == 0) return v == 2;
    int64_t limit = static_cast<int64_t>(std::sqrt(static_cast<double>(v)));
    for (int64_t d = 3; d <= limit; d += 2) {
        if (v % d == 0) return false;
    }
    return true;
}

static int64_t find_nth_prime_after(int64_t start, int count) {
    int64_t p = start;
    for (int i = 0; i < count; ++i) {
        p += 1;
        while (!is_prime_int64(p)) {
            p += 1;
        }
    }
    return p;
}

void init_ple_metadata(PleLayer& ple, int unigram_vocab_size, int ngram_vocab_base, int divisor, int seed) {
    ple.multipliers_cpu = build_layer_multipliers(unigram_vocab_size, ple.ngram_size, 0, seed);
    ple.moduli_cpu.resize(ple.ngram_heads);
    std::vector<int64_t> head_offsets(ple.ngram_heads);
    int64_t total_vocab = 0;
    for (int h = 0; h < ple.ngram_heads; ++h) {
        int64_t sz = find_nth_prime_after(ngram_vocab_base - 1, h + 1);
        ple.moduli_cpu[h] = sz;
        head_offsets[h] = total_vocab;
        total_vocab += sz;
    }
    int64_t padded_vocab = ((total_vocab + divisor - 1) / divisor) * divisor;
    int64_t base = padded_vocab / ple.split_ngram_parts;
    int64_t rem = padded_vocab % ple.split_ngram_parts;
    ple.shard_offsets.resize(ple.split_ngram_parts + 1, 0);
    for (int s = 0; s < ple.split_ngram_parts; ++s) {
        int64_t sz = base + (s < rem ? 1 : 0);
        ple.shard_offsets[s + 1] = ple.shard_offsets[s] + sz;
    }
}

static const char* kPleDepthwiseConvSource = R"(
    const uint c = thread_position_in_grid.x;
    const uint t = thread_position_in_grid.y;
    const uint b = thread_position_in_grid.z;
    if (c >= C) return;
    const size_t t_out = threads_per_grid.y;
    const size_t t_in = t_out + (K - 1) * DIL;
    const device T* src = x + ((size_t)b * t_in + t) * C + c;
    float acc = 0.0f;
    for (int i = 0; i < K; ++i) {
        acc += static_cast<float>(src[(size_t)i * DIL * C]) *
            static_cast<float>(w[c * K + i]);
    }
    y[((size_t)b * t_out + t) * C + c] = static_cast<T>(acc);
)";

static mx::fast::CustomKernelFunction& get_ple_conv_kernel() {
    static mx::fast::CustomKernelFunction kernel = mx::fast::metal_kernel(
        "omlx_qwen4_depthwise_conv1d",
        {"x", "w"},
        {"y"},
        kPleDepthwiseConvSource,
        "",
        true,
        false);
    return kernel;
}

mx::array Qwen38Engine::run_ple(
    const mx::array& hidden_states,
    const std::vector<int32_t>& input_ids_cpu) {
    init_kernels();
    int batch = hidden_states.shape(0);
    int seq_len = static_cast<int>(input_ids_cpu.size());
    if (batch != 1 || seq_len <= 0) {
        throw std::runtime_error("run_ple requires batch == 1 and seq_len >= 1");
    }

    if (ple_.history_cpu.size() != static_cast<size_t>(ple_.context_len)) {
        ple_.history_cpu.assign(ple_.context_len, ple_.eos_token_id);
    }

    int hist_len = ple_.context_len + seq_len;
    std::vector<int64_t> token_history(hist_len);
    for (int i = 0; i < ple_.context_len; ++i) {
        token_history[i] = ple_.history_cpu[i];
    }
    for (int i = 0; i < seq_len; ++i) {
        token_history[ple_.context_len + i] = static_cast<int64_t>(input_ids_cpu[i]);
    }
    for (int i = 0; i < ple_.context_len; ++i) {
        ple_.history_cpu[i] = token_history[hist_len - ple_.context_len + i];
    }
    if (seq_len > 1 || ple_.prefill_context_cpu.empty()) {
        ple_.prefill_context_cpu = ple_.history_cpu;
    }

    // Compute _shift_right_ignore_eos for shift = 0, 1, 2
    std::vector<std::vector<int64_t>> shifted(ple_.ngram_size, std::vector<int64_t>(hist_len));
    shifted[0] = token_history;
    for (int shift = 1; shift < ple_.ngram_size; ++shift) {
        int64_t prev_eos = -1;
        for (int pos = 0; pos < hist_len; ++pos) {
            int64_t seg_start = prev_eos + 1;
            int64_t pos_in_seg = pos - seg_start;
            int64_t src_pos = pos - shift;
            bool valid = (pos_in_seg >= shift) && (src_pos >= 0);
            shifted[shift][pos] = valid ? token_history[src_pos] : ple_.eos_token_id;
            if (token_history[pos] == ple_.eos_token_id) {
                prev_eos = pos;
            }
        }
    }

    // Compute head_offsets
    std::vector<int64_t> head_offsets(ple_.ngram_heads);
    int64_t cum_off = 0;
    for (int h = 0; h < ple_.ngram_heads; ++h) {
        head_offsets[h] = cum_off;
        cum_off += ple_.moduli_cpu[h];
    }

    int total_rows = seq_len * ple_.ngram_heads;
    std::vector<int64_t> host_indices(total_rows);
    for (int t = 0; t < seq_len; ++t) {
        int pos = ple_.context_len + t;
        for (int ngram = 2; ngram <= ple_.ngram_size; ++ngram) {
            int start_h = (ngram - 2) * ple_.heads_per_ngram;
            int end_h = start_h + ple_.heads_per_ngram;
            int64_t mixed = shifted[0][pos] * ple_.multipliers_cpu[0];
            for (int p = 1; p < ngram; ++p) {
                mixed ^= (shifted[p][pos] * ple_.multipliers_cpu[p]);
            }
            for (int h = start_h; h < end_h; ++h) {
                int64_t mod = ple_.moduli_cpu[h];
                int64_t rem = mixed % mod;
                if (rem < 0) rem += mod;
                host_indices[t * ple_.ngram_heads + h] = rem + head_offsets[h];
            }
        }
    }

    // Gather rows from mmap shards
    int w_cols = static_cast<int>(ple_.shards[0].weight.cols); // 20
    int s_cols = static_cast<int>(ple_.shards[0].scales.cols); // 5
    int bits = ple_.shards[0].bits;
    int group_size = ple_.shards[0].group_size;

    std::vector<uint32_t> w_buf(total_rows * w_cols);
    std::vector<uint16_t> s_buf(total_rows * s_cols);
    std::vector<uint16_t> b_buf(total_rows * s_cols);

    for (int r = 0; r < total_rows; ++r) {
        int64_t gidx = host_indices[r];
        auto it = std::upper_bound(ple_.shard_offsets.begin(), ple_.shard_offsets.end(), gidx);
        int shard_idx = static_cast<int>(std::distance(ple_.shard_offsets.begin(), it)) - 1;
        shard_idx = std::clamp(shard_idx, 0, ple_.split_ngram_parts - 1);
        int64_t local_row = gidx - ple_.shard_offsets[shard_idx];
        const auto& sh = ple_.shards[shard_idx];
        std::memcpy(
            w_buf.data() + r * w_cols,
            sh.weight.data_ptr + local_row * sh.weight.row_bytes,
            w_cols * sizeof(uint32_t));
        std::memcpy(
            s_buf.data() + r * s_cols,
            sh.scales.data_ptr + local_row * sh.scales.row_bytes,
            s_cols * sizeof(uint16_t));
        std::memcpy(
            b_buf.data() + r * s_cols,
            sh.biases.data_ptr + local_row * sh.biases.row_bytes,
            s_cols * sizeof(uint16_t));
    }

    auto w_arr = mx::array(w_buf.data(), {total_rows, w_cols}, mx::uint32);
    auto s_arr = mx::array(reinterpret_cast<const mx::bfloat16_t*>(s_buf.data()), {total_rows, s_cols}, mx::bfloat16);
    auto b_arr = mx::array(reinterpret_cast<const mx::bfloat16_t*>(b_buf.data()), {total_rows, s_cols}, mx::bfloat16);

    auto deq = mx::dequantize(w_arr, s_arr, b_arr, group_size, bits, "affine");
    auto values_emb = mx::astype(deq, mx::bfloat16) * ple_.weight_scale;
    auto embeddings = mx::reshape(values_emb, {batch, seq_len, ple_.ple_embed_dim});

    auto keys_proj = ple_.key_proj.forward(embeddings);
    auto keys = mx::reshape(
        qwen4_rms_norm(keys_proj, ple_.norm_key_weight, ple_.rms_norm_eps, ple_.hidden_size),
        {batch, seq_len, ple_.hc_count, ple_.hidden_size});
    auto values = ple_.value_proj.forward(embeddings);
    auto queries = mx::reshape(
        qwen4_rms_norm(hidden_states, ple_.norm_query_weight, ple_.rms_norm_eps, ple_.hidden_size),
        {batch, seq_len, ple_.hc_count, ple_.hidden_size});

    float sqrt_h = std::sqrt(static_cast<float>(ple_.hidden_size));
    auto gate = mx::divide(mx::sum(keys * queries, -1, true), mx::array(sqrt_h, keys.dtype()));
    auto eps_gate = mx::array(1e-6f, gate.dtype());
    gate = mx::sign(gate) * mx::sqrt(mx::maximum(mx::abs(gate), eps_gate));
    auto gated_values = mx::sigmoid(gate) * mx::expand_dims(values, -2);
    gated_values = mx::reshape(gated_values, hidden_states.shape());

    auto normed = qwen4_rms_norm(gated_values, ple_.norm_conv_weight, ple_.rms_norm_eps, ple_.hidden_size);

    if (!ple_.has_conv_state) {
        ple_.conv_state = mx::zeros({batch, ple_.short_conv_state_len, ple_.hc_hidden_size}, normed.dtype());
        ple_.has_conv_state = true;
    }
    auto conv_input = mx::concatenate({ple_.conv_state, normed}, 1);
    int total_conv_len = conv_input.shape(1);
    ple_.conv_state = mx::slice(
        conv_input,
        {0, total_conv_len - ple_.short_conv_state_len, 0},
        {batch, total_conv_len, ple_.hc_hidden_size});

    int channels = ple_.hc_hidden_size;
    int taps = ple_.conv_kernel_size;
    int dil = ple_.conv_dilation;
    auto& conv_k = get_ple_conv_kernel();
    auto conv_out = conv_k(
        {conv_input, ple_.short_conv_weight},
        {{batch, seq_len, channels}},
        {normed.dtype()},
        {channels, seq_len, batch},
        {std::min(256, channels), 1, 1},
        {{"T", normed.dtype()}, {"C", channels}, {"K", taps}, {"DIL", dil}},
        std::nullopt,
        false,
        {})[0];

    auto conv_silu = silu_compiled_({conv_out})[0];
    return gated_values + conv_silu;
}

} // namespace qwen38

#pragma once

#include <atomic>
#include <cstdint>
#include <functional>
#include <list>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

#include <mlx/mlx.h>
#include <mlx/fast.h>
#include <mlx/compile.h>

#include "budget.h"
#include "io_sched.h"

namespace mx = mlx::core;

namespace qwen38 {

// RoPE inverse frequencies, bit-identical to mlx_vlm rope_utils.compute_inv_freq (see loader.cpp).
mx::array make_rope_inv_freq(int rotary_dim = 64, float theta = 10000000.0f);

// Quantized or dense linear projection
struct QLinear {
    mx::array weight{mx::zeros({0})};
    std::optional<mx::array> scales;
    std::optional<mx::array> biases;
    std::optional<mx::array> bias;
    int group_size = 64;
    int bits = 4;
    bool is_quantized = false;
    bool reshape_2d = false;

    mx::array forward(const mx::array& x) const;
};

// Prefill logits exactly as Python: lm_head over ALL L rows (M = L quantized_matmul), then keep row L-1.
// (Slicing first gives M = 1 / a different kernel and differed by up to 0.094 on row 0.)
mx::array lm_head_last_row(const QLinear& head, const mx::array& final_h);

// SwitchGLU expert bank (Layer 0 resident or synthetic)
struct QSwitchLinear {
    mx::array weight{mx::zeros({0})};
    mx::array scales{mx::zeros({0})};
    mx::array biases{mx::zeros({0})};
    int group_size = 64;
    int bits = 2;

    mx::array forward(const mx::array& x, const mx::array& indices, bool sorted_indices = false) const;
};

// K50 v3/v4 ExpertPack + StackedSwitchMlp + ResidentCache
struct K50PackRecord {
    int expert = -1;
    std::string kind; // "tern" | "2t" | "2" | "3" | "4" | "6" | "8"
    std::string unit; // "hot" | "warm" | "mid" | "cold" | "dense"
    std::vector<int> shape;
    int row_bytes = 0;
    uint64_t code_off = 0;
    uint64_t code_nbytes = 0;
    uint64_t scale_off = 0;
    uint64_t scale_nbytes = 0;
    uint64_t bias_off = 0;
    uint64_t bias_nbytes = 0;
    int group_size = 64;
};

struct K50MaterializedWeight {
    std::string kind;
    int bits = 2;
    int group_size = 64;
    mx::array weight{mx::zeros({0})};
    mx::array scales{mx::zeros({0})};
    mx::array biases{mx::zeros({0})};
    size_t nbytes = 0;
};

class ExpertPackCpp {
public:
    int layer_idx = 0;
    std::string layer_dir;
    int fd_codes = -1;
    int fd_scales = -1;
    int fd_bias = -1;
    std::unordered_map<std::string, K50PackRecord> dense;
    std::unordered_map<std::string, K50PackRecord> experts; // key: "proj:eid"

    explicit ExpertPackCpp(const std::string& dir, int l_idx = 0);
    ~ExpertPackCpp();

    K50MaterializedWeight materialize(const K50PackRecord& rec);
    K50MaterializedWeight materialize_as_affine2(const K50PackRecord& rec);
    QLinear load_dense_qlinear(const std::string& name);
};

struct K50CacheKey {
    uint8_t layer_id = 0;
    uint8_t proj_id = 0; // 0: gate_proj, 1: up_proj, 2: down_proj
    uint16_t expert_id = 0;
    bool operator==(const K50CacheKey& o) const {
        return layer_id == o.layer_id && proj_id == o.proj_id && expert_id == o.expert_id;
    }
};

struct K50CacheKeyHash {
    size_t operator()(const K50CacheKey& k) const {
        return (static_cast<size_t>(k.layer_id) << 24) ^
               (static_cast<size_t>(k.proj_id) << 16) ^
               static_cast<size_t>(k.expert_id);
    }
};

class K50ResidentCache {
public:
    size_t budget_bytes = 0;
    size_t pinned_bytes = 0;
    size_t lru_bytes = 0;
    uint64_t hot_hits = 0;
    uint64_t lru_hits = 0;
    uint64_t misses = 0;
    uint64_t evictions = 0;

    bool in_prefill = false;
    uint64_t prefill_hot_hits = 0;
    uint64_t prefill_lru_hits = 0;
    uint64_t prefill_misses = 0;

    std::list<std::pair<K50CacheKey, K50MaterializedWeight>> lru_list;
    std::unordered_map<
        K50CacheKey,
        std::list<std::pair<K50CacheKey, K50MaterializedWeight>>::iterator,
        K50CacheKeyHash> lru_map;

    void set_budget(size_t total_budget_bytes, size_t pinned_hot_bytes);
    void clear_lru();
    void reset_stats();
    void record_hot_hit(uint64_t count = 1);
    const K50MaterializedWeight& get_or_load(
        uint8_t layer_id,
        uint8_t proj_id,
        uint16_t expert_id,
        ExpertPackCpp& pack,
        const K50PackRecord& rec);
};

struct K50StackedGroup {
    std::string kind; // "2t" | "2" | "3" | "4" | "6" | "8"
    int bits = 2;
    int group_size = 64;
    mx::array weights{mx::zeros({0})};
    mx::array scales{mx::zeros({0})};
    mx::array biases{mx::zeros({0})};
    std::vector<int32_t> e2l;
    size_t nbytes = 0;
};

struct StackedSwitchMlp {
    int layer_idx = 0;
    int num_experts = 512;
    std::shared_ptr<ExpertPackCpp> pack;
    std::map<std::string, std::map<std::string, K50StackedGroup>> groups;
    std::map<std::string, std::vector<bool>> is_stacked;
    size_t stacked_bytes = 0;

    void build(
        int l_idx,
        std::shared_ptr<ExpertPackCpp> p,
        int n_experts,
        const std::string& mode = "hot");

    mx::array proj_all(
        const std::string& proj_name,
        uint8_t proj_id,
        const mx::array& x,
        const std::vector<int32_t>& route,
        bool expand_x,
        K50ResidentCache* cache);

    mx::array forward(
        const mx::array& x,
        const mx::array& indices,
        K50ResidentCache* cache);
};

// Hyper-Connection Gated Residual (with or without block_inject_weight)
struct HyperConnection {
    int hc_count = 4;
    int hidden_size = 2560;
    int hc_lowrank = 320;
    float eps = 1e-6f;
    bool use_combine = true;
    mx::array norm_weight{mx::zeros({0})};
    mx::array norm_scale_f32{mx::zeros({0})};
    mx::array eps_arr{mx::zeros({0})};
    QLinear down;
    QLinear up;
    QLinear inject; // valid when use_combine == true
};

// GatedDeltaNet (Linear Attention, 36 layers)
struct LinearAttnLayer {
    int num_k_heads = 16;
    int num_v_heads = 48;
    int head_k_dim = 128;
    int head_v_dim = 128;
    int key_dim = 2048;
    int value_dim = 6144;
    int conv_dim = 10240;
    int conv_kernel_size = 4;
    float rms_norm_eps = 1e-6f;
    QLinear in_proj_qkv;
    QLinear in_proj_z;
    QLinear in_proj_b;
    QLinear in_proj_a;
    mx::array conv1d_weight{mx::zeros({0})};
    mx::array conv_weight_decode_f32{mx::zeros({0})};
    mx::array dt_bias{mx::zeros({0})};
    mx::array A_log{mx::zeros({0})};
    mx::array norm_weight{mx::zeros({0})};
    QLinear out_proj;
};

// Full Attention + QSA Indexer (12 layers: 3, 7, ..., 47)
struct FullAttnLayer {
    int num_heads = 24;
    int num_kv_heads = 2;
    int head_dim = 256;
    float scale = 0.0625f;
    float rms_norm_eps = 1e-6f;
    QLinear q_proj;
    QLinear k_proj;
    QLinear v_proj;
    QLinear o_proj;
    mx::array q_norm_scale{mx::zeros({0})}; // 1.0f + weight.astype(float32)
    mx::array k_norm_scale{mx::zeros({0})};
    mx::array rope_inv_freq{mx::zeros({0})};
    int indexer_n_heads = 4;
    int indexer_kv_heads = 1;
    int indexer_head_dim = 128;
    int indexer_compress_ratio = 4;
    int indexer_budget = 2048;
    int indexer_block_topk = 512;
    QLinear index_qk_proj;
    mx::array index_q_norm_scale{mx::zeros({0})};
    mx::array index_k_norm_scale{mx::zeros({0})};
    mx::array index_rope_inv_freq{mx::zeros({0})};
};

// Per-Layer Slot Cache for Offloaded MoE (Layers 1..47)
struct OffloadLayerCache {
    int fd = -1;
    int capacity = 64;
    uint64_t base_offsets[9]{};
    uint64_t row_bytes[9]{};
    uint64_t base_ptrs[9]{};
    mx::array gate_w{mx::zeros({0})};
    mx::array gate_s{mx::zeros({0})};
    mx::array gate_b{mx::zeros({0})};
    mx::array up_w{mx::zeros({0})};
    mx::array up_s{mx::zeros({0})};
    mx::array up_b{mx::zeros({0})};
    mx::array down_w{mx::zeros({0})};
    mx::array down_s{mx::zeros({0})};
    mx::array down_b{mx::zeros({0})};
    mx::array slot_inds_buf{mx::zeros({0})}; // shape (1, 1, 10) uint32
    int group_size = 64;
    int bits = 2;
};

//PLE Shard mmap spec for Layer 1
struct PleTensorSlice {
    const uint8_t* data_ptr = nullptr;
    int64_t rows = 0;
    int64_t cols = 0;
    int64_t row_bytes = 0;
    std::string dtype;
};

struct PleShardSpec {
    PleTensorSlice weight;
    PleTensorSlice scales;
    PleTensorSlice biases;
    int bits = 4;
    int group_size = 32;
};

struct MmapFile {
    int fd = -1;
    void* addr = nullptr;
    size_t size = 0;
    uint64_t data_start = 0;
    ~MmapFile();
};

// PLE Layer (Layer 1 only)
struct PleLayer {
    bool active = false;
    int ngram_size = 3;
    int context_len = 2;
    int heads_per_ngram = 8;
    int ngram_heads = 16;
    int ple_embed_dim = 2560;
    int hidden_size = 2560;
    int hc_count = 4;
    int hc_hidden_size = 10240;
    int conv_kernel_size = 4;
    int conv_dilation = 3;
    int short_conv_state_len = 9;
    int split_ngram_parts = 128;
    int64_t eos_token_id = 248044;
    float rms_norm_eps = 1e-6f;

    std::vector<int64_t> multipliers_cpu; // length 3
    std::vector<int64_t> moduli_cpu;      // length 16 (2 ngrams * 8 heads)
    std::vector<int64_t> head_offsets_cpu; // length 16 (cumulative sum of moduli)
    std::vector<int64_t> shard_offsets;   // length 129
    std::vector<PleShardSpec> shards;     // length 128
    mx::array weight_scale{mx::zeros({0})}; // scalar bfloat16/float32

    QLinear key_proj;
    QLinear value_proj;
    mx::array norm_query_weight{mx::zeros({0})}; // grouped RMSNorm (group_size=2560)
    mx::array norm_key_weight{mx::zeros({0})};   // RMSNorm (dim=2560)
    mx::array norm_conv_weight{mx::zeros({0})};  // grouped RMSNorm (group_size=2560)
    mx::array short_conv_weight{mx::zeros({0})}; // shape (10240, 4, 1) bfloat16
    mx::array gate_scale{mx::zeros({0})};        // shape (1, 1, 4, 1) bfloat16

    // State across prefill and decode
    std::vector<int64_t> history_cpu; // last 2 tokens
    std::vector<int64_t> prefill_context_cpu; // last 2 tokens from prefill (saved as layer1.s3)
    mx::array conv_state{mx::zeros({0})}; // shape (1, 9, 10240) bfloat16
    bool has_conv_state = false;
};

struct DecoderLayer {
    int layer_idx = 0;
    bool is_linear = true;
    HyperConnection attn_hc;
    HyperConnection mlp_hc;
    LinearAttnLayer lin_attn;
    FullAttnLayer full_attn;
    QLinear gate;
    QLinear shared_expert_gate;
    QLinear shared_gate_proj;
    QLinear shared_up_proj;
    QLinear shared_down_proj;
    // Layer 0 resident experts:
    bool resident_moe = false;
    QSwitchLinear res_gate_proj;
    QSwitchLinear res_up_proj;
    QSwitchLinear res_down_proj;
    // Layers 1..47 offloaded experts:
    OffloadLayerCache offload;
    // K50 v3 experts (L00: 512, L01..L47: 256):
    bool k50_moe = false;
    int num_experts = 512;
    std::shared_ptr<ExpertPackCpp> k50_pack;
    StackedSwitchMlp k50_switch;
};

// Runtime caches for 48 layers
struct LayerExecCache {
    bool is_linear = true;
    // GDN state:
    mx::array conv_state{mx::zeros({0})}; // (1, 3, 10240) bfloat16
    mx::array ssm_state{mx::zeros({0})};  // (1, 48, 128, 128) float32
    bool gdn_initialized = false;

    // Full Attention QSAKVCache state:
    mx::array keys{mx::zeros({0})};   // (1, 2, cap, 256) bfloat16
    mx::array values{mx::zeros({0})}; // (1, 2, cap, 256) bfloat16
    int offset = 0;
    mx::array index_keys{mx::zeros({0})};         // (1, cap, 128) bfloat16
    mx::array index_position_ids{mx::zeros({0})}; // (1, cap) or (3, 1, cap) int32
    int index_offset = 0;
    mx::array pooled_index_keys{mx::zeros({0})};  // (1, block_cap, 128) bfloat16
    int pooled_index_offset = 0;
};

struct EngineOptions {
    std::string model_dir;  // --model DIR, or $QWEN38_MODEL
    std::string budget_mode = "fixed"; // "fixed" | "auto"
    double budget_pct = 20.0;
    std::string alloc_mode = "even";   // "even" | "proportional" | "auto" | "uniform"
    std::string prefetch_mode = "1";   // "0"/"off" | "1"/"on"/"med" | "min" | "agg"
    int top_m = 12;
    int cap = 3;                       // prefetch_cap (PREFETCH_MAX_CAP = 3)
    int slot_cap = 0;                  // explicit per-layer LRU slot capacity override if > 0
    bool native_sched = false;
    std::string sync_mode = "layer";   // "layer" | "token"
    bool compile = true;
    bool cold = false;
    int repeat = 1;
    std::string dump_dir;
    std::vector<int> dump_steps;
    std::string state_out;
    bool no_stop = false;
    std::string io_mode = "exact";      // "exact" | "p1"
    int io_threads = 6;                 // default 6 for --io p1
    std::string budget_log;             // --budget-log FILE (default logs/budget_<pid>.jsonl)
    double force_miss_rate = 0.0;       // --force-miss-rate R (0..1)
    uint64_t force_miss_seed = 42;      // --force-miss-seed S
    // K50 v3 options:
    double resident_gb = 15.6;
    std::string stack_mode = "hot";     // "hot" | "all"
};

struct RunMetrics {
    std::vector<int32_t> tokens;
    double load_s = 0.0;
    double prefill_s = 0.0;
    double ttft_s = 0.0;
    double decode_s = 0.0;
    double tok_per_s = 0.0;
    double ms_per_tok = 0.0;
    double p50_ms = 0.0;
    double p90_ms = 0.0;
    uint64_t io_pread_bytes = 0;
    double read_mib_per_tok = 0.0;
    uint64_t demand_misses = 0;
    uint64_t lru_hits = 0;
    uint64_t prefetch_hits = 0;
    uint64_t decode_requests = 0;
    double cache_hit_rate = 0.0;
    std::vector<int> per_layer_cap;
    uint64_t slot_buffer_bytes = 0;
    uint64_t peak_rss_bytes = 0;
    uint64_t mlx_active_bytes = 0;
    uint64_t mlx_peak_bytes = 0;
    uint64_t mlx_cache_bytes = 0;
    // Dynamic budget telemetry:
    int budget_adjustments = 0;
    int min_total_slots = 0;
    int max_total_slots = 0;
    int max_pressure_level = 0;
    uint64_t max_swap_used_bytes = 0;
    // P1 IO scheduler & forced-miss telemetry:
    uint64_t urgent_reads = 0;
    uint64_t prefetch_issued = 0;
    uint64_t prefetch_cancelled = 0;
    uint64_t prefetch_promoted = 0;
    uint64_t prefetch_used = 0;
    uint64_t prefetch_wasted = 0;
    uint64_t pin_waits = 0;
    double io_wait_ms = 0.0;
    uint64_t forced_invalidations = 0;
    std::vector<P1LayerStats> p1_layer_stats;
    // K50 v3 resident cache & expert hit/miss telemetry:
    bool is_k50_v3 = false;
    uint64_t k50_pinned_bytes = 0;
    uint64_t k50_lru_budget_bytes = 0;
    uint64_t k50_lru_bytes = 0;
    uint64_t k50_lru_entries = 0;
    uint64_t k50_hot_hits = 0;
    uint64_t k50_lru_hits = 0;
    uint64_t k50_misses = 0;
    uint64_t k50_evictions = 0;
    uint64_t k50_prefill_hot_hits = 0;
    uint64_t k50_prefill_lru_hits = 0;
    uint64_t k50_prefill_misses = 0;
};

class Qwen38Engine {
public:
    explicit Qwen38Engine(const EngineOptions& opts);
    ~Qwen38Engine();

    // Validate all model weights against safetensors headers without loading tensor data
    bool check_weights() const;

    // Load weights, PLE mmap shards, and initialize expert_io slot buffers
    void load();

    // Reset KV/GDN/PLE caches and optionally reset expert LRU caches (--cold)
    void reset_caches(bool reset_expert_lru);

    // Save cache states to safetensors file (--state-out)
    void save_state(const std::string& path) const;

    // Run prefill on prompt_ids and decode n_tokens greedy tokens
    RunMetrics generate(
        const std::vector<int32_t>& prompt_ids,
        int n_tokens,
        const std::string& logits_out_path = "");

    // Standalone unit-test entry points (can be called without loading the 48-layer model):
    void init_kernels();
    static mx::array qwen4_rms_norm(
        const mx::array& x,
        const mx::array& weight,
        float eps = 1e-6f,
        int group_size = 0);
    static mx::array qwen4_rms_norm_scale(
        const mx::array& x,
        const mx::array& scale_fp32,
        float eps = 1e-6f);
    static mx::array qwen4_rms_norm_gated(
        const mx::array& x,
        const mx::array& gate,
        const mx::array& weight,
        float eps = 1e-6f,
        bool sigmoid_gate = true);

    std::tuple<mx::array, mx::array, mx::array> run_hc(
        const HyperConnection& hc,
        const mx::array& h,
        const std::optional<std::pair<mx::array, mx::array>>& write = std::nullopt);

    mx::array apply_mrope_one(
        const mx::array& x,
        const mx::array& inv_freq,
        const mx::array& position_ids);

    struct GdnIntermediates {
        mx::array mixed_qkv{mx::zeros({0})};
        mx::array z{mx::zeros({0})};
        mx::array a{mx::zeros({0})};
        mx::array b{mx::zeros({0})};
        mx::array conv_out{mx::zeros({0})};
        mx::array q{mx::zeros({0})};
        mx::array k{mx::zeros({0})};
        mx::array v{mx::zeros({0})};
        mx::array g{mx::zeros({0})};
        mx::array beta{mx::zeros({0})};
        mx::array kernel_y{mx::zeros({0})};
        mx::array state{mx::zeros({0})};
    };

    mx::array run_linear_attn(
        const LinearAttnLayer& la,
        const mx::array& x,
        LayerExecCache& cache,
        GdnIntermediates* dbg = nullptr);

    std::pair<mx::array, mx::array> gated_delta_ops(
        const mx::array& q,
        const mx::array& k,
        const mx::array& v,
        const mx::array& g,
        const mx::array& beta,
        const mx::array& state);

    std::pair<mx::array, mx::array> gated_delta_chunked(
        const mx::array& q,
        const mx::array& k,
        const mx::array& v,
        const mx::array& g,
        const mx::array& beta,
        const mx::array& state,
        int C = 64);

    mx::array run_full_attn(
        const FullAttnLayer& fa,
        const mx::array& x,
        LayerExecCache& cache,
        const std::optional<mx::array>& position_ids = std::nullopt);

    mx::array run_ple(
        const mx::array& hidden_states,
        const std::vector<int32_t>& input_ids_cpu);

    std::pair<mx::array, mx::array> route_topk(
        const QLinear& gate,
        const mx::array& x,
        int top_k = 10);

    mx::array compute_pred_next(
        const DecoderLayer& next_layer,
        const mx::array& h,
        const std::optional<std::pair<mx::array, mx::array>>& write,
        int top_m);

    // Resize slot capacity for a layer (preserving MRU experts)
    void resize_layer_capacity(int layer_idx, int new_cap);

    // Wait for any in-flight GPU kernel reading layer_idx's slot buffers 
    void wait_slot_reader(int layer_idx);

    mx::array run_slot_glu(int layer_idx, const mx::array& x, const mx::array& slot_inds, bool sorted_indices = false);
    mx::array run_moe_prefill(int layer_idx, const mx::array& x, const mx::array& topk_inds, const mx::array& scores, const mx::array& shared_y);
    mx::array run_swiglu(const mx::array& gate, const mx::array& x) {
        init_kernels();
        return swiglu_compiled_({gate, x})[0];
    }

    PleLayer& ple() { return ple_; }
    std::vector<DecoderLayer>& layers() { return layers_; }

private:
    EngineOptions opts_;
    bool kernels_initialized_ = false;
    bool loaded_ = false;

    // Compiled functions & custom Metal kernels
    std::function<std::vector<mx::array>(const std::vector<mx::array>&)> swiglu_compiled_;
    std::function<std::vector<mx::array>(const std::vector<mx::array>&)> depthwise_conv_compiled_;
    std::function<std::vector<mx::array>(const std::vector<mx::array>&)> silu_compiled_;
    std::function<std::vector<mx::array>(const std::vector<mx::array>&)> compute_g_beta_compiled_;
    std::function<std::vector<mx::array>(const std::vector<mx::array>&)> gated_delta_step_ops_compiled_;
    std::function<std::vector<mx::array>(const std::vector<mx::array>&)> hc_act_compiled_;
    std::function<std::vector<mx::array>(const std::vector<mx::array>&)> hc_tail_compiled_;

    mx::fast::CustomKernelFunction hc_ndn_kernel_;
    mx::fast::CustomKernelFunction hc_ndnw_kernel_;
    mx::fast::CustomKernelFunction hc_u2_kernel_;
    mx::fast::CustomKernelFunction hc_n_kernel_;
    mx::fast::CustomKernelFunction hc_np_kernel_;
    mx::fast::CustomKernelFunction hc_wn_kernel_;
    mx::fast::CustomKernelFunction hc_wnd_kernel_;
    mx::fast::CustomKernelFunction hc_d_kernel_;
    mx::fast::CustomKernelFunction hc_u_kernel_;
    mx::fast::CustomKernelFunction hc_ti_kernel_;
    mx::fast::CustomKernelFunction gated_delta_kernel_;
    mx::fast::CustomKernelFunction mrope_2d_kernel_;
    mx::fast::CustomKernelFunction mrope_3d_kernel_;

    // Model weights
    bool is_k50_v3_ = false;
    K50ResidentCache k50_cache_;
    QLinear embed_tokens_; // QuantizedEmbedding (weight, scales, biases, 4-bit group_size=32 or 8-bit group_size=64)
    std::vector<DecoderLayer> layers_; // size 48
    PleLayer ple_;                     // Layer 1 PLE
    HyperConnection final_mixer_;      // use_combine = false
    QLinear lm_head_;

    void load_k50_v3();

    // Open file descriptors & mmaps
    std::unordered_map<std::string, int> open_fds_;
    std::unordered_map<std::string, std::shared_ptr<MmapFile>> ple_mmaps_;

    // Per-layer execution caches
    std::vector<LayerExecCache> exec_caches_;

    // Per-layer last GPU array reading slot buffers (race safety)
    std::vector<std::optional<mx::array>> last_slot_reader_;

    // P1 IO scheduler (--io p1) and forced-miss state
    P1IoScheduler p1_sched_;
    uint64_t forced_invalidations_ = 0;

    // Dynamic memory budget state 
    std::vector<int> current_caps_; // size 48 (index 0 unused)
    std::atomic<bool> stop_monitor_{false};
    std::atomic<int> latest_pressure_level_{1};
    std::atomic<uint64_t> latest_swap_used_{0};
    std::atomic<uint64_t> latest_avail_bytes_{0};
    std::atomic<uint64_t> latest_compressed_bytes_{0};
    std::thread monitor_thread_;
    uint64_t initial_swap_bytes_ = 0;
    uint64_t prev_compressed_bytes_ = 0;
    uint64_t recommended_working_set_bytes_ = 0;
    uint64_t total_ram_bytes_ = 64ULL << 30;
    int last_shrink_step_ = -1000;
    int last_none_log_step_ = -1000;
    std::string budget_log_path_;
    int budget_adjustments_ = 0;
    int min_total_slots_ = 0;
    int max_total_slots_ = 0;
    int max_pressure_level_ = 1;
    uint64_t max_swap_used_ = 0;

    void start_memory_monitor();
    void stop_memory_monitor();
    void check_and_adjust_budget(int step_idx);
    void append_budget_log(const BudgetReadings& r, const BudgetDecision& d);

    mx::array prefill(const std::vector<int32_t>& prompt_ids);
    mx::array decode_step(int32_t token_id, int step_idx);
};

} // namespace qwen38

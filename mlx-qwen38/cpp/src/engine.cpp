#include <atomic>
#include "model.hpp"
#include "expert_io.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#ifdef __APPLE__
#include <mach/mach.h>
#include <mach/mach_host.h>
#include <mach/task.h>
#include <sys/sysctl.h>
#endif

namespace qwen38 {

static mx::array embed_lookup(const QLinear& emb, const mx::array& input_ids) {
    if (!emb.is_quantized) {
        return mx::take(emb.weight, input_ids, 0);
    }
    auto w = mx::take(emb.weight, input_ids, 0);
    auto s = mx::take(*emb.scales, input_ids, 0);
    std::optional<mx::array> b = std::nullopt;
    if (emb.biases.has_value()) {
        b = mx::take(*emb.biases, input_ids, 0);
    }
    return mx::dequantize(w, s, b, emb.group_size, emb.bits, "affine");
}

static mx::array hc_write(const mx::array& hyper_input, const mx::array& branch, const mx::array& weights) {
    auto inj = mx::expand_dims(branch, -2) * mx::expand_dims(weights, -1);
    return hyper_input + mx::reshape(inj, hyper_input.shape());
}

static int resolve_prefetch_cap(const EngineOptions& opts) {
    if (opts.prefetch_mode == "0" || opts.prefetch_mode == "off" || opts.prefetch_mode == "false") {
        return 0;
    }
    if (opts.prefetch_mode == "min") return 1;
    if (opts.prefetch_mode == "agg") return std::max(6, opts.cap);
    return (opts.cap > 0) ? opts.cap : 3;
}

static uint64_t get_recommended_working_set_bytes() {
    if (mx::metal::is_available()) {
        // In MLX 0.32.2 C++, mx::metal::device_info() is declared in metal.h but moved to
        // mx::device_info(mx::Device(mx::Device::gpu)) / mx::gpu::device_info(0) in libmlx.dylib.
        const auto& info = mx::device_info(mx::Device(mx::Device::gpu));
        auto it = info.find("max_recommended_working_set_size");
        if (it != info.end() && std::holds_alternative<size_t>(it->second)) {
            return static_cast<uint64_t>(std::get<size_t>(it->second));
        }
    }
    return 0;
}

int force_invalidate_exact_layer(
    int layer_idx,
    double rate,
    std::mt19937_64& rng,
    uint8_t* first_tensor_base_ptr,
    uint64_t first_tensor_row_bytes) {
    if (rate <= 0.0) return 0;
    c_layer_wait_demand(layer_idx);
    c_layer_wait_prefetch(layer_idx);

    int32_t eids[512];
    int32_t slots[512];
    int count = 0;
    c_layer_get_lru_state(layer_idx, eids, slots, &count);
    if (count <= 0) return 0;

    std::uniform_real_distribution<double> dist(0.0, 1.0);
    int32_t kept_eids[512];
    int32_t kept_slots[512];
    int kept = 0;
    int invalidated = 0;

    for (int i = 0; i < count; ++i) {
        if (rate >= 1.0 || dist(rng) < rate) {
            invalidated++;
            if (first_tensor_base_ptr != nullptr && first_tensor_row_bytes > 0 && slots[i] >= 0) {
                size_t scrub = std::min<size_t>(64, static_cast<size_t>(first_tensor_row_bytes));
                std::memset(
                    first_tensor_base_ptr + static_cast<uint64_t>(slots[i]) * first_tensor_row_bytes,
                    0xA5,
                    scrub);
            }
        } else {
            kept_eids[kept] = eids[i];
            kept_slots[kept] = slots[i];
            kept++;
        }
    }

    if (invalidated > 0) {
        c_layer_set_lru_state(layer_idx, kept_eids, kept_slots, kept, 0, 0, 0);
    }
    return invalidated;
}

#ifdef __APPLE__
static uint64_t get_phys_footprint_bytes() {
    task_vm_info_data_t vm_info{};
    mach_msg_type_number_t count = TASK_VM_INFO_COUNT;
    if (task_info(mach_task_self(), TASK_VM_INFO, reinterpret_cast<task_info_t>(&vm_info), &count) == KERN_SUCCESS) {
        return static_cast<uint64_t>(vm_info.phys_footprint);
    }
    return 0;
}

static void query_macos_memory(
    uint64_t& avail_bytes,
    uint64_t& swap_used_bytes,
    uint64_t& compressed_bytes,
    int& pressure_level) {
    avail_bytes = 0;
    swap_used_bytes = 0;
    compressed_bytes = 0;
    pressure_level = 1;

    mach_port_t host = mach_host_self();
    vm_size_t page_size = 16384;
    host_page_size(host, &page_size);

    vm_statistics64_data_t vm_stat{};
    mach_msg_type_number_t count = HOST_VM_INFO64_COUNT;
    if (host_statistics64(host, HOST_VM_INFO64, reinterpret_cast<host_info64_t>(&vm_stat), &count) == KERN_SUCCESS) {
        // Task 4 A.2: available = free + purgeable + clean file-backed cache ONLY (with 10% safety margin on external_page_count).
        // Never count inactive_count (which includes inactive anonymous pages).
        avail_bytes = compute_clean_avail_bytes(
            static_cast<uint64_t>(vm_stat.free_count),
            static_cast<uint64_t>(vm_stat.purgeable_count),
            static_cast<uint64_t>(vm_stat.external_page_count),
            static_cast<uint64_t>(page_size));
        compressed_bytes = static_cast<uint64_t>(vm_stat.compressor_page_count) * static_cast<uint64_t>(page_size);
    }

    int plevel = 1;
    size_t plen = sizeof(plevel);
    if (sysctlbyname("kern.memorystatus_vm_pressure_level", &plevel, &plen, nullptr, 0) == 0) {
        pressure_level = plevel;
    }

    struct xsw_usage swap_u{};
    size_t slen = sizeof(swap_u);
    if (sysctlbyname("vm.swapusage", &swap_u, &slen, nullptr, 0) == 0) {
        swap_used_bytes = swap_u.xsu_used;
    }
}
#else
static uint64_t get_phys_footprint_bytes() { return 0; }
static void query_macos_memory(uint64_t& avail_bytes, uint64_t& swap_used_bytes, uint64_t& compressed_bytes, int& pressure_level) {
    avail_bytes = 8ULL << 30;
    swap_used_bytes = 0;
    compressed_bytes = 0;
    pressure_level = 1;
}
#endif

// Effective process limit (bytes of phys_footprint), frozen before the first load():
// min(19.5e9, floor(0.8 * clean avail at start)). The self-guard and budget=auto use the same value.
static std::atomic<uint64_t> g_proc_limit_bytes{kSelfGuardFootprintBytes};
static std::atomic<uint64_t> g_peak_footprint_bytes{0};

static void check_process_memory_guard(uint64_t initial_swap_bytes) {
    uint64_t rss = get_phys_footprint_bytes();
    uint64_t prev = g_peak_footprint_bytes.load();
    while (rss > prev && !g_peak_footprint_bytes.compare_exchange_weak(prev, rss)) {}
    uint64_t avail = 0, swap_b = 0, comp_b = 0;
    int plevel = 1;
    query_macos_memory(avail, swap_b, comp_b, plevel);
    const uint64_t limit = g_proc_limit_bytes.load();
    constexpr uint64_t kMaxSwapGrowthBytes = 8ULL << 30;    // 8 GiB
    if (rss > limit) {
        std::fprintf(stderr, "[FATAL] Self-guard abort: phys_footprint=%llu bytes exceeds proc_limit=%llu bytes\n",
                     static_cast<unsigned long long>(rss), static_cast<unsigned long long>(limit));
        std::fflush(stderr);
        std::_Exit(2);
    }
    if (swap_b > initial_swap_bytes && (swap_b - initial_swap_bytes) > kMaxSwapGrowthBytes) {
        std::fprintf(stderr, "[FATAL] Self-guard abort: swap grew by %llu bytes (> 8 GiB limit)\n",
                     static_cast<unsigned long long>(swap_b - initial_swap_bytes));
        std::fflush(stderr);
        std::_Exit(2);
    }
}

Qwen38Engine::Qwen38Engine(const EngineOptions& opts) : opts_(opts) {
    mx::set_cache_limit(2ULL << 30);
    last_slot_reader_.resize(48, std::nullopt);
    current_caps_.assign(48, 0);
}

Qwen38Engine::~Qwen38Engine() {
    stop_memory_monitor();
    if (loaded_) {
        for (int l = 1; l < 48; ++l) {
            wait_slot_reader(l);
            if (opts_.io_mode == "p1") {
                p1_sched_.wait_layer_all(l);
            } else {
                c_layer_wait_demand(l);
                c_layer_wait_prefetch(l);
            }
        }
    }
    p1_sched_.shutdown();
    for (auto& kv : open_fds_) {
        if (kv.second >= 0) {
            close(kv.second);
        }
    }
    open_fds_.clear();
    ple_mmaps_.clear();
}

void Qwen38Engine::wait_slot_reader(int layer_idx) {
    if (layer_idx >= 0 && layer_idx < static_cast<int>(last_slot_reader_.size())) {
        if (last_slot_reader_[layer_idx].has_value()) {
            mx::eval({*last_slot_reader_[layer_idx]});
            last_slot_reader_[layer_idx].reset();
        }
        if (opts_.io_mode == "p1") {
            p1_sched_.unpin_layer_slots(layer_idx);
        }
    }
}

void Qwen38Engine::reset_caches(bool reset_expert_lru) {
    for (int l = 1; l < static_cast<int>(layers_.size()); ++l) {
        wait_slot_reader(l);
        if (opts_.io_mode == "p1") {
            p1_sched_.wait_layer_all(l);
            p1_sched_.reset_stats(l);
            if (reset_expert_lru) {
                p1_sched_.reset_cache(l);
            }
        } else {
            c_layer_wait_demand(l);
            c_layer_wait_prefetch(l);
            c_layer_reset_stats(l);
            if (reset_expert_lru) {
                c_layer_reset_cache(l);
            }
        }
    }
    forced_invalidations_ = 0;
    for (size_t l = 0; l < exec_caches_.size(); ++l) {
        bool is_lin = exec_caches_[l].is_linear;
        exec_caches_[l] = LayerExecCache{};
        exec_caches_[l].is_linear = is_lin;
    }
    ple_.history_cpu.clear();
    ple_.prefill_context_cpu.clear();
    ple_.conv_state = mx::zeros({0});
    ple_.has_conv_state = false;
    mx::clear_cache();
}

void Qwen38Engine::save_state(const std::string& path) const {
    if (path.empty()) return;
    std::filesystem::create_directories(std::filesystem::path(path).parent_path());
    std::unordered_map<std::string, mx::array> states;
    std::vector<mx::array> to_eval;

    for (size_t i = 0; i < exec_caches_.size(); ++i) {
        const auto& c = exec_caches_[i];
        std::string pfx = "layer" + std::to_string(i) + ".s";
        if (c.is_linear) {
            if (c.gdn_initialized && c.conv_state.size() > 0 && c.ssm_state.size() > 0) {
                states.emplace(pfx + "0", c.conv_state);
                states.emplace(pfx + "1", c.ssm_state);
                to_eval.push_back(c.conv_state);
                to_eval.push_back(c.ssm_state);
            }
            if (i == 1 && ple_.active && ple_.has_conv_state) {
                states.emplace(pfx + "2", ple_.conv_state);
                to_eval.push_back(ple_.conv_state);
                // Python cache[1].state[3] = PLE history = last context_len (2) FED tokens, updated on every
                // decode step (prefill_context_cpu is only set by prefill -> stale after decode).
                if (!ple_.history_cpu.empty()) {
                    auto ctx = mx::array(
                        ple_.history_cpu.data(),
                        {1, static_cast<int>(ple_.history_cpu.size())},
                        mx::int64);
                    states.emplace(pfx + "3", ctx);
                    to_eval.push_back(ctx);
                }
            }
        } else {
            if (c.offset > 0 && c.keys.size() > 0 && c.values.size() > 0) {
                int H = c.keys.shape(1);
                int D = c.keys.shape(3);
                auto k_valid = mx::contiguous(mx::slice(c.keys, {0, 0, 0, 0}, {1, H, c.offset, D}));
                auto v_valid = mx::contiguous(mx::slice(c.values, {0, 0, 0, 0}, {1, H, c.offset, D}));
                states.emplace(pfx + "0", k_valid);
                states.emplace(pfx + "1", v_valid);
                to_eval.push_back(k_valid);
                to_eval.push_back(v_valid);
            }
            if (c.index_offset > 0 && c.index_keys.size() > 0 && c.index_position_ids.size() > 0) {
                int ID = c.index_keys.shape(2);
                auto ik_valid = mx::contiguous(mx::slice(c.index_keys, {0, 0, 0}, {1, c.index_offset, ID}));
                mx::array ip_valid = (c.index_position_ids.ndim() == 3)
                    ? mx::contiguous(mx::slice(c.index_position_ids, {0, 0, 0}, {c.index_position_ids.shape(0), 1, c.index_offset}))
                    : mx::contiguous(mx::slice(c.index_position_ids, {0, 0}, {1, c.index_offset}));
                states.emplace(pfx + "2", ik_valid);
                states.emplace(pfx + "3", ip_valid);
                to_eval.push_back(ik_valid);
                to_eval.push_back(ip_valid);
            }
        }
    }
    if (!to_eval.empty()) {
        mx::eval(to_eval);
        mx::save_safetensors(path, states);
    }
}

static bool should_dump_step(const EngineOptions& opts, int step) {
    if (opts.dump_dir.empty()) return false;
    if (opts.dump_steps.empty()) return true;
    return std::find(opts.dump_steps.begin(), opts.dump_steps.end(), step) != opts.dump_steps.end();
}

static void save_step_dumps(
    const EngineOptions& opts,
    int step,
    std::unordered_map<std::string, mx::array>& step_tensors) {
    if (step_tensors.empty()) return;
    std::filesystem::create_directories(opts.dump_dir);
    std::vector<mx::array> eval_list;
    eval_list.reserve(step_tensors.size());
    for (auto& kv : step_tensors) {
        eval_list.push_back(kv.second);
    }
    mx::eval(eval_list);
    std::string out_path = opts.dump_dir + "/step_" + std::to_string(step) + ".safetensors";
    mx::save_safetensors(out_path, step_tensors);
    step_tensors.clear();
}

mx::array Qwen38Engine::prefill(const std::vector<int32_t>& prompt_ids) {
    int L = static_cast<int>(prompt_ids.size());
    auto input_ids = mx::array(prompt_ids.data(), {1, L}, mx::int32);
    auto x0 = embed_lookup(embed_tokens_, input_ids);
    auto h = mx::tile(x0, {1, 1, 4}); // shape (1, L, 10240)
    std::optional<std::pair<mx::array, mx::array>> write = std::nullopt;

    bool do_dump = should_dump_step(opts_, 0);
    std::unordered_map<std::string, mx::array> dump_map;

    for (int l = 0; l < 48; ++l) {
        auto& layer = layers_[l];
        if (l == 1 && ple_.active) {
            if (write.has_value()) {
                h = hc_write(h, write->first, write->second);
                write = std::nullopt;
            }
            auto ple_out = run_ple(h, prompt_ids);
            h = h + ple_out;
        }

        auto [a_in, h1, a_inj] = run_hc(layer.attn_hc, h, write);
        h = h1;
        mx::array a_out = layer.is_linear
            ? run_linear_attn(layer.lin_attn, a_in, exec_caches_[l])
            : run_full_attn(layer.full_attn, a_in, exec_caches_[l], std::nullopt);
        write = std::make_pair(a_out, a_inj);

        auto [m_in, h2, m_inj] = run_hc(layer.mlp_hc, h, write);
        h = h2;

        auto [inds, scores] = route_topk(layer.gate, m_in, 10);
        auto sh_g = layer.shared_gate_proj.forward(m_in);
        auto sh_u = layer.shared_up_proj.forward(m_in);
        auto sh_act = swiglu_compiled_({sh_g, sh_u})[0];
        auto sh_out = layer.shared_down_proj.forward(sh_act);
        auto sh_gate = mx::sigmoid(layer.shared_expert_gate.forward(m_in));
        auto shared_y = sh_gate * sh_out;

        auto moe_out = run_moe_prefill(l, m_in, inds, scores, shared_y);
        write = std::make_pair(moe_out, m_inj);

        if (do_dump) {
            std::string lp = "layer" + std::to_string(l) + ".";
            dump_map.emplace(lp + "mixed_attn", a_in);
            dump_map.emplace(lp + "branch_attn", a_out);
            dump_map.emplace(lp + "mixed_mlp", m_in);
            dump_map.emplace(lp + "inds", inds);
            dump_map.emplace(lp + "scores", scores);
            dump_map.emplace(lp + "shared_y", shared_y);
            dump_map.emplace(lp + "branch_mlp", moe_out);
            dump_map.emplace(lp + "h_pre_moe", h);
        }
    }

    auto [final_h, _w, _i] = run_hc(final_mixer_, h, write);
    auto logits = lm_head_last_row(lm_head_, final_h);
    if (do_dump) {
        dump_map.emplace("logits", mx::astype(logits, mx::float32));
        save_step_dumps(opts_, 0, dump_map);
    }
    return logits;
}

mx::array Qwen38Engine::decode_step(int32_t token_id, int step_idx) {
    const bool use_p1 = (opts_.io_mode == "p1");
    if (!use_p1) {
        c_begin_decode_token(step_idx);
    }
    std::vector<int32_t> tok_vec = {token_id};
    auto input_ids = mx::array(tok_vec.data(), {1, 1}, mx::int32);
    auto x0 = embed_lookup(embed_tokens_, input_ids);
    auto h = mx::tile(x0, {1, 1, 4}); // shape (1, 1, 10240)
    std::optional<std::pair<mx::array, mx::array>> write = std::nullopt;

    int prefetch_cap = resolve_prefetch_cap(opts_);
    int top_m = opts_.top_m;
    bool do_prefetch = (prefetch_cap > 0 && top_m > 0);

    bool do_dump = should_dump_step(opts_, step_idx + 1);
    std::unordered_map<std::string, mx::array> dump_map;

    for (int l = 0; l < 48; ++l) {
        auto& layer = layers_[l];
        if (l == 1 && ple_.active) {
            if (write.has_value()) {
                h = hc_write(h, write->first, write->second);
                write = std::nullopt;
            }
            auto ple_out = run_ple(h, tok_vec);
            h = h + ple_out;
        }

        auto [a_in, h1, a_inj] = run_hc(layer.attn_hc, h, write);
        h = h1;
        mx::array a_out = layer.is_linear
            ? run_linear_attn(layer.lin_attn, a_in, exec_caches_[l])
            : run_full_attn(layer.full_attn, a_in, exec_caches_[l], std::nullopt);
        write = std::make_pair(a_out, a_inj);

        auto [m_in, h2, m_inj] = run_hc(layer.mlp_hc, h, write);
        h = h2;

        auto [inds, scores] = route_topk(layer.gate, m_in, 10);
        auto sh_g = layer.shared_gate_proj.forward(m_in);
        auto sh_u = layer.shared_up_proj.forward(m_in);
        auto sh_act = swiglu_compiled_({sh_g, sh_u})[0];
        auto sh_out = layer.shared_down_proj.forward(sh_act);
        auto sh_gate = mx::sigmoid(layer.shared_expert_gate.forward(m_in));
        auto shared_y = sh_gate * sh_out;

        mx::array moe_out = shared_y;

        if (layer.resident_moe) {
            auto x_exp = mx::expand_dims(mx::expand_dims(m_in, -2), -2);
            auto u = layer.res_up_proj.forward(x_exp, inds, false);
            auto g = layer.res_gate_proj.forward(x_exp, inds, false);
            auto a = swiglu_compiled_({g, u})[0];
            auto y = mx::squeeze(layer.res_down_proj.forward(a, inds, false), -2);
            moe_out = mx::sum(y * mx::expand_dims(scores, -1), -2) + shared_y;

            if (do_prefetch && l + 1 < 48) {
                auto h_post_0 = hc_write(h, moe_out, m_inj);
                auto pred_next = compute_pred_next(layers_[l + 1], h_post_0, std::nullopt, top_m);
                mx::eval({h_post_0, pred_next});
                const uint32_t* p_ptr = pred_next.data<uint32_t>();
                int32_t pred_host[32];
                for (int k = 0; k < top_m; ++k) pred_host[k] = static_cast<int32_t>(p_ptr[k]);
                if (use_p1) {
                    p1_sched_.start_prefetch(1, pred_host, top_m, prefetch_cap);
                } else {
                    wait_slot_reader(l + 1);
                    if (opts_.native_sched) {
                        c_layer_decode_step_sched(0, nullptr, 0, pred_host, top_m, 1, prefetch_cap, 0);
                    } else if (top_m == 10) {
                        c_layer_start_prefetch10(
                            1, prefetch_cap,
                            pred_host[0], pred_host[1], pred_host[2], pred_host[3], pred_host[4],
                            pred_host[5], pred_host[6], pred_host[7], pred_host[8], pred_host[9]);
                    } else if (top_m == 12) {
                        c_layer_start_prefetch12(
                            1, prefetch_cap,
                            pred_host[0], pred_host[1], pred_host[2], pred_host[3],
                            pred_host[4], pred_host[5], pred_host[6], pred_host[7],
                            pred_host[8], pred_host[9], pred_host[10], pred_host[11]);
                    } else {
                        c_layer_start_prefetch(1, pred_host, top_m, prefetch_cap);
                    }
                }
            } else {
                mx::async_eval({moe_out});
            }
            write = std::make_pair(moe_out, m_inj);
        } else if (use_p1) {
            std::optional<mx::array> pred_next_opt = std::nullopt;
            if (do_prefetch && l + 1 < 48) {
                pred_next_opt = compute_pred_next(layers_[l + 1], h, std::nullopt, top_m);
                mx::eval({inds, *pred_next_opt});
            } else {
                mx::eval({inds});
            }

            wait_slot_reader(l);
            if (opts_.force_miss_rate > 0.0) {
                uint64_t mix_seed = opts_.force_miss_seed
                                  ^ (static_cast<uint64_t>(step_idx + 1) * 0x9E3779B97F4A7C15ULL)
                                  ^ (static_cast<uint64_t>(l) * 0xBF58476D1CE4E5B9ULL);
                std::mt19937_64 rng(mix_seed);
                forced_invalidations_ += static_cast<uint64_t>(
                    p1_sched_.force_invalidate(l, opts_.force_miss_rate, rng));
            }

            const uint32_t* e_ptr = inds.data<uint32_t>();
            int n_pending = p1_sched_.step_decode(l, e_ptr, 10);
            if (n_pending > 0) {
                mx::async_eval({scores, shared_y});
                p1_sched_.wait_layer_urgent(l);
            }

            auto y = run_slot_glu(l, m_in, layer.offload.slot_inds_buf, false);
            moe_out = mx::sum(y * mx::expand_dims(scores, -1), -2) + shared_y;
            last_slot_reader_[l] = moe_out;
            mx::async_eval({moe_out});

            if (pred_next_opt.has_value()) {
                const uint32_t* p_ptr = pred_next_opt->data<uint32_t>();
                int32_t pred_host[32];
                for (int k = 0; k < top_m; ++k) pred_host[k] = static_cast<int32_t>(p_ptr[k]);
                p1_sched_.start_prefetch(l + 1, pred_host, top_m, prefetch_cap);
            }
            write = std::make_pair(moe_out, m_inj);
        } else if (!opts_.native_sched) {
            // Default loader.py schedule (--native-sched 0, lines 2250-2285):
            std::optional<mx::array> pred_next_opt = std::nullopt;
            if (do_prefetch && l + 1 < 48) {
                pred_next_opt = compute_pred_next(layers_[l + 1], h, std::nullopt, top_m);
                mx::eval({inds, *pred_next_opt});
            } else {
                mx::eval({inds});
            }

            wait_slot_reader(l);
            if (opts_.force_miss_rate > 0.0) {
                uint64_t mix_seed = opts_.force_miss_seed
                                  ^ (static_cast<uint64_t>(step_idx + 1) * 0x9E3779B97F4A7C15ULL)
                                  ^ (static_cast<uint64_t>(l) * 0xBF58476D1CE4E5B9ULL);
                std::mt19937_64 rng(mix_seed);
                forced_invalidations_ += static_cast<uint64_t>(force_invalidate_exact_layer(
                    l,
                    opts_.force_miss_rate,
                    rng,
                    reinterpret_cast<uint8_t*>(layer.offload.base_ptrs[0]),
                    layer.offload.row_bytes[0]));
            }
            const uint32_t* e_ptr = inds.data<uint32_t>();
            int n_miss = c_layer_step_decode10_async(
                l, e_ptr[0], e_ptr[1], e_ptr[2], e_ptr[3], e_ptr[4],
                e_ptr[5], e_ptr[6], e_ptr[7], e_ptr[8], e_ptr[9]);
            if (n_miss > 0) {
                mx::async_eval({scores, shared_y});
            }

            auto y = run_slot_glu(l, m_in, layer.offload.slot_inds_buf, false);
            moe_out = mx::sum(y * mx::expand_dims(scores, -1), -2) + shared_y;

            if (n_miss > 0) {
                c_layer_wait_prefetch(l);
            }
            last_slot_reader_[l] = moe_out;
            mx::async_eval({moe_out});

            if (pred_next_opt.has_value()) {
                wait_slot_reader(l + 1);
                const uint32_t* p_ptr = pred_next_opt->data<uint32_t>();
                int32_t pred_host[32];
                for (int k = 0; k < top_m; ++k) pred_host[k] = static_cast<int32_t>(p_ptr[k]);
                if (top_m == 10) {
                    c_layer_start_prefetch10(
                        l + 1, prefetch_cap,
                        pred_host[0], pred_host[1], pred_host[2], pred_host[3], pred_host[4],
                        pred_host[5], pred_host[6], pred_host[7], pred_host[8], pred_host[9]);
                } else if (top_m == 12) {
                    c_layer_start_prefetch12(
                        l + 1, prefetch_cap,
                        pred_host[0], pred_host[1], pred_host[2], pred_host[3],
                        pred_host[4], pred_host[5], pred_host[6], pred_host[7],
                        pred_host[8], pred_host[9], pred_host[10], pred_host[11]);
                } else {
                    c_layer_start_prefetch(l + 1, pred_host, top_m, prefetch_cap);
                }
            }
            write = std::make_pair(moe_out, m_inj);
        } else {
            // Native C scheduler path (--native-sched 1, loader.py lines 2062-2117)
            std::optional<mx::array> pred_next_opt = std::nullopt;
            if (do_prefetch && l + 1 < 48) {
                pred_next_opt = compute_pred_next(layers_[l + 1], h, std::nullopt, top_m);
                mx::async_eval({inds, *pred_next_opt, scores, shared_y, h, m_inj});
                mx::eval({inds, *pred_next_opt});
            } else {
                mx::async_eval({inds, scores, shared_y, h, m_inj});
                mx::eval({inds});
            }

            wait_slot_reader(l);
            if (pred_next_opt.has_value()) {
                wait_slot_reader(l + 1);
            }
            if (opts_.force_miss_rate > 0.0) {
                uint64_t mix_seed = opts_.force_miss_seed
                                  ^ (static_cast<uint64_t>(step_idx + 1) * 0x9E3779B97F4A7C15ULL)
                                  ^ (static_cast<uint64_t>(l) * 0xBF58476D1CE4E5B9ULL);
                std::mt19937_64 rng(mix_seed);
                forced_invalidations_ += static_cast<uint64_t>(force_invalidate_exact_layer(
                    l,
                    opts_.force_miss_rate,
                    rng,
                    reinterpret_cast<uint8_t*>(layer.offload.base_ptrs[0]),
                    layer.offload.row_bytes[0]));
            }

            const uint32_t* e_ptr = inds.data<uint32_t>();
            int32_t eids_i32[10];
            for (int k = 0; k < 10; ++k) eids_i32[k] = static_cast<int32_t>(e_ptr[k]);

            int32_t pred_i32[32];
            int n_pred = 0;
            int next_l = -1;
            int flags = 0;
            if (pred_next_opt.has_value()) {
                const uint32_t* p_ptr = pred_next_opt->data<uint32_t>();
                for (int k = 0; k < top_m; ++k) pred_i32[k] = static_cast<int32_t>(p_ptr[k]);
                n_pred = top_m;
                next_l = l + 1;
            } else {
                flags |= SCHED_FLAG_NO_PREFETCH;
            }

            int n_miss = c_layer_decode_step_sched(
                l, eids_i32, 10,
                n_pred > 0 ? pred_i32 : nullptr,
                n_pred, next_l, prefetch_cap, flags);

            auto y = run_slot_glu(l, m_in, layer.offload.slot_inds_buf, false);
            moe_out = mx::sum(y * mx::expand_dims(scores, -1), -2) + shared_y;
            if (n_miss > 0) {
                c_layer_wait_demand(l);
            }
            last_slot_reader_[l] = moe_out;
            mx::async_eval({moe_out});
            write = std::make_pair(moe_out, m_inj);
        }

        if (do_dump) {
            std::string lp = "layer" + std::to_string(l) + ".";
            dump_map.emplace(lp + "mixed_attn", a_in);
            dump_map.emplace(lp + "branch_attn", a_out);
            dump_map.emplace(lp + "mixed_mlp", m_in);
            dump_map.emplace(lp + "inds", inds);
            dump_map.emplace(lp + "scores", scores);
            dump_map.emplace(lp + "shared_y", shared_y);
            dump_map.emplace(lp + "branch_mlp", moe_out);
            dump_map.emplace(lp + "h_pre_moe", h);
        }
    }

    auto [final_h, _w, _i] = run_hc(final_mixer_, h, write);
    auto logits = lm_head_.forward(final_h);

    if (!use_p1) {
        uint64_t tok_stats[10]{};
        c_end_decode_token(tok_stats);
    }

    if (do_dump) {
        dump_map.emplace("logits", mx::astype(logits, mx::float32));
        save_step_dumps(opts_, step_idx + 1, dump_map);
    }
    return logits;
}

void Qwen38Engine::append_budget_log(const BudgetReadings& r, const BudgetDecision& d) {
    if (budget_log_path_.empty()) {
        budget_log_path_ = !opts_.budget_log.empty()
            ? opts_.budget_log
            : ("logs/budget_" + std::to_string(getpid()) + ".jsonl");
    }
    auto parent = std::filesystem::path(budget_log_path_).parent_path();
    if (!parent.empty()) {
        std::filesystem::create_directories(parent);
    }
    std::ofstream ofs(budget_log_path_, std::ios::app);
    if (ofs) {
        ofs << format_budget_log_jsonl(r, d) << "\n";
    }
}

void Qwen38Engine::start_memory_monitor() {
    if (opts_.budget_mode != "auto") return;
    stop_monitor_.store(false);
    uint64_t avail = 0, swap_b = 0, comp_b = 0;
    int plevel = 1;
    query_macos_memory(avail, swap_b, comp_b, plevel);
    latest_avail_bytes_.store(avail);
    latest_swap_used_.store(swap_b);
    latest_compressed_bytes_.store(comp_b);
    latest_pressure_level_.store(plevel);
    initial_swap_bytes_ = swap_b;
    prev_compressed_bytes_ = comp_b;
    last_shrink_step_ = -1000;
    last_none_log_step_ = -1000;
    max_pressure_level_ = std::max(max_pressure_level_, plevel);
    max_swap_used_ = std::max(max_swap_used_, swap_b);

    uint64_t total_ram = 64ULL << 30;
#ifdef __APPLE__
    size_t rlen = sizeof(total_ram);
    sysctlbyname("hw.memsize", &total_ram, &rlen, nullptr, 0);
#endif
    total_ram_bytes_ = total_ram;
    recommended_working_set_bytes_ = get_recommended_working_set_bytes();

    uint64_t cur_slot_bytes = 0;
    for (int l = 1; l < 48; ++l) cur_slot_bytes += static_cast<uint64_t>(current_caps_[l]) * kBudgetExpertBytes;
    uint64_t footprint = get_phys_footprint_bytes();
    uint64_t non_cache_footprint = (footprint > cur_slot_bytes) ? (footprint - cur_slot_bytes) : (6ULL << 30);

    BudgetReadings r{};
    r.is_init = true;
    r.step_idx = -1;
    r.last_shrink_step = last_shrink_step_;
    r.avail_bytes = avail;
    r.swap_used_bytes = swap_b;
    r.initial_swap_bytes = initial_swap_bytes_;
    r.compressed_bytes = comp_b;
    r.prev_compressed_bytes = prev_compressed_bytes_;
    r.pressure_level = plevel;
    r.total_ram_bytes = total_ram_bytes_;
    r.non_cache_footprint_bytes = non_cache_footprint;
    r.recommended_working_set_bytes = recommended_working_set_bytes_;
    r.max_slot_cap = (opts_.slot_cap > 0) ? opts_.slot_cap : 256;
    r.proc_limit_bytes = g_proc_limit_bytes.load();
    r.headroom_bytes = kBudgetDefaultHeadroomBytes;

    BudgetDecision d = decide_budget(r, current_caps_);
    append_budget_log(r, d);
    if (!d.fits) {
        std::fprintf(stderr,
                     "[FATAL] budget=auto refuses to run: non_cache=%.2f GiB + headroom=%.2f GiB + %d slots/layer minimum "
                     "do not fit under proc_limit=%.2f GiB\n",
                     non_cache_footprint / 1073741824.0, r.headroom_bytes / 1073741824.0, kBudgetMinSlotCap,
                     r.proc_limit_bytes / 1073741824.0);
        std::fflush(stderr);
        std::_Exit(3);
    }

    int per_layer_target = d.new_caps[1];
    std::fprintf(stderr,
                 "[budget=auto] init (%s): avail_clean=%.2f GiB footprint=%.2f GiB rec_ws=%.2f GiB proc_limit=%.2f GiB headroom=%.2f GiB -> target_cap=%d slots/layer\n",
                 d.reason.c_str(),
                 avail / 1073741824.0,
                 footprint / 1073741824.0,
                 recommended_working_set_bytes_ / 1073741824.0,
                 r.proc_limit_bytes / 1073741824.0,
                 r.headroom_bytes / 1073741824.0,
                 per_layer_target);

    int new_tot = 0;
    for (int l = 1; l < 48; ++l) {
        if (current_caps_[l] != d.new_caps[l]) {
            resize_layer_capacity(l, d.new_caps[l]);
        }
        new_tot += current_caps_[l];
    }
    min_total_slots_ = new_tot;
    max_total_slots_ = new_tot;
    check_process_memory_guard(initial_swap_bytes_);

    monitor_thread_ = std::thread([this]() {
        while (!stop_monitor_.load()) {
            check_process_memory_guard(initial_swap_bytes_); // covers prefill and resizes, not only decode steps
            uint64_t a = 0, s = 0, c = 0;
            int p = 1;
            query_macos_memory(a, s, c, p);
            latest_avail_bytes_.store(a);
            latest_swap_used_.store(s);
            latest_compressed_bytes_.store(c);
            latest_pressure_level_.store(p);
            for (int i = 0; i < 25 && !stop_monitor_.load(); ++i) {
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
            }
        }
    });
}

void Qwen38Engine::stop_memory_monitor() {
    stop_monitor_.store(true);
    if (monitor_thread_.joinable()) {
        monitor_thread_.join();
    }
}

void Qwen38Engine::check_and_adjust_budget(int step_idx) {
    if (opts_.budget_mode != "auto") return;
    int plevel = latest_pressure_level_.load();
    uint64_t swap_b = latest_swap_used_.load();
    uint64_t avail_b = latest_avail_bytes_.load();
    uint64_t comp_b = latest_compressed_bytes_.load();
    max_pressure_level_ = std::max(max_pressure_level_, plevel);
    max_swap_used_ = std::max(max_swap_used_, swap_b);

    uint64_t cur_slot_bytes = 0;
    for (int l = 1; l < 48; ++l) cur_slot_bytes += static_cast<uint64_t>(current_caps_[l]) * kBudgetExpertBytes;
    uint64_t footprint = get_phys_footprint_bytes();
    uint64_t non_cache_footprint = (footprint > cur_slot_bytes) ? (footprint - cur_slot_bytes) : (6ULL << 30);

    BudgetReadings r{};
    r.is_init = false;
    r.step_idx = step_idx;
    r.last_shrink_step = last_shrink_step_;
    r.avail_bytes = avail_b;
    r.swap_used_bytes = swap_b;
    r.initial_swap_bytes = initial_swap_bytes_;
    r.compressed_bytes = comp_b;
    r.prev_compressed_bytes = prev_compressed_bytes_;
    r.pressure_level = plevel;
    r.total_ram_bytes = total_ram_bytes_;
    r.non_cache_footprint_bytes = non_cache_footprint;
    r.recommended_working_set_bytes = recommended_working_set_bytes_;
    r.max_slot_cap = (opts_.slot_cap > 0) ? opts_.slot_cap : 128;
    r.proc_limit_bytes = g_proc_limit_bytes.load();
    r.headroom_bytes = kBudgetDefaultHeadroomBytes;

    prev_compressed_bytes_ = comp_b;

    BudgetDecision d = decide_budget(r, current_caps_);
    if (d.action == "shrink" || d.action == "grow") {
        int new_total = 0;
        for (int l = 1; l < 48; ++l) {
            if (d.new_caps[l] != current_caps_[l]) {
                resize_layer_capacity(l, d.new_caps[l]);
            }
            new_total += current_caps_[l];
        }
        check_process_memory_guard(initial_swap_bytes_);
        d.new_total_slots = new_total;
        if (d.action == "shrink") {
            last_shrink_step_ = step_idx;
        }
        budget_adjustments_++;
        min_total_slots_ = std::min(min_total_slots_, new_total);
        max_total_slots_ = std::max(max_total_slots_, new_total);
        append_budget_log(r, d);
        std::fprintf(stderr,
                     "[budget=auto] step=%d %s (%s) pressure=%d avail=%.2f GiB swap_delta=%.1f MiB total_slots %d -> %d\n",
                     step_idx,
                     (d.action == "shrink" ? "SHRINK" : "GROW"),
                     d.reason.c_str(),
                     plevel,
                     avail_b / 1073741824.0,
                     (swap_b > initial_swap_bytes_ ? (swap_b - initial_swap_bytes_) / 1048576.0 : 0.0),
                     d.old_total_slots,
                     new_total);
    } else {
        // Throttled to at most one 'none' line per 8 steps
        if (last_none_log_step_ < 0 || (step_idx - last_none_log_step_) >= 8) {
            last_none_log_step_ = step_idx;
            append_budget_log(r, d);
        }
    }
}

static void drop_os_file_cache_for_cold(
    const std::unordered_map<std::string, int>& open_fds,
    const std::unordered_map<std::string, std::shared_ptr<MmapFile>>& ple_mmaps) {
    for (const auto& kv : open_fds) {
        int fd = kv.second;
        if (fd < 0) continue;
        struct stat st{};
        if (fstat(fd, &st) == 0 && st.st_size > 0) {
            void* p = mmap(nullptr, static_cast<size_t>(st.st_size), PROT_READ, MAP_SHARED, fd, 0);
            if (p != MAP_FAILED) {
                msync(p, static_cast<size_t>(st.st_size), MS_INVALIDATE);
                munmap(p, static_cast<size_t>(st.st_size));
            }
        }
    }
    for (const auto& kv : ple_mmaps) {
        const auto& mm = kv.second;
        if (mm && mm->addr && mm->addr != MAP_FAILED && mm->size > 0) {
            msync(mm->addr, mm->size, MS_INVALIDATE);
        }
    }
}

RunMetrics Qwen38Engine::generate(
    const std::vector<int32_t>& prompt_ids,
    int n_tokens,
    const std::string& logits_out_path) {
    uint64_t init_avail = 0, init_swap = 0, init_comp = 0;
    int init_plevel = 1;
    query_macos_memory(init_avail, init_swap, init_comp, init_plevel);
    if (!loaded_) {
        g_proc_limit_bytes.store(compute_proc_limit_bytes(init_avail));
        std::fprintf(stderr, "[mem] proc_limit=%llu bytes (min(19.5e9, 0.8 x clean avail %llu before load); phys_footprint)\n",
                     static_cast<unsigned long long>(g_proc_limit_bytes.load()),
                     static_cast<unsigned long long>(init_avail));
    }

    RunMetrics m{};
    auto t_load0 = std::chrono::steady_clock::now();
    if (!loaded_) {
        load();
    }
    auto t_load1 = std::chrono::steady_clock::now();
    m.load_s = std::chrono::duration<double>(t_load1 - t_load0).count();

    int n_repeats = std::max(1, opts_.repeat);
    for (int rep = 0; rep < n_repeats; ++rep) {
        reset_caches(true);
        if (opts_.cold) {
            drop_os_file_cache_for_cold(open_fds_, ple_mmaps_);
        }
        check_process_memory_guard(init_swap);

        m.tokens.clear();
        if (opts_.io_mode == "p1") {
            for (int l = 1; l < 48; ++l) p1_sched_.reset_stats(l);
        } else {
            c_set_io_accounting(1);
            for (int l = 1; l < 48; ++l) c_layer_reset_stats(l);
        }

        start_memory_monitor();

        std::vector<mx::array> saved_step_logits;
        if (!logits_out_path.empty() && rep == n_repeats - 1) {
            saved_step_logits.reserve(n_tokens);
        }

        auto t_pref0 = std::chrono::steady_clock::now();
        auto logits = prefill(prompt_ids);
        auto next_tok_arr = mx::argmax(mx::squeeze(logits, 1), -1);
        mx::eval({next_tok_arr});
        auto t_pref1 = std::chrono::steady_clock::now();
        m.prefill_s = std::chrono::duration<double>(t_pref1 - t_pref0).count();
        m.ttft_s = m.prefill_s;
        check_process_memory_guard(init_swap);
        std::fprintf(stderr, "[mem] peak_phys_footprint_after_prefill=%llu bytes proc_limit=%llu\n",
                     static_cast<unsigned long long>(g_peak_footprint_bytes.load()),
                     static_cast<unsigned long long>(g_proc_limit_bytes.load()));

        if (!logits_out_path.empty() && rep == n_repeats - 1) {
            saved_step_logits.push_back(mx::reshape(mx::astype(logits, mx::float32), {1, -1}));
        }

        // Match server.py line 161: mx.clear_cache() after prefill
        mx::clear_cache();

        int32_t next_tok = static_cast<int32_t>(next_tok_arr.item<uint32_t>());
        m.tokens.push_back(next_tok);

        std::vector<double> step_ms;
        step_ms.reserve(std::max(0, n_tokens - 1));

        auto t_dec0 = std::chrono::steady_clock::now();
        for (int d_idx = 0; d_idx < n_tokens - 1; ++d_idx) {
            if (!opts_.no_stop && (next_tok == 248044 || next_tok == 248046)) {
                break;
            }
            check_and_adjust_budget(d_idx);
            check_process_memory_guard(init_swap);

            auto ts0 = std::chrono::steady_clock::now();
            logits = decode_step(next_tok, d_idx);
            next_tok_arr = mx::argmax(mx::squeeze(logits, 1), -1);
            mx::eval({next_tok_arr});
            next_tok = static_cast<int32_t>(next_tok_arr.item<uint32_t>());
            auto ts1 = std::chrono::steady_clock::now();

            if (!logits_out_path.empty() && rep == n_repeats - 1) {
                saved_step_logits.push_back(mx::reshape(mx::astype(logits, mx::float32), {1, -1}));
            }

            step_ms.push_back(std::chrono::duration<double, std::milli>(ts1 - ts0).count());
            m.tokens.push_back(next_tok);

            // Match server.py line 194-195: if (d_idx + 1) % 32 == 0: mx.clear_cache()
            if ((d_idx + 1) % 32 == 0) {
                mx::clear_cache();
            }
        }
        auto t_dec1 = std::chrono::steady_clock::now();
        m.decode_s = std::chrono::duration<double>(t_dec1 - t_dec0).count();

        if (opts_.io_mode == "p1") {
            for (int l = 1; l < 48; ++l) {
                wait_slot_reader(l);
            }
            p1_sched_.wait_all();
        }

        stop_memory_monitor();
        check_process_memory_guard(init_swap);
        std::fprintf(stderr, "[mem] peak_phys_footprint_run=%llu bytes proc_limit=%llu\n",
                     static_cast<unsigned long long>(g_peak_footprint_bytes.load()),
                     static_cast<unsigned long long>(g_proc_limit_bytes.load()));

        if (!logits_out_path.empty() && rep == n_repeats - 1 && !saved_step_logits.empty()) {
            auto parent = std::filesystem::path(logits_out_path).parent_path();
            if (!parent.empty()) {
                std::filesystem::create_directories(parent);
            }
            auto all_logits = mx::concatenate(saved_step_logits, 0);
            mx::eval({all_logits});
            if (logits_out_path.size() >= 12 &&
                logits_out_path.substr(logits_out_path.size() - 12) == ".safetensors") {
                mx::save_safetensors(logits_out_path, {{"logits", all_logits}});
            } else {
                mx::save(logits_out_path, all_logits);
            }
        }

        if (!opts_.state_out.empty() && rep == n_repeats - 1) {
            save_state(opts_.state_out);
        }

        int dec_cnt = static_cast<int>(step_ms.size());
        if (dec_cnt > 0 && m.decode_s > 0.0) {
            m.tok_per_s = static_cast<double>(dec_cnt) / m.decode_s;
            m.ms_per_tok = (m.decode_s * 1000.0) / static_cast<double>(dec_cnt);
            std::sort(step_ms.begin(), step_ms.end());
            m.p50_ms = step_ms[dec_cnt / 2];
            m.p90_ms = step_ms[std::min(dec_cnt - 1, static_cast<int>(dec_cnt * 0.90))];
        }
    }

    int dec_tokens = std::max(1, static_cast<int>(m.tokens.size()) - 1);
    if (opts_.io_mode == "p1") {
        P1LayerStats tot = p1_sched_.get_total_stats(48);
        m.decode_requests = tot.decode_requests;
        m.lru_hits = tot.lru_hits;
        m.prefetch_hits = tot.prefetch_used;
        m.demand_misses = tot.urgent_reads;
        m.urgent_reads = tot.urgent_reads;
        m.prefetch_issued = tot.prefetch_issued;
        m.prefetch_cancelled = tot.prefetch_cancelled;
        m.prefetch_promoted = tot.prefetch_promoted;
        m.prefetch_used = tot.prefetch_used;
        m.prefetch_wasted = tot.prefetch_wasted;
        m.pin_waits = tot.pin_waits;
        m.io_wait_ms = tot.io_wait_ms();
        m.forced_invalidations = forced_invalidations_;
        m.io_pread_bytes = tot.bytes;
        if (m.decode_requests > 0) {
            m.cache_hit_rate = static_cast<double>(m.lru_hits + m.prefetch_hits) / static_cast<double>(m.decode_requests);
        }
        m.read_mib_per_tok = (static_cast<double>(tot.bytes) / (1024.0 * 1024.0)) / static_cast<double>(dec_tokens);

        m.p1_layer_stats.clear();
        m.p1_layer_stats.reserve(47);
        for (int l = 1; l < 48; ++l) {
            m.p1_layer_stats.push_back(p1_sched_.get_layer_stats(l));
        }

        std::fprintf(stderr,
                     "[io=p1] summary (layers 1..47, threads=%d): urgent_reads=%llu prefetch(issued=%llu cancelled=%llu promoted=%llu used=%llu wasted=%llu) pin_waits=%llu bytes=%llu (%.2f MiB/tok) io_wait_ms=%.2f\n",
                     opts_.io_threads,
                     static_cast<unsigned long long>(m.urgent_reads),
                     static_cast<unsigned long long>(m.prefetch_issued),
                     static_cast<unsigned long long>(m.prefetch_cancelled),
                     static_cast<unsigned long long>(m.prefetch_promoted),
                     static_cast<unsigned long long>(m.prefetch_used),
                     static_cast<unsigned long long>(m.prefetch_wasted),
                     static_cast<unsigned long long>(m.pin_waits),
                     static_cast<unsigned long long>(m.io_pread_bytes),
                     m.read_mib_per_tok,
                     m.io_wait_ms);
    } else {
        uint64_t totals[6]{};
        c_get_decode_totals(48, totals);
        // c_get_decode_totals order: [lru_hits, prefetch_hits, demand_misses, requests, demand_bytes, prefetch_bytes]
        m.lru_hits = totals[0];
        m.prefetch_hits = totals[1];
        m.demand_misses = totals[2];
        m.decode_requests = totals[3];
        uint64_t total_decode_bytes = totals[4] + totals[5];
        if (m.decode_requests > 0) {
            m.cache_hit_rate = static_cast<double>(m.lru_hits + m.prefetch_hits) / static_cast<double>(m.decode_requests);
        }
        m.io_pread_bytes = total_decode_bytes;
        m.read_mib_per_tok = (static_cast<double>(total_decode_bytes) / (1024.0 * 1024.0)) / static_cast<double>(dec_tokens);
        m.forced_invalidations = forced_invalidations_;
    }

    m.per_layer_cap.assign(current_caps_.begin() + 1, current_caps_.end());
    uint64_t total_slots = 0;
    for (int c : m.per_layer_cap) total_slots += static_cast<uint64_t>(c);
    m.slot_buffer_bytes = total_slots * 1536000ULL;
    m.peak_rss_bytes = get_phys_footprint_bytes();
    m.mlx_active_bytes = mx::get_active_memory();
    m.mlx_peak_bytes = mx::get_peak_memory();
    m.mlx_cache_bytes = mx::get_cache_memory();
    m.budget_adjustments = budget_adjustments_;
    m.min_total_slots = min_total_slots_;
    m.max_total_slots = max_total_slots_;
    m.max_pressure_level = max_pressure_level_;
    m.max_swap_used_bytes = max_swap_used_;
    return m;
}

} // namespace qwen38

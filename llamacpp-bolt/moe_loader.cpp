#include "arg.h"
#include "common.h"
#include "llama.h"
#include "sampling.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#if defined(__APPLE__)
#include <mach/mach.h>
#endif

#include "../llama.cpp/src/llama-moe-cache.h"

using json = nlohmann::json;

static double get_process_rss_gib() {
#if defined(__APPLE__)
    struct mach_task_basic_info info;
    mach_msg_type_number_t count = MACH_TASK_BASIC_INFO_COUNT;
    if (task_info(mach_task_self(), MACH_TASK_BASIC_INFO, (task_info_t)&info, &count) == KERN_SUCCESS) {
        return (double)info.resident_size / (1024.0 * 1024.0 * 1024.0);
    }
#endif
    return 0.0;
}

struct prompt_item {
    std::string id;
    std::string domain;
    std::string prompt;
};

static std::string json_escape(const std::string & s) {
    std::ostringstream o;
    for (unsigned char c : s) {
        switch (c) {
            case '"':  o << "\\\""; break;
            case '\\': o << "\\\\"; break;
            case '\b': o << "\\b";  break;
            case '\f': o << "\\f";  break;
            case '\n': o << "\\n";  break;
            case '\r': o << "\\r";  break;
            case '\t': o << "\\t";  break;
            default:
                if (c < 0x20) {
                    char buf[8];
                    snprintf(buf, sizeof(buf), "\\u%04x", c);
                    o << buf;
                } else {
                    o << c;
                }
        }
    }
    return o.str();
}

static std::vector<prompt_item> parse_prompts_json(const std::string & path) {
    std::ifstream f(path);
    if (!f.is_open()) {
        fprintf(stderr, "Error: cannot open prompts file %s\n", path.c_str());
        return {};
    }
    json j;
    f >> j;
    std::vector<prompt_item> items;
    size_t idx = 0;
    for (const auto & entry : j) {
        if (entry.is_array() && entry.size() >= 2) {
            std::string dom = entry[0].get<std::string>();
            std::string txt = entry[1].get<std::string>();
            items.push_back({dom + "_" + std::to_string(idx), dom, txt});
        } else if (entry.is_object()) {
            std::string dom = entry.value("domain", "default");
            std::string id  = entry.value("id", dom + "_" + std::to_string(idx));
            std::string txt = entry.value("prompt", "");
            items.push_back({id, dom, txt});
        }
        idx++;
    }
    return items;
}

int main(int argc, char ** argv) {
    std::string model_path   = "NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB.gguf";
    std::string prompts_path = "";
    std::string single_prompt = "Write a C function to reverse a singly linked list in-place.";
    std::string output_json  = "";
    std::string policy       = "lru";
    std::string prefetch_str = "router";
    double      cache_gb     = 8.0;
    int         n_predict    = 64;
    int         max_prompts  = 0;
    int         per_layer    = 1;
    int         prefetch_mode = 2; // 0=off, 1=learned, 2=router (default), 3=hybrid
    int         prefetch_topk = 9;
    int         prefetch_cap  = 2;
    int         pin_first    = 0;
    int         io_threads   = 8;
    int         measure_router = 0;
    int         pread_mode   = 1;
    bool        baseline_gpu = false;
    bool        domain_switch = false;

    std::vector<char *> llama_args;
    llama_args.push_back(argv[0]);

    auto parse_prefetch_arg = [&](const char * arg) {
        if (strcmp(arg, "0") == 0 || strcmp(arg, "off") == 0 || strcmp(arg, "none") == 0) {
            prefetch_mode = 0;
            prefetch_str  = "off";
        } else if (strcmp(arg, "1") == 0 || strcmp(arg, "learned") == 0 || strcmp(arg, "transition") == 0) {
            prefetch_mode = 1;
            prefetch_str  = "learned";
        } else if (strcmp(arg, "2") == 0 || strcmp(arg, "router") == 0) {
            prefetch_mode = 2;
            prefetch_str  = "router";
        } else if (strcmp(arg, "3") == 0 || strcmp(arg, "hybrid") == 0) {
            prefetch_mode = 3;
            prefetch_str  = "hybrid";
        } else {
            prefetch_mode = atoi(arg);
            prefetch_str  = (prefetch_mode == 0) ? "off" : (prefetch_mode == 2 ? "router" : (prefetch_mode == 3 ? "hybrid" : "learned"));
        }
    };

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--model") == 0 && i + 1 < argc) {
            model_path = argv[++i];
        } else if (strcmp(argv[i], "--prompts") == 0 && i + 1 < argc) {
            prompts_path = argv[++i];
        } else if (strcmp(argv[i], "--prompt") == 0 && i + 1 < argc) {
            single_prompt = argv[++i];
        } else if (strcmp(argv[i], "--out") == 0 && i + 1 < argc) {
            output_json = argv[++i];
        } else if (strcmp(argv[i], "--policy") == 0 && i + 1 < argc) {
            policy = argv[++i];
        } else if (strcmp(argv[i], "--cache-gb") == 0 && i + 1 < argc) {
            cache_gb = atof(argv[++i]);
        } else if (strcmp(argv[i], "--n-predict") == 0 && i + 1 < argc) {
            n_predict = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--max-prompts") == 0 && i + 1 < argc) {
            max_prompts = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--per-layer") == 0 && i + 1 < argc) {
            per_layer = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--prefetch") == 0 && i + 1 < argc) {
            parse_prefetch_arg(argv[++i]);
        } else if (strcmp(argv[i], "--prefetch-topk") == 0 && i + 1 < argc) {
            prefetch_topk = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--prefetch-cap") == 0 && i + 1 < argc) {
            prefetch_cap = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--pin-first-moe") == 0) {
            if (i + 1 < argc && (strcmp(argv[i + 1], "0") == 0 || strcmp(argv[i + 1], "1") == 0)) {
                pin_first = atoi(argv[++i]);
            } else {
                pin_first = 1;
            }
        } else if (strcmp(argv[i], "--no-pin-first-moe") == 0) {
            pin_first = 0;
        } else if (strcmp(argv[i], "--pin-first") == 0 && i + 1 < argc) {
            pin_first = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--io-threads") == 0 && i + 1 < argc) {
            io_threads = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--measure-router") == 0) {
            if (i + 1 < argc && (strcmp(argv[i + 1], "0") == 0 || strcmp(argv[i + 1], "1") == 0)) {
                measure_router = atoi(argv[++i]);
            } else {
                measure_router = 1;
            }
        } else if (strcmp(argv[i], "--pread") == 0 && i + 1 < argc) {
            pread_mode = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--baseline") == 0) {
            baseline_gpu = true;
        } else if (strcmp(argv[i], "--domain-switch") == 0) {
            domain_switch = true;
        } else {
            llama_args.push_back(argv[i]);
        }
    }

    // Configure environment variables for MoE cache before model load
    setenv("LLAMA_MOE_CACHE_POLICY", policy.c_str(), 1);
    setenv("LLAMA_MOE_PER_LAYER", per_layer ? "1" : "0", 1);
    setenv("LLAMA_MOE_PREFETCH", prefetch_str.c_str(), 1);
    setenv("LLAMA_MOE_PREFETCH_TOPK", std::to_string(prefetch_topk).c_str(), 1);
    setenv("LLAMA_MOE_PREFETCH_CAP", std::to_string(prefetch_cap).c_str(), 1);
    setenv("LLAMA_MOE_PIN_FIRST", pin_first ? "1" : "0", 1);
    setenv("LLAMA_MOE_IO_THREADS", std::to_string(io_threads).c_str(), 1);
    setenv("LLAMA_MOE_MEASURE_ROUTER", measure_router ? "1" : "0", 1);
    if (!baseline_gpu && pread_mode) {
        setenv("LLAMA_MOE_PREAD", "1", 1);
    } else {
        setenv("LLAMA_MOE_PREAD", "0", 1);
    }

    llama_args.push_back((char *)"-m");
    llama_args.push_back(const_cast<char *>(model_path.c_str()));
    if (!baseline_gpu) {
        llama_args.push_back((char *)"-cmoe");
    }

    common_params params;
    if (!common_params_parse((int)llama_args.size(), llama_args.data(), params, LLAMA_EXAMPLE_COMMON)) {
        return 1;
    }
    params.model.path = model_path;
    params.n_predict  = n_predict;
    params.sampling.temp = 0.0f; // deterministic greedy decoding
    params.sampling.seed = 42;
    params.n_ctx      = 2048;
    params.n_batch    = 512;
    params.n_ubatch   = 4; // keep prefill ubatches <= 4 tokens (36 experts <= 60 slots at 4 GB) so slot_map is always used
    params.warmup     = false;
    params.fit_params = false;

    if (!baseline_gpu) {
        params.load_mode = LLAMA_LOAD_MODE_NONE;
        params.moe_cache_size = (size_t)(cache_gb * 1024.0 * 1024.0 * 1024.0);
    }

    common_init();
    llama_backend_init();
    llama_numa_init(params.numa);

    auto t_load_start = std::chrono::high_resolution_clock::now();
    auto llama_init = common_init_from_params(params);
    auto t_load_end   = std::chrono::high_resolution_clock::now();
    double load_sec   = std::chrono::duration<double>(t_load_end - t_load_start).count();

    llama_model * model = llama_init->model();
    llama_context * ctx = llama_init->context();
    if (!model || !ctx) {
        fprintf(stderr, "Error: failed to initialize model/context\n");
        return 1;
    }

    llama_moe_cache::cache_stats init_st{};
    llama_moe_cache_get_ext_stats(ctx, &init_st);

    const llama_vocab * vocab = llama_model_get_vocab(model);
    std::vector<prompt_item> items;
    if (!prompts_path.empty()) {
        items = parse_prompts_json(prompts_path);
        if (max_prompts > 0 && (int)items.size() > max_prompts) {
            items.resize(max_prompts);
        }
    } else {
        items.push_back({"single_0", "code", single_prompt});
    }

    double rss_after_load = get_process_rss_gib();
    fprintf(stderr, "\n[moe_loader] mode=%s cache_gb=%.2f policy=%s per_layer=%d pread=%d io_threads=%d prefetch=%s(topk=%d,cap=%d) pin_first=%d(blk.%d,%d_exps,%.2f_MiB) load_sec=%.2f rss_gib=%.2f\n",
            baseline_gpu ? "baseline_gpu" : "moe_cache",
            baseline_gpu ? 17.05 : cache_gb,
            policy.c_str(), per_layer, pread_mode, io_threads,
            prefetch_str.c_str(), prefetch_topk, prefetch_cap,
            pin_first, init_st.pinned_layer, init_st.pinned_n_expert,
            (double)init_st.pinned_bytes / (1024.0 * 1024.0),
            load_sec, rss_after_load);

    std::ofstream out_f;
    if (!output_json.empty()) {
        out_f.open(output_json);
    }

    double total_prefill_ms = 0.0;
    double total_decode_ms  = 0.0;
    int    total_prefill_tokens = 0;
    int    total_decode_tokens  = 0;

    llama_moe_cache::cache_stats dec_acc{};
    dec_acc.pinned_layer    = init_st.pinned_layer;
    dec_acc.pinned_n_expert = init_st.pinned_n_expert;
    dec_acc.pinned_bytes    = init_st.pinned_bytes;
    dec_acc.io_threads      = init_st.io_threads;
    dec_acc.prefetch_mode   = init_st.prefetch_mode;
    dec_acc.prefetch_topk   = init_st.prefetch_topk;

    std::ostringstream prompts_json_arr;
    prompts_json_arr << "[\n";

    auto batch_set_tokens = [](llama_batch_ext * batch, const llama_token * tokens, int32_t n_tokens, llama_pos pos_0) {
        llama_batch_ext_clear(batch);
        for (int32_t i = 0; i < n_tokens; ++i) {
            const int32_t idx = llama_batch_ext_add_token(batch, 0, tokens[i]);
            const llama_pos pos = pos_0 + i;
            llama_batch_ext_set_pos(batch, idx, &pos);
        }
        llama_batch_ext_set_output_logits(batch, n_tokens - 1, true);
    };

    llama_batch_ext * batch = llama_batch_ext_init(ctx);

    for (size_t idx = 0; idx < items.size(); ++idx) {
        const auto & item = items[idx];
        llama_memory_clear(llama_get_memory(ctx), true);

        const bool add_bos = llama_vocab_get_add_bos(vocab);
        std::vector<llama_token> tokens_list = common_tokenize(ctx, item.prompt, add_bos, true);
        int n_prompt = (int)tokens_list.size();

        // Prefill prompt in chunks of n_ubatch so slot_map is used & warmed cleanly
        auto t_pre0 = std::chrono::high_resolution_clock::now();
        const int chunk_sz = 4;
        for (int p0 = 0; p0 < n_prompt; p0 += chunk_sz) {
            int n_cur = std::min(chunk_sz, n_prompt - p0);
            batch_set_tokens(batch, tokens_list.data() + p0, n_cur, p0);
            if (llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch)) {
                fprintf(stderr, "Error: llama_process failed on prefill\n");
                llama_batch_ext_free(batch);
                return 1;
            }
        }
        auto t_pre1 = std::chrono::high_resolution_clock::now();
        double prefill_ms = std::chrono::duration<double, std::milli>(t_pre1 - t_pre0).count();
        total_prefill_ms += prefill_ms;
        total_prefill_tokens += n_prompt;

        llama_moe_cache::cache_stats st0{};
        llama_moe_cache_get_ext_stats(ctx, &st0);

        common_sampler * smpl = common_sampler_init(model, params.sampling);
        for (auto id : tokens_list) {
            common_sampler_accept(smpl, id, false);
        }
        std::string gen_text;
        std::vector<int32_t> gen_tokens;
        std::vector<double> step_hit_rates;
        std::vector<double> step_latencies_ms;

        llama_pos n_past = n_prompt;
        auto t_dec0 = std::chrono::high_resolution_clock::now();

        for (int step = 0; step < n_predict; ++step) {
            llama_token new_token_id = common_sampler_sample(smpl, ctx, -1);
            common_sampler_accept(smpl, new_token_id, true);
            gen_tokens.push_back(new_token_id);

            if (llama_vocab_is_eog(vocab, new_token_id)) {
                break;
            }
            gen_text += common_token_to_piece(ctx, new_token_id);

            size_t hs0 = 0, ms0 = 0, bs0 = 0, pfs0 = 0;
            llama_moe_cache_get_stats(ctx, &hs0, &ms0, &bs0, &pfs0);
            auto ts0 = std::chrono::high_resolution_clock::now();

            batch_set_tokens(batch, &new_token_id, 1, n_past);
            n_past++;
            if (llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch)) {
                fprintf(stderr, "Error: llama_process failed at step %d\n", step);
                break;
            }

            auto ts1 = std::chrono::high_resolution_clock::now();
            size_t hs1 = 0, ms1 = 0, bs1 = 0, pfs1 = 0;
            llama_moe_cache_get_stats(ctx, &hs1, &ms1, &bs1, &pfs1);

            size_t dh = hs1 - hs0;
            size_t dm = ms1 - ms0;
            double hr = (dh + dm > 0) ? (double)dh / (double)(dh + dm) : 1.0;
            step_hit_rates.push_back(hr);
            step_latencies_ms.push_back(std::chrono::duration<double, std::milli>(ts1 - ts0).count());
        }

        auto t_dec1 = std::chrono::high_resolution_clock::now();
        double decode_ms = std::chrono::duration<double, std::milli>(t_dec1 - t_dec0).count();
        int n_dec_evals = (int)step_latencies_ms.size();

        llama_moe_cache::cache_stats st1{};
        llama_moe_cache_get_ext_stats(ctx, &st1);

        size_t dec_hits     = (st1.hits_small + st1.hits_large) - (st0.hits_small + st0.hits_large);
        size_t dec_misses   = (st1.misses_small + st1.misses_large) - (st0.misses_small + st0.misses_large);
        size_t dec_lru_hits = st1.lru_hits_small - st0.lru_hits_small;
        size_t dec_lru_miss = st1.lru_misses_small - st0.lru_misses_small;
        size_t dec_bytes    = (st1.bytes_small + st1.bytes_large) - (st0.bytes_small + st0.bytes_large);
        size_t dec_pf_sub   = st1.prefetch_submitted - st0.prefetch_submitted;
        size_t dec_pf       = st1.prefetch_hits - st0.prefetch_hits;

        total_decode_ms     += decode_ms;
        total_decode_tokens += n_dec_evals;

        dec_acc.hits_small               += dec_hits;
        dec_acc.misses_small             += dec_misses;
        dec_acc.lru_hits_small           += dec_lru_hits;
        dec_acc.lru_misses_small         += dec_lru_miss;
        dec_acc.bytes_small              += dec_bytes;
        dec_acc.prefetch_submitted       += dec_pf_sub;
        dec_acc.prefetch_hits            += dec_pf;
        dec_acc.pred_eval_transitions    += (st1.pred_eval_transitions - st0.pred_eval_transitions);
        dec_acc.router_self_hits_9       += (st1.router_self_hits_9 - st0.router_self_hits_9);
        dec_acc.router_hits_at_9         += (st1.router_hits_at_9 - st0.router_hits_at_9);
        dec_acc.router_hits_at_12        += (st1.router_hits_at_12 - st0.router_hits_at_12);
        dec_acc.router_hits_at_16        += (st1.router_hits_at_16 - st0.router_hits_at_16);
        dec_acc.router_curnorm_hits_at_9 += (st1.router_curnorm_hits_at_9 - st0.router_curnorm_hits_at_9);
        dec_acc.router_curnorm_hits_at_12+= (st1.router_curnorm_hits_at_12 - st0.router_curnorm_hits_at_12);
        dec_acc.router_curnorm_hits_at_16+= (st1.router_curnorm_hits_at_16 - st0.router_curnorm_hits_at_16);
        dec_acc.router_raw_hits_at_9     += (st1.router_raw_hits_at_9 - st0.router_raw_hits_at_9);
        dec_acc.router_raw_hits_at_12    += (st1.router_raw_hits_at_12 - st0.router_raw_hits_at_12);
        dec_acc.router_raw_hits_at_16    += (st1.router_raw_hits_at_16 - st0.router_raw_hits_at_16);
        dec_acc.learned_hits_at_9        += (st1.learned_hits_at_9 - st0.learned_hits_at_9);
        dec_acc.learned_hits_at_12       += (st1.learned_hits_at_12 - st0.learned_hits_at_12);
        dec_acc.learned_hits_at_16       += (st1.learned_hits_at_16 - st0.learned_hits_at_16);
        dec_acc.hybrid_hits_at_9         += (st1.hybrid_hits_at_9 - st0.hybrid_hits_at_9);
        dec_acc.hybrid_hits_at_12        += (st1.hybrid_hits_at_12 - st0.hybrid_hits_at_12);
        dec_acc.hybrid_hits_at_16        += (st1.hybrid_hits_at_16 - st0.hybrid_hits_at_16);
        dec_acc.miss_eval_actual_misses  += (st1.miss_eval_actual_misses - st0.miss_eval_actual_misses);
        dec_acc.router_miss_cands_9      += (st1.router_miss_cands_9 - st0.router_miss_cands_9);
        dec_acc.router_miss_hits_9       += (st1.router_miss_hits_9 - st0.router_miss_hits_9);
        dec_acc.router_miss_cands_12     += (st1.router_miss_cands_12 - st0.router_miss_cands_12);
        dec_acc.router_miss_hits_12      += (st1.router_miss_hits_12 - st0.router_miss_hits_12);
        dec_acc.router_miss_cands_16     += (st1.router_miss_cands_16 - st0.router_miss_cands_16);
        dec_acc.router_miss_hits_16      += (st1.router_miss_hits_16 - st0.router_miss_hits_16);
        dec_acc.learned_miss_cands       += (st1.learned_miss_cands - st0.learned_miss_cands);
        dec_acc.learned_miss_hits        += (st1.learned_miss_hits - st0.learned_miss_hits);
        dec_acc.learned_miss_cands_9     += (st1.learned_miss_cands_9 - st0.learned_miss_cands_9);
        dec_acc.learned_miss_hits_9      += (st1.learned_miss_hits_9 - st0.learned_miss_hits_9);
        dec_acc.hybrid_miss_cands_12     += (st1.hybrid_miss_cands_12 - st0.hybrid_miss_cands_12);
        dec_acc.hybrid_miss_hits_12      += (st1.hybrid_miss_hits_12 - st0.hybrid_miss_hits_12);
        dec_acc.active_pf_cands_scheduled+= (st1.active_pf_cands_scheduled - st0.active_pf_cands_scheduled);
        dec_acc.active_pf_cands_useful   += (st1.active_pf_cands_useful - st0.active_pf_cands_useful);

        double dec_tok_s = (decode_ms > 0.0) ? (1000.0 * n_dec_evals / decode_ms) : 0.0;
        double dec_hr    = (dec_hits + dec_misses > 0) ? (100.0 * (double)dec_hits / (double)(dec_hits + dec_misses)) : 100.0;
        double dec_lru_hr = (dec_lru_hits + dec_lru_miss > 0) ? (100.0 * (double)dec_lru_hits / (double)(dec_lru_hits + dec_lru_miss)) : 100.0;
        double mib_per_tok = (n_dec_evals > 0) ? ((double)dec_bytes / (1024.0 * 1024.0 * n_dec_evals)) : 0.0;

        fprintf(stderr, "  [%zu/%zu] id=%s (%s): decode=%.2f tok/s, hit_rate=%.2f%% (lru=%.2f%%), upload=%.2f MiB/tok, pf_hits=%zu/%zu\n",
                idx + 1, items.size(), item.id.c_str(), item.domain.c_str(),
                dec_tok_s, dec_hr, dec_lru_hr, mib_per_tok, dec_pf, dec_pf_sub);

        if (idx > 0) prompts_json_arr << ",\n";
        prompts_json_arr << "    {\n"
                         << "      \"id\": \"" << json_escape(item.id) << "\",\n"
                         << "      \"domain\": \"" << json_escape(item.domain) << "\",\n"
                         << "      \"n_prompt\": " << n_prompt << ",\n"
                         << "      \"n_decode\": " << n_dec_evals << ",\n"
                         << "      \"prefill_ms\": " << prefill_ms << ",\n"
                         << "      \"decode_ms\": " << decode_ms << ",\n"
                         << "      \"decode_tok_per_s\": " << dec_tok_s << ",\n"
                         << "      \"decode_hits\": " << dec_hits << ",\n"
                         << "      \"decode_misses\": " << dec_misses << ",\n"
                         << "      \"decode_hit_rate\": " << (dec_hr / 100.0) << ",\n"
                         << "      \"decode_lru_hit_rate\": " << (dec_lru_hr / 100.0) << ",\n"
                         << "      \"mib_per_token\": " << mib_per_tok << ",\n"
                         << "      \"prefetch_submitted\": " << dec_pf_sub << ",\n"
                         << "      \"prefetch_hits\": " << dec_pf << ",\n"
                         << "      \"text\": \"" << json_escape(gen_text) << "\",\n"
                         << "      \"tokens\": [";
        for (size_t ti = 0; ti < gen_tokens.size(); ++ti) {
            if (ti > 0) prompts_json_arr << ", ";
            prompts_json_arr << gen_tokens[ti];
        }
        prompts_json_arr << "],\n      \"step_hit_rates\": [";
        for (size_t si = 0; si < step_hit_rates.size(); ++si) {
            if (si > 0) prompts_json_arr << ", ";
            char b[32];
            snprintf(b, sizeof(b), "%.4f", step_hit_rates[si]);
            prompts_json_arr << b;
        }
        prompts_json_arr << "],\n      \"step_latencies_ms\": [";
        for (size_t si = 0; si < step_latencies_ms.size(); ++si) {
            if (si > 0) prompts_json_arr << ", ";
            char b[32];
            snprintf(b, sizeof(b), "%.2f", step_latencies_ms[si]);
            prompts_json_arr << b;
        }
        prompts_json_arr << "]\n    }";

        common_sampler_free(smpl);
    }
    llama_batch_ext_free(batch);
    (void)domain_switch;
    prompts_json_arr << "\n  ]";

    double peak_rss_gib = get_process_rss_gib();
    double avg_dec_tok_s = (total_decode_ms > 0.0) ? (1000.0 * total_decode_tokens / total_decode_ms) : 0.0;
    double avg_pre_tok_s = (total_prefill_ms > 0.0) ? (1000.0 * total_prefill_tokens / total_prefill_ms) : 0.0;
    double overall_hr = (dec_acc.hits_small + dec_acc.misses_small > 0)
                        ? ((double)dec_acc.hits_small / (double)(dec_acc.hits_small + dec_acc.misses_small))
                        : 1.0;
    double overall_lru_hr = (dec_acc.lru_hits_small + dec_acc.lru_misses_small > 0)
                        ? ((double)dec_acc.lru_hits_small / (double)(dec_acc.lru_hits_small + dec_acc.lru_misses_small))
                        : 1.0;
    double avg_mib_tok = (total_decode_tokens > 0)
                         ? ((double)dec_acc.bytes_small / (1024.0 * 1024.0 * total_decode_tokens))
                         : 0.0;

    auto safe_div = [](size_t num, size_t den) -> double {
        return den > 0 ? (double)num / (double)den : 0.0;
    };

    size_t n_trans = dec_acc.pred_eval_transitions;
    size_t n_gt_experts = n_trans * 9;
    size_t n_actual_misses = dec_acc.miss_eval_actual_misses;

    if (n_trans > 0) {
        fprintf(stderr,
                "[moe_loader] SUMMARY: decode=%.2f tok/s | hit_rate=%.2f%% (lru=%.2f%%, pf_hits=%zu/%zu) | peak_rss=%.2f GiB\n"
                "  [Router vs Learned Next-Layer Prediction across %zu transitions (%zu expert slots)]:\n"
                "    self-check @9 : %.2f%%\n"
                "    router (next_norm(inpL)): recall@9=%.2f%% | recall@12=%.2f%% (prec@12=%.2f%%) | recall@16=%.2f%% (prec@16=%.2f%%)\n"
                "    router (cur_norm(inpL)) : recall@9=%.2f%% | recall@12=%.2f%% | recall@16=%.2f%%\n"
                "    router (raw inpL)       : recall@9=%.2f%% | recall@12=%.2f%% | recall@16=%.2f%%\n"
                "    learned transition      : recall@9=%.2f%% | recall@12=%.2f%% (prec@12=%.2f%%) | recall@16=%.2f%% (prec@16=%.2f%%)\n"
                "    hybrid (router+learned) : recall@9=%.2f%% | recall@12=%.2f%% (prec@12=%.2f%%) | recall@16=%.2f%% (prec@16=%.2f%%)\n"
                "  [Cache-Miss Prediction across %zu actual misses]:\n"
                "    router_top9   : precision=%.2f%%, recall=%.2f%% (cands=%zu, hits=%zu)\n"
                "    router_top12  : precision=%.2f%%, recall=%.2f%% (cands=%zu, hits=%zu)\n"
                "    router_top16  : precision=%.2f%%, recall=%.2f%% (cands=%zu, hits=%zu)\n"
                "    learned_2x3   : precision=%.2f%%, recall=%.2f%% (cands=%zu, hits=%zu)\n"
                "    learned_top9  : precision=%.2f%%, recall=%.2f%% (cands=%zu, hits=%zu)\n"
                "    hybrid_top12  : precision=%.2f%%, recall=%.2f%% (cands=%zu, hits=%zu)\n"
                "    active_pf(cap): precision=%.2f%%, recall=%.2f%% (scheduled=%zu, useful=%zu)\n",
                avg_dec_tok_s, overall_hr * 100.0, overall_lru_hr * 100.0,
                dec_acc.prefetch_hits, dec_acc.prefetch_submitted, peak_rss_gib,
                n_trans, n_gt_experts,
                100.0 * safe_div(dec_acc.router_self_hits_9, n_gt_experts),
                100.0 * safe_div(dec_acc.router_hits_at_9, n_gt_experts),
                100.0 * safe_div(dec_acc.router_hits_at_12, n_gt_experts),
                100.0 * safe_div(dec_acc.router_hits_at_12, n_trans * 12),
                100.0 * safe_div(dec_acc.router_hits_at_16, n_gt_experts),
                100.0 * safe_div(dec_acc.router_hits_at_16, n_trans * 16),
                100.0 * safe_div(dec_acc.router_curnorm_hits_at_9, n_gt_experts),
                100.0 * safe_div(dec_acc.router_curnorm_hits_at_12, n_gt_experts),
                100.0 * safe_div(dec_acc.router_curnorm_hits_at_16, n_gt_experts),
                100.0 * safe_div(dec_acc.router_raw_hits_at_9, n_gt_experts),
                100.0 * safe_div(dec_acc.router_raw_hits_at_12, n_gt_experts),
                100.0 * safe_div(dec_acc.router_raw_hits_at_16, n_gt_experts),
                100.0 * safe_div(dec_acc.learned_hits_at_9, n_gt_experts),
                100.0 * safe_div(dec_acc.learned_hits_at_12, n_gt_experts),
                100.0 * safe_div(dec_acc.learned_hits_at_12, n_trans * 12),
                100.0 * safe_div(dec_acc.learned_hits_at_16, n_gt_experts),
                100.0 * safe_div(dec_acc.learned_hits_at_16, n_trans * 16),
                100.0 * safe_div(dec_acc.hybrid_hits_at_9, n_gt_experts),
                100.0 * safe_div(dec_acc.hybrid_hits_at_12, n_gt_experts),
                100.0 * safe_div(dec_acc.hybrid_hits_at_12, n_trans * 12),
                100.0 * safe_div(dec_acc.hybrid_hits_at_16, n_gt_experts),
                100.0 * safe_div(dec_acc.hybrid_hits_at_16, n_trans * 16),
                n_actual_misses,
                100.0 * safe_div(dec_acc.router_miss_hits_9, dec_acc.router_miss_cands_9),
                100.0 * safe_div(dec_acc.router_miss_hits_9, n_actual_misses),
                dec_acc.router_miss_cands_9, dec_acc.router_miss_hits_9,
                100.0 * safe_div(dec_acc.router_miss_hits_12, dec_acc.router_miss_cands_12),
                100.0 * safe_div(dec_acc.router_miss_hits_12, n_actual_misses),
                dec_acc.router_miss_cands_12, dec_acc.router_miss_hits_12,
                100.0 * safe_div(dec_acc.router_miss_hits_16, dec_acc.router_miss_cands_16),
                100.0 * safe_div(dec_acc.router_miss_hits_16, n_actual_misses),
                dec_acc.router_miss_cands_16, dec_acc.router_miss_hits_16,
                100.0 * safe_div(dec_acc.learned_miss_hits, dec_acc.learned_miss_cands),
                100.0 * safe_div(dec_acc.learned_miss_hits, n_actual_misses),
                dec_acc.learned_miss_cands, dec_acc.learned_miss_hits,
                100.0 * safe_div(dec_acc.learned_miss_hits_9, dec_acc.learned_miss_cands_9),
                100.0 * safe_div(dec_acc.learned_miss_hits_9, n_actual_misses),
                dec_acc.learned_miss_cands_9, dec_acc.learned_miss_hits_9,
                100.0 * safe_div(dec_acc.hybrid_miss_hits_12, dec_acc.hybrid_miss_cands_12),
                100.0 * safe_div(dec_acc.hybrid_miss_hits_12, n_actual_misses),
                dec_acc.hybrid_miss_cands_12, dec_acc.hybrid_miss_hits_12,
                100.0 * safe_div(dec_acc.active_pf_cands_useful, dec_acc.active_pf_cands_scheduled),
                100.0 * safe_div(dec_acc.active_pf_cands_useful, n_actual_misses),
                dec_acc.active_pf_cands_scheduled, dec_acc.active_pf_cands_useful);
    }

    std::ostringstream root_json;
    root_json << "{\n"
              << "  \"mode\": \"" << (baseline_gpu ? "baseline_gpu" : "moe_cache") << "\",\n"
              << "  \"model\": \"" << json_escape(model_path) << "\",\n"
              << "  \"cache_gb\": " << (baseline_gpu ? 17.05 : cache_gb) << ",\n"
              << "  \"policy\": \"" << json_escape(policy) << "\",\n"
              << "  \"per_layer\": " << per_layer << ",\n"
              << "  \"pread\": " << pread_mode << ",\n"
              << "  \"io_threads\": " << io_threads << ",\n"
              << "  \"prefetch\": " << prefetch_mode << ",\n"
              << "  \"prefetch_mode\": \"" << json_escape(prefetch_str) << "\",\n"
              << "  \"prefetch_topk\": " << prefetch_topk << ",\n"
              << "  \"prefetch_cap\": " << prefetch_cap << ",\n"
              << "  \"pin_first\": " << pin_first << ",\n"
              << "  \"pinned_layer\": " << init_st.pinned_layer << ",\n"
              << "  \"pinned_n_expert\": " << init_st.pinned_n_expert << ",\n"
              << "  \"pinned_bytes\": " << init_st.pinned_bytes << ",\n"
              << "  \"pinned_mib\": " << ((double)init_st.pinned_bytes / (1024.0 * 1024.0)) << ",\n"
              << "  \"load_sec\": " << load_sec << ",\n"
              << "  \"rss_after_load_gib\": " << rss_after_load << ",\n"
              << "  \"peak_rss_gib\": " << peak_rss_gib << ",\n"
              << "  \"prefill_tok_per_s\": " << avg_pre_tok_s << ",\n"
              << "  \"decode_tok_per_s\": " << avg_dec_tok_s << ",\n"
              << "  \"decode_hit_rate\": " << overall_hr << ",\n"
              << "  \"decode_lru_hit_rate\": " << overall_lru_hr << ",\n"
              << "  \"decode_hits\": " << dec_acc.hits_small << ",\n"
              << "  \"decode_misses\": " << dec_acc.misses_small << ",\n"
              << "  \"decode_lru_hits\": " << dec_acc.lru_hits_small << ",\n"
              << "  \"decode_lru_misses\": " << dec_acc.lru_misses_small << ",\n"
              << "  \"mib_per_token\": " << avg_mib_tok << ",\n"
              << "  \"prefetch_submitted\": " << dec_acc.prefetch_submitted << ",\n"
              << "  \"prefetch_hits\": " << dec_acc.prefetch_hits << ",\n"
              << "  \"prefetch_accuracy\": {\n"
              << "    \"transitions_evaluated\": " << n_trans << ",\n"
              << "    \"ground_truth_expert_activations\": " << n_gt_experts << ",\n"
              << "    \"router_self_check_recall_at_9\": " << safe_div(dec_acc.router_self_hits_9, n_gt_experts) << ",\n"
              << "    \"router_next_norm\": {\n"
              << "      \"recall_at_9\": " << safe_div(dec_acc.router_hits_at_9, n_gt_experts) << ",\n"
              << "      \"precision_at_9\": " << safe_div(dec_acc.router_hits_at_9, n_gt_experts) << ",\n"
              << "      \"recall_at_12\": " << safe_div(dec_acc.router_hits_at_12, n_gt_experts) << ",\n"
              << "      \"precision_at_12\": " << safe_div(dec_acc.router_hits_at_12, n_trans * 12) << ",\n"
              << "      \"recall_at_16\": " << safe_div(dec_acc.router_hits_at_16, n_gt_experts) << ",\n"
              << "      \"precision_at_16\": " << safe_div(dec_acc.router_hits_at_16, n_trans * 16) << "\n"
              << "    },\n"
              << "    \"router_cur_norm\": {\n"
              << "      \"recall_at_9\": " << safe_div(dec_acc.router_curnorm_hits_at_9, n_gt_experts) << ",\n"
              << "      \"recall_at_12\": " << safe_div(dec_acc.router_curnorm_hits_at_12, n_gt_experts) << ",\n"
              << "      \"recall_at_16\": " << safe_div(dec_acc.router_curnorm_hits_at_16, n_gt_experts) << "\n"
              << "    },\n"
              << "    \"router_raw_inpL\": {\n"
              << "      \"recall_at_9\": " << safe_div(dec_acc.router_raw_hits_at_9, n_gt_experts) << ",\n"
              << "      \"recall_at_12\": " << safe_div(dec_acc.router_raw_hits_at_12, n_gt_experts) << ",\n"
              << "      \"recall_at_16\": " << safe_div(dec_acc.router_raw_hits_at_16, n_gt_experts) << "\n"
              << "    },\n"
              << "    \"learned_transition\": {\n"
              << "      \"recall_at_9\": " << safe_div(dec_acc.learned_hits_at_9, n_gt_experts) << ",\n"
              << "      \"precision_at_9\": " << safe_div(dec_acc.learned_hits_at_9, n_gt_experts) << ",\n"
              << "      \"recall_at_12\": " << safe_div(dec_acc.learned_hits_at_12, n_gt_experts) << ",\n"
              << "      \"precision_at_12\": " << safe_div(dec_acc.learned_hits_at_12, n_trans * 12) << ",\n"
              << "      \"recall_at_16\": " << safe_div(dec_acc.learned_hits_at_16, n_gt_experts) << ",\n"
              << "      \"precision_at_16\": " << safe_div(dec_acc.learned_hits_at_16, n_trans * 16) << "\n"
              << "    },\n"
              << "    \"hybrid\": {\n"
              << "      \"recall_at_9\": " << safe_div(dec_acc.hybrid_hits_at_9, n_gt_experts) << ",\n"
              << "      \"precision_at_9\": " << safe_div(dec_acc.hybrid_hits_at_9, n_gt_experts) << ",\n"
              << "      \"recall_at_12\": " << safe_div(dec_acc.hybrid_hits_at_12, n_gt_experts) << ",\n"
              << "      \"precision_at_12\": " << safe_div(dec_acc.hybrid_hits_at_12, n_trans * 12) << ",\n"
              << "      \"recall_at_16\": " << safe_div(dec_acc.hybrid_hits_at_16, n_gt_experts) << ",\n"
              << "      \"precision_at_16\": " << safe_div(dec_acc.hybrid_hits_at_16, n_trans * 16) << "\n"
              << "    },\n"
              << "    \"cache_miss_prediction\": {\n"
              << "      \"actual_cache_misses\": " << n_actual_misses << ",\n"
              << "      \"router_top9\": {\n"
              << "        \"candidates\": " << dec_acc.router_miss_cands_9 << ",\n"
              << "        \"hits\": " << dec_acc.router_miss_hits_9 << ",\n"
              << "        \"precision\": " << safe_div(dec_acc.router_miss_hits_9, dec_acc.router_miss_cands_9) << ",\n"
              << "        \"recall\": " << safe_div(dec_acc.router_miss_hits_9, n_actual_misses) << "\n"
              << "      },\n"
              << "      \"router_top12\": {\n"
              << "        \"candidates\": " << dec_acc.router_miss_cands_12 << ",\n"
              << "        \"hits\": " << dec_acc.router_miss_hits_12 << ",\n"
              << "        \"precision\": " << safe_div(dec_acc.router_miss_hits_12, dec_acc.router_miss_cands_12) << ",\n"
              << "        \"recall\": " << safe_div(dec_acc.router_miss_hits_12, n_actual_misses) << "\n"
              << "      },\n"
              << "      \"router_top16\": {\n"
              << "        \"candidates\": " << dec_acc.router_miss_cands_16 << ",\n"
              << "        \"hits\": " << dec_acc.router_miss_hits_16 << ",\n"
              << "        \"precision\": " << safe_div(dec_acc.router_miss_hits_16, dec_acc.router_miss_cands_16) << ",\n"
              << "        \"recall\": " << safe_div(dec_acc.router_miss_hits_16, n_actual_misses) << "\n"
              << "      },\n"
              << "      \"learned_top2x3\": {\n"
              << "        \"candidates\": " << dec_acc.learned_miss_cands << ",\n"
              << "        \"hits\": " << dec_acc.learned_miss_hits << ",\n"
              << "        \"precision\": " << safe_div(dec_acc.learned_miss_hits, dec_acc.learned_miss_cands) << ",\n"
              << "        \"recall\": " << safe_div(dec_acc.learned_miss_hits, n_actual_misses) << "\n"
              << "      },\n"
              << "      \"learned_top9\": {\n"
              << "        \"candidates\": " << dec_acc.learned_miss_cands_9 << ",\n"
              << "        \"hits\": " << dec_acc.learned_miss_hits_9 << ",\n"
              << "        \"precision\": " << safe_div(dec_acc.learned_miss_hits_9, dec_acc.learned_miss_cands_9) << ",\n"
              << "        \"recall\": " << safe_div(dec_acc.learned_miss_hits_9, n_actual_misses) << "\n"
              << "      },\n"
              << "      \"hybrid_top12\": {\n"
              << "        \"candidates\": " << dec_acc.hybrid_miss_cands_12 << ",\n"
              << "        \"hits\": " << dec_acc.hybrid_miss_hits_12 << ",\n"
              << "        \"precision\": " << safe_div(dec_acc.hybrid_miss_hits_12, dec_acc.hybrid_miss_cands_12) << ",\n"
              << "        \"recall\": " << safe_div(dec_acc.hybrid_miss_hits_12, n_actual_misses) << "\n"
              << "      },\n"
              << "      \"active_scheduled\": {\n"
              << "        \"candidates\": " << dec_acc.active_pf_cands_scheduled << ",\n"
              << "        \"useful_hits\": " << dec_acc.active_pf_cands_useful << ",\n"
              << "        \"precision\": " << safe_div(dec_acc.active_pf_cands_useful, dec_acc.active_pf_cands_scheduled) << ",\n"
              << "        \"recall\": " << safe_div(dec_acc.active_pf_cands_useful, n_actual_misses) << "\n"
              << "      }\n"
              << "    }\n"
              << "  },\n"
              << "  \"results\": " << prompts_json_arr.str() << "\n"
              << "}\n";

    if (out_f.is_open()) {
        out_f << root_json.str();
        out_f.close();
    } else {
        printf("%s", root_json.str().c_str());
    }

    llama_backend_free();
    return 0;
}

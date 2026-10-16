#include "arg.h"
#include "common.h"
#include "llama.h"
#include "log.h"
#include "sampling.h"
#include <nlohmann/json.hpp>

#include <clocale>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <string>
#include <vector>

using json = nlohmann::json;

struct route_trace_state {
    FILE * out_fp = nullptr;
    int prompt_idx = 0;
    std::string domain = "default";
    std::string phase = "prefill";
    int step_idx = 0; // token offset for current eval call
    std::vector<uint8_t> host_buf;
};

static bool trace_cb_eval(struct ggml_tensor * t, bool ask, void * user_data) {
    auto * st = static_cast<route_trace_state *>(user_data);

    // Only intercept ffn_moe_topk-<il>
    if (strncmp(t->name, "ffn_moe_topk-", 13) != 0) {
        return false;
    }

    int il = -1;
    if (sscanf(t->name + 13, "%d", &il) != 1 || il < 0) {
        return false;
    }

    if (ask) {
        return true; // synchronize and read this tensor before its buffer is reused
    }

    const size_t n_bytes = ggml_nbytes(t);
    const bool is_host = ggml_backend_buffer_is_host(t->buffer);
    const uint8_t * data = nullptr;

    if (!is_host) {
        st->host_buf.resize(n_bytes);
        ggml_backend_tensor_get(t, st->host_buf.data(), 0, n_bytes);
        data = st->host_buf.data();
    } else {
        data = static_cast<const uint8_t *>(t->data);
    }

    const int64_t k = t->ne[0];
    const int64_t n_tokens = t->ne[1];

    if (st->out_fp && t->type == GGML_TYPE_I32) {
        for (int64_t tok = 0; tok < n_tokens; ++tok) {
            fprintf(st->out_fp,
                    "{\"prompt_idx\":%d,\"domain\":\"%s\",\"phase\":\"%s\",\"token_idx\":%lld,\"layer\":%d,\"experts\":[",
                    st->prompt_idx, st->domain.c_str(), st->phase.c_str(),
                    (long long)(st->step_idx + tok), il);
            for (int64_t j = 0; j < k; ++j) {
                const int32_t exp_id = *reinterpret_cast<const int32_t *>(data + tok * t->nb[1] + j * t->nb[0]);
                fprintf(st->out_fp, "%s%d", (j == 0 ? "" : ","), exp_id);
            }
            fprintf(st->out_fp, "]}\n");
        }
    }

    return true;
}

static void batch_set_tokens(llama_batch_ext * batch, const llama_token * tokens, int32_t n_tokens, llama_pos pos_0) {
    llama_batch_ext_clear(batch);
    for (int32_t i = 0; i < n_tokens; ++i) {
        const int32_t idx = llama_batch_ext_add_token(batch, 0, tokens[i]);
        const llama_pos pos = pos_0 + i;
        llama_batch_ext_set_pos(batch, idx, &pos);
    }
    llama_batch_ext_set_output_logits(batch, n_tokens - 1, true);
}

struct prompt_item {
    std::string domain;
    std::string text;
};

static bool run_single_prompt(
        llama_context * ctx,
        const common_params & params,
        route_trace_state & st,
        const prompt_item & item,
        int prompt_idx,
        json & summary_out) {
    const llama_model * model = llama_get_model(ctx);
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const bool add_bos = llama_vocab_get_add_bos(vocab);

    llama_memory_clear(llama_get_memory(ctx), true);

    std::vector<llama_token> prompt_tokens = common_tokenize(ctx, item.text, add_bos, true);
    if (prompt_tokens.empty()) {
        LOG_ERR("Empty tokens for prompt %d\n", prompt_idx);
        return false;
    }

    st.prompt_idx = prompt_idx;
    st.domain = item.domain;
    st.phase = "prefill";
    st.step_idx = 0;

    llama_batch_ext * batch = llama_batch_ext_init(ctx);
    const int32_t n_prompt = (int32_t)prompt_tokens.size();
    const int32_t n_ubatch = params.n_ubatch > 0 ? params.n_ubatch : 512;

    const int64_t t_start_us = ggml_time_us();

    // Prefill in chunks if needed
    for (int32_t i = 0; i < n_prompt; i += n_ubatch) {
        const int32_t n_cur = std::min(n_ubatch, n_prompt - i);
        st.step_idx = i;
        batch_set_tokens(batch, prompt_tokens.data() + i, n_cur, i);
        if (llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch)) {
            LOG_ERR("Failed prefill at prompt %d\n", prompt_idx);
            llama_batch_ext_free(batch);
            return false;
        }
    }

    const int64_t t_prefill_us = ggml_time_us();

    // Initialize sampler
    common_params_sampling sparams = params.sampling;
    common_sampler * smpl = common_sampler_init(model, sparams);

    for (auto id : prompt_tokens) {
        common_sampler_accept(smpl, id, false);
    }

    st.phase = "decode";
    const int n_predict = params.n_predict > 0 ? params.n_predict : 64;
    std::string gen_text;
    int n_decoded = 0;
    llama_pos cur_pos = n_prompt;

    for (int i = 0; i < n_predict; ++i) {
        llama_token new_token_id = common_sampler_sample(smpl, ctx, -1);
        common_sampler_accept(smpl, new_token_id, true);

        if (llama_vocab_is_eog(vocab, new_token_id)) {
            break;
        }

        gen_text += common_token_to_piece(ctx, new_token_id);

        st.step_idx = i;
        batch_set_tokens(batch, &new_token_id, 1, cur_pos);
        cur_pos++;
        n_decoded++;

        if (llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch)) {
            LOG_ERR("Failed decode at prompt %d token %d\n", prompt_idx, i);
            break;
        }
    }

    const int64_t t_end_us = ggml_time_us();
    const double prefill_s = (t_prefill_us - t_start_us) / 1e6;
    const double decode_s = (t_end_us - t_prefill_us) / 1e6;
    const double decode_tps = decode_s > 0 ? n_decoded / decode_s : 0.0;

    summary_out = {
        {"prompt_idx", prompt_idx},
        {"domain", item.domain},
        {"prompt", item.text},
        {"n_prompt", n_prompt},
        {"n_decoded", n_decoded},
        {"prefill_s", prefill_s},
        {"decode_s", decode_s},
        {"decode_tps", decode_tps},
        {"output", gen_text}
    };

    LOG_INF("[prompt %2d | %-4s] prompt=%3d gen=%3d prefill=%.2fs decode=%.2fs (%.2f tok/s)\n",
            prompt_idx, item.domain.c_str(), n_prompt, n_decoded, prefill_s, decode_s, decode_tps);

    if (st.out_fp) {
        fflush(st.out_fp);
    }

    common_sampler_free(smpl);
    llama_batch_ext_free(batch);
    return true;
}

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");

    std::string prompts_json_path;
    std::string out_jsonl_path = "routes.jsonl";
    std::string out_meta_path = "routes_meta.json";
    std::string single_domain = "default";
    int start_prompt = 0;
    int max_prompts = -1;
    bool append_jsonl = false;

    // Filter custom args before passing to common_params_parse
    std::vector<char *> filtered_argv;
    filtered_argv.push_back(argv[0]);
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--prompts-json") == 0 && i + 1 < argc) {
            prompts_json_path = argv[++i];
        } else if (strcmp(argv[i], "--out-jsonl") == 0 && i + 1 < argc) {
            out_jsonl_path = argv[++i];
        } else if (strcmp(argv[i], "--out-meta") == 0 && i + 1 < argc) {
            out_meta_path = argv[++i];
        } else if (strcmp(argv[i], "--domain") == 0 && i + 1 < argc) {
            single_domain = argv[++i];
        } else if (strcmp(argv[i], "--start-prompt") == 0 && i + 1 < argc) {
            start_prompt = std::stoi(argv[++i]);
        } else if (strcmp(argv[i], "--max-prompts") == 0 && i + 1 < argc) {
            max_prompts = std::stoi(argv[++i]);
        } else if (strcmp(argv[i], "--append-jsonl") == 0) {
            append_jsonl = true;
        } else {
            filtered_argv.push_back(argv[i]);
        }
    }

    common_params params;
    common_init();

    if (!common_params_parse((int)filtered_argv.size(), filtered_argv.data(), params, LLAMA_EXAMPLE_COMMON)) {
        return 1;
    }

    llama_backend_init();
    llama_numa_init(params.numa);

    route_trace_state st;
    st.out_fp = fopen(out_jsonl_path.c_str(), append_jsonl ? "a" : "w");
    if (!st.out_fp) {
        LOG_ERR("Failed to open output JSONL file: %s\n", out_jsonl_path.c_str());
        return 1;
    }

    params.cb_eval = trace_cb_eval;
    params.cb_eval_user_data = &st;
    params.warmup = false;

    auto llama_init = common_init_from_params(params);
    auto * model = llama_init->model();
    auto * ctx   = llama_init->context();

    if (model == nullptr || ctx == nullptr) {
        LOG_ERR("Failed to init model/context\n");
        fclose(st.out_fp);
        return 1;
    }

    std::vector<prompt_item> prompts;
    if (!prompts_json_path.empty()) {
        std::ifstream f(prompts_json_path);
        if (!f.is_open()) {
            LOG_ERR("Failed to open prompts json: %s\n", prompts_json_path.c_str());
            fclose(st.out_fp);
            return 1;
        }
        json j;
        f >> j;
        for (const auto & entry : j) {
            if (entry.is_array() && entry.size() >= 2) {
                prompts.push_back({entry[0].get<std::string>(), entry[1].get<std::string>()});
            } else if (entry.is_object()) {
                prompts.push_back({entry.value("domain", "default"), entry.value("prompt", "")});
            }
        }
    } else {
        prompts.push_back({single_domain, params.prompt.empty() ? "Hello" : params.prompt});
    }

    json summaries = json::array();
    int count_run = 0;
    for (size_t i = (size_t)std::max(0, start_prompt); i < prompts.size(); ++i) {
        if (max_prompts > 0 && count_run >= max_prompts) {
            break;
        }
        json s;
        if (!run_single_prompt(ctx, params, st, prompts[i], (int)i, s)) {
            LOG_ERR("Error running prompt %zu\n", i);
            break;
        }
        summaries.push_back(s);
        count_run++;
    }

    fclose(st.out_fp);
    st.out_fp = nullptr;

    // Save metadata and per-prompt summaries
    if (!out_meta_path.empty()) {
        json meta = {
            {"model", params.model.path},
            {"n_layer", llama_model_n_layer(model)},
            {"n_predict", params.n_predict},
            {"prompts", summaries}
        };
        std::ofstream mf(out_meta_path);
        mf << meta.dump(2, ' ', false, json::error_handler_t::replace) << std::endl;
    }

    llama_backend_free();
    return 0;
}

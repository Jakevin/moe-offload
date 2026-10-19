#include "model.hpp"

#include <CommonCrypto/CommonDigest.h>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace qwen38;

static std::string sha256_hex(const std::string& data) {
    unsigned char hash[CC_SHA256_DIGEST_LENGTH];
    CC_SHA256(data.data(), static_cast<CC_LONG>(data.size()), hash);
    std::ostringstream oss;
    oss << std::hex << std::setfill('0');
    for (int i = 0; i < CC_SHA256_DIGEST_LENGTH; ++i) {
        oss << std::setw(2) << static_cast<int>(hash[i]);
    }
    return oss.str();
}

static std::string format_ids_json(const std::vector<int32_t>& ids) {
    std::ostringstream oss;
    oss << "[";
    for (size_t i = 0; i < ids.size(); ++i) {
        if (i > 0) oss << ", ";
        oss << ids[i];
    }
    oss << "]";
    return oss.str();
}

static void skip_ws(const std::string& s, size_t& i) {
    while (i < s.size() && std::isspace(static_cast<unsigned char>(s[i]))) ++i;
}

static std::vector<int32_t> parse_int_array(const std::string& s, size_t& i) {
    std::vector<int32_t> out;
    skip_ws(s, i);
    if (i >= s.size() || s[i] != '[') {
        throw std::runtime_error("Expected '[' when parsing int array");
    }
    ++i;
    while (i < s.size()) {
        skip_ws(s, i);
        if (i < s.size() && s[i] == ']') {
            ++i;
            break;
        }
        if (i < s.size() && s[i] == ',') {
            ++i;
            continue;
        }
        bool neg = false;
        if (s[i] == '-') {
            neg = true;
            ++i;
        }
        int64_t val = 0;
        bool has_digits = false;
        while (i < s.size() && std::isdigit(static_cast<unsigned char>(s[i]))) {
            val = val * 10 + (s[i] - '0');
            ++i;
            has_digits = true;
        }
        if (!has_digits) {
            throw std::runtime_error("Unexpected character in int array");
        }
        out.push_back(static_cast<int32_t>(neg ? -val : val));
    }
    return out;
}

// Supports:
// 1. [id0, id1, ...]
// 2. [[id0, ...], [id0, ...]]
// 3. {"prompts": [{"id": ..., "ids": [...]}, ...]} or {"ids": [...]}
static std::vector<int32_t> load_prompt_ids(const std::string& path, int prompt_index) {
    std::ifstream ifs(path);
    if (!ifs) {
        throw std::runtime_error("Cannot open --ids file: " + path);
    }
    std::ostringstream ss;
    ss << ifs.rdbuf();
    std::string s = ss.str();

    size_t i = 0;
    skip_ws(s, i);
    if (i >= s.size()) {
        throw std::runtime_error("Empty --ids file: " + path);
    }

    if (s[i] == '[') {
        size_t j = i + 1;
        skip_ws(s, j);
        if (j < s.size() && s[j] == '[') {
            // List of lists
            int cur_idx = 0;
            i = j;
            while (i < s.size()) {
                skip_ws(s, i);
                if (i >= s.size() || s[i] == ']') break;
                if (s[i] == ',') { ++i; continue; }
                auto arr = parse_int_array(s, i);
                if (cur_idx == prompt_index) return arr;
                cur_idx++;
            }
            throw std::runtime_error("prompt_index out of range in list-of-lists --ids file");
        }
        return parse_int_array(s, i);
    }

    // Object with "prompts" or "ids"
    int cur_idx = 0;
    size_t search_pos = 0;
    while (true) {
        size_t key_pos = s.find("\"ids\"", search_pos);
        if (key_pos == std::string::npos) {
            key_pos = s.find("\"prompt_token_ids\"", search_pos);
        }
        if (key_pos == std::string::npos) break;
        size_t arr_pos = s.find('[', key_pos);
        if (arr_pos == std::string::npos) break;
        size_t p = arr_pos;
        auto arr = parse_int_array(s, p);
        if (cur_idx == prompt_index) return arr;
        cur_idx++;
        search_pos = p;
    }
    throw std::runtime_error("Could not find prompt_index=" + std::to_string(prompt_index) + " in " + path);
}

static std::vector<int> parse_csv_ints(const std::string& csv) {
    std::vector<int> res;
    std::stringstream ss(csv);
    std::string item;
    while (std::getline(ss, item, ',')) {
        if (!item.empty()) {
            res.push_back(std::stoi(item));
        }
    }
    return res;
}

int main(int argc, char** argv) {
    try {
        EngineOptions opts;
        std::string ids_path;
        std::string out_path;
        std::string logits_out_path;
        int prompt_index = 0;
        int n_tokens = 128;
        bool check_weights_only = false;

        for (int i = 1; i < argc; ++i) {
            std::string arg = argv[i];
            auto next_arg = [&]() -> std::string {
                if (i + 1 >= argc) {
                    throw std::runtime_error("Missing value for argument: " + arg);
                }
                return argv[++i];
            };

            if (arg == "--model") {
                opts.model_dir = next_arg();
            } else if (arg == "--check-weights") {
                check_weights_only = true;
            } else if (arg == "--ids") {
                ids_path = next_arg();
            } else if (arg == "--prompt-index") {
                prompt_index = std::stoi(next_arg());
            } else if (arg == "--n") {
                n_tokens = std::stoi(next_arg());
            } else if (arg == "--out") {
                out_path = next_arg();
            } else if (arg == "--logits-out") {
                logits_out_path = next_arg();
            } else if (arg == "--state-out") {
                opts.state_out = next_arg();
            } else if (arg == "--repeat") {
                opts.repeat = std::stoi(next_arg());
            } else if (arg == "--cold") {
                std::string v = next_arg();
                opts.cold = (v == "1" || v == "true" || v == "on");
            } else if (arg == "--budget") {
                std::string v = next_arg();
                if (!v.empty() && std::isdigit(static_cast<unsigned char>(v[0]))) {
                    opts.budget_mode = "fixed";
                    opts.budget_pct = std::stod(v);
                } else {
                    opts.budget_mode = v;
                }
            } else if (arg == "--budget-pct") {
                opts.budget_pct = std::stod(next_arg());
            } else if (arg == "--budget-log") {
                opts.budget_log = next_arg();
            } else if (arg == "--io") {
                opts.io_mode = next_arg();
            } else if (arg == "--io-threads") {
                opts.io_threads = std::stoi(next_arg());
            } else if (arg == "--force-miss-rate") {
                opts.force_miss_rate = std::stod(next_arg());
            } else if (arg == "--force-miss-seed") {
                opts.force_miss_seed = std::stoull(next_arg());
            } else if (arg == "--alloc") {
                opts.alloc_mode = next_arg();
            } else if (arg == "--prefetch") {
                opts.prefetch_mode = next_arg();
            } else if (arg == "--top-m") {
                opts.top_m = std::stoi(next_arg());
            } else if (arg == "--cap") {
                opts.cap = std::stoi(next_arg());
            } else if (arg == "--slot-cap") {
                opts.slot_cap = std::stoi(next_arg());
            } else if (arg == "--native-sched") {
                std::string v = next_arg();
                opts.native_sched = (v == "1" || v == "true" || v == "on");
            } else if (arg == "--sync-mode") {
                opts.sync_mode = next_arg();
            } else if (arg == "--compile") {
                std::string v = next_arg();
                opts.compile = (v == "1" || v == "true" || v == "on");
            } else if (arg == "--dump-dir") {
                opts.dump_dir = next_arg();
            } else if (arg == "--dump-steps") {
                opts.dump_steps = parse_csv_ints(next_arg());
            } else if (arg == "--no-stop") {
                opts.no_stop = true;
            } else if (arg == "--help" || arg == "-h") {
                std::cout << "Usage: qwen38_decode --model DIR [--check-weights | --ids FILE.json --n 128 --out RESULT.json] [options]\n";
                return 0;
            } else {
                throw std::runtime_error("Unknown CLI argument: " + arg);
            }
        }

        if (opts.model_dir.empty()) {
            const char* env = std::getenv("QWEN38_MODEL");
            if (env && *env) opts.model_dir = env;
            else throw std::runtime_error("--model DIR (or QWEN38_MODEL) is required");
        }

        if (check_weights_only) {
            Qwen38Engine engine(opts);
            return engine.check_weights() ? 0 : 1;
        }

        if (ids_path.empty() || out_path.empty()) {
            throw std::runtime_error("Both --ids and --out are required");
        }

        auto prompt_ids = load_prompt_ids(ids_path, prompt_index);
        Qwen38Engine engine(opts);
        RunMetrics m = engine.generate(prompt_ids, n_tokens, logits_out_path);

        std::string ids_json_arr = format_ids_json(m.tokens);
        std::string ids_sha = sha256_hex(ids_json_arr);

        std::filesystem::create_directories(std::filesystem::path(out_path).parent_path());
        std::ofstream ofs(out_path);
        if (!ofs) {
            throw std::runtime_error("Cannot open output file for writing: " + out_path);
        }

        ofs << std::fixed << std::setprecision(6);
        ofs << "{\n";
        ofs << "  \"prompt_index\": " << prompt_index << ",\n";
        ofs << "  \"prompt_len\": " << prompt_ids.size() << ",\n";
        ofs << "  \"n_generated\": " << m.tokens.size() << ",\n";
        ofs << "  \"ids_sha256\": \"" << ids_sha << "\",\n";
        ofs << "  \"ids\": " << ids_json_arr << ",\n";
        ofs << "  \"load_s\": " << m.load_s << ",\n";
        ofs << "  \"prefill_s\": " << m.prefill_s << ",\n";
        ofs << "  \"ttft_s\": " << m.ttft_s << ",\n";
        ofs << "  \"decode_s\": " << m.decode_s << ",\n";
        ofs << "  \"tok_per_s\": " << m.tok_per_s << ",\n";
        ofs << "  \"ms_per_tok\": " << m.ms_per_tok << ",\n";
        ofs << "  \"p50_ms\": " << m.p50_ms << ",\n";
        ofs << "  \"p90_ms\": " << m.p90_ms << ",\n";
        ofs << "  \"cache_hit_rate\": " << m.cache_hit_rate << ",\n";
        ofs << "  \"read_mib_per_tok\": " << m.read_mib_per_tok << ",\n";
        ofs << "  \"io_pread_bytes\": " << m.io_pread_bytes << ",\n";
        ofs << "  \"decode_requests\": " << m.decode_requests << ",\n";
        ofs << "  \"lru_hits\": " << m.lru_hits << ",\n";
        ofs << "  \"prefetch_hits\": " << m.prefetch_hits << ",\n";
        ofs << "  \"demand_misses\": " << m.demand_misses << ",\n";
        ofs << "  \"slot_buffer_bytes\": " << m.slot_buffer_bytes << ",\n";
        ofs << "  \"peak_rss_bytes\": " << m.peak_rss_bytes << ",\n";
        ofs << "  \"mlx_active_bytes\": " << m.mlx_active_bytes << ",\n";
        ofs << "  \"mlx_peak_bytes\": " << m.mlx_peak_bytes << ",\n";
        ofs << "  \"mlx_cache_bytes\": " << m.mlx_cache_bytes << ",\n";
        ofs << "  \"budget_mode\": \"" << opts.budget_mode << "\",\n";
        ofs << "  \"budget_adjustments\": " << m.budget_adjustments << ",\n";
        ofs << "  \"min_total_slots\": " << m.min_total_slots << ",\n";
        ofs << "  \"max_total_slots\": " << m.max_total_slots << ",\n";
        ofs << "  \"max_pressure_level\": " << m.max_pressure_level << ",\n";
        ofs << "  \"max_swap_used_bytes\": " << m.max_swap_used_bytes << ",\n";
        ofs << "  \"io_mode\": \"" << opts.io_mode << "\",\n";
        ofs << "  \"io_threads\": " << opts.io_threads << ",\n";
        ofs << "  \"force_miss_rate\": " << opts.force_miss_rate << ",\n";
        ofs << "  \"forced_invalidations\": " << m.forced_invalidations << ",\n";
        ofs << "  \"urgent_reads\": " << m.urgent_reads << ",\n";
        ofs << "  \"prefetch_issued\": " << m.prefetch_issued << ",\n";
        ofs << "  \"prefetch_cancelled\": " << m.prefetch_cancelled << ",\n";
        ofs << "  \"prefetch_promoted\": " << m.prefetch_promoted << ",\n";
        ofs << "  \"prefetch_used\": " << m.prefetch_used << ",\n";
        ofs << "  \"prefetch_wasted\": " << m.prefetch_wasted << ",\n";
        ofs << "  \"pin_waits\": " << m.pin_waits << ",\n";
        ofs << "  \"io_wait_ms\": " << m.io_wait_ms;
        if (!m.p1_layer_stats.empty()) {
            ofs << ",\n  \"p1_per_layer\": [\n";
            for (size_t idx = 0; idx < m.p1_layer_stats.size(); ++idx) {
                const auto& ls = m.p1_layer_stats[idx];
                ofs << "    {\"layer\": " << (idx + 1)
                    << ", \"urgent_reads\": " << ls.urgent_reads
                    << ", \"prefetch_issued\": " << ls.prefetch_issued
                    << ", \"prefetch_cancelled\": " << ls.prefetch_cancelled
                    << ", \"prefetch_promoted\": " << ls.prefetch_promoted
                    << ", \"prefetch_used\": " << ls.prefetch_used
                    << ", \"prefetch_wasted\": " << ls.prefetch_wasted
                    << ", \"pin_waits\": " << ls.pin_waits
                    << ", \"bytes\": " << ls.bytes
                    << ", \"io_wait_ms\": " << ls.io_wait_ms() << "}"
                    << (idx + 1 < m.p1_layer_stats.size() ? ",\n" : "\n");
            }
            ofs << "  ]\n";
        } else {
            ofs << "\n";
        }
        ofs << "}\n";
        ofs.close();

        // Never print token IDs or decoded text to stdout/logs; print sha256/len only.
        std::cout << "[qwen38_decode] prompt_index=" << prompt_index
                  << " prompt_len=" << prompt_ids.size()
                  << " n_generated=" << m.tokens.size()
                  << " ids_sha256=" << ids_sha
                  << " prefill_s=" << std::fixed << std::setprecision(3) << m.prefill_s
                  << " tok_per_s=" << m.tok_per_s
                  << " hit_rate=" << (m.cache_hit_rate * 100.0) << "%"
                  << " read_mib_tok=" << m.read_mib_per_tok << "\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "[qwen38_decode ERROR] " << e.what() << "\n";
        return 1;
    }
}

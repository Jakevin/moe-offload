#include "model.hpp"
#include "expert_io.h"
#include <algorithm>
#include <cctype>
#include <cstring>
#include <filesystem>
#include <fcntl.h>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace qwen38 {

// Minimal safetensors header entry
struct SafeTensorEntry {
    std::string dtype;
    std::vector<int64_t> shape;
    uint64_t data_offset_start = 0;
    uint64_t data_offset_end = 0;
    std::string src_file;
    std::string src_key;
};

struct SafeTensorHeader {
    uint64_t data_start = 0;
    std::unordered_map<std::string, SafeTensorEntry> tensors;
};

static void skip_ws(const std::string& s, size_t& i) {
    while (i < s.size() && std::isspace(static_cast<unsigned char>(s[i]))) ++i;
}

static std::string parse_json_string(const std::string& s, size_t& i) {
    skip_ws(s, i);
    if (i >= s.size() || s[i] != '"') {
        throw std::runtime_error("Expected '\"' in JSON at pos " + std::to_string(i));
    }
    ++i;
    std::string out;
    while (i < s.size() && s[i] != '"') {
        if (s[i] == '\\' && i + 1 < s.size()) {
            ++i;
            char c = s[i];
            if (c == 'n') out.push_back('\n');
            else if (c == 't') out.push_back('\t');
            else if (c == 'r') out.push_back('\r');
            else out.push_back(c);
        } else {
            out.push_back(s[i]);
        }
        ++i;
    }
    if (i < s.size() && s[i] == '"') ++i;
    return out;
}

static int64_t parse_json_int(const std::string& s, size_t& i) {
    skip_ws(s, i);
    size_t start = i;
    if (i < s.size() && (s[i] == '-' || s[i] == '+')) ++i;
    while (i < s.size() && std::isdigit(static_cast<unsigned char>(s[i]))) ++i;
    return std::stoll(s.substr(start, i - start));
}

static void skip_json_value(const std::string& s, size_t& i) {
    skip_ws(s, i);
    if (i >= s.size()) return;
    if (s[i] == '"') {
        parse_json_string(s, i);
    } else if (s[i] == '{') {
        int depth = 0;
        do {
            if (s[i] == '"') {
                parse_json_string(s, i);
                continue;
            }
            if (s[i] == '{') depth++;
            else if (s[i] == '}') depth--;
            ++i;
        } while (i < s.size() && depth > 0);
    } else if (s[i] == '[') {
        int depth = 0;
        do {
            if (s[i] == '"') {
                parse_json_string(s, i);
                continue;
            }
            if (s[i] == '[') depth++;
            else if (s[i] == ']') depth--;
            ++i;
        } while (i < s.size() && depth > 0);
    } else {
        while (i < s.size() && s[i] != ',' && s[i] != '}' && s[i] != ']' &&
               !std::isspace(static_cast<unsigned char>(s[i]))) {
            ++i;
        }
    }
}

static SafeTensorHeader parse_safetensors_header(const std::string& path) {
    int fd = open(path.c_str(), O_RDONLY);
    if (fd < 0) {
        throw std::runtime_error("Cannot open safetensors file: " + path);
    }
    uint64_t header_len = 0;
    if (pread(fd, &header_len, 8, 0) != 8) {
        close(fd);
        throw std::runtime_error("Failed to read 8-byte header size from: " + path);
    }
    std::string json_str(header_len, '\0');
    size_t remaining = header_len;
    uint64_t off = 8;
    char* dst = json_str.data();
    while (remaining > 0) {
        ssize_t n = pread(fd, dst, remaining, off);
        if (n <= 0) {
            close(fd);
            throw std::runtime_error("Failed reading safetensors JSON header: " + path);
        }
        remaining -= static_cast<size_t>(n);
        off += static_cast<uint64_t>(n);
        dst += n;
    }
    close(fd);

    SafeTensorHeader result;
    result.data_start = 8 + header_len;
    std::string fname = std::filesystem::path(path).filename().string();

    size_t i = 0;
    skip_ws(json_str, i);
    if (i >= json_str.size() || json_str[i] != '{') {
        throw std::runtime_error("Invalid safetensors JSON header in: " + path);
    }
    ++i;
    while (i < json_str.size()) {
        skip_ws(json_str, i);
        if (i >= json_str.size() || json_str[i] == '}') break;
        if (json_str[i] == ',') { ++i; continue; }
        std::string key = parse_json_string(json_str, i);
        skip_ws(json_str, i);
        if (i < json_str.size() && json_str[i] == ':') ++i;
        skip_ws(json_str, i);
        if (key == "__metadata__") {
            skip_json_value(json_str, i);
            continue;
        }
        if (i >= json_str.size() || json_str[i] != '{') {
            skip_json_value(json_str, i);
            continue;
        }
        ++i; // enter tensor object
        SafeTensorEntry entry;
        entry.src_file = fname;
        entry.src_key = key;
        while (i < json_str.size()) {
            skip_ws(json_str, i);
            if (i >= json_str.size() || json_str[i] == '}') { ++i; break; }
            if (json_str[i] == ',') { ++i; continue; }
            std::string field = parse_json_string(json_str, i);
            skip_ws(json_str, i);
            if (i < json_str.size() && json_str[i] == ':') ++i;
            skip_ws(json_str, i);
            if (field == "dtype") {
                entry.dtype = parse_json_string(json_str, i);
            } else if (field == "shape") {
                if (json_str[i] == '[') {
                    ++i;
                    while (i < json_str.size()) {
                        skip_ws(json_str, i);
                        if (json_str[i] == ']') { ++i; break; }
                        if (json_str[i] == ',') { ++i; continue; }
                        entry.shape.push_back(parse_json_int(json_str, i));
                    }
                } else {
                    skip_json_value(json_str, i);
                }
            } else if (field == "data_offsets") {
                if (json_str[i] == '[') {
                    ++i;
                    skip_ws(json_str, i);
                    entry.data_offset_start = static_cast<uint64_t>(parse_json_int(json_str, i));
                    skip_ws(json_str, i);
                    if (json_str[i] == ',') ++i;
                    skip_ws(json_str, i);
                    entry.data_offset_end = static_cast<uint64_t>(parse_json_int(json_str, i));
                    skip_ws(json_str, i);
                    if (json_str[i] == ']') ++i;
                } else {
                    skip_json_value(json_str, i);
                }
            } else {
                skip_json_value(json_str, i);
            }
        }
        result.tensors.emplace(std::move(key), std::move(entry));
    }
    return result;
}

static std::unordered_map<std::string, std::string> parse_weight_map(const std::string& index_json_path) {
    std::ifstream ifs(index_json_path);
    if (!ifs) {
        throw std::runtime_error("Cannot open index file: " + index_json_path);
    }
    std::ostringstream ss;
    ss << ifs.rdbuf();
    std::string s = ss.str();

    std::unordered_map<std::string, std::string> wmap;
    size_t pos = s.find("\"weight_map\"");
    if (pos == std::string::npos) {
        throw std::runtime_error("No weight_map in " + index_json_path);
    }
    size_t i = s.find('{', pos);
    if (i == std::string::npos) {
        throw std::runtime_error("Malformed weight_map in " + index_json_path);
    }
    ++i;
    while (i < s.size()) {
        skip_ws(s, i);
        if (i >= s.size() || s[i] == '}') break;
        if (s[i] == ',') { ++i; continue; }
        std::string k = parse_json_string(s, i);
        skip_ws(s, i);
        if (i < s.size() && s[i] == ':') ++i;
        std::string v = parse_json_string(s, i);
        wmap.emplace(std::move(k), std::move(v));
    }
    return wmap;
}

static bool starts_with(const std::string& s, const std::string& pfx) {
    return s.size() >= pfx.size() && s.compare(0, pfx.size(), pfx) == 0;
}

// Mirrors Qwen4Exp.Model.sanitize + LanguageModel.sanitize in qwen4_exp/qwen4_exp.py & language.py
static bool should_drop_weight_key(const std::string& k) {
    if (starts_with(k, "vision_tower.") || starts_with(k, "model.vision_tower.")) return true;
    if (starts_with(k, "mtp.") || starts_with(k, "model.language_model.mtp.") ||
        starts_with(k, "language_model.mtp.") || starts_with(k, "language_model.model.mtp.")) {
        return true;
    }
    if (k.find("rotary_emb.inv_freq") != std::string::npos) return true;
    return false;
}

static std::string sanitize_weight_key(const std::string& k) {
    if (starts_with(k, "model.language_model.lm_head.")) {
        return "language_model.lm_head." + k.substr(std::strlen("model.language_model.lm_head."));
    }
    if (starts_with(k, "language_model.model.lm_head.")) {
        return "language_model.lm_head." + k.substr(std::strlen("language_model.model.lm_head."));
    }
    if (starts_with(k, "model.language_model.")) {
        return "language_model.model." + k.substr(std::strlen("model.language_model."));
    }
    if (starts_with(k, "model.") && !starts_with(k, "model.layers.")) {
        return "language_model." + k;
    }
    if (starts_with(k, "layers.") || starts_with(k, "embed_tokens.") || starts_with(k, "hyper_connection_mixer.")) {
        return "language_model.model." + k;
    }
    if (starts_with(k, "lm_head.")) {
        return "language_model." + k;
    }
    return k;
}

static std::vector<std::string> list_trunk_safetensors(const std::string& model_dir) {
    std::vector<std::string> base_files;
    std::vector<std::string> model_files;
    for (const auto& entry : std::filesystem::directory_iterator(model_dir)) {
        if (!entry.is_regular_file()) continue;
        std::string fname = entry.path().filename().string();
        if (fname.size() > 12 && fname.substr(fname.size() - 12) == ".safetensors") {
            if (starts_with(fname, "base-")) {
                base_files.push_back(fname);
            } else if (starts_with(fname, "model-")) {
                model_files.push_back(fname);
            }
        }
    }
    std::sort(base_files.begin(), base_files.end());
    std::sort(model_files.begin(), model_files.end());
    std::vector<std::string> out = !base_files.empty() ? base_files : model_files;
    if (std::filesystem::exists(model_dir + "/experts-00.safetensors")) {
        out.push_back("experts-00.safetensors");
    }
    return out;
}

static std::vector<std::string> list_ple_safetensors(const std::string& model_dir) {
    std::vector<std::string> ple_files;
    for (const auto& entry : std::filesystem::directory_iterator(model_dir)) {
        if (!entry.is_regular_file()) continue;
        std::string fname = entry.path().filename().string();
        if (starts_with(fname, "ple-") && fname.size() > 12 && fname.substr(fname.size() - 12) == ".safetensors") {
            ple_files.push_back(fname);
        }
    }
    std::sort(ple_files.begin(), ple_files.end());
    return ple_files;
}

extern void init_ple_metadata(PleLayer& ple, int unigram_vocab_size, int ngram_vocab_base, int divisor, int seed);

bool Qwen38Engine::check_weights() const {
    if (std::filesystem::exists(opts_.model_dir + "/skeleton.safetensors") &&
        std::filesystem::exists(opts_.model_dir + "/nonexpert.safetensors")) {
        for (const char* req : {"/skeleton.safetensors", "/nonexpert.safetensors",
                                "/ple/ple_bits.json", "/ple/model.safetensors.index.json"}) {
            if (!std::filesystem::exists(opts_.model_dir + req)) {
                std::cerr << "[check-weights FAIL] Missing K50 v3 file: " << (opts_.model_dir + req) << "\n";
                return false;
            }
        }
        for (int l = 0; l < 48; ++l) {
            char buf[32];
            std::snprintf(buf, sizeof(buf), "/L%02d/index.json", l);
            if (!std::filesystem::exists(opts_.model_dir + buf)) {
                std::cerr << "[check-weights FAIL] Missing K50 v3 layer index: " << (opts_.model_dir + buf) << "\n";
                return false;
            }
        }
        std::cout << "[check-weights] Verified K50 v3 model directory: " << opts_.model_dir << "\n";
        return true;
    }

    // 1. Verify config.json exists and has expected Qwen4Exp parameters
    std::string cfg_path = opts_.model_dir + "/config.json";
    std::ifstream cfg_ifs(cfg_path);
    if (!cfg_ifs) {
        std::cerr << "[check-weights FAIL] Cannot open " << cfg_path << "\n";
        return false;
    }
    std::ostringstream cfg_ss;
    cfg_ss << cfg_ifs.rdbuf();
    std::string cfg_str = cfg_ss.str();
    for (const char* req : {"\"qwen4_exp\"", "\"hidden_size\": 2560", "\"num_hidden_layers\": 48",
                            "\"num_experts\": 512", "\"num_experts_per_tok\": 10",
                            "\"hc_count\": 4", "\"hc_lowrank\": 320", "\"ple_embed_dim\": 2560"}) {
        if (cfg_str.find(req) == std::string::npos) {
            std::cerr << "[check-weights FAIL] config.json missing expected setting: " << req << "\n";
            return false;
        }
    }

    // 2. Parse safetensors headers for all trunk + Layer 0 files (header bytes only, no mx::load or mmap)
    auto trunk_files = list_trunk_safetensors(opts_.model_dir);
    if (trunk_files.empty()) {
        std::cerr << "[check-weights FAIL] No base-*.safetensors found in " << opts_.model_dir << "\n";
        return false;
    }

    std::unordered_map<std::string, SafeTensorEntry> san_headers;
    size_t raw_file_keys = 0;
    for (const auto& fname : trunk_files) {
        auto hdr = parse_safetensors_header(opts_.model_dir + "/" + fname);
        for (const auto& kv : hdr.tensors) {
            if (starts_with(kv.first, "vision_tower.")) continue;
            raw_file_keys++;
            if (should_drop_weight_key(kv.first)) continue;
            std::string skey = sanitize_weight_key(kv.first);
            SafeTensorEntry e = kv.second;
            // Qwen4Exp.Model.sanitize transposes 3D conv1d.weight only if shape[1] < shape[2]
            if (skey.find("conv1d.weight") != std::string::npos && e.shape.size() == 3 && e.shape[1] < e.shape[2]) {
                std::swap(e.shape[1], e.shape[2]);
            }
            san_headers[skey] = std::move(e);
        }
    }

    // Qwen4ExpNGramEmbedding / DiskBackedShardedEmbedding synthesizes weight_scale = ones((1,), bfloat16)
    std::string ws_key = "language_model.model.layers.1.ple.ple_embedding.ngram_embedding.weight_scale";
    if (san_headers.find(ws_key) == san_headers.end()) {
        SafeTensorEntry ws_entry;
        ws_entry.dtype = "BF16";
        ws_entry.shape = {1};
        ws_entry.src_file = "<synthesized>";
        ws_entry.src_key = ws_key;
        san_headers[ws_key] = ws_entry;
    }

    // 3. Build complete expected trunk + Layer 0 tensor specifications (2628 tensors)
    struct ExpectedSpec {
        std::string key;
        std::vector<int64_t> shape;
        std::string dtype;
    };
    std::vector<ExpectedSpec> expected;
    expected.reserve(2628);

    auto add_qlinear_spec = [&](const std::string& pfx, int64_t out_f, int64_t in_f, int bits, int gs) {
        int64_t w_cols = (in_f * bits) / 32;
        int64_t s_cols = in_f / gs;
        expected.push_back({pfx + ".weight", {out_f, w_cols}, "U32"});
        expected.push_back({pfx + ".scales", {out_f, s_cols}, "BF16"});
        expected.push_back({pfx + ".biases", {out_f, s_cols}, "BF16"});
    };
    auto add_hc_spec = [&](const std::string& pfx, bool use_combine) {
        expected.push_back({pfx + ".hc_norm.weight", {10240}, "BF16"});
        add_qlinear_spec(pfx + ".input_mix_weight_down", 320, 10240, 8, 64);
        add_qlinear_spec(pfx + ".input_mix_weight_up", 10240, 320, 8, 64);
        if (use_combine) {
            add_qlinear_spec(pfx + ".block_inject_weight", 4, 10240, 8, 64);
        }
    };

    // Embed tokens, final mixer, lm_head
    add_qlinear_spec("language_model.model.embed_tokens", 248320, 2560, 4, 32);
    add_hc_spec("language_model.model.hyper_connection_mixer", false);
    add_qlinear_spec("language_model.lm_head", 248320, 2560, 4, 32);

    for (int l = 0; l < 48; ++l) {
        std::string lp = "language_model.model.layers." + std::to_string(l);
        add_hc_spec(lp + ".attn_hyper_connection", true);
        add_hc_spec(lp + ".mlp_hyper_connection", true);

        bool is_lin = (l % 4 != 3);
        if (is_lin) {
            std::string ap = lp + ".linear_attn";
            add_qlinear_spec(ap + ".in_proj_qkv", 10240, 2560, 4, 32);
            add_qlinear_spec(ap + ".in_proj_z", 6144, 2560, 4, 32);
            add_qlinear_spec(ap + ".in_proj_b", 48, 2560, 4, 32);
            add_qlinear_spec(ap + ".in_proj_a", 48, 2560, 4, 32);
            add_qlinear_spec(ap + ".out_proj", 2560, 6144, 4, 32);
            expected.push_back({ap + ".conv1d.weight", {10240, 4, 1}, "BF16"});
            expected.push_back({ap + ".dt_bias", {48}, "BF16"});
            expected.push_back({ap + ".A_log", {48}, "BF16"});
            expected.push_back({ap + ".norm.weight", {128}, "BF16"});
        } else {
            std::string ap = lp + ".self_attn";
            add_qlinear_spec(ap + ".q_proj", 12288, 2560, 4, 32);
            add_qlinear_spec(ap + ".k_proj", 512, 2560, 4, 32);
            add_qlinear_spec(ap + ".v_proj", 512, 2560, 4, 32);
            add_qlinear_spec(ap + ".o_proj", 2560, 6144, 4, 32);
            expected.push_back({ap + ".q_norm.weight", {256}, "BF16"});
            expected.push_back({ap + ".k_norm.weight", {256}, "BF16"});
            add_qlinear_spec(ap + ".indexer.index_qk_proj", 640, 2560, 4, 32);
            expected.push_back({ap + ".indexer.q_layernorm.weight", {128}, "BF16"});
            expected.push_back({ap + ".indexer.k_layernorm.weight", {128}, "BF16"});
        }

        std::string mp = lp + ".mlp";
        add_qlinear_spec(mp + ".gate", 512, 2560, 8, 64);
        add_qlinear_spec(mp + ".shared_expert_gate", 1, 2560, 8, 64);
        add_qlinear_spec(mp + ".shared_expert.gate_proj", 640, 2560, 4, 32);
        add_qlinear_spec(mp + ".shared_expert.up_proj", 640, 2560, 4, 32);
        add_qlinear_spec(mp + ".shared_expert.down_proj", 2560, 640, 4, 32);

        if (l == 0) {
            std::string sp = mp + ".switch_mlp";
            expected.push_back({sp + ".gate_proj.weight", {512, 640, 160}, "U32"});
            expected.push_back({sp + ".gate_proj.scales", {512, 640, 40}, "BF16"});
            expected.push_back({sp + ".gate_proj.biases", {512, 640, 40}, "BF16"});
            expected.push_back({sp + ".up_proj.weight", {512, 640, 160}, "U32"});
            expected.push_back({sp + ".up_proj.scales", {512, 640, 40}, "BF16"});
            expected.push_back({sp + ".up_proj.biases", {512, 640, 40}, "BF16"});
            expected.push_back({sp + ".down_proj.weight", {512, 2560, 40}, "U32"});
            expected.push_back({sp + ".down_proj.scales", {512, 2560, 10}, "BF16"});
            expected.push_back({sp + ".down_proj.biases", {512, 2560, 10}, "BF16"});
        }

        if (l == 1) {
            std::string pp = lp + ".ple";
            add_qlinear_spec(pp + ".key_proj", 10240, 2560, 4, 32);
            add_qlinear_spec(pp + ".value_proj", 2560, 2560, 4, 32);
            expected.push_back({pp + ".norm_key.weight", {10240}, "BF16"});
            expected.push_back({pp + ".norm_query.weight", {10240}, "BF16"});
            expected.push_back({pp + ".norm_conv.weight", {10240}, "BF16"});
            expected.push_back({pp + ".conv1d.weight", {10240, 4, 1}, "BF16"});
            expected.push_back({pp + ".ple_embedding.layer_multipliers", {3}, "I64"});
            expected.push_back({pp + ".ple_embedding.ngram_heads_vocab_sizes", {16}, "I64"});
            expected.push_back({pp + ".ple_embedding.ngram_heads_offsets", {16}, "I64"});
            expected.push_back({pp + ".ple_embedding.ngram_embedding.weight_scale", {1}, "BF16"});
        }
    }

    int matched_trunk = 0;
    for (const auto& exp : expected) {
        auto it = san_headers.find(exp.key);
        if (it == san_headers.end()) {
            std::cerr << "[check-weights FAIL] Missing expected trunk tensor: " << exp.key << "\n";
            return false;
        }
        const auto& got = it->second;
        if (got.shape != exp.shape || got.dtype != exp.dtype) {
            std::cerr << "[check-weights FAIL] Shape/dtype mismatch for " << exp.key
                      << ": got dtype=" << got.dtype << " expected=" << exp.dtype << "\n";
            return false;
        }
        matched_trunk++;
    }
    if (san_headers.size() != expected.size()) {
        std::cerr << "[check-weights FAIL] Sanitized header count (" << san_headers.size()
                  << ") != expected count (" << expected.size() << ")\n";
        return false;
    }

    // Also verify against out/weight_manifest.json if present
    for (const char* mpath : {"out/weight_manifest.json", "../out/weight_manifest.json"}) {
        std::ifstream mifs(mpath);
        if (!mifs) continue;
        std::ostringstream mss;
        mss << mifs.rdbuf();
        std::string ms = mss.str();
        size_t san_pos = ms.find("\"sanitized\"");
        if (san_pos == std::string::npos) break;
        size_t i = ms.find('{', san_pos);
        if (i == std::string::npos) break;
        ++i;
        int manifest_count = 0;
        int missing_in_cpp = 0;
        while (i < ms.size()) {
            skip_ws(ms, i);
            if (i >= ms.size() || ms[i] == '}') break;
            if (ms[i] == ',') { ++i; continue; }
            std::string mkey = parse_json_string(ms, i);
            skip_ws(ms, i);
            if (i < ms.size() && ms[i] == ':') ++i;
            skip_json_value(ms, i);
            manifest_count++;
            if (san_headers.find(mkey) == san_headers.end()) {
                std::cerr << "[check-weights FAIL] Missing manifest key in C++ resolution: " << mkey << "\n";
                missing_in_cpp++;
            }
        }
        int extra_in_cpp = static_cast<int>(san_headers.size()) - (manifest_count - missing_in_cpp);
        if (missing_in_cpp != 0 || extra_in_cpp != 0 || manifest_count != static_cast<int>(expected.size())) {
            std::cerr << "[check-weights FAIL] Manifest mismatch vs " << mpath
                      << ": manifest=" << manifest_count << " missing=" << missing_in_cpp
                      << " extra=" << extra_in_cpp << "\n";
            return false;
        }
        std::cout << "[check-weights] Verified against " << mpath
                  << ": manifest_keys=" << manifest_count << " missing=0 extra=0\n";
        break;
    }

    // 4. Verify all 47 offloaded expert shard files (experts-01..experts-47.safetensors)
    int matched_expert_shards = 0;
    for (int l = 1; l < 48; ++l) {
        char fname_buf[64];
        std::snprintf(fname_buf, sizeof(fname_buf), "experts-%02d.safetensors", l);
        std::string exp_path = opts_.model_dir + "/" + fname_buf;
        auto hdr = parse_safetensors_header(exp_path);
        std::string sp = "language_model.model.layers." + std::to_string(l) + ".mlp.switch_mlp.";
        struct SubSpec { const char* sub; std::vector<int64_t> shape; const char* dtype; };
        const SubSpec subs[9] = {
            {"gate_proj.weight", {512, 640, 160}, "U32"},
            {"gate_proj.scales", {512, 640, 40},  "BF16"},
            {"gate_proj.biases", {512, 640, 40},  "BF16"},
            {"up_proj.weight",   {512, 640, 160}, "U32"},
            {"up_proj.scales",   {512, 640, 40},  "BF16"},
            {"up_proj.biases",   {512, 640, 40},  "BF16"},
            {"down_proj.weight", {512, 2560, 40}, "U32"},
            {"down_proj.scales", {512, 2560, 10}, "BF16"},
            {"down_proj.biases", {512, 2560, 10}, "BF16"},
        };
        for (const auto& s : subs) {
            std::string key = sp + s.sub;
            auto it = hdr.tensors.find(key);
            if (it == hdr.tensors.end()) {
                std::cerr << "[check-weights FAIL] Missing expert tensor " << key << " in " << fname_buf << "\n";
                return false;
            }
            if (it->second.shape != s.shape || it->second.dtype != s.dtype) {
                std::cerr << "[check-weights FAIL] Expert tensor shape/dtype mismatch for " << key << "\n";
                return false;
            }
        }
        matched_expert_shards++;
    }

    // 5. Verify all 128 PLE shards across ple-*.safetensors
    PleLayer ple_meta;
    init_ple_metadata(ple_meta, 248320, 20000000, 128, 1234);
    auto ple_files = list_ple_safetensors(opts_.model_dir);
    std::unordered_map<std::string, SafeTensorEntry> ple_tensors;
    for (const auto& pfname : ple_files) {
        auto hdr = parse_safetensors_header(opts_.model_dir + "/" + pfname);
        for (const auto& kv : hdr.tensors) {
            ple_tensors.emplace(sanitize_weight_key(kv.first), kv.second);
        }
    }
    int matched_ple_shards = 0;
    std::string ple_base = "language_model.model.layers.1.ple.ple_embedding.ngram_embedding.shards.";
    for (int s = 0; s < ple_meta.split_ngram_parts; ++s) {
        int64_t expected_rows = ple_meta.shard_offsets[s + 1] - ple_meta.shard_offsets[s];
        std::string sp = ple_base + std::to_string(s);
        auto w_it = ple_tensors.find(sp + ".weight");
        auto s_it = ple_tensors.find(sp + ".scales");
        auto b_it = ple_tensors.find(sp + ".biases");
        if (w_it == ple_tensors.end() || s_it == ple_tensors.end() || b_it == ple_tensors.end()) {
            std::cerr << "[check-weights FAIL] Missing PLE shard " << s << " tensors\n";
            return false;
        }
        std::vector<int64_t> exp_w_shape = {expected_rows, 20};
        std::vector<int64_t> exp_s_shape = {expected_rows, 5};
        if (w_it->second.shape != exp_w_shape || w_it->second.dtype != "U32" ||
            s_it->second.shape != exp_s_shape || s_it->second.dtype != "BF16" ||
            b_it->second.shape != exp_s_shape || b_it->second.dtype != "BF16") {
            std::cerr << "[check-weights FAIL] PLE shard " << s << " shape/dtype mismatch\n";
            return false;
        }
        matched_ple_shards++;
    }

    std::cout << "[check-weights] " << matched_trunk << "/" << expected.size()
              << " trunk+L0 tensors matched (raw_file_keys=" << raw_file_keys << "), "
              << matched_expert_shards << "/47 expert shards matched, "
              << matched_ple_shards << "/" << ple_meta.split_ngram_parts << " PLE shards matched\n";
    return true;
}

static QLinear load_qlinear(
    const std::unordered_map<std::string, mx::array>& tensors,
    const std::string& prefix,
    int default_group_size,
    int default_bits) {
    QLinear ql;
    auto w_it = tensors.find(prefix + ".weight");
    if (w_it == tensors.end()) {
        throw std::runtime_error("Missing weight tensor: " + prefix + ".weight");
    }
    ql.weight = w_it->second;
    auto s_it = tensors.find(prefix + ".scales");
    if (s_it != tensors.end()) {
        ql.is_quantized = true;
        ql.scales = s_it->second;
        auto b_it = tensors.find(prefix + ".biases");
        if (b_it != tensors.end()) {
            ql.biases = b_it->second;
        }
        ql.group_size = default_group_size;
        ql.bits = default_bits;
    } else {
        ql.is_quantized = false;
    }
    auto bias_it = tensors.find(prefix + ".bias");
    if (bias_it != tensors.end()) {
        ql.bias = bias_it->second;
    }
    return ql;
}

static HyperConnection load_hc(
    const std::unordered_map<std::string, mx::array>& tensors,
    const std::string& prefix,
    bool use_combine) {
    HyperConnection hc;
    hc.use_combine = use_combine;
    auto nw_it = tensors.find(prefix + ".hc_norm.weight");
    if (nw_it == tensors.end()) {
        throw std::runtime_error("Missing hc_norm.weight: " + prefix);
    }
    hc.norm_weight = nw_it->second;
    hc.eps_arr = mx::array({1e-6f}, mx::float32);
    mx::eval({hc.norm_weight, hc.eps_arr});
    hc.down = load_qlinear(tensors, prefix + ".input_mix_weight_down", 64, 8);
    hc.up = load_qlinear(tensors, prefix + ".input_mix_weight_up", 64, 8);
    if (use_combine) {
        hc.inject = load_qlinear(tensors, prefix + ".block_inject_weight", 64, 8);
    }
    return hc;
}

// Mirrors mlx_vlm rope_utils.compute_inv_freq exactly (same MLX op graph on the default GPU stream):
//   1.0 / (base ** (mx.arange(0, dim, 2).astype(mx.float32) / dim))
// A CPU std::pow version differed from Python by 1-2 fp32 ulp on 11/32 lanes (found while debugging the bit-exact gate).
mx::array make_rope_inv_freq(int rotary_dim, float theta) {
    auto expo = mx::divide(
        mx::astype(mx::arange(0, rotary_dim, 2, mx::int32), mx::float32),
        mx::array(static_cast<float>(rotary_dim), mx::float32));
    auto arr = mx::divide(
        mx::array(1.0f, mx::float32),
        mx::power(mx::array(theta, mx::float32), expo));
    mx::eval({arr});
    return arr;
}

void Qwen38Engine::load() {
    if (loaded_) return;
    init_kernels();

    if (std::filesystem::exists(opts_.model_dir + "/skeleton.safetensors") &&
        std::filesystem::exists(opts_.model_dir + "/nonexpert.safetensors")) {
        load_k50_v3();
        return;
    }

    std::string index_path = opts_.model_dir + "/model.safetensors.index.json";
    std::unordered_map<std::string, std::string> weight_map;
    if (std::filesystem::exists(index_path)) {
        weight_map = parse_weight_map(index_path);
    }

    // Load trunk files (base-*.safetensors) + Layer 0 expert file (experts-00.safetensors) via mx::load_safetensors
    std::vector<std::string> trunk_files = list_trunk_safetensors(opts_.model_dir);
    if (trunk_files.empty()) {
        throw std::runtime_error("No trunk safetensors files found in " + opts_.model_dir);
    }
    std::unordered_map<std::string, mx::array> trunk_tensors;
    for (const auto& fname : trunk_files) {
        auto [loaded, _meta] = mx::load_safetensors(opts_.model_dir + "/" + fname);
        for (auto& kv : loaded) {
            if (should_drop_weight_key(kv.first)) continue;
            std::string skey = sanitize_weight_key(kv.first);
            mx::array arr = kv.second;
            if (skey.find("conv1d.weight") != std::string::npos && arr.ndim() == 3 && arr.shape(1) < arr.shape(2)) {
                arr = mx::swapaxes(arr, 1, 2);
            }
            trunk_tensors.emplace(std::move(skey), std::move(arr));
        }
    }

    // 1. Token embeddings, final hyper_connection_mixer (use_combine=false), and LM head
    // Note: Qwen4ExpModel has NO final RMSNorm before lm_head; hyper_connection_mixer outputs [B, L, 2560] directly to lm_head!
    embed_tokens_ = load_qlinear(trunk_tensors, "language_model.model.embed_tokens", 32, 4);
    final_mixer_ = load_hc(trunk_tensors, "language_model.model.hyper_connection_mixer", false);
    lm_head_ = load_qlinear(trunk_tensors, "language_model.lm_head", 32, 4);

    // 2. Initialize C expert_io worker pool (8 workers matching loader.py line 252) or P1 scheduler (--io p1)
    if (opts_.io_mode == "p1") {
        p1_sched_.set_wait_reader_callback([this](int l) { wait_slot_reader(l); });
        p1_sched_.init(opts_.io_threads);
    } else {
        c_io_init(8);
    }

    auto rope_inv_freq = make_rope_inv_freq(64, 10000000.0f);

    // Compute per-layer capacities for Layers 1..47 matching compute_layer_capacities in loader.py:347-417
    current_caps_.assign(48, 0);
    if (opts_.slot_cap > 0) {
        for (int l = 1; l < 48; ++l) {
            current_caps_[l] = std::max(16, std::min(512, opts_.slot_cap));
        }
    } else {
        double frac = (opts_.budget_pct > 1.0) ? (opts_.budget_pct / 100.0) : opts_.budget_pct;
        int per_layer_even = std::max(16, std::min(512, static_cast<int>(std::round(frac * 512.0))));
        int num_cached = 47;
        int total_slots = num_cached * per_layer_even;
        int base = total_slots / num_cached;
        int rem = total_slots % num_cached;
        for (int l = 1; l < 48; ++l) {
            current_caps_[l] = std::min(512, std::max(16, base + ((l - 1) < rem ? 1 : 0)));
        }
    }

    layers_.resize(48);
    exec_caches_.resize(48);
    last_slot_reader_.resize(48, std::nullopt);

    // 3. Load all 48 decoder layers
    for (int l = 0; l < 48; ++l) {
        auto& layer = layers_[l];
        layer.layer_idx = l;
        layer.is_linear = (l % 4 != 3);
        exec_caches_[l].is_linear = layer.is_linear;

        std::string lp = "language_model.model.layers." + std::to_string(l);
        layer.attn_hc = load_hc(trunk_tensors, lp + ".attn_hyper_connection", true);
        layer.mlp_hc = load_hc(trunk_tensors, lp + ".mlp_hyper_connection", true);

        if (layer.is_linear) {
            std::string ap = lp + ".linear_attn";
            layer.lin_attn.in_proj_qkv = load_qlinear(trunk_tensors, ap + ".in_proj_qkv", 32, 4);
            layer.lin_attn.in_proj_z = load_qlinear(trunk_tensors, ap + ".in_proj_z", 32, 4);
            layer.lin_attn.in_proj_b = load_qlinear(trunk_tensors, ap + ".in_proj_b", 32, 4);
            layer.lin_attn.in_proj_a = load_qlinear(trunk_tensors, ap + ".in_proj_a", 32, 4);
            layer.lin_attn.out_proj = load_qlinear(trunk_tensors, ap + ".out_proj", 32, 4);
            layer.lin_attn.conv1d_weight = trunk_tensors.at(ap + ".conv1d.weight");
            layer.lin_attn.dt_bias = trunk_tensors.at(ap + ".dt_bias");
            layer.lin_attn.A_log = trunk_tensors.at(ap + ".A_log");
            layer.lin_attn.norm_weight = trunk_tensors.at(ap + ".norm.weight");
        } else {
            std::string ap = lp + ".self_attn";
            layer.full_attn.q_proj = load_qlinear(trunk_tensors, ap + ".q_proj", 32, 4);
            layer.full_attn.k_proj = load_qlinear(trunk_tensors, ap + ".k_proj", 32, 4);
            layer.full_attn.v_proj = load_qlinear(trunk_tensors, ap + ".v_proj", 32, 4);
            layer.full_attn.o_proj = load_qlinear(trunk_tensors, ap + ".o_proj", 32, 4);
            auto qnw = trunk_tensors.at(ap + ".q_norm.weight");
            auto knw = trunk_tensors.at(ap + ".k_norm.weight");
            layer.full_attn.q_norm_scale = mx::array(1.0f, mx::float32) + mx::astype(qnw, mx::float32);
            layer.full_attn.k_norm_scale = mx::array(1.0f, mx::float32) + mx::astype(knw, mx::float32);
            layer.full_attn.rope_inv_freq = rope_inv_freq;

            layer.full_attn.index_qk_proj = load_qlinear(trunk_tensors, ap + ".indexer.index_qk_proj", 32, 4);
            auto iqnw = trunk_tensors.at(ap + ".indexer.q_layernorm.weight");
            auto iknw = trunk_tensors.at(ap + ".indexer.k_layernorm.weight");
            layer.full_attn.index_q_norm_scale = mx::array(1.0f, mx::float32) + mx::astype(iqnw, mx::float32);
            layer.full_attn.index_k_norm_scale = mx::array(1.0f, mx::float32) + mx::astype(iknw, mx::float32);
            layer.full_attn.index_rope_inv_freq = rope_inv_freq;
            mx::eval({
                layer.full_attn.q_norm_scale,
                layer.full_attn.k_norm_scale,
                layer.full_attn.index_q_norm_scale,
                layer.full_attn.index_k_norm_scale
            });
        }

        std::string mp = lp + ".mlp";
        layer.gate = load_qlinear(trunk_tensors, mp + ".gate", 64, 8);
        layer.shared_expert_gate = load_qlinear(trunk_tensors, mp + ".shared_expert_gate", 64, 8);
        layer.shared_gate_proj = load_qlinear(trunk_tensors, mp + ".shared_expert.gate_proj", 32, 4);
        layer.shared_up_proj = load_qlinear(trunk_tensors, mp + ".shared_expert.up_proj", 32, 4);
        layer.shared_down_proj = load_qlinear(trunk_tensors, mp + ".shared_expert.down_proj", 32, 4);

        if (l == 0) {
            layer.resident_moe = true;
            std::string sp = mp + ".switch_mlp";
            layer.res_gate_proj.weight = trunk_tensors.at(sp + ".gate_proj.weight");
            layer.res_gate_proj.scales = trunk_tensors.at(sp + ".gate_proj.scales");
            layer.res_gate_proj.biases = trunk_tensors.at(sp + ".gate_proj.biases");
            layer.res_up_proj.weight = trunk_tensors.at(sp + ".up_proj.weight");
            layer.res_up_proj.scales = trunk_tensors.at(sp + ".up_proj.scales");
            layer.res_up_proj.biases = trunk_tensors.at(sp + ".up_proj.biases");
            layer.res_down_proj.weight = trunk_tensors.at(sp + ".down_proj.weight");
            layer.res_down_proj.scales = trunk_tensors.at(sp + ".down_proj.scales");
            layer.res_down_proj.biases = trunk_tensors.at(sp + ".down_proj.biases");
        } else {
            layer.resident_moe = false;
            char fname_buf[64];
            std::snprintf(fname_buf, sizeof(fname_buf), "experts-%02d.safetensors", l);
            std::string exp_path = opts_.model_dir + "/" + fname_buf;
            int fd = -1;
            auto fd_it = open_fds_.find(exp_path);
            if (fd_it != open_fds_.end()) {
                fd = fd_it->second;
            } else {
                fd = open(exp_path.c_str(), O_RDONLY);
                if (fd < 0) {
                    throw std::runtime_error("Cannot open expert shard: " + exp_path);
                }
                open_fds_[exp_path] = fd;
            }
            auto hdr = parse_safetensors_header(exp_path);
            std::string sp = mp + ".switch_mlp.";
            std::string alt_sp = "model.language_model.layers." + std::to_string(l) + ".mlp.switch_mlp.";
            const char* subkeys[9] = {
                "gate_proj.weight", "gate_proj.scales", "gate_proj.biases",
                "up_proj.weight",   "up_proj.scales",   "up_proj.biases",
                "down_proj.weight", "down_proj.scales", "down_proj.biases"
            };
            auto& off = layer.offload;
            off.fd = fd;
            off.capacity = current_caps_[l];
            uint64_t raw_offsets[9];
            uint64_t raw_rbytes[9];
            for (int t = 0; t < 9; ++t) {
                auto it = hdr.tensors.find(sp + subkeys[t]);
                if (it == hdr.tensors.end()) {
                    it = hdr.tensors.find(alt_sp + subkeys[t]);
                }
                if (it == hdr.tensors.end()) {
                    throw std::runtime_error("Missing expert tensor " + sp + subkeys[t] + " in " + exp_path);
                }
                const auto& entry = it->second;
                raw_offsets[t] = hdr.data_start + entry.data_offset_start;
                uint64_t total_bytes = entry.data_offset_end - entry.data_offset_start;
                raw_rbytes[t] = total_bytes / static_cast<uint64_t>(entry.shape[0]);
            }
            int cap = off.capacity;
            off.gate_w = mx::zeros({cap, 640, 160}, mx::uint32);
            off.gate_s = mx::zeros({cap, 640, 40}, mx::bfloat16);
            off.gate_b = mx::zeros({cap, 640, 40}, mx::bfloat16);
            off.up_w   = mx::zeros({cap, 640, 160}, mx::uint32);
            off.up_s   = mx::zeros({cap, 640, 40}, mx::bfloat16);
            off.up_b   = mx::zeros({cap, 640, 40}, mx::bfloat16);
            off.down_w = mx::zeros({cap, 2560, 40}, mx::uint32);
            off.down_s = mx::zeros({cap, 2560, 10}, mx::bfloat16);
            off.down_b = mx::zeros({cap, 2560, 10}, mx::bfloat16);
            off.slot_inds_buf = mx::zeros({1, 1, 10}, mx::uint32);
            mx::eval({
                off.gate_w, off.gate_s, off.gate_b,
                off.up_w,   off.up_s,   off.up_b,
                off.down_w, off.down_s, off.down_b,
                off.slot_inds_buf
            });
            uint64_t raw_ptrs[9] = {
                reinterpret_cast<uint64_t>(off.gate_w.data<uint32_t>()),
                reinterpret_cast<uint64_t>(off.gate_s.data<mx::bfloat16_t>()),
                reinterpret_cast<uint64_t>(off.gate_b.data<mx::bfloat16_t>()),
                reinterpret_cast<uint64_t>(off.up_w.data<uint32_t>()),
                reinterpret_cast<uint64_t>(off.up_s.data<mx::bfloat16_t>()),
                reinterpret_cast<uint64_t>(off.up_b.data<mx::bfloat16_t>()),
                reinterpret_cast<uint64_t>(off.down_w.data<uint32_t>()),
                reinterpret_cast<uint64_t>(off.down_s.data<mx::bfloat16_t>()),
                reinterpret_cast<uint64_t>(off.down_b.data<mx::bfloat16_t>())
            };
            int perm[9] = {0, 1, 2, 3, 4, 5, 6, 7, 8};
            std::sort(perm, perm + 9, [&](int a, int b) {
                return raw_offsets[a] < raw_offsets[b];
            });
            for (int t = 0; t < 9; ++t) {
                off.base_offsets[t] = raw_offsets[perm[t]];
                off.row_bytes[t] = raw_rbytes[perm[t]];
                off.base_ptrs[t] = raw_ptrs[perm[t]];
            }
            uint64_t slot_ptr = reinterpret_cast<uint64_t>(off.slot_inds_buf.data<uint32_t>());
            if (opts_.io_mode == "p1") {
                p1_sched_.layer_init(l, fd, cap, 9, off.base_offsets, off.row_bytes, off.base_ptrs, slot_ptr);
            } else {
                c_layer_init(l, fd, cap, off.base_offsets, off.row_bytes, off.base_ptrs, slot_ptr);
            }
        }
    }

    // 4. Load Layer 1 PLE weights and mmap PLE shards (ple-*.safetensors)
    {
        std::string pp = "language_model.model.layers.1.ple";
        ple_.active = true;
        ple_.key_proj = load_qlinear(trunk_tensors, pp + ".key_proj", 32, 4);
        ple_.value_proj = load_qlinear(trunk_tensors, pp + ".value_proj", 32, 4);
        ple_.norm_key_weight = trunk_tensors.at(pp + ".norm_key.weight");
        ple_.norm_query_weight = trunk_tensors.at(pp + ".norm_query.weight");
        ple_.norm_conv_weight = trunk_tensors.at(pp + ".norm_conv.weight");
        ple_.short_conv_weight = trunk_tensors.at(pp + ".conv1d.weight");

        init_ple_metadata(ple_, 248320, 20000000, 128, 1234);

        // If layer_multipliers / ngram_heads_vocab_sizes exist in trunk_tensors, verify/load them
        auto lm_it = trunk_tensors.find(pp + ".ple_embedding.layer_multipliers");
        auto vs_it = trunk_tensors.find(pp + ".ple_embedding.ngram_heads_vocab_sizes");
        if (lm_it != trunk_tensors.end() && vs_it != trunk_tensors.end()) {
            mx::eval({lm_it->second, vs_it->second});
            const int64_t* m_ptr = lm_it->second.data<int64_t>();
            const int64_t* v_ptr = vs_it->second.data<int64_t>();
            for (int i = 0; i < ple_.ngram_size; ++i) ple_.multipliers_cpu[i] = m_ptr[i];
            for (int h = 0; h < ple_.ngram_heads; ++h) ple_.moduli_cpu[h] = v_ptr[h];
        }

        std::string emb_prefix = pp + ".ple_embedding.ngram_embedding";
        std::string ws_key = emb_prefix + ".weight_scale";
        auto ws_it = trunk_tensors.find(ws_key);
        if (ws_it != trunk_tensors.end()) {
            ple_.weight_scale = mx::astype(ws_it->second, mx::bfloat16);
        } else {
            ple_.weight_scale = mx::ones({1}, mx::bfloat16);
        }
        mx::eval({
            ple_.norm_key_weight,
            ple_.norm_query_weight,
            ple_.norm_conv_weight,
            ple_.short_conv_weight,
            ple_.weight_scale
        });

        // Build fallback map from ple-*.safetensors headers if weight_map lacks any shard keys
        std::unordered_map<std::string, SafeTensorHeader> ple_headers;
        for (const auto& pfname : list_ple_safetensors(opts_.model_dir)) {
            std::string full_path = opts_.model_dir + "/" + pfname;
            auto hdr = parse_safetensors_header(full_path);
            for (const auto& kv : hdr.tensors) {
                if (weight_map.find(kv.first) == weight_map.end()) {
                    weight_map.emplace(kv.first, pfname);
                }
            }
            ple_headers.emplace(full_path, std::move(hdr));
        }

        ple_.shards.resize(ple_.split_ngram_parts);
        for (int s = 0; s < ple_.split_ngram_parts; ++s) {
            std::string candidates[4] = {
                emb_prefix + ".shards." + std::to_string(s),
                emb_prefix + ".shard_" + std::to_string(s),
                "model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shards." + std::to_string(s),
                "model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_" + std::to_string(s),
            };
            std::string sp;
            for (const auto& cand : candidates) {
                if (weight_map.find(cand + ".weight") != weight_map.end()) {
                    sp = cand;
                    break;
                }
            }
            if (sp.empty()) {
                throw std::runtime_error("Missing PLE shard " + std::to_string(s) + " in weight_map");
            }
            auto get_slice = [&](const std::string& tkey) -> PleTensorSlice {
                const std::string& fname = weight_map.at(tkey);
                std::string full_path = opts_.model_dir + "/" + fname;
                auto mm_it = ple_mmaps_.find(full_path);
                std::shared_ptr<MmapFile> mm;
                if (mm_it != ple_mmaps_.end()) {
                    mm = mm_it->second;
                } else {
                    mm = std::make_shared<MmapFile>();
                    mm->fd = open(full_path.c_str(), O_RDONLY);
                    if (mm->fd < 0) throw std::runtime_error("Cannot open PLE shard: " + full_path);
                    struct stat st{};
                    fstat(mm->fd, &st);
                    mm->size = static_cast<size_t>(st.st_size);
                    mm->addr = mmap(nullptr, mm->size, PROT_READ, MAP_PRIVATE, mm->fd, 0);
                    if (mm->addr == MAP_FAILED) {
                        throw std::runtime_error("mmap failed for: " + full_path);
                    }
                    madvise(mm->addr, mm->size, MADV_RANDOM);
                    if (ple_headers.find(full_path) == ple_headers.end()) {
                        ple_headers.emplace(full_path, parse_safetensors_header(full_path));
                    }
                    mm->data_start = ple_headers.at(full_path).data_start;
                    ple_mmaps_.emplace(full_path, mm);
                }
                const auto& hdr = ple_headers.at(full_path);
                const auto& entry = hdr.tensors.at(tkey);
                PleTensorSlice sl;
                sl.data_ptr = static_cast<const uint8_t*>(mm->addr) + mm->data_start + entry.data_offset_start;
                sl.rows = entry.shape[0];
                sl.cols = entry.shape[1];
                uint64_t total_b = entry.data_offset_end - entry.data_offset_start;
                sl.row_bytes = static_cast<int64_t>(total_b / static_cast<uint64_t>(sl.rows));
                sl.dtype = entry.dtype;
                return sl;
            };
            ple_.shards[s].weight = get_slice(sp + ".weight");
            ple_.shards[s].scales = get_slice(sp + ".scales");
            ple_.shards[s].biases = get_slice(sp + ".biases");
            ple_.shards[s].bits = 4;
            ple_.shards[s].group_size = 32;
        }
    }

    // Evaluate all resident trunk weights so no lazy safetensors references remain
    std::vector<mx::array> to_eval;
    to_eval.reserve(trunk_tensors.size());
    for (auto& kv : trunk_tensors) {
        to_eval.push_back(kv.second);
    }
    mx::eval(to_eval);

    int total_slots = 0;
    for (int l = 1; l < 48; ++l) total_slots += current_caps_[l];
    min_total_slots_ = total_slots;
    max_total_slots_ = total_slots;

    loaded_ = true;
}

void Qwen38Engine::resize_layer_capacity(int l, int new_cap) {
    if (l <= 0 || l >= static_cast<int>(layers_.size())) return;
    auto& off = layers_[l].offload;
    int old_cap = off.capacity;
    if (new_cap == old_cap || new_cap < 16) return;

    // 1. Keep waits (slot reader + demand + prefetch) before freeing old buffers
    wait_slot_reader(l);
    if (opts_.io_mode == "p1") {
        p1_sched_.wait_layer_all(l);
    } else {
        c_layer_wait_demand(l);
        c_layer_wait_prefetch(l);
    }

    // 2. Fetch current LRU order (oldest to newest)
    std::vector<int32_t> old_eids(512);
    std::vector<int32_t> old_slots(512);
    int old_cnt = 0;
    uint64_t saved_stats[18]{};
    if (opts_.io_mode == "p1") {
        p1_sched_.get_lru_state(l, old_eids.data(), old_slots.data(), &old_cnt);
    } else {
        c_layer_get_lru_state(l, old_eids.data(), old_slots.data(), &old_cnt);
        c_layer_get_stats(l, saved_stats);
    }

    int keep = std::min(old_cnt, new_cap);
    int start_idx = old_cnt - keep; // keep the most recently used `keep` experts

    uint64_t old_raw_ptrs[9] = {
        reinterpret_cast<uint64_t>(off.gate_w.data<uint32_t>()),
        reinterpret_cast<uint64_t>(off.gate_s.data<mx::bfloat16_t>()),
        reinterpret_cast<uint64_t>(off.gate_b.data<mx::bfloat16_t>()),
        reinterpret_cast<uint64_t>(off.up_w.data<uint32_t>()),
        reinterpret_cast<uint64_t>(off.up_s.data<mx::bfloat16_t>()),
        reinterpret_cast<uint64_t>(off.up_b.data<mx::bfloat16_t>()),
        reinterpret_cast<uint64_t>(off.down_w.data<uint32_t>()),
        reinterpret_cast<uint64_t>(off.down_s.data<mx::bfloat16_t>()),
        reinterpret_cast<uint64_t>(off.down_b.data<mx::bfloat16_t>())
    };

    // Stage kept rows into a single-layer temporary buffer so we can release the old arrays
    // of this layer BEFORE allocating the new arrays of this layer (transient peak = 1 layer).
    int sorted_to_raw[9]{};
    std::vector<uint8_t> staged_rows[9];
    for (int t = 0; t < 9; ++t) {
        sorted_to_raw[t] = -1;
        for (int k = 0; k < 9; ++k) {
            if (old_raw_ptrs[k] == off.base_ptrs[t]) {
                sorted_to_raw[t] = k;
                break;
            }
        }
        size_t row_b = off.row_bytes[t];
        if (sorted_to_raw[t] >= 0 && keep > 0 && row_b > 0) {
            staged_rows[t].resize(static_cast<size_t>(keep) * row_b);
            const uint8_t* src_base = reinterpret_cast<const uint8_t*>(off.base_ptrs[t]);
            uint8_t* dst_stage = staged_rows[t].data();
            for (int i = 0; i < keep; ++i) {
                int old_s = old_slots[start_idx + i];
                std::memcpy(dst_stage + static_cast<size_t>(i) * row_b,
                            src_base + static_cast<size_t>(old_s) * row_b,
                            row_b);
            }
        }
    }

    // Release old arrays of layer l and flush MLX buffer cache before allocating new buffers
    off.gate_w = mx::zeros({0});
    off.gate_s = mx::zeros({0});
    off.gate_b = mx::zeros({0});
    off.up_w   = mx::zeros({0});
    off.up_s   = mx::zeros({0});
    off.up_b   = mx::zeros({0});
    off.down_w = mx::zeros({0});
    off.down_s = mx::zeros({0});
    off.down_b = mx::zeros({0});
    mx::clear_cache();

    // Allocate new arrays for layer l
    off.gate_w = mx::zeros({new_cap, 640, 160}, mx::uint32);
    off.gate_s = mx::zeros({new_cap, 640, 40}, mx::bfloat16);
    off.gate_b = mx::zeros({new_cap, 640, 40}, mx::bfloat16);
    off.up_w   = mx::zeros({new_cap, 640, 160}, mx::uint32);
    off.up_s   = mx::zeros({new_cap, 640, 40}, mx::bfloat16);
    off.up_b   = mx::zeros({new_cap, 640, 40}, mx::bfloat16);
    off.down_w = mx::zeros({new_cap, 2560, 40}, mx::uint32);
    off.down_s = mx::zeros({new_cap, 2560, 10}, mx::bfloat16);
    off.down_b = mx::zeros({new_cap, 2560, 10}, mx::bfloat16);
    mx::eval({
        off.gate_w, off.gate_s, off.gate_b,
        off.up_w,   off.up_s,   off.up_b,
        off.down_w, off.down_s, off.down_b
    });

    uint64_t new_raw_ptrs[9] = {
        reinterpret_cast<uint64_t>(off.gate_w.data<uint32_t>()),
        reinterpret_cast<uint64_t>(off.gate_s.data<mx::bfloat16_t>()),
        reinterpret_cast<uint64_t>(off.gate_b.data<mx::bfloat16_t>()),
        reinterpret_cast<uint64_t>(off.up_w.data<uint32_t>()),
        reinterpret_cast<uint64_t>(off.up_s.data<mx::bfloat16_t>()),
        reinterpret_cast<uint64_t>(off.up_b.data<mx::bfloat16_t>()),
        reinterpret_cast<uint64_t>(off.down_w.data<uint32_t>()),
        reinterpret_cast<uint64_t>(off.down_s.data<mx::bfloat16_t>()),
        reinterpret_cast<uint64_t>(off.down_b.data<mx::bfloat16_t>())
    };

    for (int t = 0; t < 9; ++t) {
        int k = sorted_to_raw[t];
        if (k >= 0) {
            off.base_ptrs[t] = new_raw_ptrs[k];
            if (!staged_rows[t].empty()) {
                std::memcpy(reinterpret_cast<uint8_t*>(off.base_ptrs[t]),
                            staged_rows[t].data(),
                            staged_rows[t].size());
            }
        }
    }

    std::vector<int32_t> kept_eids(keep);
    std::vector<int32_t> kept_slots(keep);
    for (int i = 0; i < keep; ++i) {
        kept_eids[i] = old_eids[start_idx + i];
        kept_slots[i] = i;
    }

    off.capacity = new_cap;
    current_caps_[l] = new_cap;
    uint64_t slot_ptr = reinterpret_cast<uint64_t>(off.slot_inds_buf.data<uint32_t>());

    if (opts_.io_mode == "p1") {
        p1_sched_.resize_layer(l, new_cap, off.base_ptrs, slot_ptr,
                               kept_eids.data(), kept_slots.data(), keep);
    } else {
        c_layer_init(l, off.fd, new_cap, off.base_offsets, off.row_bytes, off.base_ptrs, slot_ptr);
        c_layer_set_lru_state(l, kept_eids.data(), kept_slots.data(), keep,
                              saved_stats[6], saved_stats[7], saved_stats[8]);
    }
}

namespace {

struct K50Base3Lut {
    uint8_t trits[256][5];
    int8_t signed_trits[256][5];

    K50Base3Lut() {
        for (int b = 0; b < 256; ++b) {
            int v = b;
            for (int k = 0; k < 5; ++k) {
                int d = v % 3;
                v /= 3;
                trits[b][k] = static_cast<uint8_t>(d);
                signed_trits[b][k] = static_cast<int8_t>(d - 1);
            }
        }
    }
};

static const K50Base3Lut g_k50_base3_lut;

static std::vector<int> parse_k50_int_array(const std::string& s, size_t& i) {
    std::vector<int> out;
    skip_ws(s, i);
    if (i < s.size() && s[i] == '[') ++i;
    while (i < s.size()) {
        skip_ws(s, i);
        if (i < s.size() && s[i] == ']') { ++i; break; }
        if (s[i] == ',') { ++i; continue; }
        out.push_back(static_cast<int>(parse_json_int(s, i)));
    }
    return out;
}

static K50PackRecord parse_k50_pack_record(const std::string& s, size_t& i) {
    K50PackRecord r;
    skip_ws(s, i);
    if (i >= s.size() || s[i] != '{') {
        throw std::runtime_error("Expected '{' in K50PackRecord at pos " + std::to_string(i));
    }
    ++i;
    while (i < s.size()) {
        skip_ws(s, i);
        if (i < s.size() && s[i] == '}') { ++i; break; }
        if (s[i] == ',') { ++i; continue; }
        std::string key = parse_json_string(s, i);
        skip_ws(s, i);
        if (i < s.size() && s[i] == ':') ++i;
        skip_ws(s, i);
        if (i < s.size() && s.compare(i, 4, "null") == 0) {
            i += 4;
            continue;
        }
        if (key == "expert") {
            r.expert = static_cast<int>(parse_json_int(s, i));
        } else if (key == "kind") {
            r.kind = parse_json_string(s, i);
        } else if (key == "unit") {
            r.unit = parse_json_string(s, i);
        } else if (key == "shape") {
            r.shape = parse_k50_int_array(s, i);
        } else if (key == "row_bytes") {
            r.row_bytes = static_cast<int>(parse_json_int(s, i));
        } else if (key == "code_off") {
            r.code_off = static_cast<uint64_t>(parse_json_int(s, i));
        } else if (key == "code_nbytes") {
            r.code_nbytes = static_cast<uint64_t>(parse_json_int(s, i));
        } else if (key == "scale_off") {
            r.scale_off = static_cast<uint64_t>(parse_json_int(s, i));
        } else if (key == "scale_nbytes") {
            r.scale_nbytes = static_cast<uint64_t>(parse_json_int(s, i));
        } else if (key == "bias_off") {
            r.bias_off = static_cast<uint64_t>(parse_json_int(s, i));
        } else if (key == "bias_nbytes") {
            r.bias_nbytes = static_cast<uint64_t>(parse_json_int(s, i));
        } else if (key == "group_size") {
            r.group_size = static_cast<int>(parse_json_int(s, i));
        } else {
            skip_json_value(s, i);
        }
    }
    return r;
}

static size_t compute_materialized_bytes(const K50MaterializedWeight& m) {
    size_t total = m.weight.nbytes() + m.scales.nbytes();
    if (m.biases.size() > 0) total += m.biases.nbytes();
    return total;
}

static void k50_pread_exact(int fd, void* buf, size_t nbytes, uint64_t offset) {
    uint8_t* dst = static_cast<uint8_t*>(buf);
    size_t done = 0;
    while (done < nbytes) {
        ssize_t r = ::pread(fd, dst + done, nbytes - done, static_cast<off_t>(offset + done));
        if (r <= 0) {
            throw std::runtime_error("pread failed at offset " + std::to_string(offset + done));
        }
        done += static_cast<size_t>(r);
    }
}

} // namespace

ExpertPackCpp::ExpertPackCpp(const std::string& dir, int l_idx)
    : layer_idx(l_idx), layer_dir(dir) {
    fd_codes = ::open((dir + "/codes.bin").c_str(), O_RDONLY);
    fd_scales = ::open((dir + "/scales.bf16").c_str(), O_RDONLY);
    fd_bias = ::open((dir + "/bias.bf16").c_str(), O_RDONLY);
    if (fd_codes < 0 || fd_scales < 0 || fd_bias < 0) {
        throw std::runtime_error("Failed to open codes.bin/scales.bf16/bias.bf16 in " + dir);
    }

    std::ifstream ifs(dir + "/index.json");
    if (!ifs) {
        throw std::runtime_error("Cannot open " + dir + "/index.json");
    }
    std::ostringstream ss;
    ss << ifs.rdbuf();
    std::string s = ss.str();

    size_t i = 0;
    skip_ws(s, i);
    if (i >= s.size() || s[i] != '{') {
        throw std::runtime_error("Invalid index.json in " + dir);
    }
    ++i;
    while (i < s.size()) {
        skip_ws(s, i);
        if (i < s.size() && s[i] == '}') { ++i; break; }
        if (s[i] == ',') { ++i; continue; }
        std::string top_key = parse_json_string(s, i);
        skip_ws(s, i);
        if (i < s.size() && s[i] == ':') ++i;
        skip_ws(s, i);
        if (top_key == "dense") {
            if (i >= s.size() || s[i] != '{') {
                throw std::runtime_error("Expected '{' for dense in " + dir);
            }
            ++i;
            while (i < s.size()) {
                skip_ws(s, i);
                if (i < s.size() && s[i] == '}') { ++i; break; }
                if (s[i] == ',') { ++i; continue; }
                std::string rec_key = parse_json_string(s, i);
                skip_ws(s, i);
                if (i < s.size() && s[i] == ':') ++i;
                K50PackRecord rec = parse_k50_pack_record(s, i);
                dense.emplace(std::move(rec_key), std::move(rec));
            }
        } else if (top_key == "experts") {
            if (i >= s.size() || s[i] != '{') {
                throw std::runtime_error("Expected '{' for experts in " + dir);
            }
            ++i;
            while (i < s.size()) {
                skip_ws(s, i);
                if (i < s.size() && s[i] == '}') { ++i; break; }
                if (s[i] == ',') { ++i; continue; }
                std::string proj_name = parse_json_string(s, i);
                skip_ws(s, i);
                if (i < s.size() && s[i] == ':') ++i;
                skip_ws(s, i);
                if (i >= s.size() || s[i] != '[') {
                    throw std::runtime_error("Expected '[' for experts." + proj_name + " in " + dir);
                }
                ++i;
                while (i < s.size()) {
                    skip_ws(s, i);
                    if (i < s.size() && s[i] == ']') { ++i; break; }
                    if (s[i] == ',') { ++i; continue; }
                    K50PackRecord rec = parse_k50_pack_record(s, i);
                    std::string key = proj_name + ":" + std::to_string(rec.expert);
                    experts.emplace(std::move(key), std::move(rec));
                }
            }
        } else {
            skip_json_value(s, i);
        }
    }
}

ExpertPackCpp::~ExpertPackCpp() {
    if (fd_codes >= 0) ::close(fd_codes);
    if (fd_scales >= 0) ::close(fd_scales);
    if (fd_bias >= 0) ::close(fd_bias);
}

K50MaterializedWeight ExpertPackCpp::materialize(const K50PackRecord& rec) {
    return materialize_as_affine2(rec);
}

K50MaterializedWeight ExpertPackCpp::materialize_as_affine2(const K50PackRecord& rec) {
    int n = rec.shape[0];
    int k = rec.shape[1];
    int groups = k / rec.group_size;

    std::vector<uint16_t> raw_s(static_cast<size_t>(n) * groups);
    k50_pread_exact(fd_scales, raw_s.data(), rec.scale_nbytes, rec.scale_off);
    mx::array scales(
        reinterpret_cast<const mx::bfloat16_t*>(raw_s.data()),
        {n, groups},
        mx::bfloat16);

    if (rec.kind == "tern") {
        std::vector<uint8_t> raw_c(rec.code_nbytes);
        k50_pread_exact(fd_codes, raw_c.data(), rec.code_nbytes, rec.code_off);

        int words_per_row = k / 16;
        std::vector<uint32_t> packed_u32(static_cast<size_t>(n) * words_per_row, 0);
        int rb = (rec.row_bytes > 0) ? rec.row_bytes : (k / 5);
        for (int r = 0; r < n; ++r) {
            const uint8_t* src = raw_c.data() + static_cast<size_t>(r) * rb;
            uint32_t* dst = packed_u32.data() + static_cast<size_t>(r) * words_per_row;
            int b_idx = 0;
            for (int w = 0; w < words_per_row; w += 5) {
                uint8_t tr[80];
                for (int j = 0; j < 16; ++j) {
                    const uint8_t* t5 = g_k50_base3_lut.trits[src[b_idx + j]];
                    tr[j * 5 + 0] = t5[0];
                    tr[j * 5 + 1] = t5[1];
                    tr[j * 5 + 2] = t5[2];
                    tr[j * 5 + 3] = t5[3];
                    tr[j * 5 + 4] = t5[4];
                }
                b_idx += 16;
                for (int u = 0; u < 5; ++u) {
                    uint32_t word = 0;
                    const uint8_t* tp = tr + u * 16;
                    for (int j = 0; j < 16; ++j) {
                        word |= static_cast<uint32_t>(tp[j]) << (2 * j);
                    }
                    dst[w + u] = word;
                }
            }
        }
        mx::array codes(packed_u32.data(), {n, words_per_row}, mx::uint32);
        std::vector<uint16_t> raw_b_neg(raw_s.size());
        for (size_t i = 0; i < raw_s.size(); ++i) {
            raw_b_neg[i] = static_cast<uint16_t>(raw_s[i] ^ 0x8000u);
        }
        mx::array biases(
            reinterpret_cast<const mx::bfloat16_t*>(raw_b_neg.data()),
            {n, groups},
            mx::bfloat16);

        K50MaterializedWeight res;
        res.kind = "2t";
        res.bits = 2;
        res.group_size = rec.group_size;
        res.weight = std::move(codes);
        res.scales = std::move(scales);
        res.biases = std::move(biases);
        res.nbytes = compute_materialized_bytes(res);
        return res;
    }

    int words_per_row = static_cast<int>(rec.code_nbytes / (static_cast<size_t>(n) * sizeof(uint32_t)));
    std::vector<uint32_t> raw_w(static_cast<size_t>(n) * words_per_row);
    k50_pread_exact(fd_codes, raw_w.data(), rec.code_nbytes, rec.code_off);
    mx::array codes(raw_w.data(), {n, words_per_row}, mx::uint32);

    std::vector<uint16_t> raw_b(static_cast<size_t>(n) * groups);
    if (rec.bias_nbytes > 0) {
        k50_pread_exact(fd_bias, raw_b.data(), rec.bias_nbytes, rec.bias_off);
    } else {
        for (size_t i = 0; i < raw_s.size(); ++i) {
            raw_b[i] = static_cast<uint16_t>(raw_s[i] ^ 0x8000u);
        }
    }
    mx::array biases(
        reinterpret_cast<const mx::bfloat16_t*>(raw_b.data()),
        {n, groups},
        mx::bfloat16);

    int bits = (rec.kind == "2t") ? 2 : std::stoi(rec.kind);
    K50MaterializedWeight res;
    res.kind = rec.kind;
    res.bits = bits;
    res.group_size = rec.group_size;
    res.weight = std::move(codes);
    res.scales = std::move(scales);
    res.biases = std::move(biases);
    res.nbytes = compute_materialized_bytes(res);
    return res;
}

QLinear ExpertPackCpp::load_dense_qlinear(const std::string& name) {
    auto it = dense.find(name);
    if (it == dense.end()) {
        it = dense.find(name + ".weight");
    }
    if (it == dense.end()) {
        throw std::runtime_error("Missing dense weight in pack (" + layer_dir + "): " + name);
    }
    auto mat = materialize_as_affine2(it->second);
    QLinear ql;
    ql.is_quantized = true;
    ql.reshape_2d = true;
    ql.bits = mat.bits;
    ql.group_size = mat.group_size;
    ql.weight = std::move(mat.weight);
    ql.scales = std::move(mat.scales);
    ql.biases = std::move(mat.biases);
    return ql;
}

void K50ResidentCache::set_budget(size_t total_budget_bytes, size_t pinned_hot_bytes) {
    pinned_bytes = pinned_hot_bytes;
    budget_bytes = (total_budget_bytes > pinned_hot_bytes)
        ? (total_budget_bytes - pinned_hot_bytes)
        : 0;
}

void K50ResidentCache::clear_lru() {
    lru_map.clear();
    lru_list.clear();
    lru_bytes = 0;
}

void K50ResidentCache::reset_stats() {
    hot_hits = 0;
    lru_hits = 0;
    misses = 0;
    evictions = 0;
    prefill_hot_hits = 0;
    prefill_lru_hits = 0;
    prefill_misses = 0;
}

void K50ResidentCache::record_hot_hit(uint64_t count) {
    if (in_prefill) {
        prefill_hot_hits += count;
    } else {
        hot_hits += count;
    }
}

const K50MaterializedWeight& K50ResidentCache::get_or_load(
    uint8_t layer_id,
    uint8_t proj_id,
    uint16_t expert_id,
    ExpertPackCpp& pack,
    const K50PackRecord& rec) {
    K50CacheKey key{layer_id, proj_id, expert_id};
    auto it = lru_map.find(key);
    if (it != lru_map.end()) {
        if (in_prefill) {
            prefill_lru_hits++;
        } else {
            lru_hits++;
        }
        lru_list.splice(lru_list.end(), lru_list, it->second);
        return it->second->second;
    }

    if (in_prefill) {
        prefill_misses++;
    } else {
        misses++;
    }

    // On-miss: materialize and convert ternary ("tern") to 2-bit affine ("2t")
    K50MaterializedWeight val = pack.materialize_as_affine2(rec);
    size_t sz = val.nbytes;

    while (budget_bytes > 0 && (lru_bytes + sz > budget_bytes) && !lru_list.empty()) {
        auto& oldest = lru_list.front();
        lru_bytes -= oldest.second.nbytes;
        lru_map.erase(oldest.first);
        lru_list.pop_front();
        evictions++;
    }

    lru_list.emplace_back(key, std::move(val));
    auto new_it = std::prev(lru_list.end());
    lru_map[key] = new_it;
    lru_bytes += sz;
    return new_it->second;
}

void StackedSwitchMlp::build(
    int l_idx,
    std::shared_ptr<ExpertPackCpp> p,
    int n_experts,
    const std::string& mode) {
    layer_idx = l_idx;
    pack = std::move(p);
    num_experts = n_experts;
    groups.clear();
    is_stacked.clear();
    stacked_bytes = 0;

    for (const char* proj_c : {"gate_proj", "up_proj", "down_proj"}) {
        std::string proj(proj_c);
        is_stacked[proj].assign(num_experts, false);
        std::map<std::string, std::vector<int>> by_kind;
        for (int e = 0; e < num_experts; ++e) {
            std::string key = proj + ":" + std::to_string(e);
            auto it = pack->experts.find(key);
            if (it == pack->experts.end()) {
                throw std::runtime_error("Missing expert record: " + key + " in layer " + std::to_string(l_idx));
            }
            const auto& rec = it->second;
            if (mode == "hot" && rec.unit != "hot") continue;
            by_kind[rec.kind].push_back(e);
        }

        for (auto& kv : by_kind) {
            const std::string& kind = kv.first;
            const std::vector<int>& eids = kv.second;
            std::vector<mx::array> ws;
            std::vector<mx::array> ss;
            std::vector<mx::array> bs;
            ws.reserve(eids.size());
            ss.reserve(eids.size());
            bs.reserve(eids.size());

            K50StackedGroup grp;
            grp.e2l.assign(num_experts, -1);
            for (size_t local_idx = 0; local_idx < eids.size(); ++local_idx) {
                int e = eids[local_idx];
                grp.e2l[e] = static_cast<int32_t>(local_idx);
                is_stacked[proj][e] = true;
                const auto& rec = pack->experts.at(proj + ":" + std::to_string(e));
                auto mat = pack->materialize_as_affine2(rec);
                grp.kind = mat.kind;
                grp.bits = mat.bits;
                grp.group_size = mat.group_size;
                ws.push_back(std::move(mat.weight));
                ss.push_back(std::move(mat.scales));
                bs.push_back(std::move(mat.biases));
            }
            grp.weights = mx::stack(ws, 0);
            grp.scales = mx::stack(ss, 0);
            grp.biases = mx::stack(bs, 0);
            mx::eval({grp.weights, grp.scales, grp.biases});
            grp.nbytes = grp.weights.nbytes() + grp.scales.nbytes() + grp.biases.nbytes();
            stacked_bytes += grp.nbytes;
            std::string gkey = grp.kind;
            groups[proj].emplace(gkey, std::move(grp));
        }
    }
}

void Qwen38Engine::load_k50_v3() {
    is_k50_v3_ = true;

    // 1. Load skeleton.safetensors (embed + head, 8-bit group_size=64)
    auto [skel_tensors, _skel_meta] = mx::load_safetensors(opts_.model_dir + "/skeleton.safetensors");
    embed_tokens_ = load_qlinear(skel_tensors, "embed", 64, 8);
    lm_head_ = load_qlinear(skel_tensors, "head", 64, 8);
    mx::eval({
        embed_tokens_.weight, *embed_tokens_.scales, *embed_tokens_.biases,
        lm_head_.weight, *lm_head_.scales, *lm_head_.biases
    });

    // 2. Load nonexpert.safetensors (hyper-connections, norms, GDN projections, gates, L1.ple)
    auto [nonexp_tensors, _nonexp_meta] = mx::load_safetensors(opts_.model_dir + "/nonexpert.safetensors");

    auto load_k50_hc = [&](const std::string& prefix, bool use_combine) -> HyperConnection {
        HyperConnection hc;
        hc.use_combine = use_combine;
        hc.norm_weight = nonexp_tensors.at(prefix + ".hc_norm.weight");
        hc.norm_scale_f32 = mx::add(
            mx::array(1.0f, mx::float32),
            mx::astype(hc.norm_weight, mx::float32));
        hc.eps_arr = mx::array({1e-6f}, mx::float32);
        hc.down.is_quantized = false;
        hc.down.weight = nonexp_tensors.at(prefix + ".input_mix_weight_down.weight");
        hc.up.is_quantized = false;
        hc.up.weight = nonexp_tensors.at(prefix + ".input_mix_weight_up.weight");
        if (use_combine) {
            hc.inject.is_quantized = false;
            hc.inject.weight = nonexp_tensors.at(prefix + ".block_inject_weight.weight");
            mx::eval({hc.norm_weight, hc.norm_scale_f32, hc.eps_arr,
                      hc.down.weight, hc.up.weight, hc.inject.weight});
        } else {
            mx::eval({hc.norm_weight, hc.norm_scale_f32, hc.eps_arr,
                      hc.down.weight, hc.up.weight});
        }
        return hc;
    };

    auto quantize_affine8 = [](const mx::array& w_bf16) -> QLinear {
        auto q_tup = mx::quantize(mx::astype(w_bf16, mx::float32), 64, 8, "affine");
        QLinear ql;
        ql.is_quantized = true;
        ql.bits = 8;
        ql.group_size = 64;
        ql.weight = std::move(q_tup[0]);
        ql.scales = std::move(q_tup[1]);
        ql.biases = std::move(q_tup[2]);
        mx::eval({ql.weight, *ql.scales, *ql.biases});
        return ql;
    };

    final_mixer_ = load_k50_hc("hyper", false);
    auto rope_inv_freq = make_rope_inv_freq(64, 10000000.0f);

    layers_.resize(48);
    exec_caches_.resize(48);
    last_slot_reader_.resize(48, std::nullopt);
    current_caps_.assign(48, 0);

    size_t pinned_hot_bytes = 0;

    // 3. Load all 48 decoder layers (L00..L47)
    for (int l = 0; l < 48; ++l) {
        auto& layer = layers_[l];
        layer.layer_idx = l;
        layer.is_linear = (l % 4 != 3);
        layer.k50_moe = true;
        layer.num_experts = (l == 0) ? 512 : 256;
        exec_caches_[l].is_linear = layer.is_linear;

        char ldir_buf[32];
        std::snprintf(ldir_buf, sizeof(ldir_buf), "/L%02d", l);
        layer.k50_pack = std::make_shared<ExpertPackCpp>(opts_.model_dir + ldir_buf, l);
        layer.k50_switch.build(l, layer.k50_pack, layer.num_experts, opts_.stack_mode);
        pinned_hot_bytes += layer.k50_switch.stacked_bytes;

        std::string lp = "L" + std::to_string(l);
        layer.attn_hc = load_k50_hc(lp + ".attn_hyper_connection", true);
        layer.mlp_hc = load_k50_hc(lp + ".mlp_hyper_connection", true);

        if (layer.is_linear) {
            layer.lin_attn.in_proj_qkv = layer.k50_pack->load_dense_qlinear("linear_attn.in_proj_qkv");
            layer.lin_attn.in_proj_z = layer.k50_pack->load_dense_qlinear("linear_attn.in_proj_z");
            layer.lin_attn.out_proj = layer.k50_pack->load_dense_qlinear("linear_attn.out_proj");

            std::string ap = lp + ".linear_attn";
            layer.lin_attn.in_proj_b.is_quantized = false;
            layer.lin_attn.in_proj_b.weight = nonexp_tensors.at(ap + ".in_proj_b.weight");
            layer.lin_attn.in_proj_a.is_quantized = false;
            layer.lin_attn.in_proj_a.weight = nonexp_tensors.at(ap + ".in_proj_a.weight");

            mx::array raw_conv = nonexp_tensors.at(ap + ".conv1d.weight");
            if (raw_conv.ndim() == 3 && raw_conv.shape(1) < raw_conv.shape(2)) {
                raw_conv = mx::swapaxes(raw_conv, 1, 2);
            }
            layer.lin_attn.conv1d_weight = raw_conv;
            layer.lin_attn.conv_weight_decode_f32 = mx::astype(
                mx::transpose(mx::squeeze(layer.lin_attn.conv1d_weight, -1)), mx::float32);
            layer.lin_attn.dt_bias = nonexp_tensors.at(ap + ".dt_bias");
            layer.lin_attn.A_log = nonexp_tensors.at(ap + ".A_log");
            layer.lin_attn.norm_weight = nonexp_tensors.at(ap + ".norm.weight");
            mx::eval({
                layer.lin_attn.in_proj_b.weight,
                layer.lin_attn.in_proj_a.weight,
                layer.lin_attn.conv1d_weight,
                layer.lin_attn.conv_weight_decode_f32,
                layer.lin_attn.dt_bias,
                layer.lin_attn.A_log,
                layer.lin_attn.norm_weight
            });
        } else {
            layer.full_attn.q_proj = layer.k50_pack->load_dense_qlinear("self_attn.q_proj");
            layer.full_attn.k_proj = layer.k50_pack->load_dense_qlinear("self_attn.k_proj");
            layer.full_attn.v_proj = layer.k50_pack->load_dense_qlinear("self_attn.v_proj");
            layer.full_attn.o_proj = layer.k50_pack->load_dense_qlinear("self_attn.o_proj");
            layer.full_attn.index_qk_proj = layer.k50_pack->load_dense_qlinear("self_attn.indexer.index_qk_proj");

            std::string ap = lp + ".self_attn";
            auto q_nw = nonexp_tensors.at(ap + ".q_norm.weight");
            auto k_nw = nonexp_tensors.at(ap + ".k_norm.weight");
            layer.full_attn.q_norm_scale = mx::add(mx::array(1.0f, mx::float32), mx::astype(q_nw, mx::float32));
            layer.full_attn.k_norm_scale = mx::add(mx::array(1.0f, mx::float32), mx::astype(k_nw, mx::float32));
            layer.full_attn.rope_inv_freq = rope_inv_freq;

            auto iq_nw = nonexp_tensors.at(ap + ".indexer.q_layernorm.weight");
            auto ik_nw = nonexp_tensors.at(ap + ".indexer.k_layernorm.weight");
            layer.full_attn.index_q_norm_scale = mx::add(mx::array(1.0f, mx::float32), mx::astype(iq_nw, mx::float32));
            layer.full_attn.index_k_norm_scale = mx::add(mx::array(1.0f, mx::float32), mx::astype(ik_nw, mx::float32));
            layer.full_attn.index_rope_inv_freq = rope_inv_freq;
            mx::eval({
                layer.full_attn.q_norm_scale,
                layer.full_attn.k_norm_scale,
                layer.full_attn.index_q_norm_scale,
                layer.full_attn.index_k_norm_scale
            });
        }

        // Router & shared expert
        std::string mp = lp + ".mlp";
        layer.gate = quantize_affine8(nonexp_tensors.at(mp + ".gate.weight"));
        layer.shared_expert_gate = quantize_affine8(nonexp_tensors.at(mp + ".shared_expert_gate.weight"));
        layer.shared_gate_proj = layer.k50_pack->load_dense_qlinear("mlp.shared_expert.gate_proj");
        layer.shared_up_proj = layer.k50_pack->load_dense_qlinear("mlp.shared_expert.up_proj");
        layer.shared_down_proj = layer.k50_pack->load_dense_qlinear("mlp.shared_expert.down_proj");
    }

    // 4. Configure Layer 1 PLE (mixed 2..6-bit from ple/ple_bits.json + ple/model-*.safetensors)
    ple_.active = true;
    init_ple_metadata(ple_, 248320, 20000000, 128, 1234);
    ple_.key_proj.is_quantized = false;
    ple_.key_proj.weight = nonexp_tensors.at("L1.ple.key_proj.weight");
    ple_.value_proj.is_quantized = false;
    ple_.value_proj.weight = nonexp_tensors.at("L1.ple.value_proj.weight");
    ple_.norm_key_weight = nonexp_tensors.at("L1.ple.norm_key.weight");
    ple_.norm_query_weight = nonexp_tensors.at("L1.ple.norm_query.weight");
    ple_.norm_conv_weight = nonexp_tensors.at("L1.ple.norm_conv.weight");

    mx::array raw_ple_conv = nonexp_tensors.at("L1.ple.conv1d.weight");
    if (raw_ple_conv.ndim() == 3 && raw_ple_conv.shape(1) < raw_ple_conv.shape(2)) {
        raw_ple_conv = mx::swapaxes(raw_ple_conv, 1, 2);
    }
    ple_.short_conv_weight = mx::contiguous(mx::squeeze(raw_ple_conv, -1));
    ple_.weight_scale = nonexp_tensors.at("L1.ple.ple_embedding.ngram_embedding.weight_scale");

    auto mult_arr = nonexp_tensors.at("L1.ple.ple_embedding.layer_multipliers");
    auto mod_arr = nonexp_tensors.at("L1.ple.ple_embedding.ngram_heads_vocab_sizes");
    auto hoff_arr = nonexp_tensors.at("L1.ple.ple_embedding.ngram_heads_offsets");
    mx::eval({
        ple_.key_proj.weight, ple_.value_proj.weight,
        ple_.norm_key_weight, ple_.norm_query_weight, ple_.norm_conv_weight,
        ple_.short_conv_weight, ple_.weight_scale,
        mult_arr, mod_arr, hoff_arr
    });

    ple_.multipliers_cpu.assign(mult_arr.data<int64_t>(), mult_arr.data<int64_t>() + 3);
    ple_.moduli_cpu.assign(mod_arr.data<int64_t>(), mod_arr.data<int64_t>() + 16);
    ple_.head_offsets_cpu.assign(hoff_arr.data<int64_t>(), hoff_arr.data<int64_t>() + 16);

    // Parse ple/ple_bits.json
    std::vector<int> shard_bits(128, 4);
    int ple_gs = 32;
    std::string ple_prefix = "model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_";
    {
        std::string bits_path = opts_.model_dir + "/ple/ple_bits.json";
        std::ifstream bifs(bits_path);
        if (!bifs) throw std::runtime_error("Cannot open " + bits_path);
        std::ostringstream bss;
        bss << bifs.rdbuf();
        std::string bs = bss.str();
        size_t i = 0;
        skip_ws(bs, i);
        if (i < bs.size() && bs[i] == '{') ++i;
        while (i < bs.size()) {
            skip_ws(bs, i);
            if (i < bs.size() && bs[i] == '}') { ++i; break; }
            if (bs[i] == ',') { ++i; continue; }
            std::string top_k = parse_json_string(bs, i);
            skip_ws(bs, i);
            if (i < bs.size() && bs[i] == ':') ++i;
            skip_ws(bs, i);
            if (top_k == "group_size") {
                ple_gs = static_cast<int>(parse_json_int(bs, i));
            } else if (top_k == "prefix") {
                ple_prefix = parse_json_string(bs, i);
            } else if (top_k == "shards") {
                if (i < bs.size() && bs[i] == '{') ++i;
                while (i < bs.size()) {
                    skip_ws(bs, i);
                    if (i < bs.size() && bs[i] == '}') { ++i; break; }
                    if (bs[i] == ',') { ++i; continue; }
                    std::string skey = parse_json_string(bs, i);
                    int sidx = std::stoi(skey);
                    skip_ws(bs, i);
                    if (i < bs.size() && bs[i] == ':') ++i;
                    shard_bits[sidx] = static_cast<int>(parse_json_int(bs, i));
                }
            } else {
                skip_json_value(bs, i);
            }
        }
    }

    // Parse ple/model.safetensors.index.json and mmap PLE shard files
    auto ple_wmap = parse_weight_map(opts_.model_dir + "/ple/model.safetensors.index.json");
    std::unordered_map<std::string, SafeTensorHeader> ple_hdrs;
    for (const auto& kv : ple_wmap) {
        const std::string& fname = kv.second;
        if (ple_mmaps_.find(fname) == ple_mmaps_.end()) {
            std::string full_path = opts_.model_dir + "/ple/" + fname;
            auto hdr = parse_safetensors_header(full_path);
            auto mm = std::make_shared<MmapFile>();
            mm->fd = ::open(full_path.c_str(), O_RDONLY);
            if (mm->fd < 0) throw std::runtime_error("Cannot open PLE shard: " + full_path);
            struct stat st{};
            ::fstat(mm->fd, &st);
            mm->size = static_cast<size_t>(st.st_size);
            mm->addr = ::mmap(nullptr, mm->size, PROT_READ, MAP_PRIVATE, mm->fd, 0);
            if (mm->addr == MAP_FAILED) throw std::runtime_error("mmap failed for: " + full_path);
            ::madvise(mm->addr, mm->size, MADV_RANDOM);
            mm->data_start = hdr.data_start;
            ple_mmaps_[fname] = mm;
            ple_hdrs[fname] = std::move(hdr);
        }
    }

    ple_.shards.resize(128);
    auto resolve_k50_ple_slice = [&](const std::string& tname) -> PleTensorSlice {
        const std::string& fname = ple_wmap.at(tname);
        const auto& mm = ple_mmaps_.at(fname);
        const auto& entry = ple_hdrs.at(fname).tensors.at(tname);
        PleTensorSlice sl;
        sl.data_ptr = static_cast<const uint8_t*>(mm->addr) + mm->data_start + entry.data_offset_start;
        sl.rows = entry.shape[0];
        sl.cols = entry.shape[1];
        size_t total_bytes = entry.data_offset_end - entry.data_offset_start;
        sl.row_bytes = static_cast<int64_t>(total_bytes / static_cast<size_t>(sl.rows));
        sl.dtype = entry.dtype;
        return sl;
    };

    for (int s = 0; s < 128; ++s) {
        std::string sp = ple_prefix + std::to_string(s);
        ple_.shards[s].weight = resolve_k50_ple_slice(sp + ".weight");
        ple_.shards[s].scales = resolve_k50_ple_slice(sp + ".scales");
        ple_.shards[s].biases = resolve_k50_ple_slice(sp + ".biases");
        ple_.shards[s].bits = shard_bits[s];
        ple_.shards[s].group_size = ple_gs;
    }

    // 5. Configure ResidentCache budget (default 15.6 GB total = pinned hot + cold LRU)
    size_t total_budget_bytes = static_cast<size_t>(opts_.resident_gb * 1e9);
    k50_cache_.set_budget(total_budget_bytes, pinned_hot_bytes);

    loaded_ = true;
}

} // namespace qwen38

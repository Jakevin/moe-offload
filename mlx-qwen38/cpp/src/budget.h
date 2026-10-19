#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <sstream>
#include <string>
#include <vector>

namespace qwen38 {

constexpr uint64_t kBudgetExpertBytes = 1536000ULL;
constexpr int kBudgetOffloadLayers = 47;
constexpr int kBudgetMinSlotCap = 16;
constexpr int kBudgetMaxSlotCap = 256;
// Process cap shared with the engine self-guard (engine.cpp check_process_memory_guard). Unit: bytes of
// task phys_footprint (includes compressed + Metal/IOKit-owned memory, excludes clean file-backed pages); this is
// NOT the ps RSS that sup/run_wd.sh watches (KiB, resident incl. mapped file pages, excludes compressed).
constexpr uint64_t kSelfGuardFootprintBytes = 19500000000ULL;  // 19.5 GB hard cap
constexpr uint64_t kBudgetProcLimitBytes = kSelfGuardFootprintBytes; // legacy name (was 20 GiB > self-guard)
constexpr uint64_t kBudgetDefaultHeadroomBytes = 3ULL << 30;   // MLX buffer cache limit (2 GiB) + 1 GiB transient
// Effective process limit = min(19.5e9, floor(0.8 * clean avail sampled before load)).
inline uint64_t compute_proc_limit_bytes(uint64_t avail_clean_start) {
    const uint64_t eighty = static_cast<uint64_t>(static_cast<long double>(avail_clean_start) * 0.8L);
    return std::min(kSelfGuardFootprintBytes, eighty);
}
constexpr uint64_t kBudgetLowAvailBytes = 3584ULL << 20;     // 3.5 GiB shrink threshold
constexpr uint64_t kBudgetPlentyAvailBytes = 7168ULL << 20;  // 7.0 GiB grow threshold
constexpr uint64_t kBudgetMaxSwapDeltaBytes = 256ULL << 20;  // 256 MiB swap growth over run start
constexpr uint64_t kBudgetMaxCompDeltaBytes = 512ULL << 20;  // 512 MiB compressor growth since last check
constexpr int kBudgetHysteresisSteps = 32;                   // no grow within 32 steps after a shrink

// Compute available clean memory from vm_statistics64 page counts.
// Never counts inactive_count (which includes inactive anonymous pages).
// Because macOS vm_statistics64 provides no separate dirty-file page counter,
// external_page_count is an upper bound on clean file-backed cache; we apply a
// 10% safety margin on external_page_count (in addition to os_reserve in decide_budget).
inline uint64_t compute_clean_avail_bytes(
    uint64_t free_pages,
    uint64_t purgeable_pages,
    uint64_t external_pages,
    uint64_t page_size) {
    uint64_t safe_external_pages = external_pages - (external_pages / 10);
    return (free_pages + purgeable_pages + safe_external_pages) * page_size;
}

struct BudgetReadings {
    bool is_init = false;
    int step_idx = 8;
    int last_shrink_step = -1000;
    uint64_t avail_bytes = 0;
    uint64_t swap_used_bytes = 0;
    uint64_t initial_swap_bytes = 0;
    uint64_t compressed_bytes = 0;
    uint64_t prev_compressed_bytes = 0;
    int pressure_level = 1;
    uint64_t total_ram_bytes = 64ULL << 30;
    uint64_t non_cache_footprint_bytes = 6ULL << 30;
    uint64_t recommended_working_set_bytes = 0; // from mx::metal::device_info()["max_recommended_working_set_size"]
    int max_slot_cap = 128;
    uint64_t proc_limit_bytes = kSelfGuardFootprintBytes; // effective, frozen before load
    uint64_t headroom_bytes = 0;                          // transient reserve (prefill / MLX cache / resize)
};

struct BudgetDecision {
    std::string action; // "init" | "none" | "shrink" | "grow"
    std::string reason;
    std::vector<int> new_caps;
    int old_total_slots = 0;
    int new_total_slots = 0;
    uint64_t target_slot_bytes = 0;
    uint64_t max_allowed_slot_bytes = 0;
    bool fits = true; // false: even kBudgetMinSlotCap per layer does not fit under the caps -> caller must refuse/abort
};

inline BudgetDecision decide_budget(
    const BudgetReadings& r,
    const std::vector<int>& current_caps) {
    BudgetDecision d;
    d.new_caps = current_caps;
    if (d.new_caps.empty()) {
        d.new_caps.assign(48, 0);
    }
    const size_t first_l = (d.new_caps.size() == 47) ? 0 : 1;
    const size_t end_l = std::min(d.new_caps.size(), first_l + static_cast<size_t>(kBudgetOffloadLayers));

    for (size_t l = first_l; l < end_l; ++l) {
        d.old_total_slots += d.new_caps[l];
    }

    const uint64_t cur_slot_bytes = static_cast<uint64_t>(d.old_total_slots) * kBudgetExpertBytes;
    const uint64_t os_reserve = std::max(7ULL << 30, static_cast<uint64_t>(r.total_ram_bytes * 0.15));

    const uint64_t reserved = r.non_cache_footprint_bytes + r.headroom_bytes;
    const uint64_t max_proc_cache = (r.proc_limit_bytes > reserved) ? (r.proc_limit_bytes - reserved) : 0ULL;

    uint64_t max_allowed = max_proc_cache;
    bool ws_is_binding_cap = false;
    if (r.recommended_working_set_bytes > 0) {
        const uint64_t max_ws_cache = (r.recommended_working_set_bytes > reserved)
            ? (r.recommended_working_set_bytes - reserved)
            : 0ULL;
        if (max_ws_cache <= max_proc_cache) {
            max_allowed = max_ws_cache;
            ws_is_binding_cap = true;
        }
    }
    d.max_allowed_slot_bytes = max_allowed;

    const uint64_t reclaimable_avail = r.avail_bytes + cur_slot_bytes;
    const uint64_t min_cache_bytes = static_cast<uint64_t>(kBudgetOffloadLayers * kBudgetMinSlotCap) * kBudgetExpertBytes;
    const uint64_t usable_avail = (reclaimable_avail > os_reserve)
        ? (reclaimable_avail - os_reserve)
        : min_cache_bytes;

    const uint64_t raw_cap_u = max_allowed / (static_cast<uint64_t>(kBudgetOffloadLayers) * kBudgetExpertBytes);
    d.fits = (raw_cap_u >= static_cast<uint64_t>(kBudgetMinSlotCap));
    // Never clamp UP past the cap: if the minimum does not fit, fits=false and the caller refuses / aborts.
    const int ws_max_per_layer = std::clamp(
        static_cast<int>(std::min<uint64_t>(raw_cap_u, kBudgetMaxSlotCap)),
        kBudgetMinSlotCap,
        kBudgetMaxSlotCap);

    if (r.is_init) {
        uint64_t target_bytes = std::min(usable_avail, max_allowed);
        d.target_slot_bytes = target_bytes;
        const int max_init_cap = (r.max_slot_cap > 0)
            ? std::min(r.max_slot_cap, kBudgetMaxSlotCap)
            : kBudgetMaxSlotCap;
        const int raw_per_layer = static_cast<int>(
            target_bytes / (static_cast<uint64_t>(kBudgetOffloadLayers) * kBudgetExpertBytes));
        int per_layer_target = std::clamp(raw_per_layer, kBudgetMinSlotCap, std::min(max_init_cap, ws_max_per_layer));

        const bool swap_grew = (r.swap_used_bytes > r.initial_swap_bytes) &&
                               ((r.swap_used_bytes - r.initial_swap_bytes) > kBudgetMaxSwapDeltaBytes);
        const bool comp_grew = (r.compressed_bytes > r.prev_compressed_bytes) &&
                               ((r.compressed_bytes - r.prev_compressed_bytes) > kBudgetMaxCompDeltaBytes);
        if (r.pressure_level >= 2 || swap_grew || comp_grew || r.avail_bytes < kBudgetLowAvailBytes) {
            per_layer_target = std::max(kBudgetMinSlotCap, std::min(per_layer_target, static_cast<int>(per_layer_target * 0.85)));
        }

        d.new_total_slots = 0;
        for (size_t l = first_l; l < end_l; ++l) {
            d.new_caps[l] = per_layer_target;
            d.new_total_slots += per_layer_target;
        }
        d.action = "init";
        if (!d.fits) {
            d.reason = "insufficient_for_min_cap";
        } else if (ws_is_binding_cap && max_allowed < usable_avail) {
            d.reason = "working_set_cap";
        } else if (r.avail_bytes < kBudgetLowAvailBytes || usable_avail <= min_cache_bytes) {
            d.reason = "low_avail";
        } else {
            d.reason = "init_sizing";
        }
        return d;
    }

    // Step-level check
    const bool pressure_high = (r.pressure_level >= 2);
    const bool swap_grew = (r.swap_used_bytes > r.initial_swap_bytes) &&
                           ((r.swap_used_bytes - r.initial_swap_bytes) > kBudgetMaxSwapDeltaBytes);
    const bool comp_grew = (r.compressed_bytes > r.prev_compressed_bytes) &&
                           ((r.compressed_bytes - r.prev_compressed_bytes) > kBudgetMaxCompDeltaBytes);
    const bool avail_low = (r.avail_bytes < kBudgetLowAvailBytes);
    const bool over_ws_cap = (cur_slot_bytes > max_allowed); // process cap applies even when Metal rec_ws is 0

    if (pressure_high || swap_grew || comp_grew || avail_low || over_ws_cap) {
        std::string trigger;
        if (pressure_high) {
            trigger = (r.pressure_level >= 4) ? "pressure_critical" : "pressure_warn";
        } else if (swap_grew) {
            trigger = "swap_growth";
        } else if (comp_grew) {
            trigger = "compressor_growth";
        } else if (avail_low) {
            trigger = "low_avail";
        } else {
            trigger = "working_set_cap";
        }

        const bool severe = (r.pressure_level >= 4) || (r.avail_bytes < (2048ULL << 20));
        const double factor = severe ? 0.70 : 0.85;

        d.new_total_slots = 0;
        for (size_t l = first_l; l < end_l; ++l) {
            int new_c = d.new_caps[l];
            if (pressure_high || swap_grew || comp_grew || avail_low) {
                new_c = std::max(kBudgetMinSlotCap, static_cast<int>(std::floor(new_c * factor)));
            }
            new_c = std::max(kBudgetMinSlotCap, std::min(new_c, ws_max_per_layer));
            d.new_caps[l] = new_c;
            d.new_total_slots += new_c;
        }
        d.target_slot_bytes = static_cast<uint64_t>(d.new_total_slots) * kBudgetExpertBytes;
        if (d.new_total_slots < d.old_total_slots) {
            d.action = "shrink";
            d.reason = trigger;
        } else {
            d.action = "none";
            d.reason = trigger + "_at_min";
        }
        return d;
    }

    // Check grow conditions
    if (r.pressure_level == 1 && r.avail_bytes > kBudgetPlentyAvailBytes) {
        const bool in_hysteresis = (r.last_shrink_step >= 0) &&
                                   ((r.step_idx - r.last_shrink_step) <= kBudgetHysteresisSteps);
        if (in_hysteresis) {
            d.action = "none";
            d.reason = "hysteresis";
            d.new_total_slots = d.old_total_slots;
            d.target_slot_bytes = cur_slot_bytes;
            return d;
        }

        if (r.step_idx <= 0 || (r.step_idx % 8 != 0)) {
            d.action = "none";
            d.reason = "cadence";
            d.new_total_slots = d.old_total_slots;
            d.target_slot_bytes = cur_slot_bytes;
            return d;
        }

        const int user_max_cap = (r.max_slot_cap > 0) ? r.max_slot_cap : 128;
        const int avail_per_layer = std::clamp(
            static_cast<int>(usable_avail / (static_cast<uint64_t>(kBudgetOffloadLayers) * kBudgetExpertBytes)),
            kBudgetMinSlotCap,
            kBudgetMaxSlotCap);
        const int max_cap = std::min({user_max_cap, ws_max_per_layer, avail_per_layer});

        d.new_total_slots = 0;
        for (size_t l = first_l; l < end_l; ++l) {
            int cur_c = d.new_caps[l];
            int new_c = (cur_c < max_cap) ? std::min(max_cap, cur_c + 4) : cur_c;
            d.new_caps[l] = new_c;
            d.new_total_slots += new_c;
        }
        d.target_slot_bytes = static_cast<uint64_t>(d.new_total_slots) * kBudgetExpertBytes;
        if (d.new_total_slots > d.old_total_slots) {
            d.action = "grow";
            d.reason = (ws_is_binding_cap && max_cap == ws_max_per_layer && ws_max_per_layer < user_max_cap)
                ? "plenty_avail_ws_capped"
                : "plenty_avail";
        } else {
            d.action = "none";
            d.reason = (ws_is_binding_cap && ws_max_per_layer <= user_max_cap)
                ? "working_set_cap"
                : "at_cap";
        }
        return d;
    }

    d.action = "none";
    d.reason = "normal";
    d.new_total_slots = d.old_total_slots;
    d.target_slot_bytes = cur_slot_bytes;
    return d;
}

inline std::string format_budget_log_jsonl(const BudgetReadings& r, const BudgetDecision& d) {
    const size_t first_l = (d.new_caps.size() == 47) ? 0 : 1;
    const int per_layer_cap = (first_l < d.new_caps.size()) ? d.new_caps[first_l] : 0;
    std::ostringstream oss;
    oss << "{"
        << "\"event\":\"" << (r.is_init ? "init" : "step") << "\","
        << "\"step\":" << (r.is_init ? -1 : r.step_idx) << ","
        << "\"action\":\"" << d.action << "\","
        << "\"reason\":\"" << d.reason << "\","
        << "\"avail_bytes\":" << r.avail_bytes << ","
        << "\"swap_used_bytes\":" << r.swap_used_bytes << ","
        << "\"initial_swap_bytes\":" << r.initial_swap_bytes << ","
        << "\"compressed_bytes\":" << r.compressed_bytes << ","
        << "\"prev_compressed_bytes\":" << r.prev_compressed_bytes << ","
        << "\"pressure_level\":" << r.pressure_level << ","
        << "\"non_cache_footprint_bytes\":" << r.non_cache_footprint_bytes << ","
        << "\"recommended_working_set_bytes\":" << r.recommended_working_set_bytes << ","
        << "\"max_allowed_slot_bytes\":" << d.max_allowed_slot_bytes << ","
        << "\"proc_limit_bytes\":" << r.proc_limit_bytes << ","
        << "\"headroom_bytes\":" << r.headroom_bytes << ","
        << "\"fits\":" << (d.fits ? "true" : "false") << ","
        << "\"old_total_slots\":" << d.old_total_slots << ","
        << "\"new_total_slots\":" << d.new_total_slots << ","
        << "\"per_layer_cap\":" << per_layer_cap << ","
        << "\"last_shrink_step\":" << r.last_shrink_step
        << "}";
    return oss.str();
}

} // namespace qwen38

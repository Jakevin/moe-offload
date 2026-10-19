#include "budget.h"

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <vector>

using namespace qwen38;

static std::vector<int> make_caps(int per_layer) {
    std::vector<int> caps(48, 0);
    for (int l = 1; l < 48; ++l) {
        caps[l] = per_layer;
    }
    return caps;
}

static void test_clean_avail_formula() {
    const uint64_t page_size = 16384; // 16 KiB macOS arm64 page size
    const uint64_t free_p = 100000;
    const uint64_t purge_p = 20000;
    const uint64_t ext_p = 200000; // 10% safety margin -> 180000 pages
    uint64_t avail = compute_clean_avail_bytes(free_p, purge_p, ext_p, page_size);
    uint64_t expected = (100000ULL + 20000ULL + 180000ULL) * page_size;
    assert(avail == expected);
    std::printf("[PASS] test_clean_avail_formula (avail=%llu MiB)\n",
                static_cast<unsigned long long>(avail >> 20));
}

static void test_plenty_of_memory() {
    // 1. Init sizing with plenty of memory
    BudgetReadings r;
    r.is_init = true;
    r.avail_bytes = 24ULL << 30; // 24 GiB clean avail
    r.total_ram_bytes = 64ULL << 30;
    r.non_cache_footprint_bytes = 6ULL << 30;
    r.recommended_working_set_bytes = 48ULL << 30;
    r.max_slot_cap = 256;

    auto d_init = decide_budget(r, make_caps(64));
    assert(d_init.action == "init");
    assert(d_init.reason == "init_sizing");
    assert(d_init.new_caps[1] > 64);
    assert(static_cast<uint64_t>(d_init.new_total_slots) * kBudgetExpertBytes + r.non_cache_footprint_bytes
           <= r.recommended_working_set_bytes);

    // 2. Step-level grow when plenty of memory is available (> 7 GiB) and no recent shrink
    BudgetReadings r_step;
    r_step.is_init = false;
    r_step.step_idx = 16;
    r_step.last_shrink_step = -1000;
    r_step.avail_bytes = 18ULL << 30;
    r_step.pressure_level = 1;
    r_step.total_ram_bytes = 64ULL << 30;
    r_step.non_cache_footprint_bytes = 6ULL << 30;
    r_step.recommended_working_set_bytes = 48ULL << 30;
    r_step.max_slot_cap = 128;

    auto d_grow = decide_budget(r_step, make_caps(64));
    assert(d_grow.action == "grow");
    assert(d_grow.reason == "plenty_avail");
    assert(d_grow.new_caps[1] == 68);
    assert(d_grow.new_total_slots == 47 * 68);
    std::printf("[PASS] test_plenty_of_memory (init_cap=%d, grow_cap=%d)\n",
                d_init.new_caps[1], d_grow.new_caps[1]);
}

static void test_low_avail() {
    BudgetReadings r;
    r.is_init = false;
    r.step_idx = 8;
    r.avail_bytes = 2800ULL << 20; // 2.8 GiB (< 3.5 GiB threshold)
    r.pressure_level = 1;
    r.total_ram_bytes = 64ULL << 30;
    r.non_cache_footprint_bytes = 6ULL << 30;
    r.recommended_working_set_bytes = 48ULL << 30;

    auto d = decide_budget(r, make_caps(80));
    assert(d.action == "shrink");
    assert(d.reason == "low_avail");
    assert(d.new_caps[1] == 68); // 80 * 0.85 = 68

    // Severe low avail (< 2.0 GiB) -> 0.70x shrink
    r.avail_bytes = 1500ULL << 20;
    auto d_sev = decide_budget(r, make_caps(80));
    assert(d_sev.action == "shrink");
    assert(d_sev.reason == "low_avail");
    assert(d_sev.new_caps[1] == 56); // 80 * 0.70 = 56
    std::printf("[PASS] test_low_avail (mild=%d, severe=%d)\n",
                d.new_caps[1], d_sev.new_caps[1]);
}

static void test_pressure_warn() {
    BudgetReadings r;
    r.is_init = false;
    r.step_idx = 8;
    r.avail_bytes = 10ULL << 30; // Even if avail looks okay, kernel pressure_level >= 2 shrinks
    r.pressure_level = 2;
    r.total_ram_bytes = 64ULL << 30;
    r.non_cache_footprint_bytes = 6ULL << 30;

    auto d_warn = decide_budget(r, make_caps(100));
    assert(d_warn.action == "shrink");
    assert(d_warn.reason == "pressure_warn");
    assert(d_warn.new_caps[1] == 85);

    r.pressure_level = 4;
    auto d_crit = decide_budget(r, make_caps(100));
    assert(d_crit.action == "shrink");
    assert(d_crit.reason == "pressure_critical");
    assert(d_crit.new_caps[1] == 70);
    std::printf("[PASS] test_pressure_warn (warn=%d, crit=%d)\n",
                d_warn.new_caps[1], d_crit.new_caps[1]);
}

static void test_swap_and_compressor_growth() {
    // 1. Swap growth > 256 MiB over run start triggers shrink even when pressure == 1 and avail > 3.5 GiB
    BudgetReadings r_swap;
    r_swap.is_init = false;
    r_swap.step_idx = 24;
    r_swap.avail_bytes = 8ULL << 30;
    r_swap.pressure_level = 1;
    r_swap.initial_swap_bytes = 100ULL << 20;
    r_swap.swap_used_bytes = 400ULL << 20; // +300 MiB (> 256 MiB)

    auto d_swap = decide_budget(r_swap, make_caps(80));
    assert(d_swap.action == "shrink");
    assert(d_swap.reason == "swap_growth");
    assert(d_swap.new_caps[1] == 68);

    // 2. Compressor growth > 512 MiB since last check triggers shrink BEFORE swap starts
    BudgetReadings r_comp;
    r_comp.is_init = false;
    r_comp.step_idx = 24;
    r_comp.avail_bytes = 8ULL << 30;
    r_comp.pressure_level = 1;
    r_comp.prev_compressed_bytes = 1024ULL << 20;
    r_comp.compressed_bytes = 1600ULL << 20; // +576 MiB (> 512 MiB)

    auto d_comp = decide_budget(r_comp, make_caps(80));
    assert(d_comp.action == "shrink");
    assert(d_comp.reason == "compressor_growth");
    assert(d_comp.new_caps[1] == 68);
    std::printf("[PASS] test_swap_and_compressor_growth (swap_cap=%d, comp_cap=%d)\n",
                d_swap.new_caps[1], d_comp.new_caps[1]);
}

static void test_recommended_working_set_cap() {
    // Suppose non_cache_footprint = 6 GiB, recommendedMaxWorkingSetSize = 9 GiB
    // Then max_allowed_slot_bytes = 3 GiB = 3,221,225,472 bytes.
    // Per layer max = floor(3 GiB / (47 * 1,536,000)) = 44 slots/layer.
    BudgetReadings r;
    r.is_init = true;
    r.avail_bytes = 32ULL << 30; // Huge available RAM
    r.total_ram_bytes = 64ULL << 30;
    r.non_cache_footprint_bytes = 6ULL << 30;
    r.recommended_working_set_bytes = 9ULL << 30;
    r.max_slot_cap = 256;

    auto d_init = decide_budget(r, make_caps(64));
    assert(d_init.action == "init");
    assert(d_init.reason == "working_set_cap");
    assert(d_init.new_caps[1] == 44);
    uint64_t total_slot_bytes = static_cast<uint64_t>(d_init.new_total_slots) * kBudgetExpertBytes;
    assert(r.non_cache_footprint_bytes + total_slot_bytes <= r.recommended_working_set_bytes);

    // Step check when current caps exceed recommendedMaxWorkingSetSize -> shrinks to fit
    r.is_init = false;
    r.step_idx = 8;
    auto d_step_over = decide_budget(r, make_caps(64));
    assert(d_step_over.action == "shrink");
    assert(d_step_over.reason == "working_set_cap");
    assert(d_step_over.new_caps[1] == 44);

    // Step check when current caps == 42 and plenty of memory -> grows only up to 44 (not 46)
    auto d_step_grow = decide_budget(r, make_caps(42));
    assert(d_step_grow.action == "grow");
    assert(d_step_grow.new_caps[1] == 44);

    // Step check when already at working_set_cap (44) -> action "none", reason "working_set_cap"
    auto d_step_at = decide_budget(r, make_caps(44));
    assert(d_step_at.action == "none");
    assert(d_step_at.reason == "working_set_cap");
    assert(d_step_at.new_caps[1] == 44);

    std::printf("[PASS] test_recommended_working_set_cap (ws_max_per_layer=%d)\n",
                d_init.new_caps[1]);
}

static void test_hysteresis() {
    // Shrink happened at step 8. For steps 9..40 (within 32 steps), grow must be blocked by hysteresis.
    BudgetReadings r;
    r.is_init = false;
    r.last_shrink_step = 8;
    r.avail_bytes = 20ULL << 30; // Plenty of memory
    r.pressure_level = 1;
    r.total_ram_bytes = 64ULL << 30;
    r.non_cache_footprint_bytes = 6ULL << 30;
    r.recommended_working_set_bytes = 48ULL << 30;
    r.max_slot_cap = 128;

    for (int step : {8, 16, 24, 32, 40}) {
        r.step_idx = step;
        auto d = decide_budget(r, make_caps(50));
        assert(d.action == "none");
        assert(d.reason == "hysteresis");
        assert(d.new_caps[1] == 50);
    }

    // At step 48 (48 - 8 = 40 > 32), hysteresis has elapsed and grow is allowed.
    r.step_idx = 48;
    auto d_after = decide_budget(r, make_caps(50));
    assert(d_after.action == "grow");
    assert(d_after.reason == "plenty_avail");
    assert(d_after.new_caps[1] == 54);

    std::string jsonl = format_budget_log_jsonl(r, d_after);
    assert(jsonl.find("\"action\":\"grow\"") != std::string::npos);
    assert(jsonl.find("\"reason\":\"plenty_avail\"") != std::string::npos);

    std::printf("[PASS] test_hysteresis (blocked steps 8..40, grew at step 48 to %d)\n",
                d_after.new_caps[1]);
}

// ---- 2026-10-18: process cap = min(19.5e9, 0.8 x clean avail at start) + transient headroom (review fixes) ----
#define CHECK(c) do { if (!(c)) { std::printf("[FAIL] %s:%d %s\n", __FILE__, __LINE__, #c); std::exit(1); } } while (0)
static uint64_t slots_bytes(const BudgetDecision& d) { return static_cast<uint64_t>(d.new_total_slots) * kBudgetExpertBytes; }

static void test_proc_limit_real_numbers() {
    // Numbers from logs/c5_p1_auto_budget.jsonl (the run that hit the 19.5 GB self-guard with 237 slots/layer).
    BudgetReadings r; r.is_init = true; r.avail_bytes = 34909257728ULL; r.total_ram_bytes = 64ULL << 30;
    r.non_cache_footprint_bytes = 4304072352ULL; r.recommended_working_set_bytes = 55662788608ULL; r.max_slot_cap = 256;
    r.proc_limit_bytes = compute_proc_limit_bytes(34909257728ULL + 0); r.headroom_bytes = kBudgetDefaultHeadroomBytes;
    CHECK(r.proc_limit_bytes == kSelfGuardFootprintBytes);  // 0.8 x 34.9e9 = 27.9e9 > 19.5e9
    auto d = decide_budget(r, make_caps(102));
    CHECK(d.fits && d.action == "init");
    CHECK(d.new_caps[1] <= 165 && d.new_caps[1] >= 160);
    CHECK(slots_bytes(d) + r.non_cache_footprint_bytes + r.headroom_bytes <= kSelfGuardFootprintBytes);
    std::printf("[PASS] test_proc_limit_real_numbers (cap=%d slots/layer, predicted=%.2f GB + headroom <= 19.5 GB)\n",
                d.new_caps[1], (slots_bytes(d) + r.non_cache_footprint_bytes) / 1e9);
}

static void test_proc_limit_80pct_binding() {
    BudgetReadings r; r.is_init = true; r.avail_bytes = 14ULL << 30; r.total_ram_bytes = 64ULL << 30;
    r.non_cache_footprint_bytes = 4304072352ULL; r.recommended_working_set_bytes = 55662788608ULL; r.max_slot_cap = 256;
    r.proc_limit_bytes = compute_proc_limit_bytes(15ULL * 1000000000ULL);  // 0.8 x 15e9 = 12e9 < 19.5e9
    r.headroom_bytes = kBudgetDefaultHeadroomBytes;
    CHECK(r.proc_limit_bytes == 12000000000ULL);
    auto d = decide_budget(r, make_caps(102));
    CHECK(d.fits);
    CHECK(slots_bytes(d) + r.non_cache_footprint_bytes + r.headroom_bytes <= 12000000000ULL);
    std::printf("[PASS] test_proc_limit_80pct_binding (cap=%d slots/layer under 12e9)\n", d.new_caps[1]);
}

static void test_min_cap_does_not_fit_refuses() {
    BudgetReadings r; r.is_init = true; r.avail_bytes = 8ULL << 30; r.total_ram_bytes = 64ULL << 30;
    r.non_cache_footprint_bytes = 4304072352ULL; r.recommended_working_set_bytes = 55662788608ULL; r.max_slot_cap = 256;
    r.proc_limit_bytes = 7000000000ULL; r.headroom_bytes = kBudgetDefaultHeadroomBytes;  // 4.3e9 + 3.2e9 > 7e9: underflow guard
    auto d = decide_budget(r, make_caps(102));
    CHECK(!d.fits); CHECK(d.reason == "insufficient_for_min_cap"); CHECK(d.max_allowed_slot_bytes == 0);
    r.proc_limit_bytes = 8500000000ULL;  // 0.98e9 left < 16 x 47 x 1.536e6 = 1.155e9
    auto d2 = decide_budget(r, make_caps(102));
    CHECK(!d2.fits);
    std::printf("[PASS] test_min_cap_does_not_fit_refuses (no clamp-up past the cap, no unsigned underflow)\n");
}

static void test_step_shrinks_by_proc_cap_without_metal_ws() {
    BudgetReadings r; r.is_init = false; r.step_idx = 3; r.avail_bytes = 20ULL << 30; r.pressure_level = 1;
    r.total_ram_bytes = 64ULL << 30; r.non_cache_footprint_bytes = 4927695616ULL;
    r.recommended_working_set_bytes = 0;  // Metal limit unknown: the process cap must still apply
    r.max_slot_cap = 128; r.proc_limit_bytes = kSelfGuardFootprintBytes; r.headroom_bytes = kBudgetDefaultHeadroomBytes;
    auto d = decide_budget(r, make_caps(237));
    CHECK(d.action == "shrink"); CHECK(d.fits);
    CHECK(slots_bytes(d) + r.non_cache_footprint_bytes + r.headroom_bytes <= kSelfGuardFootprintBytes);
    // grow never exceeds the same cap
    r.step_idx = 64; r.last_shrink_step = -1000; r.avail_bytes = 30ULL << 30; r.max_slot_cap = 256;
    auto caps = d.new_caps;
    for (int i = 0; i < 200; ++i) { r.step_idx += 8; auto g = decide_budget(r, caps); caps = g.new_caps;
        CHECK(static_cast<uint64_t>(g.new_total_slots) * kBudgetExpertBytes + r.non_cache_footprint_bytes + r.headroom_bytes
              <= kSelfGuardFootprintBytes); }
    std::printf("[PASS] test_step_shrinks_by_proc_cap_without_metal_ws (237 -> %d, grow capped at %d)\n", d.new_caps[1], caps[1]);
}

static void test_assertions_are_live() {
    bool fired = false;
#ifdef NDEBUG
    fired = false;
#else
    fired = true;
#endif
    CHECK(fired);  // test target must be compiled without NDEBUG so assert() checks above are real
    std::printf("[PASS] test_assertions_are_live (NDEBUG not defined)\n");
}

int main() {
    test_clean_avail_formula();
    test_plenty_of_memory();
    test_low_avail();
    test_pressure_warn();
    test_swap_and_compressor_growth();
    test_recommended_working_set_cap();
    test_hysteresis();
    test_assertions_are_live();
    test_proc_limit_real_numbers();
    test_proc_limit_80pct_binding();
    test_min_cap_does_not_fit_refuses();
    test_step_shrinks_by_proc_cap_without_metal_ws();
    std::printf("ALL BUDGET TESTS PASSED\n");
    return 0;
}

#ifdef NDEBUG
#error "test_io_sched must be built with assertions enabled (-UNDEBUG)"
#endif
#include "io_sched.h"

#include <atomic>
#include <cassert>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <filesystem>
#include <future>
#include <mutex>
#include <random>
#include <string>
#include <thread>
#include <unistd.h>
#include <vector>

using namespace qwen38;

namespace {

constexpr int kTestLayers = 4;       // layers 1..4
constexpr int kTestExperts = 32;     // experts 0..31 per layer
constexpr int kNumTensors = 9;
constexpr uint64_t kRowBytes[kNumTensors] = {
    1024, 256, 256,
    1024, 256, 256,
    1024, 256, 256
};

static inline uint64_t expected_word(int layer, int eid, int tensor, size_t word_idx) {
    uint64_t x = (static_cast<uint64_t>(layer) << 48)
               ^ (static_cast<uint64_t>(eid) << 32)
               ^ (static_cast<uint64_t>(tensor) << 24)
               ^ static_cast<uint64_t>(word_idx);
    x ^= (x >> 30);
    x *= 0xBF58476D1CE4E5B9ULL;
    x ^= (x >> 27);
    x *= 0x94D049BB133111EBULL;
    x ^= (x >> 31);
    return x;
}

struct SynthExpertFile {
    std::string path;
    int fd = -1;
    uint64_t base_offsets[kTestLayers + 1][kNumTensors]{};
    uint64_t total_bytes = 0;

    void create(const std::string& file_path) {
        path = file_path;
        std::filesystem::create_directories(std::filesystem::path(path).parent_path());

        uint64_t offset = 0;
        for (int l = 1; l <= kTestLayers; ++l) {
            for (int t = 0; t < kNumTensors; ++t) {
                base_offsets[l][t] = offset;
                offset += static_cast<uint64_t>(kTestExperts) * kRowBytes[t];
            }
        }
        total_bytes = offset;
        assert(total_bytes < (16ULL << 20)); // Rule: keep test data files < 16 MB

        std::vector<uint8_t> image(total_bytes, 0);
        for (int l = 1; l <= kTestLayers; ++l) {
            for (int t = 0; t < kNumTensors; ++t) {
                uint64_t rb = kRowBytes[t];
                size_t n_words = rb / sizeof(uint64_t);
                for (int e = 0; e < kTestExperts; ++e) {
                    uint64_t* dst = reinterpret_cast<uint64_t*>(
                        image.data() + base_offsets[l][t] + static_cast<uint64_t>(e) * rb);
                    for (size_t w = 0; w < n_words; ++w) {
                        dst[w] = expected_word(l, e, t, w);
                    }
                }
            }
        }

        int wfd = ::open(path.c_str(), O_CREAT | O_TRUNC | O_WRONLY, 0644);
        assert(wfd >= 0);
        size_t written = 0;
        while (written < image.size()) {
            ssize_t n = ::write(wfd, image.data() + written, image.size() - written);
            assert(n > 0);
            written += static_cast<size_t>(n);
        }
        ::close(wfd);

        fd = ::open(path.c_str(), O_RDONLY);
        assert(fd >= 0);
    }

    ~SynthExpertFile() {
        if (fd >= 0) {
            ::close(fd);
        }
        // Rule: do NOT delete tests/tmp/ files
    }
};

struct LayerSlotBuffers {
    int capacity = 0;
    std::vector<uint8_t> storage[kNumTensors];
    uint64_t base_ptrs[kNumTensors]{};
    std::vector<uint32_t> slot_inds;

    void allocate(int cap, int k_max = 10) {
        capacity = cap;
        for (int t = 0; t < kNumTensors; ++t) {
            storage[t].assign(static_cast<size_t>(cap) * kRowBytes[t], 0xCD);
            base_ptrs[t] = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(storage[t].data()));
        }
        slot_inds.assign(static_cast<size_t>(k_max), 0);
    }

    void verify_slot_bytes(int layer, int eid, uint32_t slot) const {
        assert(slot < static_cast<uint32_t>(capacity));
        for (int t = 0; t < kNumTensors; ++t) {
            uint64_t rb = kRowBytes[t];
            size_t n_words = rb / sizeof(uint64_t);
            const uint64_t* ptr = reinterpret_cast<const uint64_t*>(
                storage[t].data() + static_cast<uint64_t>(slot) * rb);
            for (size_t w = 0; w < n_words; ++w) {
                uint64_t actual = ptr[w];
                uint64_t exp = expected_word(layer, eid, t, w);
                if (actual != exp) {
                    std::fprintf(stderr,
                                 "[FAIL] Byte mismatch at layer=%d eid=%d slot=%u tensor=%d word=%zu: got=0x%llx expected=0x%llx\n",
                                 layer, eid, slot, t, w,
                                 static_cast<unsigned long long>(actual),
                                 static_cast<unsigned long long>(exp));
                    std::abort();
                }
            }
        }
    }
};

static void test_cancel_and_promote(const SynthExpertFile& sf) {
    // Use 0 background worker threads first so prefetches stay queued (tasks_started == 0)
    // until step_decode runs, deterministically testing queued cancel + queued promote + URGENT drain.
    P1IoScheduler sched;
    sched.init(0);

    LayerSlotBuffers buf;
    buf.allocate(4, 4);
    sched.layer_init(
        1, sf.fd, 4, kNumTensors,
        sf.base_offsets[1], kRowBytes, buf.base_ptrs,
        static_cast<uint64_t>(reinterpret_cast<uintptr_t>(buf.slot_inds.data())));

    // Initial synchronous load of experts {0, 1, 2, 3} into slots {0, 1, 2, 3}
    int32_t init_eids[4] = {0, 1, 2, 3};
    int32_t init_slots[4] = {0, 1, 2, 3};
    sched.load_sync(1, init_eids, init_slots, 4, true);
    sched.set_lru_state(1, init_eids, init_slots, 4, 4, 0, 4);
    sched.unpin_layer_slots(1);
    for (int i = 0; i < 4; ++i) {
        buf.verify_slot_bytes(1, init_eids[i], static_cast<uint32_t>(init_slots[i]));
    }

    // Issue prefetch for {10, 11, 12} (max_fetch = 3).
    // With 0 worker threads, all 3 prefetches are queued in prefetch_queue_ with tasks_started == 0.
    int32_t preds[3] = {10, 11, 12};
    sched.start_prefetch(1, preds, 3, 3);

    // Now step_decode actually routes to {0, 10, 20, 21}:
    // - 11 and 12 are unneeded queued prefetches -> MUST be cancelled (prefetch_cancelled == 2),
    //   and their evicted experts from {0,1,2,3} restored!
    // - 10 is a needed queued prefetch -> MUST be promoted to URGENT (prefetch_promoted == 1, prefetch_used == 1)
    //   without re-issuing a duplicate read!
    // - 20 and 21 are demand misses -> urgent_reads == 2.
    uint32_t route_eids[4] = {0, 10, 20, 21};
    uint32_t out_slots[4]{};
    int n_pending = sched.step_decode(1, route_eids, 4, out_slots);
    assert(n_pending > 0);
    sched.wait_layer_urgent(1);

    for (int i = 0; i < 4; ++i) {
        buf.verify_slot_bytes(1, static_cast<int>(route_eids[i]), out_slots[i]);
    }
    sched.unpin_layer_slots(1);

    P1LayerStats st = sched.get_layer_stats(1);
    assert(st.prefetch_issued == 3);
    assert(st.prefetch_cancelled == 2);
    assert(st.prefetch_promoted == 1);
    assert(st.prefetch_used == 1);
    assert(st.urgent_reads == 2);

    sched.shutdown();
    std::printf("[PASS] test_cancel_and_promote (issued=%llu cancelled=%llu promoted=%llu used=%llu urgent=%llu)\n",
                static_cast<unsigned long long>(st.prefetch_issued),
                static_cast<unsigned long long>(st.prefetch_cancelled),
                static_cast<unsigned long long>(st.prefetch_promoted),
                static_cast<unsigned long long>(st.prefetch_used),
                static_cast<unsigned long long>(st.urgent_reads));
}

static void test_pin_waits_and_reader_callback(const SynthExpertFile& sf) {
    P1IoScheduler sched;
    sched.init(4);

    LayerSlotBuffers buf;
    buf.allocate(6, 4); // capacity = 6, k = 4
    sched.layer_init(
        1, sf.fd, 6, kNumTensors,
        sf.base_offsets[1], kRowBytes, buf.base_ptrs,
        static_cast<uint64_t>(reinterpret_cast<uintptr_t>(buf.slot_inds.data())));

    // Load 6 initial experts
    int32_t init_eids[6] = {0, 1, 2, 3, 4, 5};
    int32_t init_slots[6] = {0, 1, 2, 3, 4, 5};
    sched.load_sync(1, init_eids, init_slots, 6, true);
    sched.set_lru_state(1, init_eids, init_slots, 6, 6, 0, 6);

    std::atomic<bool> gpu_done{false};
    std::future<void> gpu_future;

    sched.set_wait_reader_callback([&](int layer_idx) {
        if (gpu_future.valid()) {
            gpu_future.wait();
        }
        assert(gpu_done.load(std::memory_order_acquire));
        sched.unpin_layer_slots(layer_idx);
    });

    // Decode step 1: routes to {0, 1, 2, 3} -> pins 4 slots, leaving only 2 unpinned slots ({4, 5})
    uint32_t step1_eids[4] = {0, 1, 2, 3};
    uint32_t step1_slots[4]{};
    int pend1 = sched.step_decode(1, step1_eids, 4, step1_slots);
    if (pend1 > 0) sched.wait_layer_urgent(1);

    // Launch simulated async GPU kernel that holds the 4 pinned slots for 15ms and verifies their bytes
    gpu_future = std::async(std::launch::async, [&]() {
        std::this_thread::sleep_for(std::chrono::milliseconds(15));
        for (int i = 0; i < 4; ++i) {
            buf.verify_slot_bytes(1, static_cast<int>(step1_eids[i]), step1_slots[i]);
        }
        gpu_done.store(true, std::memory_order_release);
    });

    // While step 1's GPU reader is still running and holding 4 pinned slots, trigger a prefetch
    // for 3 new experts {10, 11, 12}. Only 2 unpinned slots exist, so the 3rd allocation MUST
    // hit saw_pinned == true, increment pin_waits, and invoke wait_reader_callback!
    int32_t preds[3] = {10, 11, 12};
    sched.start_prefetch(1, preds, 3, 3);
    sched.wait_layer_all(1);

    P1LayerStats st = sched.get_layer_stats(1);
    assert(st.pin_waits >= 1);
    assert(gpu_done.load(std::memory_order_acquire));

    sched.shutdown();
    std::printf("[PASS] test_pin_waits_and_reader_callback (pin_waits=%llu)\n",
                static_cast<unsigned long long>(st.pin_waits));
}

static void test_high_concurrency_stress(const SynthExpertFile& sf) {
    constexpr int kCap = 12;
    constexpr int kTopK = 4;
    constexpr int kTopM = 6;
    constexpr int kPrefetchCap = 3;
    constexpr int kSteps = 250;

    P1IoScheduler sched;
    sched.init(8); // 8 concurrent IO worker threads

    LayerSlotBuffers bufs[kTestLayers + 1];
    for (int l = 1; l <= kTestLayers; ++l) {
        bufs[l].allocate(kCap, kTopK);
        sched.layer_init(
            l, sf.fd, kCap, kNumTensors,
            sf.base_offsets[l], kRowBytes, bufs[l].base_ptrs,
            static_cast<uint64_t>(reinterpret_cast<uintptr_t>(bufs[l].slot_inds.data())));

        int32_t init_eids[kCap];
        int32_t init_slots[kCap];
        for (int i = 0; i < kCap; ++i) {
            init_eids[i] = i;
            init_slots[i] = i;
        }
        sched.load_sync(l, init_eids, init_slots, kCap, true);
        sched.set_lru_state(l, init_eids, init_slots, kCap, kCap, 0, kCap);
    }

    std::mutex gpu_mu[kTestLayers + 1];
    std::future<void> gpu_readers[kTestLayers + 1];

    auto wait_gpu_reader = [&](int l) {
        std::future<void> fut;
        {
            std::lock_guard<std::mutex> g(gpu_mu[l]);
            if (gpu_readers[l].valid()) {
                fut = std::move(gpu_readers[l]);
            }
        }
        if (fut.valid()) {
            fut.wait();
        }
        sched.unpin_layer_slots(l);
    };

    sched.set_wait_reader_callback([&](int l) {
        wait_gpu_reader(l);
    });

    std::mt19937_64 rng(123456789ULL);
    uint64_t total_gpu_verifications = 0;

    for (int step = 0; step < kSteps; ++step) {
        // Test dynamic resize_layer mid-run at step 100 (shrink to 8) and step 160 (grow to 14)
        if (step == 100 || step == 160) {
            int new_cap = (step == 100) ? 8 : 14;
            for (int l = 1; l <= kTestLayers; ++l) {
                wait_gpu_reader(l);
                sched.wait_layer_all(l);

                int32_t lru_eids[512]{};
                int32_t lru_slots[512]{};
                int lru_cnt = 0;
                sched.get_lru_state(l, lru_eids, lru_slots, &lru_cnt);

                int keep_cnt = std::min(lru_cnt, new_cap);
                int start_idx = lru_cnt - keep_cnt; // keep MRU tail
                std::vector<int32_t> kept_eids(keep_cnt);
                std::vector<int32_t> new_slots(keep_cnt);

                LayerSlotBuffers new_buf;
                new_buf.allocate(new_cap, kTopK);
                for (int i = 0; i < keep_cnt; ++i) {
                    kept_eids[i] = lru_eids[start_idx + i];
                    int old_s = lru_slots[start_idx + i];
                    assert(old_s >= 0);
                    new_slots[i] = i;
                    for (int t = 0; t < kNumTensors; ++t) {
                        uint64_t rb = kRowBytes[t];
                        std::memcpy(
                            new_buf.storage[t].data() + static_cast<size_t>(i) * rb,
                            bufs[l].storage[t].data() + static_cast<size_t>(old_s) * rb,
                            rb);
                    }
                }
                bufs[l] = std::move(new_buf);
                sched.resize_layer(
                    l, new_cap, bufs[l].base_ptrs,
                    static_cast<uint64_t>(reinterpret_cast<uintptr_t>(bufs[l].slot_inds.data())),
                    kept_eids.data(), new_slots.data(), keep_cnt);
            }
        }

        for (int l = 1; l <= kTestLayers; ++l) {
            // Pick kTopK distinct random experts for layer l
            uint32_t actual_eids[kTopK];
            uint64_t used_mask = 0;
            for (int i = 0; i < kTopK; ++i) {
                int e = 0;
                do {
                    e = static_cast<int>(rng() % kTestExperts);
                } while (used_mask & (1ULL << e));
                used_mask |= (1ULL << e);
                actual_eids[i] = static_cast<uint32_t>(e);
            }

            // Pick kTopM predictions for layer l+1
            int32_t pred_next[kTopM];
            uint64_t pred_mask = 0;
            for (int i = 0; i < kTopM; ++i) {
                int e = 0;
                do {
                    e = static_cast<int>(rng() % kTestExperts);
                } while (pred_mask & (1ULL << e));
                pred_mask |= (1ULL << e);
                pred_next[i] = e;
            }

            // Wait for layer l's previous GPU reader before mutating layer l's slots
            wait_gpu_reader(l);

            // Forced-miss stress on every 3rd step
            if (step % 3 == 1) {
                sched.force_invalidate(l, 0.30, rng);
            }

            uint32_t slots[kTopK]{};
            int n_pending = sched.step_decode(l, actual_eids, kTopK, slots);
            if (n_pending > 0) {
                sched.wait_layer_urgent(l);
            }

            // Launch simulated async GPU kernel reading layer l's pinned slots concurrently with next layer's IO
            std::vector<uint32_t> eids_copy(actual_eids, actual_eids + kTopK);
            std::vector<uint32_t> slots_copy(slots, slots + kTopK);
            {
                std::lock_guard<std::mutex> g(gpu_mu[l]);
                gpu_readers[l] = std::async(std::launch::async, [&, l, eids_copy, slots_copy]() {
                    for (int i = 0; i < kTopK; ++i) {
                        bufs[l].verify_slot_bytes(l, static_cast<int>(eids_copy[i]), slots_copy[i]);
                    }
                });
            }
            total_gpu_verifications += kTopK;

            // Start prefetch for next layer while next layer's previous GPU reader may still be running!
            if (l + 1 <= kTestLayers) {
                sched.start_prefetch(l + 1, pred_next, kTopM, kPrefetchCap);
            }
        }
    }

    for (int l = 1; l <= kTestLayers; ++l) {
        wait_gpu_reader(l);
    }
    sched.wait_all();

    P1LayerStats tot = sched.get_total_stats(kTestLayers + 1);
    assert(tot.decode_requests == static_cast<uint64_t>(kSteps * kTestLayers * kTopK));
    assert(tot.urgent_reads > 0);
    assert(tot.prefetch_issued > 0);
    assert(tot.prefetch_cancelled + tot.prefetch_wasted + tot.prefetch_used == tot.prefetch_issued);
    assert(tot.forced_invalidations > 0);
    assert(tot.bytes > 0);

    sched.shutdown();
    std::printf(
        "[PASS] test_high_concurrency_stress (steps=%d verifications=%llu urgent=%llu pref_issued=%llu "
        "pref_cancelled=%llu pref_promoted=%llu pref_used=%llu pref_wasted=%llu pin_waits=%llu forced_inv=%llu bytes=%llu io_wait_ms=%.2f)\n",
        kSteps,
        static_cast<unsigned long long>(total_gpu_verifications),
        static_cast<unsigned long long>(tot.urgent_reads),
        static_cast<unsigned long long>(tot.prefetch_issued),
        static_cast<unsigned long long>(tot.prefetch_cancelled),
        static_cast<unsigned long long>(tot.prefetch_promoted),
        static_cast<unsigned long long>(tot.prefetch_used),
        static_cast<unsigned long long>(tot.prefetch_wasted),
        static_cast<unsigned long long>(tot.pin_waits),
        static_cast<unsigned long long>(tot.forced_invalidations),
        static_cast<unsigned long long>(tot.bytes),
        tot.io_wait_ms());
}

} // namespace

int main() {
    SynthExpertFile sf;
    sf.create("tests/tmp/synth_experts_p1.bin");

    test_cancel_and_promote(sf);
    test_pin_waits_and_reader_callback(sf);
    test_high_concurrency_stress(sf);

    std::printf("ALL IO_SCHED TESTS PASSED\n");
    return 0;
}

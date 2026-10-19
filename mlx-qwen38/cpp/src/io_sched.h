#pragma once

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <random>
#include <sys/uio.h>
#include <thread>
#include <vector>

namespace qwen38 {

constexpr int kP1MaxLayers = 64;
constexpr int kP1NumExperts = 512;
constexpr int kP1MaxTensors = 9;
constexpr uint64_t kP1DefaultExpertBytes = 1536000ULL;

struct P1LayerStats {
    uint64_t decode_requests = 0;
    uint64_t lru_hits = 0;
    uint64_t urgent_reads = 0;
    uint64_t prefetch_issued = 0;
    uint64_t prefetch_cancelled = 0;
    uint64_t prefetch_promoted = 0;
    uint64_t prefetch_used = 0;
    uint64_t prefetch_wasted = 0;
    uint64_t pin_waits = 0;
    uint64_t bytes = 0;
    uint64_t io_wait_ns = 0;
    uint64_t forced_invalidations = 0;

    uint64_t prefill_requests = 0;
    uint64_t prefill_lru_hits = 0;
    uint64_t prefill_demand_misses = 0;
    uint64_t prefill_demand_bytes = 0;

    double io_wait_ms() const {
        return static_cast<double>(io_wait_ns) / 1e6;
    }
};

class P1IoScheduler {
public:
    P1IoScheduler();
    ~P1IoScheduler();

    P1IoScheduler(const P1IoScheduler&) = delete;
    P1IoScheduler& operator=(const P1IoScheduler&) = delete;

    void init(int num_threads = 6);
    void shutdown();

    void set_wait_reader_callback(std::function<void(int)> cb);

    void layer_init(
        int layer_idx,
        int fd,
        int capacity,
        int num_tensors,
        const uint64_t* base_offsets,
        const uint64_t* row_bytes,
        const uint64_t* base_ptrs,
        uint64_t slot_inds_ptr);

    void resize_layer(
        int layer_idx,
        int new_capacity,
        const uint64_t* new_base_ptrs,
        uint64_t slot_inds_ptr,
        const int32_t* ordered_eids,
        const int32_t* ordered_slots,
        int count);

    void reset_cache(int layer_idx);
    void reset_stats(int layer_idx);

    // Enqueue next-layer router predictions into PREFETCH queue (cancellable & promotable)
    void start_prefetch(
        int layer_idx,
        const int32_t* pred_eids,
        int pred_count,
        int max_fetch);

    // Perform routing lookup for layer_idx:
    // - Cancels unneeded queued prefetches for layer_idx
    // - Promotes needed queued prefetches to URGENT (or waits on needed in-flight prefetches)
    // - Enqueues demand misses into URGENT queue (coalesced preadv)
    // - Pins the k active slots in out_slots / slot_inds_ptr
    // Returns number of pending IO operations on layer_idx that must complete before GPU read.
    int step_decode(
        int layer_idx,
        const uint32_t* eids,
        int k,
        uint32_t* out_slots = nullptr);

    // Wait until all pending IO writes for layer_idx's active slots are complete.
    // Must be called before dispatching GPU reads on layer_idx's slots.
    void wait_layer_urgent(int layer_idx);

    // Wait for all pending IO (both urgent and prefetch) on layer_idx or globally.
    void wait_layer_all(int layer_idx);
    void wait_all();

    // Unpin all slots pinned by step_decode on layer_idx (called when GPU reader completes).
    void unpin_layer_slots(int layer_idx);

    // Explicit pin / unpin for specific slots (used by unit tests and custom pipelines).
    void pin_slots(int layer_idx, const uint32_t* slots, int count);
    void unpin_slots(int layer_idx, const uint32_t* slots, int count);
    int get_slot_pin_count(int layer_idx, int slot);

    // Forced-miss stress mode: invalidate random fraction `rate` (0..1) of cached experts in layer_idx.
    int force_invalidate(int layer_idx, double rate, std::mt19937_64& rng);

    // Synchronous load for prefill
    void load_sync(
        int layer_idx,
        const int32_t* eids,
        const int32_t* slots,
        int count,
        bool is_prefill);

    void get_lru_state(
        int layer_idx,
        int32_t* out_eids,
        int32_t* out_slots,
        int* out_count);

    void set_lru_state(
        int layer_idx,
        const int32_t* ordered_eids,
        const int32_t* ordered_slots,
        int count,
        uint64_t add_prefill_requests,
        uint64_t add_prefill_hits,
        uint64_t add_prefill_misses);

    P1LayerStats get_layer_stats(int layer_idx) const;
    P1LayerStats get_total_stats(int num_layers = 48) const;
    int num_threads() const { return num_workers_; }

private:
    struct ExpertReadOp {
        int layer_idx = 0;
        int32_t eid = -1;
        int16_t slot = -1;
        int16_t prev_eid = -1;
        bool is_prefetch = false;
        bool promoted = false;
        bool cancelled = false;
        bool resolved = false;
        int tasks_started = 0;
        int tasks_completed = 0;
        int total_tasks = 0;
        bool done = false;
    };

    struct IoTask {
        int layer_idx = 0;
        int fd = -1;
        off_t offset = 0;
        int iov_cnt = 0;
        bool is_prefill = false;
        struct iovec iov[16]{};
        std::shared_ptr<ExpertReadOp> ops[16];
    };

    struct LayerState {
        bool initialized = false;
        int fd = -1;
        int capacity = 0;
        int num_tensors = kP1MaxTensors;
        uint64_t base_offsets[kP1MaxTensors]{};
        uint64_t row_bytes[kP1MaxTensors]{};
        uint8_t* base_ptrs[kP1MaxTensors]{};
        uint32_t* slot_inds_ptr = nullptr;
        uint64_t expert_bytes = kP1DefaultExpertBytes;

        int16_t eid_to_slot[kP1NumExperts]{};
        int16_t slot_to_eid[kP1NumExperts]{};
        int16_t prev_slot[kP1NumExperts]{};
        int16_t next_slot[kP1NumExperts]{};
        int16_t lru_head = -1;
        int16_t lru_tail = -1;
        int16_t free_slots[kP1NumExperts]{};
        int free_count = 0;

        int pin_count[kP1NumExperts]{};
        std::vector<int16_t> gpu_pinned_slots;
        std::shared_ptr<ExpertReadOp> active_io[kP1NumExperts];

        bool has_pre_prefetch_snapshot = false;
        uint64_t pre_prefetch_bitmap[8]{};
        std::vector<std::shared_ptr<ExpertReadOp>> last_prefetch_ops;
        std::vector<std::shared_ptr<ExpertReadOp>> pending_urgent_ops;

        P1LayerStats stats;
        std::atomic<uint64_t> atomic_bytes{0};
    };

    void worker_loop();
    bool pop_next_task_locked(IoTask& out_task);
    void execute_task(IoTask& task);

    void lru_remove_slot_locked(LayerState& L, int16_t slot);
    void lru_append_mru_locked(LayerState& L, int16_t slot);
    void lru_prepend_lru_locked(LayerState& L, int32_t eid, int16_t slot);
    void lru_move_to_mru_locked(LayerState& L, int16_t slot);
    void lru_insert_eid_slot_locked(LayerState& L, int32_t eid, int16_t slot);

    int16_t alloc_slot_locked(
        int layer_idx,
        std::unique_lock<std::mutex>& lock,
        const uint64_t protect_bm[8],
        int16_t* out_prev_eid);

    void enqueue_coalesced_urgent_locked(
        int layer_idx,
        LayerState& L,
        const std::vector<std::shared_ptr<ExpertReadOp>>& ops);

    void enqueue_single_prefetch_locked(
        int layer_idx,
        LayerState& L,
        const std::shared_ptr<ExpertReadOp>& op);

    mutable std::mutex mutex_;
    std::condition_variable work_cv_;
    std::condition_variable done_cv_;

    std::deque<IoTask> urgent_queue_;
    std::deque<IoTask> prefetch_queue_;

    LayerState layers_[kP1MaxLayers];
    std::vector<std::thread> workers_;
    int num_workers_ = 6;
    bool initialized_ = false;
    bool shutdown_ = false;

    std::function<void(int)> wait_reader_fn_;
};

// Helper for --io exact forced-miss mode (uses public c_layer_get/set_lru_state without touching third_party/expert_io.c)
int force_invalidate_exact_layer(
    int layer_idx,
    double rate,
    std::mt19937_64& rng,
    uint8_t* first_tensor_base_ptr,
    uint64_t first_tensor_row_bytes);

} // namespace qwen38

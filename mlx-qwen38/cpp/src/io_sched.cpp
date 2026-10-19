#include "io_sched.h"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <unistd.h>

namespace qwen38 {

P1IoScheduler::P1IoScheduler() = default;

P1IoScheduler::~P1IoScheduler() {
    shutdown();
}

void P1IoScheduler::init(int num_threads) {
    std::unique_lock<std::mutex> lock(mutex_);
    if (initialized_) return;
    num_workers_ = std::clamp(num_threads, 1, 32);
    shutdown_ = false;
    urgent_queue_.clear();
    prefetch_queue_.clear();
    workers_.reserve(static_cast<size_t>(num_workers_));
    for (int i = 0; i < num_workers_; ++i) {
        workers_.emplace_back(&P1IoScheduler::worker_loop, this);
    }
    initialized_ = true;
}

void P1IoScheduler::shutdown() {
    std::vector<std::thread> to_join;
    {
        std::unique_lock<std::mutex> lock(mutex_);
        if (!initialized_) return;
        shutdown_ = true;
        work_cv_.notify_all();
        done_cv_.notify_all();
        to_join.swap(workers_);
        initialized_ = false;
    }
    for (auto& t : to_join) {
        if (t.joinable()) {
            t.join();
        }
    }
}

void P1IoScheduler::set_wait_reader_callback(std::function<void(int)> cb) {
    std::unique_lock<std::mutex> lock(mutex_);
    wait_reader_fn_ = std::move(cb);
}

void P1IoScheduler::lru_remove_slot_locked(LayerState& L, int16_t slot) {
    int16_t p = L.prev_slot[slot];
    int16_t n = L.next_slot[slot];
    if (p >= 0) {
        L.next_slot[p] = n;
    } else {
        L.lru_head = n;
    }
    if (n >= 0) {
        L.prev_slot[n] = p;
    } else {
        L.lru_tail = p;
    }
    L.prev_slot[slot] = -1;
    L.next_slot[slot] = -1;
}

void P1IoScheduler::lru_append_mru_locked(LayerState& L, int16_t slot) {
    L.prev_slot[slot] = L.lru_tail;
    L.next_slot[slot] = -1;
    if (L.lru_tail >= 0) {
        L.next_slot[L.lru_tail] = slot;
    } else {
        L.lru_head = slot;
    }
    L.lru_tail = slot;
}

void P1IoScheduler::lru_prepend_lru_locked(LayerState& L, int32_t eid, int16_t slot) {
    L.eid_to_slot[eid] = slot;
    L.slot_to_eid[slot] = static_cast<int16_t>(eid);
    L.prev_slot[slot] = -1;
    L.next_slot[slot] = L.lru_head;
    if (L.lru_head >= 0) {
        L.prev_slot[L.lru_head] = slot;
    } else {
        L.lru_tail = slot;
    }
    L.lru_head = slot;
}

void P1IoScheduler::lru_move_to_mru_locked(LayerState& L, int16_t slot) {
    if (L.lru_tail == slot) return;
    lru_remove_slot_locked(L, slot);
    lru_append_mru_locked(L, slot);
}

void P1IoScheduler::lru_insert_eid_slot_locked(LayerState& L, int32_t eid, int16_t slot) {
    if (L.eid_to_slot[eid] >= 0) {
        int16_t existing = L.eid_to_slot[eid];
        lru_move_to_mru_locked(L, existing);
        return;
    }
    L.eid_to_slot[eid] = slot;
    L.slot_to_eid[slot] = static_cast<int16_t>(eid);
    lru_append_mru_locked(L, slot);
}

void P1IoScheduler::layer_init(
    int layer_idx,
    int fd,
    int capacity,
    int num_tensors,
    const uint64_t* base_offsets,
    const uint64_t* row_bytes,
    const uint64_t* base_ptrs,
    uint64_t slot_inds_ptr) {
    if (layer_idx < 0 || layer_idx >= kP1MaxLayers) return;
    wait_layer_all(layer_idx);

    std::unique_lock<std::mutex> lock(mutex_);
    LayerState& L = layers_[layer_idx];
    L.initialized = true;
    L.fd = fd;
    L.capacity = std::clamp(capacity, 1, kP1NumExperts);
    L.num_tensors = std::clamp(num_tensors, 1, kP1MaxTensors);
    uint64_t sum_rb = 0;
    for (int t = 0; t < L.num_tensors; ++t) {
        L.base_offsets[t] = base_offsets[t];
        L.row_bytes[t] = row_bytes[t];
        L.base_ptrs[t] = reinterpret_cast<uint8_t*>(static_cast<uintptr_t>(base_ptrs[t]));
        sum_rb += row_bytes[t];
    }
    L.expert_bytes = (sum_rb > 0) ? sum_rb : kP1DefaultExpertBytes;
    L.slot_inds_ptr = reinterpret_cast<uint32_t*>(static_cast<uintptr_t>(slot_inds_ptr));

    for (int i = 0; i < kP1NumExperts; ++i) {
        L.eid_to_slot[i] = -1;
        L.slot_to_eid[i] = -1;
        L.prev_slot[i] = -1;
        L.next_slot[i] = -1;
        L.pin_count[i] = 0;
        L.active_io[i].reset();
    }
    L.gpu_pinned_slots.clear();
    L.lru_head = -1;
    L.lru_tail = -1;
    L.free_count = L.capacity;
    for (int i = 0; i < L.capacity; ++i) {
        L.free_slots[i] = static_cast<int16_t>(L.capacity - 1 - i);
    }
    L.has_pre_prefetch_snapshot = false;
    std::memset(L.pre_prefetch_bitmap, 0, sizeof(L.pre_prefetch_bitmap));
    L.last_prefetch_ops.clear();
    L.pending_urgent_ops.clear();
    L.stats = P1LayerStats{};
    L.atomic_bytes.store(0, std::memory_order_relaxed);
}

void P1IoScheduler::resize_layer(
    int layer_idx,
    int new_capacity,
    const uint64_t* new_base_ptrs,
    uint64_t slot_inds_ptr,
    const int32_t* ordered_eids,
    const int32_t* ordered_slots,
    int count) {
    if (layer_idx < 0 || layer_idx >= kP1MaxLayers) return;
    wait_layer_all(layer_idx);

    std::unique_lock<std::mutex> lock(mutex_);
    LayerState& L = layers_[layer_idx];
    L.capacity = std::clamp(new_capacity, 1, kP1NumExperts);
    for (int t = 0; t < L.num_tensors; ++t) {
        L.base_ptrs[t] = reinterpret_cast<uint8_t*>(static_cast<uintptr_t>(new_base_ptrs[t]));
    }
    L.slot_inds_ptr = reinterpret_cast<uint32_t*>(static_cast<uintptr_t>(slot_inds_ptr));

    for (int i = 0; i < kP1NumExperts; ++i) {
        L.eid_to_slot[i] = -1;
        L.slot_to_eid[i] = -1;
        L.prev_slot[i] = -1;
        L.next_slot[i] = -1;
        L.pin_count[i] = 0;
        L.active_io[i].reset();
    }
    L.gpu_pinned_slots.clear();
    L.lru_head = -1;
    L.lru_tail = -1;
    L.has_pre_prefetch_snapshot = false;
    std::memset(L.pre_prefetch_bitmap, 0, sizeof(L.pre_prefetch_bitmap));
    L.last_prefetch_ops.clear();
    L.pending_urgent_ops.clear();

    uint8_t used_slot[kP1NumExperts]{};
    for (int i = 0; i < count; ++i) {
        int32_t eid = ordered_eids[i];
        int16_t slot = static_cast<int16_t>(ordered_slots[i]);
        if (slot >= 0 && slot < L.capacity && eid >= 0 && eid < kP1NumExperts) {
            used_slot[slot] = 1;
            lru_insert_eid_slot_locked(L, eid, slot);
        }
    }

    L.free_count = 0;
    for (int s = L.capacity - 1; s >= 0; --s) {
        if (!used_slot[s]) {
            L.free_slots[L.free_count++] = static_cast<int16_t>(s);
        }
    }
}

void P1IoScheduler::reset_cache(int layer_idx) {
    if (layer_idx < 0 || layer_idx >= kP1MaxLayers) return;
    wait_layer_all(layer_idx);

    std::unique_lock<std::mutex> lock(mutex_);
    LayerState& L = layers_[layer_idx];
    for (int i = 0; i < kP1NumExperts; ++i) {
        L.eid_to_slot[i] = -1;
        L.slot_to_eid[i] = -1;
        L.prev_slot[i] = -1;
        L.next_slot[i] = -1;
        L.pin_count[i] = 0;
        L.active_io[i].reset();
    }
    L.gpu_pinned_slots.clear();
    L.lru_head = -1;
    L.lru_tail = -1;
    L.free_count = L.capacity;
    for (int i = 0; i < L.capacity; ++i) {
        L.free_slots[i] = static_cast<int16_t>(L.capacity - 1 - i);
    }
    L.has_pre_prefetch_snapshot = false;
    std::memset(L.pre_prefetch_bitmap, 0, sizeof(L.pre_prefetch_bitmap));
    L.last_prefetch_ops.clear();
    L.pending_urgent_ops.clear();
}

void P1IoScheduler::reset_stats(int layer_idx) {
    if (layer_idx < 0 || layer_idx >= kP1MaxLayers) return;
    std::unique_lock<std::mutex> lock(mutex_);
    LayerState& L = layers_[layer_idx];
    L.stats = P1LayerStats{};
    L.atomic_bytes.store(0, std::memory_order_relaxed);
}

bool P1IoScheduler::pop_next_task_locked(IoTask& out_task) {
    // Priority 1: URGENT queue is always served before PREFETCH queue
    while (!urgent_queue_.empty()) {
        out_task = std::move(urgent_queue_.front());
        urgent_queue_.pop_front();
        bool any_valid = false;
        for (int k = 0; k < out_task.iov_cnt; ++k) {
            if (out_task.ops[k] && !out_task.ops[k]->cancelled) {
                out_task.ops[k]->tasks_started++;
                any_valid = true;
            }
        }
        if (any_valid) {
            return true;
        }
    }

    // Priority 2: PREFETCH queue
    while (!prefetch_queue_.empty()) {
        out_task = std::move(prefetch_queue_.front());
        prefetch_queue_.pop_front();
        auto& op = out_task.ops[0];
        if (!op || op->cancelled || op->promoted) {
            continue;
        }
        op->tasks_started++;
        return true;
    }

    return false;
}

void P1IoScheduler::execute_task(IoTask& task) {
    uint64_t total_read = 0;
    if (task.iov_cnt == 1) {
        uint8_t* buf = static_cast<uint8_t*>(task.iov[0].iov_base);
        size_t len = task.iov[0].iov_len;
        off_t off = task.offset;
        while (len > 0) {
            ssize_t n = ::pread(task.fd, buf, len, off);
            if (n <= 0) break;
            buf += n;
            len -= static_cast<size_t>(n);
            off += n;
            total_read += static_cast<uint64_t>(n);
        }
    } else if (task.iov_cnt > 1) {
        size_t expected = 0;
        for (int k = 0; k < task.iov_cnt; ++k) {
            expected += task.iov[k].iov_len;
        }
        ssize_t n = ::preadv(task.fd, task.iov, task.iov_cnt, task.offset);
        if (n > 0) {
            total_read += static_cast<uint64_t>(n);
        }
        if (n < 0 || static_cast<size_t>(n) < expected) {
            // Fallback for partial/short preadv
            size_t done_bytes = (n > 0) ? static_cast<size_t>(n) : 0;
            off_t cur_off = task.offset;
            for (int k = 0; k < task.iov_cnt; ++k) {
                size_t iov_len = task.iov[k].iov_len;
                uint8_t* iov_buf = static_cast<uint8_t*>(task.iov[k].iov_base);
                if (done_bytes >= iov_len) {
                    done_bytes -= iov_len;
                } else {
                    size_t rem = iov_len - done_bytes;
                    uint8_t* p = iov_buf + done_bytes;
                    off_t off = cur_off + static_cast<off_t>(done_bytes);
                    done_bytes = 0;
                    while (rem > 0) {
                        ssize_t r = ::pread(task.fd, p, rem, off);
                        if (r <= 0) break;
                        p += r;
                        rem -= static_cast<size_t>(r);
                        off += r;
                        total_read += static_cast<uint64_t>(r);
                    }
                }
                cur_off += static_cast<off_t>(iov_len);
            }
        }
    }

    if (!task.is_prefill && task.layer_idx >= 0 && task.layer_idx < kP1MaxLayers && total_read > 0) {
        layers_[task.layer_idx].atomic_bytes.fetch_add(total_read, std::memory_order_relaxed);
    }

    std::unique_lock<std::mutex> lock(mutex_);
    bool notify = false;
    for (int k = 0; k < task.iov_cnt; ++k) {
        if (auto& op = task.ops[k]) {
            op->tasks_completed++;
            if (op->tasks_completed >= op->total_tasks) {
                op->done = true;
                notify = true;
            }
        }
    }
    if (notify) {
        done_cv_.notify_all();
    }
}

void P1IoScheduler::worker_loop() {
    while (true) {
        IoTask task;
        {
            std::unique_lock<std::mutex> lock(mutex_);
            while (!shutdown_ && !pop_next_task_locked(task)) {
                work_cv_.wait(lock);
            }
            if (shutdown_ && task.iov_cnt == 0) {
                return;
            }
        }
        execute_task(task);
    }
}

int16_t P1IoScheduler::alloc_slot_locked(
    int layer_idx,
    std::unique_lock<std::mutex>& lock,
    const uint64_t protect_bm[8],
    int16_t* out_prev_eid) {
    LayerState& L = layers_[layer_idx];
    if (L.free_count > 0) {
        if (out_prev_eid) *out_prev_eid = -1;
        return L.free_slots[--L.free_count];
    }

    bool counted_pin_wait = false;
    while (true) {
        bool saw_pinned = false;
        bool saw_active_io = false;
        int16_t cur = L.lru_head;
        while (cur >= 0) {
            int16_t eid = L.slot_to_eid[cur];
            bool prot = (eid >= 0) && ((protect_bm[eid >> 6] & (1ULL << (eid & 63))) != 0);
            bool pinned = (L.pin_count[cur] > 0);
            bool busy_io = (L.active_io[cur] != nullptr && !L.active_io[cur]->done);
            if (!prot && !pinned && !busy_io) {
                lru_remove_slot_locked(L, cur);
                if (eid >= 0) {
                    L.eid_to_slot[eid] = -1;
                }
                L.slot_to_eid[cur] = -1;
                L.active_io[cur].reset();
                if (out_prev_eid) *out_prev_eid = eid;
                return cur;
            }
            if (!prot && pinned) saw_pinned = true;
            if (!prot && busy_io) saw_active_io = true;
            cur = L.next_slot[cur];
        }

        // Second pass: only if all slots were in protect_bm, check if any slot is pinned or busy
        if (!saw_pinned && !saw_active_io) {
            cur = L.lru_head;
            while (cur >= 0) {
                bool pinned = (L.pin_count[cur] > 0);
                bool busy_io = (L.active_io[cur] != nullptr && !L.active_io[cur]->done);
                if (pinned) saw_pinned = true;
                if (busy_io) saw_active_io = true;
                cur = L.next_slot[cur];
            }
        }

        if (saw_pinned) {
            if (!counted_pin_wait) {
                L.stats.pin_waits++;
                counted_pin_wait = true;
            }
            auto cb = wait_reader_fn_;
            if (cb) {
                lock.unlock();
                cb(layer_idx);
                lock.lock();
                continue;
            }
            done_cv_.wait(lock);
            continue;
        }

        if (saw_active_io) {
            IoTask task;
            if (pop_next_task_locked(task)) {
                lock.unlock();
                execute_task(task);
                lock.lock();
            } else {
                done_cv_.wait(lock);
            }
            continue;
        }

        // Fallback: all slots were in protect_bm (only possible if capacity <= popcount(protect_bm)),
        // pick oldest unpinned & non-busy slot.
        cur = L.lru_head;
        while (cur >= 0) {
            if (L.pin_count[cur] == 0 && (!L.active_io[cur] || L.active_io[cur]->done)) {
                int16_t eid = L.slot_to_eid[cur];
                lru_remove_slot_locked(L, cur);
                if (eid >= 0) L.eid_to_slot[eid] = -1;
                L.slot_to_eid[cur] = -1;
                L.active_io[cur].reset();
                if (out_prev_eid) *out_prev_eid = eid;
                return cur;
            }
            cur = L.next_slot[cur];
        }
    }
}

void P1IoScheduler::enqueue_single_prefetch_locked(
    int layer_idx,
    LayerState& L,
    const std::shared_ptr<ExpertReadOp>& op) {
    // Pass 0: large weight tensors (row_bytes > 100000); Pass 1: scale/bias tensors (row_bytes <= 100000)
    for (int pass = 0; pass < 2; ++pass) {
        for (int t = 0; t < L.num_tensors; ++t) {
            uint64_t rb = L.row_bytes[t];
            bool is_large = (rb > 100000ULL);
            if ((pass == 0 && !is_large) || (pass == 1 && is_large)) continue;

            IoTask task;
            task.layer_idx = layer_idx;
            task.fd = L.fd;
            task.offset = static_cast<off_t>(L.base_offsets[t] + static_cast<uint64_t>(op->eid) * rb);
            task.iov_cnt = 1;
            task.iov[0].iov_base = L.base_ptrs[t] + static_cast<uint64_t>(op->slot) * rb;
            task.iov[0].iov_len = static_cast<size_t>(rb);
            task.ops[0] = op;
            prefetch_queue_.push_back(std::move(task));
        }
    }
}

void P1IoScheduler::enqueue_coalesced_urgent_locked(
    int layer_idx,
    LayerState& L,
    const std::vector<std::shared_ptr<ExpertReadOp>>& ops) {
    if (ops.empty()) return;

    std::vector<std::shared_ptr<ExpertReadOp>> sorted = ops;
    std::sort(sorted.begin(), sorted.end(), [](const auto& a, const auto& b) {
        return a->eid < b->eid;
    });

    struct Run {
        size_t start;
        size_t len;
    };
    std::vector<Run> runs;
    size_t cur_start = 0;
    for (size_t i = 1; i < sorted.size(); ++i) {
        if (sorted[i]->eid != sorted[i - 1]->eid + 1 || (i - cur_start) >= 16) {
            runs.push_back({cur_start, i - cur_start});
            cur_start = i;
        }
    }
    runs.push_back({cur_start, sorted.size() - cur_start});

    for (int pass = 0; pass < 2; ++pass) {
        for (int t = 0; t < L.num_tensors; ++t) {
            uint64_t rb = L.row_bytes[t];
            bool is_large = (rb > 100000ULL);
            if ((pass == 0 && !is_large) || (pass == 1 && is_large)) continue;

            uint64_t base_off = L.base_offsets[t];
            uint8_t* base_ptr = L.base_ptrs[t];

            for (const auto& r : runs) {
                IoTask task;
                task.layer_idx = layer_idx;
                task.fd = L.fd;
                int32_t first_eid = sorted[r.start]->eid;
                task.offset = static_cast<off_t>(base_off + static_cast<uint64_t>(first_eid) * rb);
                task.iov_cnt = static_cast<int>(r.len);
                for (size_t k = 0; k < r.len; ++k) {
                    const auto& op = sorted[r.start + k];
                    task.iov[k].iov_base = base_ptr + static_cast<uint64_t>(op->slot) * rb;
                    task.iov[k].iov_len = static_cast<size_t>(rb);
                    task.ops[k] = op;
                }
                urgent_queue_.push_back(std::move(task));
            }
        }
    }
}

void P1IoScheduler::start_prefetch(
    int layer_idx,
    const int32_t* pred_eids,
    int pred_count,
    int max_fetch) {
    if (layer_idx < 0 || layer_idx >= kP1MaxLayers || pred_eids == nullptr || pred_count <= 0 || max_fetch <= 0) {
        return;
    }

    std::unique_lock<std::mutex> lock(mutex_);
    LayerState& L = layers_[layer_idx];
    if (!L.initialized) return;

    // Resolve any unconsumed prior prefetch ops on this layer
    if (!L.last_prefetch_ops.empty()) {
        for (auto& op : L.last_prefetch_ops) {
            if (!op || op->resolved) continue;
            op->resolved = true;
            if (op->tasks_started == 0) {
                op->cancelled = true;
                L.stats.prefetch_cancelled++;
                if (L.slot_to_eid[op->slot] == op->eid) {
                    lru_remove_slot_locked(L, op->slot);
                    L.eid_to_slot[op->eid] = -1;
                    L.slot_to_eid[op->slot] = -1;
                    L.active_io[op->slot].reset();
                    if (op->prev_eid >= 0 && op->prev_eid < kP1NumExperts && L.eid_to_slot[op->prev_eid] < 0) {
                        lru_prepend_lru_locked(L, op->prev_eid, op->slot);
                    } else {
                        L.free_slots[L.free_count++] = op->slot;
                    }
                }
            } else {
                L.stats.prefetch_wasted++;
            }
        }
        L.last_prefetch_ops.clear();
    }

    std::memset(L.pre_prefetch_bitmap, 0, sizeof(L.pre_prefetch_bitmap));
    for (int e = 0; e < kP1NumExperts; ++e) {
        if (L.eid_to_slot[e] >= 0) {
            L.pre_prefetch_bitmap[e >> 6] |= (1ULL << (e & 63));
        }
    }
    L.has_pre_prefetch_snapshot = true;

    uint64_t protect_bm[8]{};
    int32_t missing_pred[32];
    int n_missing = 0;
    int cap_fetch = std::min(max_fetch, 32);

    for (int i = 0; i < pred_count; ++i) {
        int32_t eid = pred_eids[i];
        if (eid < 0 || eid >= kP1NumExperts) continue;
        bool already_prot = (protect_bm[eid >> 6] & (1ULL << (eid & 63))) != 0;
        protect_bm[eid >> 6] |= (1ULL << (eid & 63));
        if (!already_prot && L.eid_to_slot[eid] < 0 && n_missing < cap_fetch) {
            missing_pred[n_missing++] = eid;
        }
    }

    if (n_missing == 0) {
        return;
    }

    // Allocate slots in reverse rank order so highest-ranked prediction is MRU
    for (int i = 0; i < n_missing; ++i) {
        int32_t eid = missing_pred[n_missing - 1 - i];
        if (L.eid_to_slot[eid] >= 0) continue;
        int16_t prev_eid = -1;
        int16_t slot = alloc_slot_locked(layer_idx, lock, protect_bm, &prev_eid);
        lru_insert_eid_slot_locked(L, eid, slot);

        auto op = std::make_shared<ExpertReadOp>();
        op->layer_idx = layer_idx;
        op->eid = eid;
        op->slot = slot;
        op->prev_eid = prev_eid;
        op->is_prefetch = true;
        op->total_tasks = L.num_tensors;

        L.active_io[slot] = op;
        L.last_prefetch_ops.push_back(op);
        L.stats.prefetch_issued++;
        enqueue_single_prefetch_locked(layer_idx, L, op);
    }

    work_cv_.notify_all();
}

int P1IoScheduler::step_decode(
    int layer_idx,
    const uint32_t* eids,
    int k,
    uint32_t* out_slots) {
    if (layer_idx < 0 || layer_idx >= kP1MaxLayers || eids == nullptr || k <= 0) {
        return 0;
    }

    std::unique_lock<std::mutex> lock(mutex_);
    LayerState& L = layers_[layer_idx];
    if (!L.initialized) return 0;

    L.stats.decode_requests += static_cast<uint64_t>(k);

    uint64_t actual_bm[8]{};
    for (int i = 0; i < k; ++i) {
        uint32_t eid = eids[i];
        if (eid < static_cast<uint32_t>(kP1NumExperts)) {
            actual_bm[eid >> 6] |= (1ULL << (eid & 63));
        }
    }

    std::vector<std::shared_ptr<ExpertReadOp>> urgent_ops_to_enqueue;

    // Step 1: Resolve prefetches for layer_idx (cancel unneeded unstarted, promote needed unstarted, wait on needed in-flight)
    if (!L.last_prefetch_ops.empty()) {
        // 1a: Cancel unneeded unstarted prefetches first (frees their slots / restores prev_eid)
        for (auto& op : L.last_prefetch_ops) {
            if (!op || op->resolved) continue;
            bool needed = (actual_bm[op->eid >> 6] & (1ULL << (op->eid & 63))) != 0;
            if (!needed) {
                op->resolved = true;
                if (op->tasks_started == 0) {
                    op->cancelled = true;
                    L.stats.prefetch_cancelled++;
                    if (L.slot_to_eid[op->slot] == op->eid) {
                        lru_remove_slot_locked(L, op->slot);
                        L.eid_to_slot[op->eid] = -1;
                        L.slot_to_eid[op->slot] = -1;
                        L.active_io[op->slot].reset();
                        if (op->prev_eid >= 0 && op->prev_eid < kP1NumExperts && L.eid_to_slot[op->prev_eid] < 0) {
                            lru_prepend_lru_locked(L, op->prev_eid, op->slot);
                            L.pre_prefetch_bitmap[op->prev_eid >> 6] |= (1ULL << (op->prev_eid & 63));
                        } else {
                            L.free_slots[L.free_count++] = op->slot;
                        }
                    }
                } else {
                    L.stats.prefetch_wasted++;
                }
            }
        }

        // 1b: Promote needed prefetches
        for (auto& op : L.last_prefetch_ops) {
            if (!op || op->resolved) continue;
            bool needed = (actual_bm[op->eid >> 6] & (1ULL << (op->eid & 63))) != 0;
            if (needed) {
                op->resolved = true;
                L.stats.prefetch_used++;
                if (op->tasks_started == 0) {
                    op->promoted = true;
                    op->is_prefetch = false;
                    L.stats.prefetch_promoted++;
                    urgent_ops_to_enqueue.push_back(op);
                } else if (!op->done) {
                    // Already started (in-flight): move any remaining unstarted tensor tasks of op from prefetch_queue_ to urgent_queue_
                    for (auto it = prefetch_queue_.begin(); it != prefetch_queue_.end();) {
                        if (it->ops[0] == op) {
                            urgent_queue_.push_back(std::move(*it));
                            it = prefetch_queue_.erase(it);
                        } else {
                            ++it;
                        }
                    }
                }
            }
        }
        L.last_prefetch_ops.clear();
    }

    // Purge cancelled/promoted entries from prefetch_queue_
    for (auto it = prefetch_queue_.begin(); it != prefetch_queue_.end();) {
        const auto& op = it->ops[0];
        if (!op || op->cancelled || op->promoted) {
            it = prefetch_queue_.erase(it);
        } else {
            ++it;
        }
    }

    // Step 2: Check each requested expert in eids[0..k-1]
    uint64_t protect_bm[8];
    std::memcpy(protect_bm, actual_bm, sizeof(protect_bm));

    for (int i = 0; i < k; ++i) {
        int32_t eid = static_cast<int32_t>(eids[i]);
        if (eid < 0 || eid >= kP1NumExperts) continue;
        if (L.eid_to_slot[eid] >= 0) {
            if (!L.has_pre_prefetch_snapshot ||
                (L.pre_prefetch_bitmap[eid >> 6] & (1ULL << (eid & 63))) != 0) {
                L.stats.lru_hits++;
            }
        } else {
            L.stats.urgent_reads++;
            int16_t prev_eid = -1;
            int16_t slot = alloc_slot_locked(layer_idx, lock, protect_bm, &prev_eid);
            lru_insert_eid_slot_locked(L, eid, slot);

            auto op = std::make_shared<ExpertReadOp>();
            op->layer_idx = layer_idx;
            op->eid = eid;
            op->slot = slot;
            op->prev_eid = prev_eid;
            op->is_prefetch = false;
            op->total_tasks = L.num_tensors;

            L.active_io[slot] = op;
            urgent_ops_to_enqueue.push_back(op);
        }
    }
    L.has_pre_prefetch_snapshot = false;

    if (!urgent_ops_to_enqueue.empty()) {
        enqueue_coalesced_urgent_locked(layer_idx, L, urgent_ops_to_enqueue);
    }

    // Step 3: Populate slot indices, move to MRU, PIN active slots, and collect pending urgent ops
    L.pending_urgent_ops.clear();
    uint32_t* dst_slots = L.slot_inds_ptr;
    for (int i = 0; i < k; ++i) {
        int32_t eid = static_cast<int32_t>(eids[i]);
        int16_t slot = L.eid_to_slot[eid];
        lru_move_to_mru_locked(L, slot);
        if (dst_slots != nullptr) {
            dst_slots[i] = static_cast<uint32_t>(slot);
        }
        if (out_slots != nullptr) {
            out_slots[i] = static_cast<uint32_t>(slot);
        }
        L.pin_count[slot]++;
        L.gpu_pinned_slots.push_back(slot);
        if (L.active_io[slot] && !L.active_io[slot]->done) {
            L.pending_urgent_ops.push_back(L.active_io[slot]);
        }
    }

    if (!L.pending_urgent_ops.empty() || !urgent_ops_to_enqueue.empty()) {
        work_cv_.notify_all();
    }

    return static_cast<int>(L.pending_urgent_ops.size());
}

void P1IoScheduler::wait_layer_urgent(int layer_idx) {
    if (layer_idx < 0 || layer_idx >= kP1MaxLayers) return;
    std::unique_lock<std::mutex> lock(mutex_);
    LayerState& L = layers_[layer_idx];
    if (L.pending_urgent_ops.empty()) return;

    bool any_pending = false;
    for (const auto& op : L.pending_urgent_ops) {
        if (op && !op->done) {
            any_pending = true;
            break;
        }
    }
    if (!any_pending) {
        L.pending_urgent_ops.clear();
        return;
    }

    auto t0 = std::chrono::steady_clock::now();
    std::vector<std::shared_ptr<ExpertReadOp>> wait_ops = L.pending_urgent_ops;

    while (true) {
        bool all_done = true;
        for (const auto& op : wait_ops) {
            if (op && !op->done) {
                all_done = false;
                break;
            }
        }
        if (all_done) break;

        // Help drain URGENT queue while waiting
        IoTask task;
        if (!urgent_queue_.empty()) {
            task = std::move(urgent_queue_.front());
            urgent_queue_.pop_front();
            bool valid = false;
            for (int k = 0; k < task.iov_cnt; ++k) {
                if (task.ops[k] && !task.ops[k]->cancelled) {
                    task.ops[k]->tasks_started++;
                    valid = true;
                }
            }
            if (valid) {
                lock.unlock();
                execute_task(task);
                lock.lock();
                continue;
            }
        }
        done_cv_.wait(lock);
    }

    auto t1 = std::chrono::steady_clock::now();
    L.stats.io_wait_ns += static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count());
    L.pending_urgent_ops.clear();
}

void P1IoScheduler::wait_layer_all(int layer_idx) {
    if (layer_idx < 0 || layer_idx >= kP1MaxLayers) return;
    wait_layer_urgent(layer_idx);

    std::unique_lock<std::mutex> lock(mutex_);
    LayerState& L = layers_[layer_idx];
    if (!L.initialized) return;
    while (true) {
        bool all_done = true;
        for (int s = 0; s < L.capacity; ++s) {
            if (L.active_io[s] && !L.active_io[s]->done && !L.active_io[s]->cancelled) {
                all_done = false;
                break;
            }
        }
        if (all_done) break;

        IoTask task;
        if (pop_next_task_locked(task)) {
            lock.unlock();
            execute_task(task);
            lock.lock();
        } else {
            done_cv_.wait(lock);
        }
    }
}

void P1IoScheduler::wait_all() {
    for (int l = 0; l < kP1MaxLayers; ++l) {
        wait_layer_all(l);
    }
}

void P1IoScheduler::unpin_layer_slots(int layer_idx) {
    if (layer_idx < 0 || layer_idx >= kP1MaxLayers) return;
    std::unique_lock<std::mutex> lock(mutex_);
    LayerState& L = layers_[layer_idx];
    if (L.gpu_pinned_slots.empty()) return;
    for (int16_t slot : L.gpu_pinned_slots) {
        if (slot >= 0 && slot < kP1NumExperts && L.pin_count[slot] > 0) {
            L.pin_count[slot]--;
        }
    }
    L.gpu_pinned_slots.clear();
    done_cv_.notify_all();
}

void P1IoScheduler::pin_slots(int layer_idx, const uint32_t* slots, int count) {
    if (layer_idx < 0 || layer_idx >= kP1MaxLayers || slots == nullptr || count <= 0) return;
    std::unique_lock<std::mutex> lock(mutex_);
    LayerState& L = layers_[layer_idx];
    for (int i = 0; i < count; ++i) {
        uint32_t s = slots[i];
        if (s < static_cast<uint32_t>(kP1NumExperts)) {
            L.pin_count[s]++;
        }
    }
}

void P1IoScheduler::unpin_slots(int layer_idx, const uint32_t* slots, int count) {
    if (layer_idx < 0 || layer_idx >= kP1MaxLayers || slots == nullptr || count <= 0) return;
    std::unique_lock<std::mutex> lock(mutex_);
    LayerState& L = layers_[layer_idx];
    for (int i = 0; i < count; ++i) {
        uint32_t s = slots[i];
        if (s < static_cast<uint32_t>(kP1NumExperts) && L.pin_count[s] > 0) {
            L.pin_count[s]--;
        }
    }
    done_cv_.notify_all();
}

int P1IoScheduler::get_slot_pin_count(int layer_idx, int slot) {
    if (layer_idx < 0 || layer_idx >= kP1MaxLayers || slot < 0 || slot >= kP1NumExperts) return 0;
    std::unique_lock<std::mutex> lock(mutex_);
    return layers_[layer_idx].pin_count[slot];
}

int P1IoScheduler::force_invalidate(int layer_idx, double rate, std::mt19937_64& rng) {
    if (layer_idx < 0 || layer_idx >= kP1MaxLayers || rate <= 0.0) return 0;

    std::unique_lock<std::mutex> lock(mutex_);
    LayerState& L = layers_[layer_idx];
    if (!L.initialized) return 0;

    std::uniform_real_distribution<double> dist(0.0, 1.0);
    int invalidated = 0;

    // Collect resident unpinned slots in LRU order
    std::vector<int16_t> slots_to_check;
    int16_t cur = L.lru_head;
    while (cur >= 0) {
        slots_to_check.push_back(cur);
        cur = L.next_slot[cur];
    }

    std::vector<int16_t> chosen_slots;
    for (int16_t slot : slots_to_check) {
        if (L.pin_count[slot] > 0) continue;
        if (L.slot_to_eid[slot] < 0) continue;
        if (rate >= 1.0 || dist(rng) < rate) {
            chosen_slots.push_back(slot);
        }
    }

    for (int16_t slot : chosen_slots) {
        if (L.pin_count[slot] > 0) continue;
        int16_t eid = L.slot_to_eid[slot];
        if (eid < 0) continue;

        auto active = L.active_io[slot];
        if (active && !active->done && !active->cancelled) {
            if (active->is_prefetch && active->tasks_started == 0) {
                active->resolved = true;
                active->cancelled = true;
                L.stats.prefetch_cancelled++;
            } else {
                // Wait for in-flight IO on this slot before scrubbing/freeing it
                while (!active->done && !active->cancelled) {
                    IoTask task;
                    if (pop_next_task_locked(task)) {
                        lock.unlock();
                        execute_task(task);
                        lock.lock();
                    } else {
                        done_cv_.wait(lock);
                    }
                }
            }
        }

        // Re-verify after potential unlock
        if (L.pin_count[slot] > 0) continue;
        eid = L.slot_to_eid[slot];
        if (eid < 0) continue;

        for (auto& op : L.last_prefetch_ops) {
            if (op && !op->resolved && op->slot == slot) {
                op->resolved = true;
                L.stats.prefetch_wasted++;
            }
        }
        lru_remove_slot_locked(L, slot);
        L.eid_to_slot[eid] = -1;
        L.slot_to_eid[slot] = -1;
        L.active_io[slot].reset();
        L.pre_prefetch_bitmap[eid >> 6] &= ~(1ULL << (eid & 63));
        // Scrub first 64 bytes of slot so any bug failing to reload this expert from disk is caught
        if (L.base_ptrs[0] != nullptr && L.row_bytes[0] > 0) {
            size_t scrub = std::min<size_t>(64, static_cast<size_t>(L.row_bytes[0]));
            std::memset(L.base_ptrs[0] + static_cast<uint64_t>(slot) * L.row_bytes[0], 0xA5, scrub);
        }
        L.free_slots[L.free_count++] = slot;
        invalidated++;
    }

    L.stats.forced_invalidations += static_cast<uint64_t>(invalidated);
    return invalidated;
}

void P1IoScheduler::load_sync(
    int layer_idx,
    const int32_t* eids,
    const int32_t* slots,
    int count,
    bool is_prefill) {
    if (layer_idx < 0 || layer_idx >= kP1MaxLayers || count <= 0) return;
    wait_layer_all(layer_idx);

    std::unique_lock<std::mutex> lock(mutex_);
    LayerState& L = layers_[layer_idx];
    if (!L.initialized) return;

    if (is_prefill) {
        L.stats.prefill_demand_bytes += static_cast<uint64_t>(count) * L.expert_bytes;
    }

    std::vector<std::shared_ptr<ExpertReadOp>> ops;
    ops.reserve(static_cast<size_t>(count));
    for (int i = 0; i < count; ++i) {
        auto op = std::make_shared<ExpertReadOp>();
        op->layer_idx = layer_idx;
        op->eid = eids[i];
        op->slot = static_cast<int16_t>(slots[i]);
        op->is_prefetch = false;
        op->total_tasks = L.num_tensors;
        L.active_io[op->slot] = op;
        ops.push_back(op);
    }

    size_t prev_q_size = urgent_queue_.size();
    enqueue_coalesced_urgent_locked(layer_idx, L, ops);
    if (is_prefill) {
        for (size_t qi = prev_q_size; qi < urgent_queue_.size(); ++qi) {
            urgent_queue_[qi].is_prefill = true;
        }
    }
    L.pending_urgent_ops = ops;
    work_cv_.notify_all();
    lock.unlock();

    wait_layer_urgent(layer_idx);
}

void P1IoScheduler::get_lru_state(
    int layer_idx,
    int32_t* out_eids,
    int32_t* out_slots,
    int* out_count) {
    if (layer_idx < 0 || layer_idx >= kP1MaxLayers) {
        if (out_count) *out_count = 0;
        return;
    }
    wait_layer_all(layer_idx);

    std::unique_lock<std::mutex> lock(mutex_);
    LayerState& L = layers_[layer_idx];
    int cnt = 0;
    int16_t cur = L.lru_head;
    while (cur >= 0) {
        out_eids[cnt] = static_cast<int32_t>(L.slot_to_eid[cur]);
        out_slots[cnt] = static_cast<int32_t>(cur);
        cnt++;
        cur = L.next_slot[cur];
    }
    *out_count = cnt;
}

void P1IoScheduler::set_lru_state(
    int layer_idx,
    const int32_t* ordered_eids,
    const int32_t* ordered_slots,
    int count,
    uint64_t add_prefill_requests,
    uint64_t add_prefill_hits,
    uint64_t add_prefill_misses) {
    if (layer_idx < 0 || layer_idx >= kP1MaxLayers) return;
    wait_layer_all(layer_idx);

    std::unique_lock<std::mutex> lock(mutex_);
    LayerState& L = layers_[layer_idx];
    L.stats.prefill_requests += add_prefill_requests;
    L.stats.prefill_lru_hits += add_prefill_hits;
    L.stats.prefill_demand_misses += add_prefill_misses;

    for (int i = 0; i < kP1NumExperts; ++i) {
        L.eid_to_slot[i] = -1;
        L.slot_to_eid[i] = -1;
        L.prev_slot[i] = -1;
        L.next_slot[i] = -1;
        L.active_io[i].reset();
    }
    L.lru_head = -1;
    L.lru_tail = -1;
    L.has_pre_prefetch_snapshot = false;
    L.last_prefetch_ops.clear();
    L.pending_urgent_ops.clear();

    uint8_t used_slot[kP1NumExperts]{};
    for (int i = 0; i < count; ++i) {
        int32_t eid = ordered_eids[i];
        int16_t slot = static_cast<int16_t>(ordered_slots[i]);
        if (slot >= 0 && slot < L.capacity && eid >= 0 && eid < kP1NumExperts) {
            used_slot[slot] = 1;
            lru_insert_eid_slot_locked(L, eid, slot);
        }
    }

    L.free_count = 0;
    for (int s = L.capacity - 1; s >= 0; --s) {
        if (!used_slot[s]) {
            L.free_slots[L.free_count++] = static_cast<int16_t>(s);
        }
    }
}

P1LayerStats P1IoScheduler::get_layer_stats(int layer_idx) const {
    if (layer_idx < 0 || layer_idx >= kP1MaxLayers) return P1LayerStats{};
    std::unique_lock<std::mutex> lock(mutex_);
    P1LayerStats st = layers_[layer_idx].stats;
    st.bytes = layers_[layer_idx].atomic_bytes.load(std::memory_order_relaxed);
    return st;
}

P1LayerStats P1IoScheduler::get_total_stats(int num_layers) const {
    std::unique_lock<std::mutex> lock(mutex_);
    P1LayerStats tot{};
    int lim = std::min(num_layers, kP1MaxLayers);
    for (int l = 1; l < lim; ++l) {
        const auto& L = layers_[l];
        if (!L.initialized) continue;
        tot.decode_requests += L.stats.decode_requests;
        tot.lru_hits += L.stats.lru_hits;
        tot.urgent_reads += L.stats.urgent_reads;
        tot.prefetch_issued += L.stats.prefetch_issued;
        tot.prefetch_cancelled += L.stats.prefetch_cancelled;
        tot.prefetch_promoted += L.stats.prefetch_promoted;
        tot.prefetch_used += L.stats.prefetch_used;
        tot.prefetch_wasted += L.stats.prefetch_wasted;
        tot.pin_waits += L.stats.pin_waits;
        tot.bytes += L.atomic_bytes.load(std::memory_order_relaxed);
        tot.io_wait_ns += L.stats.io_wait_ns;
        tot.forced_invalidations += L.stats.forced_invalidations;
        tot.prefill_requests += L.stats.prefill_requests;
        tot.prefill_lru_hits += L.stats.prefill_lru_hits;
        tot.prefill_demand_misses += L.stats.prefill_demand_misses;
        tot.prefill_demand_bytes += L.stats.prefill_demand_bytes;
    }
    return tot;
}

} // namespace qwen38

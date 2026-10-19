/*
 * expert_io.c - Native C pthread I/O pool + O(1) LRU slot cache for Qwen3.8-Flash-Next MLX Offload
 *
 * Build on macOS (Apple Silicon):
 *   clang -O3 -shared -fPIC -lpthread expert_io.c -o libexpert_io.dylib
 */

#include <pthread.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <sys/uio.h>
#include <time.h>
#include <unistd.h>

#define MAX_LAYERS 64
#define NUM_EXPERTS 512
#define NUM_TENSORS 9
#define EXPERT_BYTES 1536000ULL
#define MAX_IO_WORKERS 16
#define MAX_TASKS_PER_BATCH (NUM_TENSORS * NUM_EXPERTS)

typedef struct {
    int fd;
    off_t offset;
    int iov_cnt;
    struct iovec *iov_ptr;
    struct iovec inline_iov[16];
} IoTask;

typedef struct {
    IoTask tasks[MAX_TASKS_PER_BATCH];
    struct iovec overflow_iovs[NUM_TENSORS * NUM_EXPERTS];
    int overflow_used;
    atomic_int num_tasks;
    atomic_int next_task;
    atomic_int remaining;
    uint64_t generation;
} IoBatch;

typedef struct {
    int initialized;
    int fd;
    int capacity;
    uint64_t base_offsets[NUM_TENSORS];
    uint64_t row_bytes[NUM_TENSORS];
    uint8_t *base_ptrs[NUM_TENSORS];
    uint32_t *slot_inds_ptr;

    /* O(1) doubly-linked list LRU over slots [0..capacity-1] */
    int16_t eid_to_slot[NUM_EXPERTS];
    int16_t slot_to_eid[NUM_EXPERTS];
    int16_t prev_slot[NUM_EXPERTS];
    int16_t next_slot[NUM_EXPERTS];
    int16_t lru_head; /* oldest (LRU) */
    int16_t lru_tail; /* newest (MRU) */
    int16_t free_slots[NUM_EXPERTS];
    int free_count;

    /* Prefetch tracking */
    int has_pre_prefetch_snapshot;
    uint64_t pre_prefetch_bitmap[8];
    uint64_t last_pred_bitmap[8];
    uint64_t last_prefetched_bitmap[8];

    /* Layer stats */
    uint64_t decode_requests;
    uint64_t decode_lru_hits;
    uint64_t decode_prefetch_hits;
    uint64_t decode_demand_misses;
    uint64_t decode_demand_bytes;
    uint64_t decode_prefetch_bytes;

    uint64_t prefill_requests;
    uint64_t prefill_lru_hits;
    uint64_t prefill_demand_misses;
    uint64_t prefill_demand_bytes;

    uint64_t pred_topm_hits;
    uint64_t pred_topm_total_pred;
    uint64_t pred_topm_total_actual;

    uint64_t miss_pref_hits;
    uint64_t miss_pref_total_fetched;
    uint64_t miss_pref_total_actual_misses;

    uint64_t uncoalesced_tensor_reads;
    uint64_t coalesced_preadv_calls;

    /* Per-expert last decode token step when evicted (-1 = never / currently resident) */
    int32_t last_evicted_token[NUM_EXPERTS];
} LayerCache;

static LayerCache g_layers[MAX_LAYERS];

/* Optional per-token IO accounting state (default OFF) */
static int g_io_accounting_enabled = 0;
static int32_t g_cache_token_step = 0;
static int32_t g_cur_decode_token = -1;
static uint64_t g_tok_lru_hits = 0;
static uint64_t g_tok_demand_misses = 0;
static uint64_t g_tok_prefetch_loads = 0;
static uint64_t g_tok_prefetch_hits = 0;
static uint64_t g_tok_wasted_prefetches = 0;
static uint64_t g_tok_demand_rereads_16 = 0;
static uint64_t g_tok_demand_rereads_64 = 0;
static uint64_t g_tok_prefetch_rereads_16 = 0;
static uint64_t g_tok_prefetch_rereads_64 = 0;
static _Atomic uint64_t g_tok_pread_bytes = 0;

/* Global persistent worker pool */
static pthread_t g_workers[MAX_IO_WORKERS];
static int g_num_workers = 0;
static int g_pool_initialized = 0;
static atomic_int g_shutdown = 0;

static pthread_mutex_t g_pool_mutex = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t g_work_cond = PTHREAD_COND_INITIALIZER;
static pthread_cond_t g_done_cond = PTHREAD_COND_INITIALIZER;

/* 4-slot batch ring buffer so consecutive batches never alias in-flight worker pointers */
#define BATCH_RING_SIZE 4
static IoBatch g_batch_ring[BATCH_RING_SIZE];
static uint64_t g_batch_alloc_idx = 0;
static _Atomic(IoBatch *) g_prior_prefetch_batch = NULL;
static _Atomic(IoBatch *) g_demand_batch = NULL;
static _Atomic(IoBatch *) g_current_batch = NULL;
static IoBatch *g_async_batch_ptr = NULL;
static IoBatch *g_layer_demand_batch[MAX_LAYERS];
static IoBatch *g_layer_prefetch_batch[MAX_LAYERS];
static uint32_t g_split_hit_slots[MAX_LAYERS][16];
static uint32_t g_split_miss_slots[MAX_LAYERS][16];
static int32_t g_split_inv_order[MAX_LAYERS][16];
static uint64_t g_work_seq = 0;

#define SCHED_FLAG_SYNC_DEMAND 1
#define SCHED_FLAG_NO_PREFETCH 2
#define SCHED_FLAG_SPLIT       4

static inline void execute_one_task(const IoTask *t) {
    if (t->iov_cnt == 1) {
        uint8_t *buf = (uint8_t *)t->iov_ptr[0].iov_base;
        size_t len = t->iov_ptr[0].iov_len;
        off_t off = t->offset;
        while (len > 0) {
            ssize_t n = pread(t->fd, buf, len, off);
            if (n <= 0) break;
            if (g_io_accounting_enabled) {
                atomic_fetch_add_explicit(&g_tok_pread_bytes, (uint64_t)n, memory_order_relaxed);
            }
            buf += n;
            len -= (size_t)n;
            off += n;
        }
    } else {
        ssize_t n = preadv(t->fd, t->iov_ptr, t->iov_cnt, t->offset);
        if (g_io_accounting_enabled && n > 0) {
            atomic_fetch_add_explicit(&g_tok_pread_bytes, (uint64_t)n, memory_order_relaxed);
        }
    }
}

static inline int try_run_batch_task(IoBatch *b) {
    if (b == NULL) return 0;
    int n = atomic_load_explicit(&b->num_tasks, memory_order_acquire);
    if (n <= 0) return 0;
    int idx = atomic_fetch_add_explicit(&b->next_task, 1, memory_order_relaxed);
    if (idx >= n) return 0;

    execute_one_task(&b->tasks[idx]);

    if (atomic_fetch_sub_explicit(&b->remaining, 1, memory_order_acq_rel) == 1) {
        pthread_mutex_lock(&g_pool_mutex);
        if (atomic_load_explicit(&g_prior_prefetch_batch, memory_order_relaxed) == b) {
            atomic_store_explicit(&g_prior_prefetch_batch, NULL, memory_order_release);
        }
        if (atomic_load_explicit(&g_demand_batch, memory_order_relaxed) == b) {
            atomic_store_explicit(&g_demand_batch, NULL, memory_order_release);
        }
        if (atomic_load_explicit(&g_current_batch, memory_order_relaxed) == b) {
            atomic_store_explicit(&g_current_batch, NULL, memory_order_release);
        }
        pthread_cond_broadcast(&g_done_cond);
        pthread_mutex_unlock(&g_pool_mutex);
    }
    return 1;
}

static inline int batch_has_unclaimed_tasks(IoBatch *b) {
    if (b == NULL) return 0;
    int n = atomic_load_explicit(&b->num_tasks, memory_order_acquire);
    if (n <= 0) return 0;
    return atomic_load_explicit(&b->next_task, memory_order_relaxed) < n;
}

static inline int pool_has_unclaimed_work(void) {
    return batch_has_unclaimed_tasks(atomic_load_explicit(&g_prior_prefetch_batch, memory_order_acquire)) ||
           batch_has_unclaimed_tasks(atomic_load_explicit(&g_demand_batch, memory_order_acquire)) ||
           batch_has_unclaimed_tasks(atomic_load_explicit(&g_current_batch, memory_order_acquire));
}

static inline int try_run_any_active_batch(void) {
    IoBatch *b0 = atomic_load_explicit(&g_prior_prefetch_batch, memory_order_acquire);
    if (b0 != NULL && try_run_batch_task(b0)) return 1;
    IoBatch *b1 = atomic_load_explicit(&g_demand_batch, memory_order_acquire);
    if (b1 != NULL && try_run_batch_task(b1)) return 1;
    IoBatch *b2 = atomic_load_explicit(&g_current_batch, memory_order_acquire);
    if (b2 != NULL && try_run_batch_task(b2)) return 1;
    return 0;
}

static void *worker_thread_main(void *arg) {
    (void)arg;
    pthread_mutex_lock(&g_pool_mutex);
    uint64_t seen_seq = g_work_seq;
    pthread_mutex_unlock(&g_pool_mutex);

    while (!atomic_load_explicit(&g_shutdown, memory_order_relaxed)) {
        if (try_run_any_active_batch()) {
            continue;
        }

        /* Brief user-space spin (~30 us) to avoid kernel condvar wakeup latency between layers */
        int found = 0;
        for (int spin = 0; spin < 2000; ++spin) {
            if (pool_has_unclaimed_work()) {
                found = 1;
                break;
            }
            #if defined(__aarch64__)
            __asm__ volatile("yield");
            #endif
        }
        if (found) continue;

        pthread_mutex_lock(&g_pool_mutex);
        while (!atomic_load_explicit(&g_shutdown, memory_order_relaxed) &&
               !pool_has_unclaimed_work() &&
               g_work_seq == seen_seq) {
            pthread_cond_wait(&g_work_cond, &g_pool_mutex);
        }
        seen_seq = g_work_seq;
        pthread_mutex_unlock(&g_pool_mutex);
    }
    return NULL;
}

/* Coalesce (eids, slots) into contiguous eid runs and populate IoBatch */
static int build_coalesced_batch(
    LayerCache *L,
    IoBatch *B,
    const int32_t *eids,
    const int32_t *slots,
    int count
) {
    if (count <= 0) return 0;

    /* Sort pairs by ascending eid (insertion sort; count <= 16 in decode, <= 512 in prefill) */
    int32_t s_eids[NUM_EXPERTS];
    int32_t s_slots[NUM_EXPERTS];
    for (int i = 0; i < count; ++i) {
        int32_t e = eids[i];
        int32_t sl = slots[i];
        int j = i - 1;
        while (j >= 0 && s_eids[j] > e) {
            s_eids[j + 1] = s_eids[j];
            s_slots[j + 1] = s_slots[j];
            --j;
        }
        s_eids[j + 1] = e;
        s_slots[j + 1] = sl;
    }

    /* Identify contiguous runs */
    int run_starts[NUM_EXPERTS];
    int run_lens[NUM_EXPERTS];
    int num_runs = 0;

    int cur_start = 0;
    for (int i = 1; i < count; ++i) {
        if (s_eids[i] != s_eids[i - 1] + 1) {
            run_starts[num_runs] = cur_start;
            run_lens[num_runs] = i - cur_start;
            num_runs++;
            cur_start = i;
        }
    }
    run_starts[num_runs] = cur_start;
    run_lens[num_runs] = count - cur_start;
    num_runs++;

    L->uncoalesced_tensor_reads += (uint64_t)count * NUM_TENSORS;
    L->coalesced_preadv_calls += (uint64_t)num_runs * NUM_TENSORS;

    B->overflow_used = 0;
    int task_idx = 0;

    /* Pass 0: weight tensors (row_bytes > 100000) first (LPT scheduling).
     * Pass 1: scale/bias tensors (row_bytes <= 100000) second. */
    for (int pass = 0; pass < 2; ++pass) {
        for (int t = 0; t < NUM_TENSORS; ++t) {
            uint64_t rb = L->row_bytes[t];
            int is_large = (rb > 100000ULL);
            if ((pass == 0 && !is_large) || (pass == 1 && is_large)) continue;

            uint64_t base_off = L->base_offsets[t];
            uint8_t *base_ptr = L->base_ptrs[t];

            for (int r = 0; r < num_runs; ++r) {
                int r_start = run_starts[r];
                int r_len = run_lens[r];
                int32_t first_eid = s_eids[r_start];

                IoTask *task = &B->tasks[task_idx++];
                task->fd = L->fd;
                task->offset = (off_t)(base_off + (uint64_t)first_eid * rb);
                task->iov_cnt = r_len;

                struct iovec *iov;
                if (r_len <= 16) {
                    iov = task->inline_iov;
                } else {
                    iov = &B->overflow_iovs[B->overflow_used];
                    B->overflow_used += r_len;
                }
                task->iov_ptr = iov;

                for (int k = 0; k < r_len; ++k) {
                    int32_t sl = s_slots[r_start + k];
                    iov[k].iov_base = base_ptr + (uint64_t)sl * rb;
                    iov[k].iov_len = (size_t)rb;
                }
            }
        }
    }
    return task_idx;
}

static void wait_for_batch(IoBatch *B) {
    if (B == NULL) return;
    while (try_run_batch_task(B)) {}

    if (atomic_load_explicit(&B->remaining, memory_order_acquire) > 0) {
        for (int spin = 0; spin < 2000; ++spin) {
            if (atomic_load_explicit(&B->remaining, memory_order_acquire) <= 0) {
                break;
            }
            #if defined(__aarch64__)
            __asm__ volatile("yield");
            #endif
        }
        if (atomic_load_explicit(&B->remaining, memory_order_acquire) > 0) {
            pthread_mutex_lock(&g_pool_mutex);
            while (atomic_load_explicit(&B->remaining, memory_order_acquire) > 0) {
                struct timespec ts;
                clock_gettime(CLOCK_REALTIME, &ts);
                ts.tv_nsec += 5000000L;
                if (ts.tv_nsec >= 1000000000L) {
                    ts.tv_sec += 1;
                    ts.tv_nsec -= 1000000000L;
                }
                pthread_cond_timedwait(&g_done_cond, &g_pool_mutex, &ts);
                pthread_mutex_unlock(&g_pool_mutex);
                while (try_run_batch_task(B)) {}
                pthread_mutex_lock(&g_pool_mutex);
            }
            pthread_mutex_unlock(&g_pool_mutex);
        }
    }
    atomic_store_explicit(&B->num_tasks, 0, memory_order_release);
}

static inline IoBatch *alloc_ring_batch(void) {
    IoBatch *B = &g_batch_ring[g_batch_alloc_idx++ & (BATCH_RING_SIZE - 1)];
    if (atomic_load_explicit(&B->num_tasks, memory_order_acquire) > 0) {
        wait_for_batch(B);
    }
    return B;
}

static void wait_async_batch(void) {
    if (g_async_batch_ptr != NULL) {
        IoBatch *B = g_async_batch_ptr;
        g_async_batch_ptr = NULL;
        wait_for_batch(B);
    }
    IoBatch *Bp = atomic_load_explicit(&g_prior_prefetch_batch, memory_order_acquire);
    if (Bp != NULL) wait_for_batch(Bp);
    IoBatch *Bd = atomic_load_explicit(&g_demand_batch, memory_order_acquire);
    if (Bd != NULL) wait_for_batch(Bd);
    IoBatch *Bc = atomic_load_explicit(&g_current_batch, memory_order_acquire);
    if (Bc != NULL) wait_for_batch(Bc);
    for (int i = 0; i < MAX_LAYERS; ++i) {
        if (g_layer_prefetch_batch[i] != NULL) {
            IoBatch *B = g_layer_prefetch_batch[i];
            g_layer_prefetch_batch[i] = NULL;
            wait_for_batch(B);
        }
        if (g_layer_demand_batch[i] != NULL) {
            IoBatch *B = g_layer_demand_batch[i];
            g_layer_demand_batch[i] = NULL;
            wait_for_batch(B);
        }
    }
}

static void dispatch_sync_batch(LayerCache *L, const int32_t *eids, const int32_t *slots, int count) {
    if (count <= 0) return;
    wait_async_batch();

    IoBatch *B = alloc_ring_batch();
    int n_tasks = build_coalesced_batch(L, B, eids, slots, count);
    atomic_store_explicit(&B->next_task, 0, memory_order_relaxed);
    atomic_store_explicit(&B->remaining, n_tasks, memory_order_relaxed);
    atomic_store_explicit(&B->num_tasks, n_tasks, memory_order_release);

    pthread_mutex_lock(&g_pool_mutex);
    atomic_store_explicit(&g_current_batch, B, memory_order_release);
    g_work_seq++;
    pthread_cond_broadcast(&g_work_cond);
    pthread_mutex_unlock(&g_pool_mutex);

    wait_for_batch(B);
}

static void dispatch_async_batch(int layer_idx, LayerCache *L, const int32_t *eids, const int32_t *slots, int count) {
    (void)layer_idx;
    if (count <= 0) return;
    wait_async_batch();

    IoBatch *B = alloc_ring_batch();
    int n_tasks = build_coalesced_batch(L, B, eids, slots, count);
    atomic_store_explicit(&B->next_task, 0, memory_order_relaxed);
    atomic_store_explicit(&B->remaining, n_tasks, memory_order_relaxed);
    atomic_store_explicit(&B->num_tasks, n_tasks, memory_order_release);
    g_async_batch_ptr = B;

    pthread_mutex_lock(&g_pool_mutex);
    atomic_store_explicit(&g_current_batch, B, memory_order_release);
    g_work_seq++;
    pthread_cond_broadcast(&g_work_cond);
    pthread_mutex_unlock(&g_pool_mutex);
}

/* Doubly-linked list LRU helpers */
static inline void lru_remove_slot(LayerCache *L, int16_t slot) {
    int16_t p = L->prev_slot[slot];
    int16_t n = L->next_slot[slot];
    if (p >= 0) {
        L->next_slot[p] = n;
    } else {
        L->lru_head = n;
    }
    if (n >= 0) {
        L->prev_slot[n] = p;
    } else {
        L->lru_tail = p;
    }
    L->prev_slot[slot] = -1;
    L->next_slot[slot] = -1;
}

static inline void lru_append_mru(LayerCache *L, int16_t slot) {
    L->prev_slot[slot] = L->lru_tail;
    L->next_slot[slot] = -1;
    if (L->lru_tail >= 0) {
        L->next_slot[L->lru_tail] = slot;
    } else {
        L->lru_head = slot;
    }
    L->lru_tail = slot;
}

static inline void lru_move_to_mru(LayerCache *L, int16_t slot) {
    if (L->lru_tail == slot) return;
    lru_remove_slot(L, slot);
    lru_append_mru(L, slot);
}

static inline void lru_insert_eid_slot(LayerCache *L, int32_t eid, int16_t slot) {
    if (L->eid_to_slot[eid] >= 0) {
        int16_t existing = L->eid_to_slot[eid];
        lru_move_to_mru(L, existing);
        return;
    }
    L->eid_to_slot[eid] = slot;
    L->slot_to_eid[slot] = (int16_t)eid;
    lru_append_mru(L, slot);
}

static inline int16_t lru_alloc_slot(LayerCache *L, const uint64_t protected_bm[8]) {
    if (L->free_count > 0) {
        return L->free_slots[--L->free_count];
    }
    int16_t cur = L->lru_head;
    while (cur >= 0) {
        int16_t eid = L->slot_to_eid[cur];
        if (eid >= 0 && !(protected_bm[eid >> 6] & (1ULL << (eid & 63)))) {
            lru_remove_slot(L, cur);
            L->eid_to_slot[eid] = -1;
            L->slot_to_eid[cur] = -1;
            if (g_io_accounting_enabled && g_cur_decode_token >= 0) {
                L->last_evicted_token[eid] = g_cur_decode_token;
            }
            return cur;
        }
        cur = L->next_slot[cur];
    }
    /* Fallback: pop oldest (lru_head) */
    int16_t victim = L->lru_head;
    int16_t eid = L->slot_to_eid[victim];
    lru_remove_slot(L, victim);
    if (eid >= 0) {
        L->eid_to_slot[eid] = -1;
        if (g_io_accounting_enabled && g_cur_decode_token >= 0) {
            L->last_evicted_token[eid] = g_cur_decode_token;
        }
    }
    L->slot_to_eid[victim] = -1;
    return victim;
}

/* Public C API exported to Python ctypes */

void c_io_init(int num_threads) {
    if (g_pool_initialized) return;
    if (num_threads < 1) num_threads = 1;
    if (num_threads > MAX_IO_WORKERS) num_threads = MAX_IO_WORKERS;
    g_num_workers = num_threads;
    atomic_store(&g_shutdown, 0);
    atomic_store(&g_prior_prefetch_batch, NULL);
    atomic_store(&g_demand_batch, NULL);
    atomic_store(&g_current_batch, NULL);
    g_async_batch_ptr = NULL;
    memset(g_layer_demand_batch, 0, sizeof(g_layer_demand_batch));
    memset(g_layer_prefetch_batch, 0, sizeof(g_layer_prefetch_batch));
    for (int b = 0; b < BATCH_RING_SIZE; ++b) {
        atomic_store(&g_batch_ring[b].num_tasks, 0);
        atomic_store(&g_batch_ring[b].next_task, 0);
        atomic_store(&g_batch_ring[b].remaining, 0);
    }

    for (int i = 0; i < g_num_workers; ++i) {
        pthread_create(&g_workers[i], NULL, worker_thread_main, NULL);
    }
    g_pool_initialized = 1;
}

void c_layer_reset_cache(int layer_idx) {
    if (layer_idx < 0 || layer_idx >= MAX_LAYERS) return;
    wait_async_batch();
    LayerCache *L = &g_layers[layer_idx];
    for (int i = 0; i < NUM_EXPERTS; ++i) {
        L->eid_to_slot[i] = -1;
        L->slot_to_eid[i] = -1;
        L->prev_slot[i] = -1;
        L->next_slot[i] = -1;
        L->last_evicted_token[i] = -1;
    }
    L->lru_head = -1;
    L->lru_tail = -1;
    L->free_count = L->capacity;
    /* Match Python list(range(capacity - 1, -1, -1)) so pop() returns 0, 1, 2, ... */
    for (int i = 0; i < L->capacity; ++i) {
        L->free_slots[i] = (int16_t)(L->capacity - 1 - i);
    }
    L->has_pre_prefetch_snapshot = 0;
    if (layer_idx == 1) {
        g_cache_token_step = 0;
        g_cur_decode_token = -1;
    }
}

void c_layer_reset_stats(int layer_idx) {
    if (layer_idx < 0 || layer_idx >= MAX_LAYERS) return;
    LayerCache *L = &g_layers[layer_idx];
    L->decode_requests = 0;
    L->decode_lru_hits = 0;
    L->decode_prefetch_hits = 0;
    L->decode_demand_misses = 0;
    L->decode_demand_bytes = 0;
    L->decode_prefetch_bytes = 0;

    L->prefill_requests = 0;
    L->prefill_lru_hits = 0;
    L->prefill_demand_misses = 0;
    L->prefill_demand_bytes = 0;

    L->pred_topm_hits = 0;
    L->pred_topm_total_pred = 0;
    L->pred_topm_total_actual = 0;

    L->miss_pref_hits = 0;
    L->miss_pref_total_fetched = 0;
    L->miss_pref_total_actual_misses = 0;

    L->uncoalesced_tensor_reads = 0;
    L->coalesced_preadv_calls = 0;
}

void c_layer_init(
    int layer_idx,
    int fd,
    int capacity,
    const uint64_t *base_offsets,
    const uint64_t *row_bytes,
    const uint64_t *base_ptrs,
    uint64_t slot_inds_ptr
) {
    if (layer_idx < 0 || layer_idx >= MAX_LAYERS) return;
    LayerCache *L = &g_layers[layer_idx];
    memset(L, 0, sizeof(LayerCache));
    L->initialized = 1;
    L->fd = fd;
    L->capacity = capacity;
    for (int t = 0; t < NUM_TENSORS; ++t) {
        L->base_offsets[t] = base_offsets[t];
        L->row_bytes[t] = row_bytes[t];
        L->base_ptrs[t] = (uint8_t *)(uintptr_t)base_ptrs[t];
    }
    L->slot_inds_ptr = (uint32_t *)(uintptr_t)slot_inds_ptr;
    c_layer_reset_cache(layer_idx);
    c_layer_reset_stats(layer_idx);
}

void c_layer_wait_prefetch(int layer_idx) {
    (void)layer_idx;
    wait_async_batch();
}

void c_layer_start_prefetch(
    int layer_idx,
    const int32_t *pred_eids,
    int pred_count,
    int max_fetch
) {
    if (layer_idx < 0 || layer_idx >= MAX_LAYERS) return;
    wait_async_batch();
    LayerCache *L = &g_layers[layer_idx];

    /* Snapshot current LRU cache residency bitmap */
    memset(L->pre_prefetch_bitmap, 0, sizeof(L->pre_prefetch_bitmap));
    for (int e = 0; e < NUM_EXPERTS; ++e) {
        if (L->eid_to_slot[e] >= 0) {
            L->pre_prefetch_bitmap[e >> 6] |= (1ULL << (e & 63));
        }
    }
    memset(L->last_pred_bitmap, 0, sizeof(L->last_pred_bitmap));
    memset(L->last_prefetched_bitmap, 0, sizeof(L->last_prefetched_bitmap));
    L->has_pre_prefetch_snapshot = 1;

    uint64_t protected_bm[8] = {0};
    int32_t missing_pred[32];
    int n_missing = 0;
    int cap_fetch = (max_fetch > 0 && max_fetch < 32) ? max_fetch : 32;

    for (int i = 0; i < pred_count; ++i) {
        int32_t eid = pred_eids[i];
        if (eid < 0 || eid >= NUM_EXPERTS) continue;
        L->last_pred_bitmap[eid >> 6] |= (1ULL << (eid & 63));
        protected_bm[eid >> 6] |= (1ULL << (eid & 63));
        if (L->eid_to_slot[eid] < 0 && n_missing < cap_fetch) {
            missing_pred[n_missing++] = eid;
            L->last_prefetched_bitmap[eid >> 6] |= (1ULL << (eid & 63));
        }
    }

    if (n_missing == 0) {
        return;
    }

    /* Insert in reverse rank order so highest-ranked predictions are newest in LRU */
    int32_t rev_missing[32];
    int32_t assigned_slots[32];
    for (int i = 0; i < n_missing; ++i) {
        int32_t eid = missing_pred[n_missing - 1 - i];
        int16_t slot = lru_alloc_slot(L, protected_bm);
        if (g_io_accounting_enabled && g_cur_decode_token >= 0) {
            int32_t ev = L->last_evicted_token[eid];
            if (ev >= 0 && g_cur_decode_token >= ev) {
                int32_t dist = g_cur_decode_token - ev;
                if (dist <= 16) g_tok_prefetch_rereads_16++;
                if (dist <= 64) g_tok_prefetch_rereads_64++;
            }
            L->last_evicted_token[eid] = -1;
        }
        lru_insert_eid_slot(L, eid, slot);
        rev_missing[i] = eid;
        assigned_slots[i] = (int32_t)slot;
    }

    if (g_io_accounting_enabled) {
        g_tok_prefetch_loads += (uint64_t)n_missing;
    }
    L->decode_prefetch_bytes += (uint64_t)n_missing * EXPERT_BYTES;
    dispatch_async_batch(layer_idx, L, rev_missing, assigned_slots, n_missing);
}

int c_layer_step_decode_ex(int layer_idx, const uint32_t *eids, int k, int async_io) {
    if (layer_idx < 0 || layer_idx >= MAX_LAYERS) return 0;
    wait_async_batch();
    LayerCache *L = &g_layers[layer_idx];

    L->decode_requests += (uint64_t)k;

    uint64_t actual_bm[8] = {0};
    for (int i = 0; i < k; ++i) {
        uint32_t eid = eids[i];
        actual_bm[eid >> 6] |= (1ULL << (eid & 63));
    }

    int32_t demand_missing[16];
    int n_missing = 0;

    if (L->has_pre_prefetch_snapshot) {
        uint64_t overlap_pred = 0;
        uint64_t tot_actual = 0;
        uint64_t tot_pred = 0;
        uint64_t miss_hits = 0;
        uint64_t tot_actual_misses = 0;
        uint64_t tot_fetched = 0;

        for (int b = 0; b < 8; ++b) {
            uint64_t act = actual_bm[b];
            uint64_t prd = L->last_pred_bitmap[b];
            uint64_t pre = L->pre_prefetch_bitmap[b];
            uint64_t fet = L->last_prefetched_bitmap[b];
            uint64_t act_miss = act & ~pre;

            overlap_pred += (uint64_t)__builtin_popcountll(act & prd);
            tot_actual += (uint64_t)__builtin_popcountll(act);
            tot_pred += (uint64_t)__builtin_popcountll(prd);

            miss_hits += (uint64_t)__builtin_popcountll(act_miss & fet);
            tot_actual_misses += (uint64_t)__builtin_popcountll(act_miss);
            tot_fetched += (uint64_t)__builtin_popcountll(fet);
        }

        L->pred_topm_hits += overlap_pred;
        L->pred_topm_total_actual += tot_actual;
        L->pred_topm_total_pred += tot_pred;

        L->miss_pref_hits += miss_hits;
        L->miss_pref_total_actual_misses += tot_actual_misses;
        L->miss_pref_total_fetched += tot_fetched;

        uint64_t tok_lru_h = 0;
        uint64_t tok_pref_h = 0;
        for (int i = 0; i < k; ++i) {
            uint32_t eid = eids[i];
            if (L->eid_to_slot[eid] >= 0) {
                if (L->pre_prefetch_bitmap[eid >> 6] & (1ULL << (eid & 63))) {
                    L->decode_lru_hits++;
                    tok_lru_h++;
                } else {
                    L->decode_prefetch_hits++;
                    tok_pref_h++;
                }
            } else {
                L->decode_demand_misses++;
                demand_missing[n_missing++] = (int32_t)eid;
            }
        }
        if (g_io_accounting_enabled) {
            g_tok_lru_hits += tok_lru_h;
            g_tok_prefetch_hits += tok_pref_h;
            if (tot_fetched >= tok_pref_h) {
                g_tok_wasted_prefetches += (tot_fetched - tok_pref_h);
            }
            g_tok_demand_misses += (uint64_t)n_missing;
        }
        L->has_pre_prefetch_snapshot = 0;
    } else {
        uint64_t tok_lru_h = 0;
        for (int i = 0; i < k; ++i) {
            uint32_t eid = eids[i];
            if (L->eid_to_slot[eid] >= 0) {
                L->decode_lru_hits++;
                tok_lru_h++;
            } else {
                L->decode_demand_misses++;
                demand_missing[n_missing++] = (int32_t)eid;
            }
        }
        if (g_io_accounting_enabled) {
            g_tok_lru_hits += tok_lru_h;
            g_tok_demand_misses += (uint64_t)n_missing;
        }
    }

    int32_t demand_slots[16];
    if (n_missing > 0) {
        for (int i = 0; i < n_missing; ++i) {
            int32_t eid = demand_missing[i];
            int16_t slot = lru_alloc_slot(L, actual_bm);
            if (g_io_accounting_enabled && g_cur_decode_token >= 0) {
                int32_t ev = L->last_evicted_token[eid];
                if (ev >= 0 && g_cur_decode_token >= ev) {
                    int32_t dist = g_cur_decode_token - ev;
                    if (dist <= 16) g_tok_demand_rereads_16++;
                    if (dist <= 64) g_tok_demand_rereads_64++;
                }
                L->last_evicted_token[eid] = -1;
            }
            lru_insert_eid_slot(L, eid, slot);
            demand_slots[i] = (int32_t)slot;
        }
        L->decode_demand_bytes += (uint64_t)n_missing * EXPERT_BYTES;
    }

    uint32_t *out_slots = L->slot_inds_ptr;
    for (int i = 0; i < k; ++i) {
        uint32_t eid = eids[i];
        int16_t slot = L->eid_to_slot[eid];
        lru_move_to_mru(L, slot);
        out_slots[i] = (uint32_t)slot;
    }

    if (n_missing > 0) {
        if (async_io) {
            dispatch_async_batch(layer_idx, L, demand_missing, demand_slots, n_missing);
        } else {
            dispatch_sync_batch(L, demand_missing, demand_slots, n_missing);
        }
    }

    return n_missing;
}

int c_layer_step_decode(int layer_idx, const uint32_t *eids, int k) {
    return c_layer_step_decode_ex(layer_idx, eids, k, 0);
}

/* Fast register-arg wrapper for K=10 decode steps */
int c_layer_step_decode10(
    int layer_idx,
    uint32_t e0, uint32_t e1, uint32_t e2, uint32_t e3, uint32_t e4,
    uint32_t e5, uint32_t e6, uint32_t e7, uint32_t e8, uint32_t e9
) {
    uint32_t eids[10] = {e0, e1, e2, e3, e4, e5, e6, e7, e8, e9};
    return c_layer_step_decode_ex(layer_idx, eids, 10, 0);
}

int c_layer_step_decode10_async(
    int layer_idx,
    uint32_t e0, uint32_t e1, uint32_t e2, uint32_t e3, uint32_t e4,
    uint32_t e5, uint32_t e6, uint32_t e7, uint32_t e8, uint32_t e9
) {
    uint32_t eids[10] = {e0, e1, e2, e3, e4, e5, e6, e7, e8, e9};
    return c_layer_step_decode_ex(layer_idx, eids, 10, 1);
}

void c_layer_load_sync(
    int layer_idx,
    const int32_t *eids,
    const int32_t *slots,
    int count,
    int is_prefill
) {
    if (layer_idx < 0 || layer_idx >= MAX_LAYERS || count <= 0) return;
    wait_async_batch();
    LayerCache *L = &g_layers[layer_idx];
    if (is_prefill) {
        L->prefill_demand_bytes += (uint64_t)count * EXPERT_BYTES;
    }
    dispatch_sync_batch(L, eids, slots, count);
}

void c_layer_get_lru_state(
    int layer_idx,
    int32_t *out_eids,
    int32_t *out_slots,
    int *out_count
) {
    if (layer_idx < 0 || layer_idx >= MAX_LAYERS) {
        *out_count = 0;
        return;
    }
    wait_async_batch();
    LayerCache *L = &g_layers[layer_idx];
    int cnt = 0;
    int16_t cur = L->lru_head;
    while (cur >= 0) {
        out_eids[cnt] = (int32_t)L->slot_to_eid[cur];
        out_slots[cnt] = (int32_t)cur;
        cnt++;
        cur = L->next_slot[cur];
    }
    *out_count = cnt;
}

void c_layer_set_lru_state(
    int layer_idx,
    const int32_t *ordered_eids,
    const int32_t *ordered_slots,
    int count,
    uint64_t add_prefill_requests,
    uint64_t add_prefill_hits,
    uint64_t add_prefill_misses
) {
    if (layer_idx < 0 || layer_idx >= MAX_LAYERS) return;
    wait_async_batch();
    LayerCache *L = &g_layers[layer_idx];
    L->prefill_requests += add_prefill_requests;
    L->prefill_lru_hits += add_prefill_hits;
    L->prefill_demand_misses += add_prefill_misses;

    for (int i = 0; i < NUM_EXPERTS; ++i) {
        L->eid_to_slot[i] = -1;
        L->slot_to_eid[i] = -1;
        L->prev_slot[i] = -1;
        L->next_slot[i] = -1;
    }
    L->lru_head = -1;
    L->lru_tail = -1;
    L->has_pre_prefetch_snapshot = 0;

    uint8_t used_slot[NUM_EXPERTS] = {0};
    for (int i = 0; i < count; ++i) {
        int32_t eid = ordered_eids[i];
        int16_t slot = (int16_t)ordered_slots[i];
        used_slot[slot] = 1;
        if (eid >= 0 && eid < NUM_EXPERTS) {
            L->last_evicted_token[eid] = -1;
        }
        lru_insert_eid_slot(L, eid, slot);
    }

    L->free_count = 0;
    for (int s = L->capacity - 1; s >= 0; --s) {
        if (!used_slot[s]) {
            L->free_slots[L->free_count++] = (int16_t)s;
        }
    }
}

void c_layer_get_stats(int layer_idx, uint64_t *out_stats) {
    if (layer_idx < 0 || layer_idx >= MAX_LAYERS) return;
    LayerCache *L = &g_layers[layer_idx];
    out_stats[0] = L->decode_requests;
    out_stats[1] = L->decode_lru_hits;
    out_stats[2] = L->decode_prefetch_hits;
    out_stats[3] = L->decode_demand_misses;
    out_stats[4] = L->decode_demand_bytes;
    out_stats[5] = L->decode_prefetch_bytes;
    out_stats[6] = L->prefill_requests;
    out_stats[7] = L->prefill_lru_hits;
    out_stats[8] = L->prefill_demand_misses;
    out_stats[9] = L->prefill_demand_bytes;
    out_stats[10] = L->pred_topm_hits;
    out_stats[11] = L->pred_topm_total_actual;
    out_stats[12] = L->pred_topm_total_pred;
    out_stats[13] = L->miss_pref_hits;
    out_stats[14] = L->miss_pref_total_actual_misses;
    out_stats[15] = L->miss_pref_total_fetched;
    out_stats[16] = L->uncoalesced_tensor_reads;
    out_stats[17] = L->coalesced_preadv_calls;
}

void c_layer_start_prefetch10(
    int layer_idx,
    int max_fetch,
    int32_t e0, int32_t e1, int32_t e2, int32_t e3, int32_t e4,
    int32_t e5, int32_t e6, int32_t e7, int32_t e8, int32_t e9
) {
    int32_t preds[10] = {e0, e1, e2, e3, e4, e5, e6, e7, e8, e9};
    c_layer_start_prefetch(layer_idx, preds, 10, max_fetch);
}

void c_layer_start_prefetch12(
    int layer_idx,
    int max_fetch,
    int32_t e0, int32_t e1, int32_t e2, int32_t e3, int32_t e4, int32_t e5,
    int32_t e6, int32_t e7, int32_t e8, int32_t e9, int32_t e10, int32_t e11
) {
    int32_t preds[12] = {e0, e1, e2, e3, e4, e5, e6, e7, e8, e9, e10, e11};
    c_layer_start_prefetch(layer_idx, preds, 12, max_fetch);
}

void c_get_decode_totals(int num_layers, uint64_t *out6) {
    uint64_t lru_h = 0, pref_h = 0, dem_m = 0, req = 0, dem_b = 0, pref_b = 0;
    if (num_layers > MAX_LAYERS) num_layers = MAX_LAYERS;
    for (int il = 1; il < num_layers; ++il) {
        LayerCache *L = &g_layers[il];
        lru_h += L->decode_lru_hits;
        pref_h += L->decode_prefetch_hits;
        dem_m += L->decode_demand_misses;
        req += L->decode_requests;
        dem_b += L->decode_demand_bytes;
        pref_b += L->decode_prefetch_bytes;
    }
    out6[0] = lru_h;
    out6[1] = pref_h;
    out6[2] = dem_m;
    out6[3] = req;
    out6[4] = dem_b;
    out6[5] = pref_b;
}

void c_set_io_accounting(int enabled) {
    wait_async_batch();
    g_io_accounting_enabled = enabled ? 1 : 0;
    if (!g_io_accounting_enabled) {
        g_cur_decode_token = -1;
    }
}

void c_begin_decode_token(int token_idx) {
    (void)token_idx;
    wait_async_batch();
    g_cur_decode_token = g_cache_token_step++;
    g_tok_lru_hits = 0;
    g_tok_demand_misses = 0;
    g_tok_prefetch_loads = 0;
    g_tok_prefetch_hits = 0;
    g_tok_wasted_prefetches = 0;
    g_tok_demand_rereads_16 = 0;
    g_tok_demand_rereads_64 = 0;
    g_tok_prefetch_rereads_16 = 0;
    g_tok_prefetch_rereads_64 = 0;
    atomic_store_explicit(&g_tok_pread_bytes, 0ULL, memory_order_relaxed);
}

void c_end_decode_token(uint64_t *out10) {
    wait_async_batch();
    out10[0] = g_tok_lru_hits;
    out10[1] = g_tok_demand_misses;
    out10[2] = g_tok_prefetch_loads;
    out10[3] = g_tok_prefetch_hits;
    out10[4] = g_tok_wasted_prefetches;
    out10[5] = g_tok_demand_rereads_16;
    out10[6] = g_tok_demand_rereads_64;
    out10[7] = g_tok_prefetch_rereads_16;
    out10[8] = g_tok_prefetch_rereads_64;
    out10[9] = atomic_load_explicit(&g_tok_pread_bytes, memory_order_acquire);
}

uint64_t c_get_native_buffer_bytes(void) {
    return (uint64_t)(sizeof(g_layers) + sizeof(g_batch_ring));
}

static int g_split_n_hit[MAX_LAYERS];
static int g_split_n_miss[MAX_LAYERS];
static int g_split_k[MAX_LAYERS];

void c_layer_wait_demand(int layer_idx) {
    if (layer_idx < 0 || layer_idx >= MAX_LAYERS) return;
    IoBatch *Bp = g_layer_prefetch_batch[layer_idx];
    if (Bp != NULL) {
        g_layer_prefetch_batch[layer_idx] = NULL;
        wait_for_batch(Bp);
    }
    IoBatch *Bd = g_layer_demand_batch[layer_idx];
    if (Bd != NULL) {
        g_layer_demand_batch[layer_idx] = NULL;
        wait_for_batch(Bd);
    }
}

void c_layer_get_split_inds(
    int layer_idx,
    uint32_t *out_hit_slots,
    uint32_t *out_miss_slots,
    int32_t *out_inv_order
) {
    if (layer_idx < 0 || layer_idx >= MAX_LAYERS) return;
    int nh = g_split_n_hit[layer_idx];
    int nm = g_split_n_miss[layer_idx];
    int k = g_split_k[layer_idx];
    if (nh > 0 && out_hit_slots != NULL) {
        memcpy(out_hit_slots, g_split_hit_slots[layer_idx], (size_t)nh * sizeof(uint32_t));
    }
    if (nm > 0 && out_miss_slots != NULL) {
        memcpy(out_miss_slots, g_split_miss_slots[layer_idx], (size_t)nm * sizeof(uint32_t));
    }
    if (k > 0 && out_inv_order != NULL) {
        memcpy(out_inv_order, g_split_inv_order[layer_idx], (size_t)k * sizeof(int32_t));
    }
}

/*
 * Single C entry point per layer for native_sched decode:
 * 1. Performs LRU lookup/update for current layer `layer_idx` eids[0..k-1], fills `slot_inds_ptr`,
 *    and prepares coalesced demand-miss batch (if n_miss > 0).
 * 2. Performs LRU lookup/allocation for `next_layer` predicted experts pred[0..n_pred-1] capped at
 *    `prefetch_cap`, and prepares coalesced prefetch batch (if n_pref_missing > 0).
 * 3. Publishes both batches to the worker pool in a single wakeup with strict demand-first priority
 *    (current-layer prior prefetch -> current-layer demand -> next-layer prefetch).
 * 4. Returns n_miss for `layer_idx`. Caller waits for `layer_idx` demand reads via
 *    `c_layer_wait_demand(layer_idx)` before dispatching the GPU expert matmul.
 */
int c_layer_decode_step_sched(
    int layer_idx,
    const int32_t *eids,
    int k,
    const int32_t *pred,
    int n_pred,
    int next_layer,
    int prefetch_cap,
    int flags
) {
    /* Ensure any legacy async batch or previous layer demand batch is drained */
    if (g_async_batch_ptr != NULL) {
        IoBatch *Ba = g_async_batch_ptr;
        g_async_batch_ptr = NULL;
        if (layer_idx > 0 && layer_idx < MAX_LAYERS && g_layer_prefetch_batch[layer_idx] == NULL) {
            g_layer_prefetch_batch[layer_idx] = Ba;
        } else {
            wait_for_batch(Ba);
        }
    }
    IoBatch *prev_dem = atomic_load_explicit(&g_demand_batch, memory_order_acquire);
    if (prev_dem != NULL) {
        wait_for_batch(prev_dem);
    }
    IoBatch *prev_prior = atomic_load_explicit(&g_prior_prefetch_batch, memory_order_acquire);
    if (prev_prior != NULL &&
        (layer_idx <= 0 || layer_idx >= MAX_LAYERS || prev_prior != g_layer_prefetch_batch[layer_idx])) {
        wait_for_batch(prev_prior);
    }

    int n_missing = 0;
    IoBatch *B_dem = NULL;
    IoBatch *B_pref = NULL;

    /* Step 1: Current layer LRU lookup/update and demand-miss batch construction */
    if (layer_idx > 0 && layer_idx < MAX_LAYERS && eids != NULL && k > 0) {
        if (k > 16) k = 16;
        LayerCache *L = &g_layers[layer_idx];
        L->decode_requests += (uint64_t)k;

        uint64_t actual_bm[8] = {0};
        for (int i = 0; i < k; ++i) {
            int32_t eid = eids[i];
            if (eid >= 0 && eid < NUM_EXPERTS) {
                actual_bm[eid >> 6] |= (1ULL << (eid & 63));
            }
        }

        /* Protect both current-layer active experts and any in-flight prefetched experts for this layer */
        uint64_t protect_bm[8];
        for (int b = 0; b < 8; ++b) {
            protect_bm[b] = actual_bm[b] | (L->has_pre_prefetch_snapshot ? L->last_prefetched_bitmap[b] : 0ULL);
        }

        int32_t demand_missing[16];
        uint8_t is_hit[16] = {0};

        if (L->has_pre_prefetch_snapshot) {
            uint64_t overlap_pred = 0;
            uint64_t tot_actual = 0;
            uint64_t tot_pred = 0;
            uint64_t miss_hits = 0;
            uint64_t tot_actual_misses = 0;
            uint64_t tot_fetched = 0;

            for (int b = 0; b < 8; ++b) {
                uint64_t act = actual_bm[b];
                uint64_t prd = L->last_pred_bitmap[b];
                uint64_t pre = L->pre_prefetch_bitmap[b];
                uint64_t fet = L->last_prefetched_bitmap[b];
                uint64_t act_miss = act & ~pre;

                overlap_pred += (uint64_t)__builtin_popcountll(act & prd);
                tot_actual += (uint64_t)__builtin_popcountll(act);
                tot_pred += (uint64_t)__builtin_popcountll(prd);

                miss_hits += (uint64_t)__builtin_popcountll(act_miss & fet);
                tot_actual_misses += (uint64_t)__builtin_popcountll(act_miss);
                tot_fetched += (uint64_t)__builtin_popcountll(fet);
            }

            L->pred_topm_hits += overlap_pred;
            L->pred_topm_total_actual += tot_actual;
            L->pred_topm_total_pred += tot_pred;

            L->miss_pref_hits += miss_hits;
            L->miss_pref_total_actual_misses += tot_actual_misses;
            L->miss_pref_total_fetched += tot_fetched;

            uint64_t tok_lru_h = 0;
            uint64_t tok_pref_h = 0;
            for (int i = 0; i < k; ++i) {
                int32_t eid = eids[i];
                if (L->eid_to_slot[eid] >= 0) {
                    is_hit[i] = 1;
                    if (L->pre_prefetch_bitmap[eid >> 6] & (1ULL << (eid & 63))) {
                        L->decode_lru_hits++;
                        tok_lru_h++;
                    } else {
                        L->decode_prefetch_hits++;
                        tok_pref_h++;
                    }
                } else {
                    L->decode_demand_misses++;
                    demand_missing[n_missing++] = eid;
                }
            }
            if (g_io_accounting_enabled) {
                g_tok_lru_hits += tok_lru_h;
                g_tok_prefetch_hits += tok_pref_h;
                if (tot_fetched >= tok_pref_h) {
                    g_tok_wasted_prefetches += (tot_fetched - tok_pref_h);
                }
                g_tok_demand_misses += (uint64_t)n_missing;
            }
            L->has_pre_prefetch_snapshot = 0;
        } else {
            uint64_t tok_lru_h = 0;
            for (int i = 0; i < k; ++i) {
                int32_t eid = eids[i];
                if (L->eid_to_slot[eid] >= 0) {
                    is_hit[i] = 1;
                    L->decode_lru_hits++;
                    tok_lru_h++;
                } else {
                    L->decode_demand_misses++;
                    demand_missing[n_missing++] = eid;
                }
            }
            if (g_io_accounting_enabled) {
                g_tok_lru_hits += tok_lru_h;
                g_tok_demand_misses += (uint64_t)n_missing;
            }
        }

        int32_t demand_slots[16];
        if (n_missing > 0) {
            for (int i = 0; i < n_missing; ++i) {
                int32_t eid = demand_missing[i];
                int16_t slot = lru_alloc_slot(L, protect_bm);
                if (g_io_accounting_enabled && g_cur_decode_token >= 0) {
                    int32_t ev = L->last_evicted_token[eid];
                    if (ev >= 0 && g_cur_decode_token >= ev) {
                        int32_t dist = g_cur_decode_token - ev;
                        if (dist <= 16) g_tok_demand_rereads_16++;
                        if (dist <= 64) g_tok_demand_rereads_64++;
                    }
                    L->last_evicted_token[eid] = -1;
                }
                lru_insert_eid_slot(L, eid, slot);
                demand_slots[i] = (int32_t)slot;
            }
            L->decode_demand_bytes += (uint64_t)n_missing * EXPERT_BYTES;
        }

        uint32_t *out_slots = L->slot_inds_ptr;
        for (int i = 0; i < k; ++i) {
            int32_t eid = eids[i];
            int16_t slot = L->eid_to_slot[eid];
            lru_move_to_mru(L, slot);
            out_slots[i] = (uint32_t)slot;
        }

        if ((flags & SCHED_FLAG_SPLIT) && n_missing > 0 && n_missing < k) {
            int h_idx = 0;
            int m_idx = 0;
            int n_hit = k - n_missing;
            for (int i = 0; i < k; ++i) {
                if (is_hit[i]) {
                    g_split_hit_slots[layer_idx][h_idx] = out_slots[i];
                    g_split_inv_order[layer_idx][i] = h_idx;
                    h_idx++;
                } else {
                    g_split_miss_slots[layer_idx][m_idx] = out_slots[i];
                    g_split_inv_order[layer_idx][i] = n_hit + m_idx;
                    m_idx++;
                }
            }
            g_split_n_hit[layer_idx] = n_hit;
            g_split_n_miss[layer_idx] = n_missing;
            g_split_k[layer_idx] = k;
        }

        if (n_missing > 0) {
            B_dem = alloc_ring_batch();
            int n_dem_tasks = build_coalesced_batch(L, B_dem, demand_missing, demand_slots, n_missing);
            atomic_store_explicit(&B_dem->next_task, 0, memory_order_relaxed);
            atomic_store_explicit(&B_dem->remaining, n_dem_tasks, memory_order_relaxed);
            atomic_store_explicit(&B_dem->num_tasks, n_dem_tasks, memory_order_release);
            g_layer_demand_batch[layer_idx] = B_dem;
        }
    }

    /* Step 2: Next-layer speculative prefetch LRU setup and batch construction (queued AFTER demand) */
    if (!(flags & SCHED_FLAG_NO_PREFETCH) &&
        next_layer > 0 && next_layer < MAX_LAYERS &&
        pred != NULL && n_pred > 0 && prefetch_cap > 0) {
        if (g_layer_prefetch_batch[next_layer] != NULL) {
            IoBatch *Bo = g_layer_prefetch_batch[next_layer];
            g_layer_prefetch_batch[next_layer] = NULL;
            wait_for_batch(Bo);
        }

        LayerCache *Ln = &g_layers[next_layer];
        memset(Ln->pre_prefetch_bitmap, 0, sizeof(Ln->pre_prefetch_bitmap));
        for (int e = 0; e < NUM_EXPERTS; ++e) {
            if (Ln->eid_to_slot[e] >= 0) {
                Ln->pre_prefetch_bitmap[e >> 6] |= (1ULL << (e & 63));
            }
        }
        memset(Ln->last_pred_bitmap, 0, sizeof(Ln->last_pred_bitmap));
        memset(Ln->last_prefetched_bitmap, 0, sizeof(Ln->last_prefetched_bitmap));
        Ln->has_pre_prefetch_snapshot = 1;

        uint64_t next_prot_bm[8] = {0};
        int32_t missing_pred[32];
        int n_pref_missing = 0;
        int cap_fetch = (prefetch_cap > 0 && prefetch_cap < 32) ? prefetch_cap : 32;

        for (int i = 0; i < n_pred; ++i) {
            int32_t eid = pred[i];
            if (eid < 0 || eid >= NUM_EXPERTS) continue;
            Ln->last_pred_bitmap[eid >> 6] |= (1ULL << (eid & 63));
            next_prot_bm[eid >> 6] |= (1ULL << (eid & 63));
            if (Ln->eid_to_slot[eid] < 0 && n_pref_missing < cap_fetch &&
                !(Ln->last_prefetched_bitmap[eid >> 6] & (1ULL << (eid & 63)))) {
                missing_pred[n_pref_missing++] = eid;
                Ln->last_prefetched_bitmap[eid >> 6] |= (1ULL << (eid & 63));
            }
        }

        if (n_pref_missing > 0) {
            int32_t rev_missing[32];
            int32_t assigned_slots[32];
            for (int i = 0; i < n_pref_missing; ++i) {
                int32_t eid = missing_pred[n_pref_missing - 1 - i];
                int16_t slot = lru_alloc_slot(Ln, next_prot_bm);
                if (g_io_accounting_enabled && g_cur_decode_token >= 0) {
                    int32_t ev = Ln->last_evicted_token[eid];
                    if (ev >= 0 && g_cur_decode_token >= ev) {
                        int32_t dist = g_cur_decode_token - ev;
                        if (dist <= 16) g_tok_prefetch_rereads_16++;
                        if (dist <= 64) g_tok_prefetch_rereads_64++;
                    }
                    Ln->last_evicted_token[eid] = -1;
                }
                lru_insert_eid_slot(Ln, eid, slot);
                rev_missing[i] = eid;
                assigned_slots[i] = (int32_t)slot;
            }

            if (g_io_accounting_enabled) {
                g_tok_prefetch_loads += (uint64_t)n_pref_missing;
            }
            Ln->decode_prefetch_bytes += (uint64_t)n_pref_missing * EXPERT_BYTES;

            B_pref = alloc_ring_batch();
            int n_pref_tasks = build_coalesced_batch(Ln, B_pref, rev_missing, assigned_slots, n_pref_missing);
            atomic_store_explicit(&B_pref->next_task, 0, memory_order_relaxed);
            atomic_store_explicit(&B_pref->remaining, n_pref_tasks, memory_order_relaxed);
            atomic_store_explicit(&B_pref->num_tasks, n_pref_tasks, memory_order_release);
            g_layer_prefetch_batch[next_layer] = B_pref;
        }
    }

    /* Step 3: Publish demand (high priority) and next-layer prefetch (low priority) in a single wakeup */
    IoBatch *cur_layer_pref = (layer_idx > 0 && layer_idx < MAX_LAYERS) ? g_layer_prefetch_batch[layer_idx] : NULL;
    if (B_dem != NULL || B_pref != NULL || cur_layer_pref != NULL) {
        pthread_mutex_lock(&g_pool_mutex);
        if (cur_layer_pref != NULL &&
            atomic_load_explicit(&cur_layer_pref->remaining, memory_order_acquire) > 0) {
            atomic_store_explicit(&g_prior_prefetch_batch, cur_layer_pref, memory_order_release);
        }
        if (B_dem != NULL) {
            atomic_store_explicit(&g_demand_batch, B_dem, memory_order_release);
        }
        if (B_pref != NULL || atomic_load_explicit(&g_current_batch, memory_order_relaxed) == cur_layer_pref) {
            atomic_store_explicit(&g_current_batch, B_pref, memory_order_release);
        }
        if (B_dem != NULL || B_pref != NULL) {
            g_work_seq++;
            pthread_cond_broadcast(&g_work_cond);
        }
        pthread_mutex_unlock(&g_pool_mutex);
    }

    if (layer_idx > 0 && layer_idx < MAX_LAYERS) {
        if ((flags & SCHED_FLAG_SYNC_DEMAND) || n_missing == 0) {
            /* When n_missing == 0, ensure any in-flight prefetch for layer_idx is complete now
             * so Python does not need a second C call on 100%-hit layers. */
            c_layer_wait_demand(layer_idx);
        } else if ((flags & SCHED_FLAG_SPLIT) && cur_layer_pref != NULL) {
            /* When splitting hits and misses, ensure prefetch-hit slots for layer_idx are ready
             * before Python dispatches y_hit. */
            g_layer_prefetch_batch[layer_idx] = NULL;
            wait_for_batch(cur_layer_pref);
        }
    }

    return n_missing;
}



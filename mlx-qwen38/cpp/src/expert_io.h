#pragma once
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

enum {
    SCHED_FLAG_SYNC_DEMAND = 1,
    SCHED_FLAG_NO_PREFETCH = 2,
    SCHED_FLAG_SPLIT       = 4
};

void c_io_init(int num_threads);
void c_io_shutdown(void);

void c_layer_reset_cache(int layer_idx);
void c_layer_reset_stats(int layer_idx);

void c_layer_init(
    int layer_idx,
    int fd,
    int capacity,
    const uint64_t *base_offsets,
    const uint64_t *row_bytes,
    const uint64_t *base_ptrs,
    uint64_t slot_inds_ptr);

void c_layer_wait_prefetch(int layer_idx);
void c_layer_wait_demand(int layer_idx);

void c_layer_start_prefetch(
    int layer_idx,
    const int32_t *pred_eids,
    int pred_count,
    int max_fetch);

void c_layer_start_prefetch10(
    int layer_idx,
    int max_fetch,
    int32_t e0, int32_t e1, int32_t e3_or_2, int32_t e3, int32_t e4,
    int32_t e5, int32_t e6, int32_t e7, int32_t e8, int32_t e9);

void c_layer_start_prefetch12(
    int layer_idx,
    int max_fetch,
    int32_t e0, int32_t e1, int32_t e2, int32_t e3, int32_t e4, int32_t e5,
    int32_t e6, int32_t e7, int32_t e8, int32_t e9, int32_t e10, int32_t e11);

int c_layer_step_decode_ex(int layer_idx, const uint32_t *eids, int k, int async_io);
int c_layer_step_decode(int layer_idx, const uint32_t *eids, int k);

int c_layer_step_decode10(
    int layer_idx,
    uint32_t e0, uint32_t e1, uint32_t e2, uint32_t e3, uint32_t e4,
    uint32_t e5, uint32_t e6, uint32_t e7, uint32_t e8, uint32_t e9);

int c_layer_step_decode10_async(
    int layer_idx,
    uint32_t e0, uint32_t e1, uint32_t e2, uint32_t e3, uint32_t e4,
    uint32_t e5, uint32_t e6, uint32_t e7, uint32_t e8, uint32_t e9);

void c_layer_load_sync(
    int layer_idx,
    const int32_t *eids,
    const int32_t *slots,
    int count,
    int is_prefill);

void c_layer_get_lru_state(
    int layer_idx,
    int32_t *out_eids,
    int32_t *out_slots,
    int *out_count);

void c_layer_set_lru_state(
    int layer_idx,
    const int32_t *ordered_eids,
    const int32_t *ordered_slots,
    int count,
    uint64_t add_prefill_requests,
    uint64_t add_prefill_hits,
    uint64_t add_prefill_misses);

void c_layer_get_stats(int layer_idx, uint64_t *out_stats);
void c_get_decode_totals(int num_layers, uint64_t *out6);
void c_set_io_accounting(int enabled);
void c_begin_decode_token(int token_idx);
void c_end_decode_token(uint64_t *out10);
uint64_t c_get_native_buffer_bytes(void);

void c_layer_get_split_inds(
    int layer_idx,
    uint32_t *out_hit_slots,
    uint32_t *out_miss_slots,
    int32_t *out_inv_order);

int c_layer_decode_step_sched(
    int layer_idx,
    const int32_t *eids,
    int k,
    const int32_t *pred,
    int n_pred,
    int next_layer,
    int prefetch_cap,
    int flags);

#ifdef __cplusplus
}
#endif

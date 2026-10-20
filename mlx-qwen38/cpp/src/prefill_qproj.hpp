// Prefill expert projection for the slot-cache MoE path (C++ twin of mlx-qwen38/prefill_qproj.py).
// k_hdr fix: the sorted gather_qmm path only passes slots 0..k_hdr-1 (k_hdr = max(rows,16)/4) to the
// kernel, so it is taken only when every local slot of the chunk (0..n_chunk-1) is < k_hdr; otherwise
// the unsorted kernel over all slots is used. Unit test: tests/test_prefill_qproj.cpp.
#pragma once
#include <mlx/mlx.h>
#include <algorithm>
#include <optional>
#include <vector>

namespace qwen38 {
namespace mx = mlx::core;

inline mx::array prefill_qproj(const mx::array& w,
                               const mx::array& s,
                               const mx::array& b,
                               const mx::array& x_in,
                               const mx::array& idx_in,
                               bool use_sorted,
                               int n_chunk,
                               int group_size,
                               int bits) {
    int k_hdr_chk = std::max(1, std::min(static_cast<int>(w.shape(0)), std::max(static_cast<int>(x_in.shape(0)), 16) / 4));
    if (use_sorted && n_chunk <= k_hdr_chk) {
        int orig_n = x_in.shape(0);
        mx::array x_use = x_in;
        mx::array idx_use = idx_in;
        if (orig_n < 16) {
            int pad_n = 16 - orig_n;
            mx::Shape x_start(x_in.ndim(), 0);
            x_start[0] = orig_n - 1;
            mx::Shape x_stop = x_in.shape();
            auto last_x = mx::slice(x_in, x_start, x_stop);
            std::vector<mx::array> x_cat = {x_in};
            for (int p = 0; p < pad_n; ++p) x_cat.push_back(last_x);
            x_use = mx::concatenate(x_cat, 0);

            auto last_i = mx::slice(idx_in, {orig_n - 1}, {orig_n});
            std::vector<mx::array> i_cat = {idx_in};
            for (int p = 0; p < pad_n; ++p) i_cat.push_back(last_i);
            idx_use = mx::concatenate(i_cat, 0);
        }
        int k_hdr = std::max(1, std::min(static_cast<int>(w.shape(0)), static_cast<int>(x_use.shape(0)) / 4));
        auto slice_hdr = [&](const mx::array& arr) {
            mx::Shape stop = arr.shape();
            stop[0] = k_hdr;
            return mx::slice(arr, mx::Shape(arr.ndim(), 0), stop);
        };
        auto out = mx::gather_qmm(
            x_use, slice_hdr(w), slice_hdr(s), slice_hdr(b),
            std::nullopt, idx_use, true, group_size, bits, "affine", true);
        if (orig_n < 16) {
            mx::Shape o_stop = out.shape();
            o_stop[0] = orig_n;
            out = mx::slice(out, mx::Shape(out.ndim(), 0), o_stop);
        }
        return out;
    }
    return mx::gather_qmm(
        x_in, w, s, b, std::nullopt, idx_in,
        true, group_size, bits, "affine", false);
}

}  // namespace qwen38

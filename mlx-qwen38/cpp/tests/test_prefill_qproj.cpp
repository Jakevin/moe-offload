// Fixture test for the k_hdr fix in src/prefill_qproj.hpp. Random weights only, no model needed.
//   ./build/test_prefill_qproj         fixed code: must PASS
//   ./build/test_prefill_qproj --old   negative control, pre-fix logic (n_chunk ignored): must FAIL
// Per chunk shape (rows, distinct experts): output matches a dequantized reference matmul, and when the
// chunk has more experts than k_hdr it is bit-identical to the unsorted gather_qmm over all slots.
#include "prefill_qproj.hpp"
#include <cstdio>
#include <cstring>
#include <random>

namespace mx = mlx::core;

int main(int argc, char** argv) {
    bool old = argc > 1 && std::strcmp(argv[1], "--old") == 0;
    std::printf("MLX %s\n", mx::version());
    const int CAP = 102, D = 1024, O = 384, GS = 64, BITS = 2;
    std::mt19937 g(1);
    mx::random::seed(0);
    auto W = mx::astype(mx::multiply(mx::random::normal({CAP, O, D}), mx::array(0.05f)), mx::bfloat16);
    auto qv = mx::quantize(W, GS, BITS);
    auto w = qv[0], s = qv[1], b = qv[2];
    auto deq = mx::dequantize(w, s, b, GS, BITS);
    mx::eval({w, s, b, deq});
    int shapes[][2] = {{63, 19}, {190, 55}, {404, 102}, {128, 33}, {99, 26}, {22, 9}, {73, 25}, {372, 102},
                       {61, 17}, {80, 25}, {587, 102}, {246, 39}, {64, 16}, {2000, 90}, {5, 2}, {1500, 102}};
    int fails = 0, n = 0;
    for (auto& sh : shapes) {
        int rows = sh[0], ne = sh[1];
        ++n;
        std::vector<uint32_t> idx(rows);
        for (int r = 0; r < rows; ++r) idx[r] = r < ne ? r : g() % ne;
        std::sort(idx.begin(), idx.end());
        auto ia = mx::array(idx.data(), {rows}, mx::uint32);
        auto x = mx::astype(mx::random::normal({rows, 1, D}), mx::bfloat16);
        // old logic == the fixed function with n_chunk forced to 0 (always takes the sorted path)
        auto out = qwen38::prefill_qproj(w, s, b, x, ia, true, old ? 0 : ne, GS, BITS);
        auto un = mx::gather_qmm(x, w, s, b, std::nullopt, ia, true, GS, BITS, "affine", false);
        auto ref = mx::squeeze(mx::matmul(mx::astype(x, mx::float32), mx::swapaxes(mx::take(deq, ia, 0), -1, -2)), 1);
        auto err = mx::max(mx::abs(mx::subtract(mx::reshape(mx::astype(out, mx::float32), {rows, O}), ref)), 1);
        auto tol = mx::multiply(mx::max(mx::abs(ref)), mx::array(0.02f));
        auto nbad = mx::sum(mx::greater(err, tol));
        auto eq = mx::array_equal(out, un);
        mx::eval({nbad, eq});
        int k_hdr = std::max(1, std::min(CAP, std::max(rows, 16) / 4));
        int over = 0;
        for (auto v : idx) over += v >= (uint32_t)k_hdr;
        int bad = nbad.item<int32_t>();
        bool e = eq.item<bool>();
        bool ok = bad == 0 && (ne <= k_hdr || e);
        fails += !ok;
        std::printf("rows=%5d experts=%3d k_hdr=%3d rows_slot>=k_hdr=%4d wrong_rows=%4d ==unsorted=%d %s\n",
                    rows, ne, k_hdr, over, bad, (int)e, ok ? "OK" : "FAIL");
    }
    std::printf("%s (%d/%d shapes failed)\n", fails ? "FAIL" : "PASS", fails, n);
    return fails ? 1 : 0;
}

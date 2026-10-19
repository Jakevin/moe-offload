#include "model.hpp"
#include <cmath>
#include <stdexcept>

namespace qwen38 {

static const char* kHcV2Header = R"(
using namespace metal;

template <int bits>
inline constexpr short hc_pack_factor() {
    return bits == 5 ? 8 : (bits == 6 ? 4 : 32 / bits);
}

template <int bits>
inline constexpr short hc_bytes_per_pack() {
    constexpr int power_of_2_bits = (bits & (bits - 1)) == 0;
    return power_of_2_bits ? 4 : (bits == 5 ? 5 : 3);
}

template <typename T, int N, int bits>
inline float hc_load_vector(const device T* x, thread float* xt) {
    float sum = 0.0f;
    if (bits == 4) {
        for (int i = 0; i < N; i += 4) {
            sum += x[i] + x[i + 1] + x[i + 2] + x[i + 3];
            xt[i] = x[i];
            xt[i + 1] = x[i + 1] / 16.0f;
            xt[i + 2] = x[i + 2] / 256.0f;
            xt[i + 3] = x[i + 3] / 4096.0f;
        }
    } else if (bits == 5) {
        for (int i = 0; i < N; i += 8) {
            sum += x[i] + x[i + 1] + x[i + 2] + x[i + 3]
                + x[i + 4] + x[i + 5] + x[i + 6] + x[i + 7];
            xt[i] = x[i];
            xt[i + 1] = x[i + 1] / 32.0f;
            xt[i + 2] = x[i + 2] / 4.0f;
            xt[i + 3] = x[i + 3] / 128.0f;
            xt[i + 4] = x[i + 4] / 16.0f;
            xt[i + 5] = x[i + 5] / 2.0f;
            xt[i + 6] = x[i + 6] / 64.0f;
            xt[i + 7] = x[i + 7] / 8.0f;
        }
    } else if (bits == 6) {
        for (int i = 0; i < N; i += 4) {
            sum += x[i] + x[i + 1] + x[i + 2] + x[i + 3];
            xt[i] = x[i];
            xt[i + 1] = x[i + 1] / 64.0f;
            xt[i + 2] = x[i + 2] / 16.0f;
            xt[i + 3] = x[i + 3] / 4.0f;
        }
    } else if (bits == 8) {
        for (int i = 0; i < N; ++i) {
            sum += x[i];
            xt[i] = x[i];
        }
    }
    return sum;
}

template <int N, int bits>
inline float hc_qdot(
    const device uint8_t* w,
    const thread float* xt,
    float scale,
    float bias,
    float sum
) {
    float accum = 0.0f;
    if (bits == 4) {
        const device uint16_t* ws = (const device uint16_t*)w;
        for (int i = 0; i < N / 4; ++i) {
            accum +=
                (xt[4 * i] * (ws[i] & 0x000f)
                 + xt[4 * i + 1] * (ws[i] & 0x00f0)
                 + xt[4 * i + 2] * (ws[i] & 0x0f00)
                 + xt[4 * i + 3] * (ws[i] & 0xf000));
        }
    } else if (bits == 5) {
        for (int i = 0; i < N / 8; ++i) {
            xt += 8 * i;
            w += 5 * i;
            accum += (w[0] & 0x1f) * xt[0];
            accum += (w[0] & 0xe0) * xt[1];
            accum += (w[1] & 0x3) * (xt[1] * 256.0f);
            accum += (w[1] & 0x7c) * xt[2];
            accum += (w[1] & 0x80) * xt[3];
            accum += (w[2] & 0xf) * (xt[3] * 256.0f);
            accum += (w[2] & 0xf0) * xt[4];
            accum += (w[3] & 0x1) * (xt[4] * 256.0f);
            accum += (w[3] & 0x3e) * xt[5];
            accum += (w[3] & 0xc0) * xt[6];
            accum += (w[4] & 0x7) * (xt[6] * 256.0f);
            accum += (w[4] & 0xf8) * xt[7];
        }
    } else if (bits == 6) {
        for (int i = 0; i < N / 4; ++i) {
            xt += 4 * i;
            w += 3 * i;
            accum += (w[0] & 0x3f) * xt[0];
            accum += (w[0] & 0xc0) * xt[1];
            accum += (w[1] & 0x0f) * (xt[1] * 256.0f);
            accum += (w[1] & 0xf0) * xt[2];
            accum += (w[2] & 0x03) * (xt[2] * 256.0f);
            accum += (w[2] & 0xfc) * xt[3];
        }
    } else if (bits == 8) {
        for (int i = 0; i < N; ++i) accum += xt[i] * w[i];
    }
    return scale * accum + sum * bias;
}

template <typename T, int N, int bits>
inline float hc_load_vector_tg(const threadgroup T* x, thread float* xt) {
    float sum = 0.0f;
    if (bits == 4) {
        for (int i = 0; i < N; i += 4) {
            sum += x[i] + x[i + 1] + x[i + 2] + x[i + 3];
            xt[i] = x[i];
            xt[i + 1] = x[i + 1] / 16.0f;
            xt[i + 2] = x[i + 2] / 256.0f;
            xt[i + 3] = x[i + 3] / 4096.0f;
        }
    } else if (bits == 5) {
        for (int i = 0; i < N; i += 8) {
            sum += x[i] + x[i + 1] + x[i + 2] + x[i + 3]
                + x[i + 4] + x[i + 5] + x[i + 6] + x[i + 7];
            xt[i] = x[i];
            xt[i + 1] = x[i + 1] / 32.0f;
            xt[i + 2] = x[i + 2] / 4.0f;
            xt[i + 3] = x[i + 3] / 128.0f;
            xt[i + 4] = x[i + 4] / 16.0f;
            xt[i + 5] = x[i + 5] / 2.0f;
            xt[i + 6] = x[i + 6] / 64.0f;
            xt[i + 7] = x[i + 7] / 8.0f;
        }
    } else if (bits == 6) {
        for (int i = 0; i < N; i += 4) {
            sum += x[i] + x[i + 1] + x[i + 2] + x[i + 3];
            xt[i] = x[i];
            xt[i + 1] = x[i + 1] / 64.0f;
            xt[i + 2] = x[i + 2] / 16.0f;
            xt[i + 3] = x[i + 3] / 4.0f;
        }
    } else if (bits == 8) {
        for (int i = 0; i < N; ++i) {
            sum += x[i];
            xt[i] = x[i];
        }
    }
    return sum;
}
)";

static const char* kHcNdnSource = R"(
    const uint gy = threadgroup_position_in_grid.y;
    const uint row = threadgroup_position_in_grid.z;
    const uint t = thread_index_in_threadgroup;
    const uint sg = simdgroup_index_in_threadgroup;
    const uint lane = thread_index_in_simdgroup;
    const uint s = gy & 3;
    const uint g = gy >> 2;
    constexpr int PER = (H + 255) / 256;
    constexpr int KS = 4;
    constexpr int SLICE = K / KS;
    constexpr int GROUPS = K / 64;
    constexpr int PR = R + HC;
    constexpr int ROWS_TG = 8 * RPS;
    constexpr int NDG = R / ROWS_TG;
    static_assert(HC == 4 && K == 4 * H && R % ROWS_TG == 0, "down K slice ks is stream ks");
    static_assert(PER <= NDG + INJ, "one stored strip per threadgroup");
    threadgroup T xs[H];
    threadgroup float npart[8];
    float v[PER];
    T y_mine = T(0.0f);
    T n_mine = T(0.0f);
    const size_t base = (size_t)row * K + (size_t)s * H;
    const device T* wp = w + (size_t)s * H;
    float wv[PER];
    for (int i = 0; i < PER; ++i) {
        const int k = t + i * 256;
        wv[i] = (k < H) ? float(wp[k]) : 0.0f;
    }
    const float e = eps[0];
    float ss = 0.0f;
    const device T* xp = x + base;
    for (int i = 0; i < PER; ++i) {
        const int k = t + i * 256;
        v[i] = (k < H) ? float(xp[k]) : 0.0f;
        ss += v[i] * v[i];
    }
    ss = simd_sum(ss);
    if (lane == 0) npart[sg] = ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float tot = 0.0f;
    for (int i = 0; i < 8; ++i) tot += npart[i];
    const float inv = metal::rsqrt(tot / float(H) + e);
    for (int i = 0; i < PER; ++i) {
        const int k = t + i * 256;
        if (k < H) {
            const T y = T(v[i] * inv * (1.0f + wv[i]));
            xs[k] = y;
            if (i == int(g)) n_mine = y;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const bool down_tg = g < NDG;
    const int ks = int(s);
    device float* pp = parts + ((size_t)row * KS + ks) * PR;
    if (down_tg) {
        constexpr int PF = hc_pack_factor<BITS_D>();
        constexpr int BP = hc_bytes_per_pack<BITS_D>();
        constexpr int ROW_BYTES = K * BP / PF;
        constexpr int PPT = 2;
        constexpr int VPT = PF * PPT;
        constexpr int BLOCK = VPT * 32;
        constexpr int SCALE_STEP = 64 / VPT;
        const int out_row = int(g) * ROWS_TG + int(sg) * RPS;
        const device uint8_t* wq = (const device uint8_t*)down_w
            + out_row * ROW_BYTES + ks * (SLICE * BP / PF) + int(lane) * PPT * BP;
        const device T* sp = down_s + out_row * GROUPS + ks * (SLICE / 64)
            + int(lane) / SCALE_STEP;
        const device T* bp = down_b + out_row * GROUPS + ks * (SLICE / 64)
            + int(lane) / SCALE_STEP;
        const threadgroup T* xp_tg = xs + int(lane) * VPT;
        float result[RPS] = {0.0f};
        float xv[VPT];
        constexpr int TAIL = SLICE % BLOCK;
        for (int k = 0; k < SLICE - TAIL; k += BLOCK) {
            float sum = hc_load_vector_tg<T, VPT, BITS_D>(xp_tg, xv);
            for (int r = 0; r < RPS; ++r) {
                result[r] += hc_qdot<VPT, BITS_D>(
                    wq + r * ROW_BYTES, xv,
                    float(sp[r * GROUPS]), float(bp[r * GROUPS]), sum);
            }
            wq += BLOCK * BP / PF;
            sp += BLOCK / 64;
            bp += BLOCK / 64;
            xp_tg += BLOCK;
        }
        if (TAIL > 0 && int(lane) * VPT < TAIL) {
            float sum = hc_load_vector_tg<T, VPT, BITS_D>(xp_tg, xv);
            for (int r = 0; r < RPS; ++r) {
                result[r] += hc_qdot<VPT, BITS_D>(
                    wq + r * ROW_BYTES, xv,
                    float(sp[r * GROUPS]), float(bp[r * GROUPS]), sum);
            }
        }
        for (int r = 0; r < RPS; ++r) {
            float v_sum = simd_sum(result[r]);
            if (lane == 0) pp[out_row + r] = v_sum;
        }
    } else if (INJ != 0 && int(sg) < HC) {
        constexpr int PF = hc_pack_factor<BITS_I>();
        constexpr int BP = hc_bytes_per_pack<BITS_I>();
        constexpr int ROW_BYTES = K * BP / PF;
        constexpr int VPT = PF;
        constexpr int BLOCK = VPT * 32;
        constexpr int SCALE_STEP = 64 / VPT;
        const int r = int(sg);
        const device uint8_t* wq = (const device uint8_t*)inject_w
            + r * ROW_BYTES + ks * (SLICE * BP / PF) + int(lane) * BP;
        const device T* sp = inject_s + r * GROUPS + ks * (SLICE / 64)
            + int(lane) / SCALE_STEP;
        const device T* bp = inject_b + r * GROUPS + ks * (SLICE / 64)
            + int(lane) / SCALE_STEP;
        const threadgroup T* xp_tg = xs + int(lane) * VPT;
        float result = 0.0f;
        float xv[VPT];
        constexpr int TAIL = SLICE % BLOCK;
        for (int k = 0; k < SLICE - TAIL; k += BLOCK) {
            float sum = hc_load_vector_tg<T, VPT, BITS_I>(xp_tg, xv);
            result += hc_qdot<VPT, BITS_I>(wq, xv, float(sp[0]), float(bp[0]), sum);
            wq += BLOCK * BP / PF;
            sp += BLOCK / 64;
            bp += BLOCK / 64;
            xp_tg += BLOCK;
        }
        if (TAIL > 0 && int(lane) * VPT < TAIL) {
            float sum = hc_load_vector_tg<T, VPT, BITS_I>(xp_tg, xv);
            result += hc_qdot<VPT, BITS_I>(wq, xv, float(sp[0]), float(bp[0]), sum);
        }
        float v_sum = simd_sum(result);
        if (lane == 0) pp[R + r] = v_sum;
    }
    if (int(g) < PER && int(t) + int(g) * 256 < H) {
        const size_t at = base + int(t) + int(g) * 256;
        xn[at] = n_mine;
    }
)";

static const char* kHcNdnwSource = R"(
    const uint gy = threadgroup_position_in_grid.y;
    const uint row = threadgroup_position_in_grid.z;
    const uint t = thread_index_in_threadgroup;
    const uint sg = simdgroup_index_in_threadgroup;
    const uint lane = thread_index_in_simdgroup;
    const uint s = gy & 3;
    const uint g = gy >> 2;
    constexpr int PER = (H + 255) / 256;
    constexpr int KS = 4;
    constexpr int SLICE = K / KS;
    constexpr int GROUPS = K / 64;
    constexpr int PR = R + HC;
    constexpr int ROWS_TG = 8 * RPS;
    constexpr int NDG = R / ROWS_TG;
    static_assert(HC == 4 && K == 4 * H && R % ROWS_TG == 0, "down K slice ks is stream ks");
    static_assert(PER <= NDG + INJ, "one stored strip per threadgroup");
    threadgroup T xs[H];
    threadgroup float npart[8];
    float v[PER];
    T y_mine = T(0.0f);
    T n_mine = T(0.0f);
    const size_t base = (size_t)row * K + (size_t)s * H;
    const device T* wp = w + (size_t)s * H;
    float wv[PER];
    for (int i = 0; i < PER; ++i) {
        const int k = t + i * 256;
        wv[i] = (k < H) ? float(wp[k]) : 0.0f;
    }
    const float e = eps[0];
    float ss = 0.0f;
    const device T* bq = branch + (size_t)row * H;
    const float g_s = float(gate[(size_t)row * HC + s]);
    for (int i = 0; i < PER; ++i) {
        const int k = t + i * 256;
        v[i] = 0.0f;
        if (k < H) {
            const T y = T(float(x[base + k]) + float(T(float(bq[k]) * g_s)));
            if (i == int(g)) y_mine = y;
            v[i] = float(y);
        }
        ss += v[i] * v[i];
    }
    ss = simd_sum(ss);
    if (lane == 0) npart[sg] = ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float tot = 0.0f;
    for (int i = 0; i < 8; ++i) tot += npart[i];
    const float inv = metal::rsqrt(tot / float(H) + e);
    for (int i = 0; i < PER; ++i) {
        const int k = t + i * 256;
        if (k < H) {
            const T y = T(v[i] * inv * (1.0f + wv[i]));
            xs[k] = y;
            if (i == int(g)) n_mine = y;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const bool down_tg = g < NDG;
    const int ks = int(s);
    device float* pp = parts + ((size_t)row * KS + ks) * PR;
    if (down_tg) {
        constexpr int PF = hc_pack_factor<BITS_D>();
        constexpr int BP = hc_bytes_per_pack<BITS_D>();
        constexpr int ROW_BYTES = K * BP / PF;
        constexpr int PPT = 2;
        constexpr int VPT = PF * PPT;
        constexpr int BLOCK = VPT * 32;
        constexpr int SCALE_STEP = 64 / VPT;
        const int out_row = int(g) * ROWS_TG + int(sg) * RPS;
        const device uint8_t* wq = (const device uint8_t*)down_w
            + out_row * ROW_BYTES + ks * (SLICE * BP / PF) + int(lane) * PPT * BP;
        const device T* sp = down_s + out_row * GROUPS + ks * (SLICE / 64)
            + int(lane) / SCALE_STEP;
        const device T* bp = down_b + out_row * GROUPS + ks * (SLICE / 64)
            + int(lane) / SCALE_STEP;
        const threadgroup T* xp_tg = xs + int(lane) * VPT;
        float result[RPS] = {0.0f};
        float xv[VPT];
        constexpr int TAIL = SLICE % BLOCK;
        for (int k = 0; k < SLICE - TAIL; k += BLOCK) {
            float sum = hc_load_vector_tg<T, VPT, BITS_D>(xp_tg, xv);
            for (int r = 0; r < RPS; ++r) {
                result[r] += hc_qdot<VPT, BITS_D>(
                    wq + r * ROW_BYTES, xv,
                    float(sp[r * GROUPS]), float(bp[r * GROUPS]), sum);
            }
            wq += BLOCK * BP / PF;
            sp += BLOCK / 64;
            bp += BLOCK / 64;
            xp_tg += BLOCK;
        }
        if (TAIL > 0 && int(lane) * VPT < TAIL) {
            float sum = hc_load_vector_tg<T, VPT, BITS_D>(xp_tg, xv);
            for (int r = 0; r < RPS; ++r) {
                result[r] += hc_qdot<VPT, BITS_D>(
                    wq + r * ROW_BYTES, xv,
                    float(sp[r * GROUPS]), float(bp[r * GROUPS]), sum);
            }
        }
        for (int r = 0; r < RPS; ++r) {
            float v_sum = simd_sum(result[r]);
            if (lane == 0) pp[out_row + r] = v_sum;
        }
    } else if (INJ != 0 && int(sg) < HC) {
        constexpr int PF = hc_pack_factor<BITS_I>();
        constexpr int BP = hc_bytes_per_pack<BITS_I>();
        constexpr int ROW_BYTES = K * BP / PF;
        constexpr int VPT = PF;
        constexpr int BLOCK = VPT * 32;
        constexpr int SCALE_STEP = 64 / VPT;
        const int r = int(sg);
        const device uint8_t* wq = (const device uint8_t*)inject_w
            + r * ROW_BYTES + ks * (SLICE * BP / PF) + int(lane) * BP;
        const device T* sp = inject_s + r * GROUPS + ks * (SLICE / 64)
            + int(lane) / SCALE_STEP;
        const device T* bp = inject_b + r * GROUPS + ks * (SLICE / 64)
            + int(lane) / SCALE_STEP;
        const threadgroup T* xp_tg = xs + int(lane) * VPT;
        float result = 0.0f;
        float xv[VPT];
        constexpr int TAIL = SLICE % BLOCK;
        for (int k = 0; k < SLICE - TAIL; k += BLOCK) {
            float sum = hc_load_vector_tg<T, VPT, BITS_I>(xp_tg, xv);
            result += hc_qdot<VPT, BITS_I>(wq, xv, float(sp[0]), float(bp[0]), sum);
            wq += BLOCK * BP / PF;
            sp += BLOCK / 64;
            bp += BLOCK / 64;
            xp_tg += BLOCK;
        }
        if (TAIL > 0 && int(lane) * VPT < TAIL) {
            float sum = hc_load_vector_tg<T, VPT, BITS_I>(xp_tg, xv);
            result += hc_qdot<VPT, BITS_I>(wq, xv, float(sp[0]), float(bp[0]), sum);
        }
        float v_sum = simd_sum(result);
        if (lane == 0) pp[R + r] = v_sum;
    }
    if (int(g) < PER && int(t) + int(g) * 256 < H) {
        const size_t at = base + int(t) + int(g) * 256;
        xn[at] = n_mine;
        y_out[at] = y_mine;
    }
)";

static const char* kHcU2Source = R"(
    constexpr int PF = hc_pack_factor<BITS_U>();
    constexpr int BP = hc_bytes_per_pack<BITS_U>();
    constexpr int ROW_BYTES = R * BP / PF;
    constexpr int GROUPS_R = R / 64;
    constexpr int CH = 2 * PF;
    constexpr int CPG = 64 / CH;
    constexpr int NC = R / CH;
    constexpr int NCP = NC + 1;
    constexpr int HB = NO / HC;
    constexpr int JS = TGS / NC;
    constexpr int KS = 4;
    constexpr int PR = R + HC;
    static_assert(HC == 4 && NO % 32 == 0 && TGS % NC == 0 && JS > 0,
                  "up tiles: four streams, whole simdgroups, whole chunk rows");
    threadgroup T acts[R];
    threadgroup float q[NO * NCP];
    const uint t = thread_index_in_threadgroup;
    const int r = threadgroup_position_in_grid.z;
    const int h0 = int(threadgroup_position_in_grid.y) * HB;
    const int s = int(t & 3);
    const int h = h0 + int(t >> 2);
    const int n = s * H + h;
    const float xnv = int(t) < NO ? float(xn[(size_t)r * K + n]) : 0.0f;
    const device float* pp = parts + (size_t)r * KS * PR;
    for (int i = int(t); i < R; i += TGS) {
        float v = pp[i] + pp[PR + i] + pp[2 * PR + i] + pp[3 * PR + i];
        v = v / float(HC);
        acts[i] = T(v / (1.0f + metal::exp(-v)));
    }
    const bool inj_thread = INJ != 0 && threadgroup_position_in_grid.y == 0 && int(t) < HC;
    T injv = T(0.0f);
    if (inj_thread) {
        const int i = R + int(t);
        float v = pp[i] + pp[PR + i] + pp[2 * PR + i] + pp[3 * PR + i];
        v = v / float(HC);
        injv = T(2.0f / (1.0f + metal::exp(-v)));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    {
        const int c = int(t) % NC;
        const int gq = c / CPG;
        const int e = c * CH;
        float xv[CH];
        float sum = hc_load_vector_tg<T, CH, BITS_U>(acts + e, xv);
        for (int j = int(t) / NC; j < NO; j += JS) {
            const int nj = (j / HB) * H + h0 + j % HB;
            const device uint8_t* w = (const device uint8_t*)up_w + (size_t)nj * ROW_BYTES;
            const float sc = float(up_s[(size_t)nj * GROUPS_R + gq]);
            const float bi = float(up_b[(size_t)nj * GROUPS_R + gq]);
            q[j * NCP + c] = hc_qdot<CH, BITS_U>(w + e * BP / PF, xv, sc, bi, sum);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (int(t) >= NO) return;
    const threadgroup float* qp = q + (s * HB + int(t >> 2)) * NCP;
    float acc = 0.0f;
    for (int c = 0; c < NC; ++c) {
        acc += qp[c];
    }
    float gate = 1.0f / (1.0f + metal::exp(-acc));
    float v = gate * xnv;
    v += simd_shuffle_down(v, 1);
    v += simd_shuffle_down(v, 2);
    if (s == 0) mixed[(size_t)r * H + h] = T(v / float(HC));
    if (inj_thread) inj[(size_t)r * HC + int(t)] = injv;
)";

static const char* kHcNSource = R"(
    const uint row = threadgroup_position_in_grid.z;
    const uint s = threadgroup_position_in_grid.y;
    const uint t = thread_index_in_threadgroup;
    const uint sg = simdgroup_index_in_threadgroup;
    const uint lane = thread_index_in_simdgroup;
    threadgroup float part[8];
    constexpr int PER = (H + 255) / 256;
    const device T* xp = x + (size_t)row * K + (size_t)s * H;
    const device T* wp = w + (size_t)s * H;
    device T* op = xn + (size_t)row * K + (size_t)s * H;
    float v[PER];
    float ss = 0.0f;
    for (int i = 0; i < PER; ++i) {
        const int k = t + i * 256;
        v[i] = (k < H) ? float(xp[k]) : 0.0f;
        ss += v[i] * v[i];
    }
    ss = simd_sum(ss);
    if (lane == 0) part[sg] = ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float tot = 0.0f;
    for (int i = 0; i < 8; ++i) tot += part[i];
    const float inv = metal::rsqrt(tot / float(H) + eps[0]);
    for (int i = 0; i < PER; ++i) {
        const int k = t + i * 256;
        if (k < H) op[k] = T(v[i] * inv * (1.0f + float(wp[k])));
    }
)";

static const char* kHcNpSource = R"(
    const uint row = threadgroup_position_in_grid.z;
    const uint s = threadgroup_position_in_grid.y;
    const uint t = thread_index_in_threadgroup;
    const uint sg = simdgroup_index_in_threadgroup;
    const uint lane = thread_index_in_simdgroup;
    threadgroup float part[8];
    constexpr int PER = (H + 255) / 256;
    const device T* xp = x + (size_t)row * K + (size_t)s * H;
    const device T* wp = w + (size_t)s * H;
    device T* op = xn + (size_t)row * K + (size_t)s * H;
    float v[PER];
    float ss = 0.0f;
    for (int i = 0; i < PER; ++i) {
        const int k = t + i * 256;
        v[i] = (k < H) ? float(xp[k]) : 0.0f;
        ss += v[i] * v[i];
    }
    ss = simd_sum(ss);
    if (lane == 0) part[sg] = ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float tot = 0.0f;
    for (int i = 0; i < 8; ++i) tot += part[i];
    const float inv = metal::precise::rsqrt(tot / float(H) + eps[0]);
    for (int i = 0; i < PER; ++i) {
        const int k = t + i * 256;
        if (k < H) op[k] = T(v[i] * inv * (1.0f + float(wp[k])));
    }
)";

static const char* kHcWnSource = R"(
    const uint row = threadgroup_position_in_grid.z;
    const uint s = threadgroup_position_in_grid.y;
    const uint t = thread_index_in_threadgroup;
    const uint sg = simdgroup_index_in_threadgroup;
    const uint lane = thread_index_in_simdgroup;
    threadgroup float part[8];
    constexpr int PER = (H + 255) / 256;
    const size_t base = (size_t)row * K + (size_t)s * H;
    const device T* bp = branch + (size_t)row * H;
    const float g = float(gate[(size_t)row * HC + s]);
    const device T* wp = w + (size_t)s * H;
    float v[PER];
    float ss = 0.0f;
    for (int i = 0; i < PER; ++i) {
        const int k = t + i * 256;
        v[i] = 0.0f;
        if (k < H) {
            const T y = T(float(x[base + k]) + float(T(float(bp[k]) * g)));
            y_out[base + k] = y;
            v[i] = float(y);
        }
        ss += v[i] * v[i];
    }
    ss = simd_sum(ss);
    if (lane == 0) part[sg] = ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float tot = 0.0f;
    for (int i = 0; i < 8; ++i) tot += part[i];
    const float inv = metal::precise::rsqrt(tot / float(H) + eps[0]);
    for (int i = 0; i < PER; ++i) {
        const int k = t + i * 256;
        if (k < H) xn[base + k] = T(v[i] * inv * (1.0f + float(wp[k])));
    }
)";

static const char* kHcWndSource = R"(
    const uint row = threadgroup_position_in_grid.z;
    const uint s = threadgroup_position_in_grid.y;
    const uint t = thread_index_in_threadgroup;
    const uint sg = simdgroup_index_in_threadgroup;
    const uint lane = thread_index_in_simdgroup;
    threadgroup float part[8];
    constexpr int PER = (H + 255) / 256;
    const size_t base = (size_t)row * K + (size_t)s * H;
    const device T* bp = branch + (size_t)row * H;
    const float g = float(gate[(size_t)row * HC + s]);
    const device T* wp = w + (size_t)s * H;
    float v[PER];
    float ss = 0.0f;
    for (int i = 0; i < PER; ++i) {
        const int k = t + i * 256;
        v[i] = 0.0f;
        if (k < H) {
            const T y = T(float(x[base + k]) + float(T(float(bp[k]) * g)));
            y_out[base + k] = y;
            v[i] = float(y);
        }
        ss += v[i] * v[i];
    }
    ss = simd_sum(ss);
    if (lane == 0) part[sg] = ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float tot = 0.0f;
    for (int i = 0; i < 8; ++i) tot += part[i];
    const float inv = metal::rsqrt(tot / float(H) + eps[0]);
    for (int i = 0; i < PER; ++i) {
        const int k = t + i * 256;
        if (k < H) xn[base + k] = T(v[i] * inv * (1.0f + float(wp[k])));
    }
)";

static const char* kHcDSource = R"(
    const uint tg = threadgroup_position_in_grid.y;
    const uint row = threadgroup_position_in_grid.z;
    const uint sg = simdgroup_index_in_threadgroup;
    const uint lane = thread_index_in_simdgroup;
    constexpr int KS = 4;
    constexpr int SLICE = K / KS;
    constexpr int GROUPS = K / 64;
    threadgroup float part[KS][8];
    const device T* x = xn + (size_t)row * K;
    const int rg = int(sg & 1);
    const int ks = int(sg >> 1);
    if (tg < R / 8) {
        constexpr int PF = hc_pack_factor<BITS_D>();
        constexpr int BP = hc_bytes_per_pack<BITS_D>();
        constexpr int ROW_BYTES = K * BP / PF;
        constexpr int PPT = 2;
        constexpr int VPT = PF * PPT;
        constexpr int BLOCK = VPT * 32;
        constexpr int SCALE_STEP = 64 / VPT;
        const int out_row = int(tg) * 8 + rg * 4;
        const device uint8_t* wp = (const device uint8_t*)down_w
            + out_row * ROW_BYTES + ks * (SLICE * BP / PF) + int(lane) * PPT * BP;
        const device T* sp = down_s + out_row * GROUPS + ks * (SLICE / 64)
            + int(lane) / SCALE_STEP;
        const device T* bp = down_b + out_row * GROUPS + ks * (SLICE / 64)
            + int(lane) / SCALE_STEP;
        const device T* xp = x + ks * SLICE + int(lane) * VPT;
        float result[4] = {0.0f};
        float xv[VPT];
        constexpr int TAIL = SLICE % BLOCK;
        for (int k = 0; k < SLICE - TAIL; k += BLOCK) {
            float sum = hc_load_vector<T, VPT, BITS_D>(xp, xv);
            for (int r = 0; r < 4; ++r) {
                result[r] += hc_qdot<VPT, BITS_D>(
                    wp + r * ROW_BYTES, xv,
                    float(sp[r * GROUPS]), float(bp[r * GROUPS]), sum);
            }
            wp += BLOCK * BP / PF;
            sp += BLOCK / 64;
            bp += BLOCK / 64;
            xp += BLOCK;
        }
        if (TAIL > 0 && int(lane) * VPT < TAIL) {
            float sum = hc_load_vector<T, VPT, BITS_D>(xp, xv);
            for (int r = 0; r < 4; ++r) {
                result[r] += hc_qdot<VPT, BITS_D>(
                    wp + r * ROW_BYTES, xv,
                    float(sp[r * GROUPS]), float(bp[r * GROUPS]), sum);
            }
        }
        for (int r = 0; r < 4; ++r) {
            float v = simd_sum(result[r]);
            if (lane == 0) part[ks][rg * 4 + r] = v;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sg < 2 && lane < 4) {
            const int r = int(lane);
            float v = part[0][sg * 4 + r] + part[1][sg * 4 + r]
                + part[2][sg * 4 + r] + part[3][sg * 4 + r];
            v = v / float(HC);
            act[(size_t)row * R + int(tg) * 8 + int(sg) * 4 + r]
                = T(v / (1.0f + metal::exp(-v)));
        }
        return;
    }
    if (INJ == 0 || rg != 0) return;
    {
        constexpr int PF = hc_pack_factor<BITS_I>();
        constexpr int BP = hc_bytes_per_pack<BITS_I>();
        constexpr int ROW_BYTES = K * BP / PF;
        constexpr int VPT = PF;
        constexpr int BLOCK = VPT * 32;
        constexpr int SCALE_STEP = 64 / VPT;
        const device uint8_t* wp = (const device uint8_t*)inject_w
            + ks * (SLICE * BP / PF) + int(lane) * BP;
        const device T* sp = inject_s + ks * (SLICE / 64) + int(lane) / SCALE_STEP;
        const device T* bp = inject_b + ks * (SLICE / 64) + int(lane) / SCALE_STEP;
        const device T* xp = x + ks * SLICE + int(lane) * VPT;
        float result[HC] = {0.0f};
        float xv[VPT];
        constexpr int TAIL = SLICE % BLOCK;
        for (int k = 0; k < SLICE - TAIL; k += BLOCK) {
            float sum = hc_load_vector<T, VPT, BITS_I>(xp, xv);
            for (int r = 0; r < HC; ++r) {
                result[r] += hc_qdot<VPT, BITS_I>(
                    wp + r * ROW_BYTES, xv,
                    float(sp[r * GROUPS]), float(bp[r * GROUPS]), sum);
            }
            wp += BLOCK * BP / PF;
            sp += BLOCK / 64;
            bp += BLOCK / 64;
            xp += BLOCK;
        }
        if (TAIL > 0 && int(lane) * VPT < TAIL) {
            float sum = hc_load_vector<T, VPT, BITS_I>(xp, xv);
            for (int r = 0; r < HC; ++r) {
                result[r] += hc_qdot<VPT, BITS_I>(
                    wp + r * ROW_BYTES, xv,
                    float(sp[r * GROUPS]), float(bp[r * GROUPS]), sum);
            }
        }
        for (int r = 0; r < HC; ++r) {
            float v = simd_sum(result[r]);
            if (lane == 0) part[ks][r] = v;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sg == 0 && lane < HC) {
            const int r = int(lane);
            float v = part[0][r] + part[1][r] + part[2][r] + part[3][r];
            v = v / float(HC);
            inj[(size_t)row * HC + r] = T(2.0f / (1.0f + metal::exp(-v)));
        }
    }
)";

static const char* kHcUSource = R"(
    const uint g = threadgroup_position_in_grid.y;
    const uint t = thread_index_in_threadgroup;
    const int h = int(g) * 64 + int(t >> 2);
    const int s = int(t & 3);
    const int n = s * H + h;
    constexpr int PF = hc_pack_factor<BITS_U>();
    constexpr int BP = hc_bytes_per_pack<BITS_U>();
    constexpr int ROW_BYTES = R * BP / PF;
    constexpr int GROUPS_R = R / 64;
    constexpr int CH = 2 * PF;
    constexpr int CPG = 64 / CH;
    const device uint8_t* w = (const device uint8_t*)up_w + (size_t)n * ROW_BYTES;
    const device T* sp = up_s + (size_t)n * GROUPS_R;
    const device T* bp = up_b + (size_t)n * GROUPS_R;
    float xv[CH];
    const int r = threadgroup_position_in_grid.z;
    const device T* a = act + (size_t)r * R;
    float acc = 0.0f;
    for (int gq = 0; gq < GROUPS_R; ++gq) {
        const float sc = float(sp[gq]);
        const float bi = float(bp[gq]);
        for (int c = 0; c < CPG; ++c) {
            const int e = gq * 64 + c * CH;
            float sum = hc_load_vector<T, CH, BITS_U>(a + e, xv);
            acc += hc_qdot<CH, BITS_U>(w + e * BP / PF, xv, sc, bi, sum);
        }
    }
    float gate = 1.0f / (1.0f + metal::exp(-acc));
    float v = gate * float(xn[(size_t)r * K + n]);
    v += simd_shuffle_down(v, 1);
    v += simd_shuffle_down(v, 2);
    if (s == 0) mixed[(size_t)r * H + h] = T(v / float(HC));
)";

static const char* kHcTiSource = R"(
    const uint row = threadgroup_position_in_grid.z;
    const uint t = thread_index_in_threadgroup;
    const uint sg = simdgroup_index_in_threadgroup;
    const uint lane = thread_index_in_simdgroup;
    constexpr int PER_H = (H + 255) / 256;
    constexpr int PF = hc_pack_factor<BITS_I>();
    constexpr int BP = hc_bytes_per_pack<BITS_I>();
    constexpr int ROW_BYTES = K * BP / PF;
    constexpr int GROUPS = K / 64;
    constexpr int PER = K / 256;
    constexpr int VT = H / PER;
    static_assert(K == HC * H && H % PER == 0 && VT % 32 == 0 && HC * VT == 256,
                  "inject chunks must tile each stream");
    threadgroup T xs[H];
    threadgroup float part[HC][8];
    const device T* up_r = up + (size_t)row * K;
    const device T* xn_r = xn + (size_t)row * K;
    float acc[PER_H];
    for (int s = 0; s < HC; ++s) {
        for (int i = 0; i < PER_H; ++i) {
            const int h = int(t) + i * 256;
            if (h < H) {
                const int n = s * H + h;
                const T x = xn_r[n];
                xs[h] = x;
                const T g = T(1.0f / (1.0f + metal::exp(-float(up_r[n]))));
                const T p = T(float(g) * float(x));
                acc[i] = s == 0 ? float(p) : float(T(acc[i] + float(p)));
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (int(t) < VT) {
            float res[HC] = {0.0f};
            float xv[PF];
            const int e0 = int(t) * PER;
            for (int e = e0; e < e0 + PER; e += PF) {
                const float sum = hc_load_vector_tg<T, PF, BITS_I>(xs + e, xv);
                const int ge = s * H + e;
                const int g = ge / 64;
                for (int r = 0; r < HC; ++r) {
                    res[r] += hc_qdot<PF, BITS_I>(
                        (const device uint8_t*)inject_w + r * ROW_BYTES + ge * BP / PF,
                        xv, float(inject_s[r * GROUPS + g]), float(inject_b[r * GROUPS + g]),
                        sum);
                }
            }
            for (int r = 0; r < HC; ++r) {
                const float v = simd_sum(res[r]);
                if (lane == 0) part[r][s * (VT / 32) + int(sg)] = v;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (int i = 0; i < PER_H; ++i) {
        const int h = int(t) + i * 256;
        if (h < H) mixed[(size_t)row * H + h] = T(acc[i] * (1.0f / float(HC)));
    }
    if (t < HC) {
        float v = 0.0f;
        for (int i = 0; i < 8; ++i) v += part[t][i];
        const float q = float(T(float(T(v)) / float(HC)));
        const float gate = float(T(1.0f / (1.0f + metal::exp(-q))));
        inj[(size_t)row * HC + t] = T(2.0f * gate);
    }
)";

static const char* kGatedDeltaStepSource = R"(
    auto n = thread_position_in_grid.z;
    auto b_idx = n / Hv;
    auto hv_idx = n % Hv;
    auto hk_idx = hv_idx / (Hv / Hk);
    constexpr int n_per_t = Dk / 32;

    auto q_ = q + b_idx * T * Hk * Dk + hk_idx * Dk;
    auto k_ = k + b_idx * T * Hk * Dk + hk_idx * Dk;
    auto v_ = v + b_idx * T * Hv * Dv + hv_idx * Dv;
    y += b_idx * T * Hv * Dv + hv_idx * Dv;

    auto dk_idx = thread_position_in_threadgroup.x;
    auto dv_idx = thread_position_in_grid.y;

    auto i_state = state_in + (n * Dv + dv_idx) * Dk;
    auto o_state = state_out + (n * Dv + dv_idx) * Dk;

    float state[n_per_t];
    for (int i = 0; i < n_per_t; ++i) {
      auto s_idx = n_per_t * dk_idx + i;
      state[i] = static_cast<float>(i_state[s_idx]);
    }

    auto g_ = g + b_idx * T * Hv;
    auto beta_ = beta + b_idx * T * Hv;

    for (int t = 0; t < T; ++t) {
      float kv_mem = 0.0f;
      for (int i = 0; i < n_per_t; ++i) {
        auto s_idx = n_per_t * dk_idx + i;
        state[i] = state[i] * g_[hv_idx];
        kv_mem += state[i] * k_[s_idx];
      }
      kv_mem = simd_sum(kv_mem);

      auto delta = (v_[dv_idx] - kv_mem) * beta_[hv_idx];

      float out = 0.0f;
      for (int i = 0; i < n_per_t; ++i) {
        auto s_idx = n_per_t * dk_idx + i;
        state[i] = state[i] + k_[s_idx] * delta;
        out += state[i] * q_[s_idx];
      }
      out = simd_sum(out);
      if (thread_index_in_simdgroup == 0) {
        y[dv_idx] = static_cast<InT>(out);
      }
      q_ += Hk * Dk;
      k_ += Hk * Dk;
      v_ += Hv * Dv;
      y += Hv * Dv;
      g_ += Hv;
      beta_ += Hv;
    }
    for (int i = 0; i < n_per_t; ++i) {
      auto s_idx = n_per_t * dk_idx + i;
      o_state[s_idx] = static_cast<StT>(state[i]);
    }
)";

static const char* kMropeApplyHalfSplit64_2dSource = R"(
    constexpr uint32_t D = 256;
    constexpr uint32_t R = 64;
    constexpr uint32_t HALF = 32;

    uint32_t d = thread_position_in_grid.x;
    uint32_t l = thread_position_in_grid.y;
    uint32_t bh = thread_position_in_grid.z;

    if (d >= D) return;

    uint32_t H = x_shape[1];
    uint32_t L = x_shape[2];
    uint32_t b = bh / H;
    uint32_t h = bh % H;
    uint64_t base = ((uint64_t)b * H + h) * L * D + (uint64_t)l * D;

    if (d >= R) {
        out[base + d] = x[base + d];
        return;
    }

    uint32_t k = (d < HALF) ? d : (d - HALF);
    uint32_t pos_Lp = position_ids_shape[1];
    int32_t pos = position_ids[b * pos_Lp + l];
    float angle = (float)pos * inv_freq[k];
    float c = metal::cos(angle);
    float s = metal::sin(angle);

    float x_lo = (float)x[base + k];
    float x_hi = (float)x[base + k + HALF];

    float val = (d < HALF) ? (x_lo * c - x_hi * s) : (x_hi * c + x_lo * s);
    out[base + d] = (T)val;
)";

static const char* kMropeApplyHalfSplit64_3dSource = R"(
    constexpr uint32_t D = 256;
    constexpr uint32_t R = 64;
    constexpr uint32_t HALF = 32;

    uint32_t d = thread_position_in_grid.x;
    uint32_t l = thread_position_in_grid.y;
    uint32_t bh = thread_position_in_grid.z;

    if (d >= D) return;

    uint32_t B = x_shape[0];
    uint32_t H = x_shape[1];
    uint32_t L = x_shape[2];
    uint32_t b = bh / H;
    uint32_t h = bh % H;
    uint64_t base = ((uint64_t)b * H + h) * L * D + (uint64_t)l * D;

    if (d >= R) {
        out[base + d] = x[base + d];
        return;
    }

    uint32_t k = (d < HALF) ? d : (d - HALF);
    uint32_t rem = k % 3;
    uint32_t plane = 0;
    if (rem == 1 && k < 33) {
        plane = 1;
    } else if (rem == 2 && k < 30) {
        plane = 2;
    }

    uint32_t pos_Lp = position_ids_shape[2];
    int32_t pos = position_ids[(plane * B + b) * pos_Lp + l];
    float angle = (float)pos * inv_freq[k];
    float c = metal::cos(angle);
    float s = metal::sin(angle);

    float x_lo = (float)x[base + k];
    float x_hi = (float)x[base + k + HALF];

    float val = (d < HALF) ? (x_lo * c - x_hi * s) : (x_hi * c + x_lo * s);
    out[base + d] = (T)val;
)";

mx::array QLinear::forward(const mx::array& x) const {
    mx::array y = is_quantized
        ? mx::quantized_matmul(x, weight, *scales, biases, true, group_size, bits, "affine")
        : mx::matmul(x, mx::transpose(weight));
    if (bias.has_value()) {
        y = y + *bias;
    }
    return y;
}

mx::array lm_head_last_row(const QLinear& head, const mx::array& final_h) {
    auto full = head.forward(final_h);  // [B, L, V], same rows as Python lm(inp).logits
    int L = full.shape(1);
    return mx::slice(full, {0, L - 1, 0}, {full.shape(0), L, full.shape(2)});
}

mx::array QSwitchLinear::forward(const mx::array& x, const mx::array& indices, bool sorted_indices) const {
    return mx::gather_qmm(
        x,
        weight,
        scales,
        biases,
        std::nullopt,
        indices,
        true,
        group_size,
        bits,
        "affine",
        sorted_indices);
}

mx::array Qwen38Engine::qwen4_rms_norm(
    const mx::array& x,
    const mx::array& weight,
    float eps,
    int group_size) {
    auto dtype = x.dtype();
    auto scale = mx::array(1.0f, mx::float32) + mx::astype(weight, mx::float32);
    if (group_size <= 0) {
        return mx::astype(mx::fast::rms_norm(x, scale, eps), dtype);
    }
    auto shape = x.shape();
    mx::Shape grouped_shape = shape;
    grouped_shape.pop_back();
    grouped_shape.push_back(shape.back() / group_size);
    grouped_shape.push_back(group_size);
    auto y = mx::reshape(mx::astype(x, mx::float32), grouped_shape);
    y = mx::fast::rms_norm(y, std::nullopt, eps) *
        mx::reshape(scale, {shape.back() / group_size, group_size});
    return mx::astype(mx::reshape(y, shape), dtype);
}

mx::array Qwen38Engine::qwen4_rms_norm_scale(
    const mx::array& x,
    const mx::array& scale_fp32,
    float eps) {
    return mx::astype(mx::fast::rms_norm(x, scale_fp32, eps), x.dtype());
}

mx::array Qwen38Engine::qwen4_rms_norm_gated(
    const mx::array& x,
    const mx::array& gate,
    const mx::array& weight,
    float eps,
    bool sigmoid_gate) {
    auto dtype = x.dtype();
    auto y = mx::astype(mx::fast::rms_norm(x, weight, eps), mx::float32);
    auto g = mx::astype(gate, mx::float32);
    if (sigmoid_gate) {
        g = mx::sigmoid(g);
    } else {
        g = g * mx::sigmoid(g);
    }
    return mx::astype(y * g, dtype);
}

void Qwen38Engine::init_kernels() {
    if (kernels_initialized_) return;

    swiglu_compiled_ = mx::compile([](const std::vector<mx::array>& inputs) -> std::vector<mx::array> {
        const auto& gate = inputs[0];
        const auto& x = inputs[1];
        return {(gate * mx::sigmoid(gate)) * x};
    }, true);

    depthwise_conv_compiled_ = mx::compile([](const std::vector<mx::array>& inputs) -> std::vector<mx::array> {
        const auto& conv_in = inputs[0];
        const auto& w_t_f32 = inputs[1];
        auto out = mx::sum(mx::astype(conv_in, mx::float32) * mx::expand_dims(w_t_f32, 0), 1);
        return {mx::expand_dims(mx::astype(out, conv_in.dtype()), 1)};
    }, true);

    silu_compiled_ = mx::compile([](const std::vector<mx::array>& inputs) -> std::vector<mx::array> {
        const auto& x = inputs[0];
        return {x * mx::sigmoid(x)};
    }, true);

    compute_g_beta_compiled_ = mx::compile([](const std::vector<mx::array>& inputs) -> std::vector<mx::array> {
        const auto& A_log = inputs[0];
        const auto& a = inputs[1];
        const auto& b = inputs[2];
        const auto& dt_bias = inputs[3];
        auto adt = a + dt_bias;
        auto g = mx::exp(-mx::exp(mx::astype(A_log, mx::float32)) *
                         mx::logaddexp(adt, mx::array(0.0f, adt.dtype())));
        return {g, mx::sigmoid(b)};
    }, true);

    gated_delta_step_ops_compiled_ = mx::compile([](const std::vector<mx::array>& inputs) -> std::vector<mx::array> {
        const auto& q = inputs[0];
        const auto& k = inputs[1];
        const auto& v = inputs[2];
        const auto& g = inputs[3];
        const auto& beta = inputs[4];
        auto state = inputs[5];
        auto decay = (g.ndim() == 2)
            ? mx::expand_dims(g, {-2, -1})
            : mx::expand_dims(g, -2);
        state = state * decay;
        auto kv_mem = mx::sum(state * mx::expand_dims(k, -2), -1);
        auto delta = (v - kv_mem) * mx::expand_dims(beta, -1);
        state = state + mx::expand_dims(k, -2) * mx::expand_dims(delta, -1);
        auto y = mx::sum(state * mx::expand_dims(q, -2), -1);
        return {mx::astype(y, q.dtype()), state};
    }, false);

    hc_act_compiled_ = mx::compile([](const std::vector<mx::array>& inputs) -> std::vector<mx::array> {
        const auto& y = inputs[0];
        auto v = y / mx::array(4.0f, y.dtype());
        return {v * mx::sigmoid(v)};
    }, true);

    hc_tail_compiled_ = mx::compile([](const std::vector<mx::array>& inputs) -> std::vector<mx::array> {
        const auto& up = inputs[0];
        const auto& normed = inputs[1];
        constexpr int hidden = 2560;
        auto gate = mx::sigmoid(up);
        auto slice3d = [](const mx::array& a, int start, int stop) {
            return mx::slice(a, {0, 0, start}, {a.shape(0), a.shape(1), stop});
        };
        auto acc = slice3d(gate, 0, hidden) * slice3d(normed, 0, hidden);
        for (int g = 1; g < 4; ++g) {
            int lo = g * hidden;
            acc = acc + slice3d(gate, lo, lo + hidden) * slice3d(normed, lo, lo + hidden);
        }
        return {acc * mx::array(0.25f, acc.dtype())};
    }, false);

    hc_ndn_kernel_ = mx::fast::metal_kernel(
        "omlx_qwen4_hc_decode_norm_down",
        {"x", "w", "eps", "down_w", "down_s", "down_b", "inject_w", "inject_s", "inject_b"},
        {"xn", "parts"}, kHcNdnSource, kHcV2Header, true, false);
    hc_ndnw_kernel_ = mx::fast::metal_kernel(
        "omlx_qwen4_hc_decode_write_norm_down",
        {"x", "branch", "gate", "w", "eps", "down_w", "down_s", "down_b", "inject_w", "inject_s", "inject_b"},
        {"y_out", "xn", "parts"}, kHcNdnwSource, kHcV2Header, true, false);
    hc_u2_kernel_ = mx::fast::metal_kernel(
        "omlx_qwen4_hc_decode_up",
        {"xn", "parts", "up_w", "up_s", "up_b"},
        {"mixed", "inj"}, kHcU2Source, kHcV2Header, true, false);

    hc_n_kernel_ = mx::fast::metal_kernel(
        "omlx_qwen4_hc_fused_norm", {"x", "w", "eps"}, {"xn"}, kHcNSource, "", true, false);
    hc_np_kernel_ = mx::fast::metal_kernel(
        "omlx_qwen4_hc_prefill_norm", {"x", "w", "eps"}, {"xn"}, kHcNpSource, "", true, false);
    hc_wn_kernel_ = mx::fast::metal_kernel(
        "omlx_qwen4_hc_prefill_write_norm", {"x", "branch", "gate", "w", "eps"}, {"y_out", "xn"}, kHcWnSource, "", true, false);
    hc_wnd_kernel_ = mx::fast::metal_kernel(
        "omlx_qwen4_hc_fused_write_norm", {"x", "branch", "gate", "w", "eps"}, {"y_out", "xn"}, kHcWndSource, "", true, false);
    hc_d_kernel_ = mx::fast::metal_kernel(
        "omlx_qwen4_hc_fused_down",
        {"xn", "down_w", "down_s", "down_b", "inject_w", "inject_s", "inject_b"},
        {"act", "inj"}, kHcDSource, kHcV2Header, true, false);
    hc_u_kernel_ = mx::fast::metal_kernel(
        "omlx_qwen4_hc_fused_up",
        {"xn", "act", "up_w", "up_s", "up_b"},
        {"mixed"}, kHcUSource, kHcV2Header, true, false);
    hc_ti_kernel_ = mx::fast::metal_kernel(
        "omlx_qwen4_hc_prefill_tail_inject",
        {"up", "xn", "inject_w", "inject_s", "inject_b"},
        {"mixed", "inj"}, kHcTiSource, kHcV2Header, true, false);

    gated_delta_kernel_ = mx::fast::metal_kernel(
        "gated_delta_step",
        {"q", "k", "v", "g", "beta", "state_in", "T"},
        {"y", "state_out"}, kGatedDeltaStepSource, "", true, false);
    mrope_2d_kernel_ = mx::fast::metal_kernel(
        "mrope_apply_half_split_64_2d",
        {"x", "inv_freq", "position_ids"},
        {"out"}, kMropeApplyHalfSplit64_2dSource, "", true, false);
    mrope_3d_kernel_ = mx::fast::metal_kernel(
        "mrope_apply_half_split_64_3d",
        {"x", "inv_freq", "position_ids"},
        {"out"}, kMropeApplyHalfSplit64_3dSource, "", true, false);

    kernels_initialized_ = true;
}

static mx::array invert_unit_lower(const mx::array& Tmat, int C) {
    int d0 = Tmat.shape(0);
    int d1 = Tmat.shape(1);
    int d2 = Tmat.shape(2);
    auto X = mx::broadcast_to(mx::eye(C, mx::float32), Tmat.shape()) + mx::zeros_like(Tmat);
    std::vector<mx::array> rows;
    rows.reserve(C);
    rows.push_back(mx::slice(X, {0, 0, 0, 0, 0}, {d0, d1, d2, 1, C}));
    for (int i = 1; i < C; ++i) {
        auto Tij = mx::slice(Tmat, {0, 0, 0, i, 0}, {d0, d1, d2, i + 1, i});
        auto Xj = mx::concatenate(rows, -2);
        auto Xi = mx::slice(X, {0, 0, 0, i, 0}, {d0, d1, d2, i + 1, C});
        rows.push_back(Xi - mx::matmul(Tij, Xj));
    }
    return mx::concatenate(rows, -2);
}

std::pair<mx::array, mx::array> Qwen38Engine::gated_delta_ops(
    const mx::array& q,
    const mx::array& k,
    const mx::array& v,
    const mx::array& g,
    const mx::array& beta,
    const mx::array& state) {
    init_kernels();
    int B = q.shape(0);
    int T = q.shape(1);
    int Hk = q.shape(2);
    int Dk = q.shape(3);
    int Hv = v.shape(2);
    int Dv = v.shape(3);
    int repeat_factor = Hv / Hk;
    auto q_in = (repeat_factor > 1) ? mx::repeat(q, repeat_factor, -2) : q;
    auto k_in = (repeat_factor > 1) ? mx::repeat(k, repeat_factor, -2) : k;
    auto cur_state = state;
    std::vector<mx::array> ys;
    ys.reserve(T);
    for (int t = 0; t < T; ++t) {
        auto qt = mx::squeeze(mx::slice(q_in, {0, t, 0, 0}, {B, t + 1, Hv, Dk}), 1);
        auto kt = mx::squeeze(mx::slice(k_in, {0, t, 0, 0}, {B, t + 1, Hv, Dk}), 1);
        auto vt = mx::squeeze(mx::slice(v, {0, t, 0, 0}, {B, t + 1, Hv, Dv}), 1);
        auto gt = mx::squeeze(mx::slice(g, {0, t, 0}, {B, t + 1, Hv}), 1);
        auto bt = mx::squeeze(mx::slice(beta, {0, t, 0}, {B, t + 1, Hv}), 1);
        auto outs = gated_delta_step_ops_compiled_({qt, kt, vt, gt, bt, cur_state});
        ys.push_back(outs[0]);
        cur_state = outs[1];
    }
    auto y = mx::stack(ys, 1);
    return {y, cur_state};
}

std::pair<mx::array, mx::array> Qwen38Engine::gated_delta_chunked(
    const mx::array& q,
    const mx::array& k,
    const mx::array& v,
    const mx::array& g,
    const mx::array& beta,
    const mx::array& state,
    int C) {
    init_kernels();
    int B = q.shape(0);
    int T = q.shape(1);
    int Hk = q.shape(2);
    int Dk = q.shape(3);
    int Hv = v.shape(2);
    int Dv = v.shape(3);
    auto in_dtype = v.dtype();
    int rf = Hv / Hk;
    auto q_in = (rf > 1) ? mx::repeat(q, rf, -2) : q;
    auto k_in = (rf > 1) ? mx::repeat(k, rf, -2) : k;
    auto v_in = v;
    auto g_in = g;
    auto beta_in = beta;

    int pad = (C - (T % C)) % C;
    if (pad > 0) {
        q_in = mx::concatenate({q_in, mx::zeros({B, pad, Hv, Dk}, q_in.dtype())}, 1);
        k_in = mx::concatenate({k_in, mx::zeros({B, pad, Hv, Dk}, k_in.dtype())}, 1);
        v_in = mx::concatenate({v_in, mx::zeros({B, pad, Hv, Dv}, v_in.dtype())}, 1);
        g_in = mx::concatenate({g_in, mx::ones({B, pad, Hv}, g_in.dtype())}, 1);
        beta_in = mx::concatenate({beta_in, mx::zeros({B, pad, Hv}, beta_in.dtype())}, 1);
    }
    int Tp = T + pad;
    int nC = Tp / C;

    auto rc = [&](const mx::array& x, int D) {
        return mx::astype(
            mx::transpose(mx::reshape(x, {B, nC, C, Hv, D}), {0, 3, 1, 2, 4}),
            mx::float32);
    };
    auto qc = rc(q_in, Dk);
    auto kc = rc(k_in, Dk);
    auto vc = rc(v_in, Dv);
    auto gc = mx::transpose(mx::reshape(g_in, {B, nC, C, Hv}), {0, 3, 1, 2});
    auto betac = mx::transpose(mx::reshape(beta_in, {B, nC, C, Hv}), {0, 3, 1, 2});

    auto lcg = mx::cumsum(
        mx::log(mx::clip(gc, mx::array(1e-6f, gc.dtype()), mx::array(1.0f, gc.dtype()))),
        -1);
    auto cumg = mx::exp(lcg);
    auto lower_incl = mx::tril(mx::ones({C, C}, mx::float32), 0);
    auto strict_lower = mx::tril(mx::ones({C, C}, mx::float32), -1);
    auto diff = mx::where(
        lower_incl > mx::array(0.0f, mx::float32),
        mx::expand_dims(lcg, -1) - mx::expand_dims(lcg, -2),
        mx::array(-1e30f, lcg.dtype()));
    auto dr = mx::exp(diff);

    auto KK = mx::matmul(kc, mx::swapaxes(kc, -1, -2));
    auto A = mx::expand_dims(betac, -1) * dr * KK * strict_lower;
    auto Tinv = invert_unit_lower(mx::eye(C, mx::float32) + A, C);

    auto kbeta = mx::expand_dims(betac * cumg, -1) * kc;
    auto U0 = mx::matmul(Tinv, mx::expand_dims(betac, -1) * vc);
    auto Kt = mx::matmul(Tinv, kbeta);
    auto M = dr * mx::matmul(qc, mx::swapaxes(kc, -1, -2)) * lower_incl;
    auto Qeff = mx::expand_dims(cumg, -1) * qc - mx::matmul(M, Kt);
    auto MU0 = mx::matmul(M, U0);
    auto cumg_last = mx::squeeze(mx::slice(cumg, {0, 0, 0, C - 1}, {B, Hv, nC, C}), -1);
    auto lcg_last = mx::expand_dims(mx::squeeze(mx::slice(lcg, {0, 0, 0, C - 1}, {B, Hv, nC, C}), -1), -1);
    auto ratio_last = mx::exp(lcg_last - lcg);

    auto S = mx::astype(state, mx::float32);
    std::vector<mx::array> ys;
    ys.reserve(nC);
    auto slice_chunk_5d = [&](const mx::array& a, int c, int D) {
        return mx::squeeze(mx::slice(a, {0, 0, c, 0, 0}, {B, Hv, c + 1, C, D}), 2);
    };
    for (int c = 0; c < nC; ++c) {
        auto St = mx::swapaxes(S, -1, -2);
        auto MU0_c = slice_chunk_5d(MU0, c, Dv);
        auto Qeff_c = slice_chunk_5d(Qeff, c, Dk);
        ys.push_back(MU0_c + mx::matmul(Qeff_c, St));
        auto U0_c = slice_chunk_5d(U0, c, Dv);
        auto Kt_c = slice_chunk_5d(Kt, c, Dk);
        auto Uc = U0_c - mx::matmul(Kt_c, St);
        auto rl_c = mx::expand_dims(
            mx::squeeze(mx::slice(ratio_last, {0, 0, c, 0}, {B, Hv, c + 1, C}), 2),
            -1);
        auto Uscaled = rl_c * Uc;
        auto cgl_c = mx::expand_dims(
            mx::squeeze(mx::slice(cumg_last, {0, 0, c}, {B, Hv, c + 1}), 2),
            {-2, -1});
        auto kc_c = slice_chunk_5d(kc, c, Dk);
        S = cgl_c * S + mx::matmul(mx::swapaxes(Uscaled, -1, -2), kc_c);
    }
    auto Y_full = mx::reshape(
        mx::transpose(mx::stack(ys, 2), {0, 2, 3, 1, 4}),
        {B, Tp, Hv, Dv});
    auto Y = mx::astype(mx::slice(Y_full, {0, 0, 0, 0}, {B, T, Hv, Dv}), in_dtype);
    mx::async_eval({Y, S});
    return {Y, S};
}

std::tuple<mx::array, mx::array, mx::array> Qwen38Engine::run_hc(
    const HyperConnection& hc,
    const mx::array& h,
    const std::optional<std::pair<mx::array, mx::array>>& write) {
    init_kernels();
    int batch = h.shape(0);
    int seq = h.shape(1);
    int rows = batch * seq;
    int hc_count = hc.hc_count;
    int hidden = hc.hidden_size;
    int lowrank = hc.hc_lowrank;
    int width = hc_count * hidden;
    auto dtype = h.dtype();
    auto flat = mx::reshape(h, {rows, width});
    int inj_flag = hc.use_combine ? 1 : 0;

    const QLinear& inj_proj = hc.use_combine ? hc.inject : hc.down;

    if (rows <= 6) {
        int rps = (rows <= 2) ? 1 : 2;
        int down_row_groups = lowrank / (8 * rps) + inj_flag;
        mx::array written = flat;
        mx::array normed = flat;
        mx::array parts = flat;

        std::vector<std::pair<std::string, mx::fast::TemplateArg>> tmpl_ndn = {
            {"T", dtype},
            {"BITS_D", hc.down.bits},
            {"BITS_I", inj_proj.bits},
            {"K", width},
            {"H", hidden},
            {"R", lowrank},
            {"HC", hc_count},
            {"INJ", inj_flag},
            {"RPS", rps}
        };
        std::tuple<int, int, int> grid_ndn = {256, hc_count * down_row_groups, rows};
        std::tuple<int, int, int> tgroup_ndn = {256, 1, 1};

        if (!write.has_value()) {
            auto outs = hc_ndn_kernel_(
                {flat, hc.norm_weight, hc.eps_arr,
                 hc.down.weight, *hc.down.scales, *hc.down.biases,
                 inj_proj.weight, *inj_proj.scales, *inj_proj.biases},
                {{rows, width}, {rows, hc_count, lowrank + hc_count}},
                {dtype, mx::float32},
                grid_ndn, tgroup_ndn, tmpl_ndn, std::nullopt, false, {});
            normed = outs[0];
            parts = outs[1];
            written = h;
        } else {
            auto branch = mx::reshape(write->first, {rows, hidden});
            auto gate = mx::reshape(write->second, {rows, hc_count});
            auto outs = hc_ndnw_kernel_(
                {flat, branch, gate, hc.norm_weight, hc.eps_arr,
                 hc.down.weight, *hc.down.scales, *hc.down.biases,
                 inj_proj.weight, *inj_proj.scales, *inj_proj.biases},
                {{rows, width}, {rows, width}, {rows, hc_count, lowrank + hc_count}},
                {dtype, dtype, mx::float32},
                grid_ndn, tgroup_ndn, tmpl_ndn, std::nullopt, false, {});
            written = mx::reshape(outs[0], h.shape());
            normed = outs[1];
            parts = outs[2];
        }

        int pack_factor = (hc.up.bits == 5) ? 8 : ((hc.up.bits == 6) ? 4 : (32 / hc.up.bits));
        int chunks = lowrank / (2 * pack_factor);
        int threads = chunks * (320 / chunks);
        std::vector<std::pair<std::string, mx::fast::TemplateArg>> tmpl_u2 = {
            {"T", dtype},
            {"BITS_U", hc.up.bits},
            {"K", width},
            {"R", lowrank},
            {"HC", hc_count},
            {"H", hidden},
            {"INJ", inj_flag},
            {"NO", 32},
            {"TGS", threads}
        };
        std::tuple<int, int, int> grid_u2 = {threads, width / 32, rows};
        std::tuple<int, int, int> tgroup_u2 = {threads, 1, 1};
        auto outs_u2 = hc_u2_kernel_(
            {normed, parts, hc.up.weight, *hc.up.scales, *hc.up.biases},
            {{rows, hidden}, {rows, hc_count}},
            {dtype, dtype},
            grid_u2, tgroup_u2, tmpl_u2, std::nullopt, false, {});
        auto mixed = mx::reshape(outs_u2[0], {batch, seq, hidden});
        auto inj = mx::reshape(outs_u2[1], {batch, seq, hc_count});
        return {mixed, written, inj};
    }

    if (rows <= 16) {
        mx::array written = h;
        mx::array normed = flat;
        if (!write.has_value()) {
            normed = hc_n_kernel_(
                {flat, hc.norm_weight, hc.eps_arr},
                {{rows, width}}, {dtype},
                {256, hc_count, rows}, {256, 1, 1},
                {{"T", dtype}, {"K", width}, {"H", hidden}},
                std::nullopt, false, {})[0];
        } else {
            auto branch = mx::reshape(write->first, {rows, hidden});
            auto gate = mx::reshape(write->second, {rows, hc_count});
            auto outs = hc_wnd_kernel_(
                {flat, branch, gate, hc.norm_weight, hc.eps_arr},
                {{rows, width}, {rows, width}}, {dtype, dtype},
                {256, hc_count, rows}, {256, 1, 1},
                {{"T", dtype}, {"K", width}, {"H", hidden}, {"HC", hc_count}},
                std::nullopt, false, {});
            written = mx::reshape(outs[0], h.shape());
            normed = outs[1];
            }
        auto outs_d = hc_d_kernel_(
            {normed, hc.down.weight, *hc.down.scales, *hc.down.biases,
             inj_proj.weight, *inj_proj.scales, *inj_proj.biases},
            {{rows, lowrank}, {rows, hc_count}}, {dtype, dtype},
            {32, 8 * (lowrank / 8 + 1), rows}, {32, 8, 1},
            {{"T", dtype}, {"BITS_D", hc.down.bits}, {"BITS_I", inj_proj.bits},
             {"K", width}, {"R", lowrank}, {"HC", hc_count}, {"INJ", inj_flag}},
            std::nullopt, false, {});
        auto mixed = hc_u_kernel_(
            {normed, outs_d[0], hc.up.weight, *hc.up.scales, *hc.up.biases},
            {{rows, hidden}}, {dtype},
            {256, hidden / 64, rows}, {256, 1, 1},
            {{"T", dtype}, {"BITS_U", hc.up.bits}, {"K", width},
             {"R", lowrank}, {"HC", hc_count}, {"H", hidden}},
            std::nullopt, false, {})[0];
        return {mx::reshape(mixed, {batch, seq, hidden}),
                written,
                mx::reshape(outs_d[1], {batch, seq, hc_count})};
    }

    // Prefill path (rows > 16): bit-identical to hc_fused.prefill_forward
    mx::array written = h;
    mx::array normed = flat;
    if (!write.has_value()) {
        normed = hc_np_kernel_(
            {flat, hc.norm_weight, hc.eps_arr},
            {{rows, width}}, {dtype},
            {256, hc_count, rows}, {256, 1, 1},
            {{"T", dtype}, {"K", width}, {"H", hidden}},
            std::nullopt, false, {})[0];
    } else {
        auto branch = mx::reshape(write->first, {rows, hidden});
        auto gate = mx::reshape(write->second, {rows, hc_count});
        auto outs = hc_wn_kernel_(
            {flat, branch, gate, hc.norm_weight, hc.eps_arr},
            {{rows, width}, {rows, width}}, {dtype, dtype},
            {256, hc_count, rows}, {256, 1, 1},
            {{"T", dtype}, {"K", width}, {"H", hidden}, {"HC", hc_count}},
            std::nullopt, false, {});
        written = mx::reshape(outs[0], h.shape());
        normed = outs[1];
    }
    auto normed_3d = mx::reshape(normed, h.shape());
    auto mix = hc_act_compiled_({hc.down.forward(normed_3d)})[0];
    auto up = hc.up.forward(mix);
    if (!hc.use_combine) {
        auto mixed = hc_tail_compiled_({up, normed_3d})[0];
        return {mixed, written, mx::zeros({batch, seq, hc_count}, dtype)};
    }
    auto outs_ti = hc_ti_kernel_(
        {mx::reshape(up, {rows, width}), normed,
         hc.inject.weight, *hc.inject.scales, *hc.inject.biases},
        {{rows, hidden}, {rows, hc_count}}, {dtype, dtype},
        {256, 1, rows}, {256, 1, 1},
        {{"T", dtype}, {"BITS_I", hc.inject.bits}, {"K", width}, {"H", hidden}, {"HC", hc_count}},
        std::nullopt, false, {});
    return {mx::reshape(outs_ti[0], {batch, seq, hidden}),
            written,
            mx::reshape(outs_ti[1], {batch, seq, hc_count})};
}

mx::array Qwen38Engine::apply_mrope_one(
    const mx::array& x,
    const mx::array& inv_freq,
    const mx::array& position_ids) {
    init_kernels();
    int B = x.shape(0);
    int H = x.shape(1);
    int L = x.shape(2);
    int D = x.shape(3);
    std::tuple<int, int, int> grid = {D, L, B * H};
    std::tuple<int, int, int> tgroup = {64, 4, 1};
    std::vector<std::pair<std::string, mx::fast::TemplateArg>> tmpl = {{"T", x.dtype()}};
    if (position_ids.ndim() == 2) {
        return mrope_2d_kernel_(
            {x, inv_freq, position_ids},
            {x.shape()}, {x.dtype()},
            grid, tgroup, tmpl, std::nullopt, false, {})[0];
    }
    return mrope_3d_kernel_(
        {x, inv_freq, position_ids},
        {x.shape()}, {x.dtype()},
        grid, tgroup, tmpl, std::nullopt, false, {})[0];
}

} // namespace qwen38

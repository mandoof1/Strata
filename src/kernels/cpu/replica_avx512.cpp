// src/kernels/cpu/replica_avx512.cpp - STRATA_CPU_PARITY: the GPU's native expert arithmetic in AVX-512.
//
// The same values as the GPU's native_expert_grouped, bit for bit (expert_replica checks it against the GPU on every
// layer): q8_1 activations, each 32-value part's integer dot and the format's integer scale step, (dw * d8) * sumi
// accumulated by fma in row_dot_80_sub16's / row_dot_40_sub8's / row_dot_20_sub16's lanes and order, then their
// butterfly; the SwiGLU with sexp.  The i-quant weight decode is iq_avx512.cpp's (grid magnitudes, a sign mask per
// byte).  Gate/up IQ3_XXS, IQ3_S, IQ2_S, IQ4_XS; down IQ4_NL, Q2_0; n_embd 2560, n_ff 640.
#include "strata/kernels/cpu/replica_avx512.hpp"
#include "strata/kernels/sexp.hpp"

#define GGML_COMMON_DECL_CPP
#define GGML_COMMON_IMPL_CPP
#include "ggml-common.h"

#include <immintrin.h>

#include <cmath>
#include <cstdlib>
#include <cstring>

namespace strata::kernels::cpu::replica {
namespace {

inline float h2f(uint16_t h) { return _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128((int) h))); }
inline uint32_t u32(const uint8_t* p) { uint32_t v; std::memcpy(&v, p, 4); return v; }
inline uint16_t u16(const uint8_t* p) { uint16_t v; std::memcpy(&v, p, 2); return v; }
inline uint64_t u64(const uint8_t* p) { uint64_t v; std::memcpy(&v, p, 8); return v; }

// ---- 64 values (chunk j of a 256-value block) -> grid magnitudes and a sign mask (iq_avx512.cpp's decode)
// IQ3_S: the 16 grid entries of a chunk by one vpgatherdd instead of 16 scalar loads - the same table values, so the
// same integer sums; the scalar loads bound this kernel (Zen 5, 8 threads: 49 -> 60 GB/s of weights in a scratch
// benchmark).  STRATA_REPLICA_GATHER=0 keeps the scalar loads.
// iq3b: read on first use, not in a static initializer of this -mavx512* file (see k_idx below)
bool g_iq3s_gather_on() {
    static const bool on = [] { const char* e = std::getenv("STRATA_REPLICA_GATHER"); return !(e != nullptr && e[0] == '0'); }();
    return on;
}
template <int TY> struct Dec;
template <> struct Dec<21> {   // IQ3_S
    static constexpr int bytes = 110;
    static inline void decode(const uint8_t* b, int j, __m512i& g, __mmask64& m) {
        if (g_iq3s_gather_on()) {
            const __m512i idx = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*) (b + 2 + 16 * j)));
            const __mmask16 hi = (__mmask16) (b[66 + 2 * j] | (b[66 + 2 * j + 1] << 8));   // bit k: index k's 9th bit
            g = _mm512_i32gather_epi32(_mm512_mask_add_epi32(idx, hi, idx, _mm512_set1_epi32(256)), (const void*) iq3s_grid, 4);
            m = _cvtu64_mask64(u64(b + 74 + 8 * j));
            return;
        }
        const uint8_t* q = b + 2 + 16 * j;
        const uint32_t h0 = b[66 + 2 * j], h1 = b[66 + 2 * j + 1];
#define G3(k, h, kk) (int) iq3s_grid[q[k] | (((h >> kk) & 1) << 8)]
        g = _mm512_set_epi32(G3(15, h1, 7), G3(14, h1, 6), G3(13, h1, 5), G3(12, h1, 4), G3(11, h1, 3), G3(10, h1, 2),
                             G3(9, h1, 1), G3(8, h1, 0), G3(7, h0, 7), G3(6, h0, 6), G3(5, h0, 5), G3(4, h0, 4),
                             G3(3, h0, 3), G3(2, h0, 2), G3(1, h0, 1), G3(0, h0, 0));
#undef G3
        m = _cvtu64_mask64(u64(b + 74 + 8 * j));
    }
    // the integer scale of part sb (0..7) of a block
    static inline int ls(const uint8_t* b, int sb) { return 1 + 2 * ((b[106 + sb / 2] >> (4 * (sb % 2))) & 0x0F); }
};
template <> struct Dec<18> {   // IQ3_XXS
    static constexpr int bytes = 98;
    static inline void decode(const uint8_t* b, int j, __m512i& g, __mmask64& m) {
        const uint8_t* q = b + 2 + 16 * j;
        g = _mm512_set_epi32((int) iq3xxs_grid[q[15]], (int) iq3xxs_grid[q[14]], (int) iq3xxs_grid[q[13]], (int) iq3xxs_grid[q[12]],
                             (int) iq3xxs_grid[q[11]], (int) iq3xxs_grid[q[10]], (int) iq3xxs_grid[q[9]], (int) iq3xxs_grid[q[8]],
                             (int) iq3xxs_grid[q[7]], (int) iq3xxs_grid[q[6]], (int) iq3xxs_grid[q[5]], (int) iq3xxs_grid[q[4]],
                             (int) iq3xxs_grid[q[3]], (int) iq3xxs_grid[q[2]], (int) iq3xxs_grid[q[1]], (int) iq3xxs_grid[q[0]]);
        const uint32_t a = u32(b + 2 + 64 + 8 * j), c = u32(b + 2 + 64 + 8 * j + 4);
        const uint64_t s = (uint64_t) ksigns_iq2xs[a & 127] | ((uint64_t) ksigns_iq2xs[(a >> 7) & 127] << 8) |
                           ((uint64_t) ksigns_iq2xs[(a >> 14) & 127] << 16) | ((uint64_t) ksigns_iq2xs[(a >> 21) & 127] << 24) |
                           ((uint64_t) ksigns_iq2xs[c & 127] << 32) | ((uint64_t) ksigns_iq2xs[(c >> 7) & 127] << 40) |
                           ((uint64_t) ksigns_iq2xs[(c >> 14) & 127] << 48) | ((uint64_t) ksigns_iq2xs[(c >> 21) & 127] << 56);
        m = _cvtu64_mask64(s);
    }
    static inline int ls(const uint8_t* b, int sb) { return b[2 + 64 + 4 * sb + 3] >> 4; }
};
template <> struct Dec<22> {   // IQ2_S
    static constexpr int bytes = 82;
    static inline void decode(const uint8_t* b, int j, __m512i& g, __mmask64& m) {
        const uint8_t* qs = b + 2 + 8 * j;
        const uint8_t h0 = b[66 + 2 * j], h1 = b[66 + 2 * j + 1];
        g = _mm512_set_epi64((long long) iq2s_grid[qs[7] | ((h1 << 2) & 0x300)], (long long) iq2s_grid[qs[6] | ((h1 << 4) & 0x300)],
                             (long long) iq2s_grid[qs[5] | ((h1 << 6) & 0x300)], (long long) iq2s_grid[qs[4] | ((h1 << 8) & 0x300)],
                             (long long) iq2s_grid[qs[3] | ((h0 << 2) & 0x300)], (long long) iq2s_grid[qs[2] | ((h0 << 4) & 0x300)],
                             (long long) iq2s_grid[qs[1] | ((h0 << 6) & 0x300)], (long long) iq2s_grid[qs[0] | ((h0 << 8) & 0x300)]);
        m = _cvtu64_mask64(u64(b + 2 + 32 + 8 * j));
    }
    static inline int ls0(const uint8_t* b, int sb) { return b[74 + sb] & 0x0F; }
    static inline int ls1(const uint8_t* b, int sb) { return b[74 + sb] >> 4; }
};

template <> struct Dec<23> {   // IQ4_XS: d, scales_h, scales_l[4], qs[128] (sub-block sb: qs[16 sb ..], low nibbles first)
    static constexpr int bytes = 136;
    static inline void decode(const uint8_t* b, int j, __m512i& g, __mmask64& m) {
        const __m128i tbl = _mm_loadu_si128((const __m128i*) kvalues_iq4nl), mk = _mm_set1_epi8(0x0F);
        const __m128i q0 = _mm_loadu_si128((const __m128i*) (b + 8 + 32 * j)), q1 = _mm_loadu_si128((const __m128i*) (b + 8 + 32 * j + 16));
        const __m128i l0 = _mm_shuffle_epi8(tbl, _mm_and_si128(q0, mk)), h0 = _mm_shuffle_epi8(tbl, _mm_and_si128(_mm_srli_epi16(q0, 4), mk));
        const __m128i l1 = _mm_shuffle_epi8(tbl, _mm_and_si128(q1, mk)), h1 = _mm_shuffle_epi8(tbl, _mm_and_si128(_mm_srli_epi16(q1, 4), mk));
        const __m512i w = _mm512_inserti64x4(_mm512_castsi256_si512(_mm256_set_m128i(h0, l0)), _mm256_set_m128i(h1, l1), 1);
        m = _mm512_movepi8_mask(w);
        g = _mm512_abs_epi8(w);
    }
    static inline int ls(const uint8_t* b, int sb) {
        return (((b[4 + sb / 2] >> (4 * (sb % 2))) & 0x0F) | (((u16(b + 2) >> (2 * sb)) & 3) << 4)) - 32;
    }
};

// C's truncating division by 2 and 4 (the GPU's int `/`)
inline __m512i div2(__m512i v) { return _mm512_srai_epi32(_mm512_add_epi32(v, _mm512_srli_epi32(v, 31)), 1); }
inline __m512i div4(__m512i v) { return _mm512_srai_epi32(_mm512_add_epi32(v, _mm512_srli_epi32(_mm512_srai_epi32(v, 31), 30)), 2); }

// lane l of v + lane l ^ o, o = 8, 4, 2, 1 (the 16-lane butterfly; each level's adds are commutative, so lane 0 has
// the GPU's value), then lane 0
inline float butterfly16(__m512 s) {
    s = _mm512_add_ps(s, _mm512_shuffle_f32x4(s, s, 0x4E));
    s = _mm512_add_ps(s, _mm512_shuffle_f32x4(s, s, 0xB1));
    s = _mm512_add_ps(s, _mm512_permute_ps(s, 0x4E));
    s = _mm512_add_ps(s, _mm512_permute_ps(s, 0xB1));
    return _mm512_cvtss_f32(s);
}
inline float butterfly8(__m256 s) {
    s = _mm256_add_ps(s, _mm256_permute2f128_ps(s, s, 0x01));
    s = _mm256_add_ps(s, _mm256_permute_ps(s, 0x4E));
    s = _mm256_add_ps(s, _mm256_permute_ps(s, 0xB1));
    return _mm256_cvtss_f32(s);
}

// the half-part sums of block i: after the reduction lane 4L + j of the result is chunk j's 128-bit lane L, i.e. the
// 16 values 64 j + 16 L .. of the block = part 2 j + L / 2, half L % 2.  idx0 / idx1 pick, for the 16 parts of two
// blocks (lanes: parts 8 q2 + sb), half 0 / half 1 out of (block 2q2 | block 2q2+1).
struct Idx { __m512i h0, h1; };
Idx make_idx() {
    alignas(64) int a[16], b[16];
    for (int p = 0; p < 16; ++p) {
        const int blk = p / 8, sb = p % 8, j = sb / 2, L0 = (sb % 2) * 2;
        a[p] = blk * 16 + 4 * L0 + j;
        b[p] = blk * 16 + 4 * (L0 + 1) + j;
    }
    return {_mm512_load_si512(a), _mm512_load_si512(b)};
}
// iq3b: a function-local static, built on first use - a namespace-scope one ran AVX-512 instructions in the static
// initializer at program start and killed the engine with SIGILL on CPUs without AVX-512 (this file is built with
// -mavx512*; only STRATA_CPU_PARITY on an AVX-512 CPU calls into it)
const Idx& k_idx() {
    static const Idx v = make_idx();
    return v;
}

inline __m512i reduce4(__m512i a0, __m512i a1, __m512i a2, __m512i a3) {
    const __m512i t0 = _mm512_add_epi32(_mm512_unpacklo_epi32(a0, a1), _mm512_unpackhi_epi32(a0, a1));
    const __m512i t1 = _mm512_add_epi32(_mm512_unpacklo_epi32(a2, a3), _mm512_unpackhi_epi32(a2, a3));
    return _mm512_add_epi32(_mm512_unpacklo_epi64(t0, t1), _mm512_unpackhi_epi64(t0, t1));
}

// one gate or up row (n_embd 2560: 10 blocks, 80 parts) against NT tokens
template <int TY, int NT>
inline void row80(const uint8_t* row, const ActX* const* y, float* res) {
    constexpr int NB = 10;
    __m512i H[NT][NB];   // per token: the 16 half-part sums of each block
    const __m512i zero = _mm512_setzero_si512();
    for (int i = 0; i < NB; ++i) {
        const uint8_t* blk = row + (size_t) i * Dec<TY>::bytes;
        _mm_prefetch((const char*) blk + 2048, _MM_HINT_T0);
        _mm_prefetch((const char*) blk + 2048 + 64, _MM_HINT_T0);
        __m512i acc[NT][4];
        for (int j = 0; j < 4; ++j) {
            __m512i g;
            __mmask64 m;
            Dec<TY>::decode(blk, j, g, m);
            for (int t = 0; t < NT; ++t) {
                const __m512i yv = _mm512_loadu_si512((const void*) (y[t]->qs + 256 * i + 64 * j));
                acc[t][j] = _mm512_dpbusd_epi32(zero, g, _mm512_mask_sub_epi8(yv, m, zero, yv));
            }
        }
        for (int t = 0; t < NT; ++t) H[t][i] = reduce4(acc[t][0], acc[t][1], acc[t][2], acc[t][3]);
    }
    // per part: the integer scale(s) and dw (the row's, shared by the tokens), 16 parts (two blocks) a vector
    __m512i LA[5], LB[5];
    __m512 DW[5];
    for (int q = 0; q < 5; ++q) {
        const uint8_t* b0 = row + (size_t) (2 * q) * Dec<TY>::bytes;
        const uint8_t* b1 = b0 + Dec<TY>::bytes;
        DW[q] = _mm512_insertf32x8(_mm512_set1_ps(h2f(u16(b0))), _mm256_set1_ps(h2f(u16(b1))), 1);
        LB[q] = zero;
        if constexpr (TY == 21) {   // scales[4] at 106: part sb = nibble sb % 2 of byte sb / 2
            const __m128i v = _mm_set_epi32(0, 0, (int) u32(b1 + 106), (int) u32(b0 + 106));
            const __m128i dup = _mm_shuffle_epi8(v, _mm_set_epi8(7, 7, 6, 6, 5, 5, 4, 4, 3, 3, 2, 2, 1, 1, 0, 0));
            const __m512i nib = _mm512_and_si512(_mm512_srlv_epi32(_mm512_cvtepu8_epi32(dup),
                                                                   _mm512_set_epi32(4, 0, 4, 0, 4, 0, 4, 0, 4, 0, 4, 0, 4, 0, 4, 0)),
                                                 _mm512_set1_epi32(15));
            LA[q] = _mm512_add_epi32(_mm512_add_epi32(nib, nib), _mm512_set1_epi32(1));
        } else if constexpr (TY == 23) {   // ls - 32 (6-bit scales; one layer of the model)
            alignas(64) int l[16];
            for (int sb = 0; sb < 8; ++sb) { l[sb] = Dec<23>::ls(b0, sb); l[8 + sb] = Dec<23>::ls(b1, sb); }
            LA[q] = _mm512_load_si512(l);
        } else if constexpr (TY == 18) {   // the top nibble of each of the 8 aux words at 66
            LA[q] = _mm512_srli_epi32(_mm512_inserti64x4(_mm512_castsi256_si512(_mm256_loadu_si256((const __m256i*) (b0 + 66))),
                                                         _mm256_loadu_si256((const __m256i*) (b1 + 66)), 1), 28);
        } else {   // IQ2_S: scales[8] at 74, low / high nibble
            const __m512i sc = _mm512_cvtepu8_epi32(_mm_set_epi64x((long long) u64(b1 + 74), (long long) u64(b0 + 74)));
            LA[q] = _mm512_and_si512(sc, _mm512_set1_epi32(15));
            LB[q] = _mm512_srli_epi32(sc, 4);
        }
    }
    const __m512 zf = _mm512_setzero_ps();
    for (int t = 0; t < NT; ++t) {
        __m512 D[5], S[5];
        for (int q = 0; q < 5; ++q) {
            const __m512i h0 = _mm512_permutex2var_epi32(H[t][2 * q], k_idx().h0, H[t][2 * q + 1]);
            const __m512i h1 = _mm512_permutex2var_epi32(H[t][2 * q], k_idx().h1, H[t][2 * q + 1]);
            __m512i si;
            if constexpr (TY == 22) {   // (s0 * ls0 + s1 * ls1 + (s0 + s1) / 2) / 4
                si = div4(_mm512_add_epi32(_mm512_add_epi32(_mm512_mullo_epi32(h0, LA[q]), _mm512_mullo_epi32(h1, LB[q])),
                                           div2(_mm512_add_epi32(h0, h1))));
            } else if constexpr (TY == 18) {   // (ls * s + s / 2) / 2
                const __m512i sm = _mm512_add_epi32(h0, h1);
                si = div2(_mm512_add_epi32(_mm512_mullo_epi32(LA[q], sm), div2(sm)));
            } else {   // IQ3_S: s * ls, IQ4_XS: s * (ls - 32)
                si = _mm512_mullo_epi32(_mm512_add_epi32(h0, h1), LA[q]);
            }
            D[q] = _mm512_mul_ps(DW[q], _mm512_loadu_ps(y[t]->d8 + 16 * q));
            S[q] = _mm512_cvtepi32_ps(si);
        }
        // row_dot_80_sub16's lane l: a = q0 [fma] q2 [fma] q4, b = q1 [fma] q3, then a + b and the butterfly
        __m512 a = _mm512_fmadd_ps(D[0], S[0], zf);
        a = _mm512_fmadd_ps(D[2], S[2], a);
        a = _mm512_fmadd_ps(D[4], S[4], a);
        __m512 b = _mm512_fmadd_ps(D[1], S[1], zf);
        b = _mm512_fmadd_ps(D[3], S[3], b);
        res[t] = butterfly16(_mm512_add_ps(a, b));
    }
}

template <int TY, int NT>
void gu_rows_t(const uint8_t* blob, size_t gu_row, size_t up_off, const ActX* const* act, float* const* hout, int r0, int r1) {
    float g[NT], u[NT];
    for (int r = r0; r < r1; ++r) {
        row80<TY, NT>(blob + (size_t) r * gu_row, act, g);
        row80<TY, NT>(blob + up_off + (size_t) r * gu_row, act, u);
        for (int t = 0; t < NT; ++t) hout[t][r] = (g[t] / (1.0f + strata::kernels::sexp(-g[t]))) * u[t];
    }
}
template <int TY>
void gu_rows_nt(int nt, const uint8_t* blob, size_t gu_row, size_t up_off, const ActX* const* act, float* const* h, int r0, int r1) {
    for (int t0 = 0; t0 < nt; t0 += 4) {
        const int k = nt - t0 < 4 ? nt - t0 : 4;
        switch (k) {
            case 1: gu_rows_t<TY, 1>(blob, gu_row, up_off, act + t0, h + t0, r0, r1); break;
            case 2: gu_rows_t<TY, 2>(blob, gu_row, up_off, act + t0, h + t0, r0, r1); break;
            case 3: gu_rows_t<TY, 3>(blob, gu_row, up_off, act + t0, h + t0, r0, r1); break;
            default: gu_rows_t<TY, 4>(blob, gu_row, up_off, act + t0, h + t0, r0, r1); break;
        }
    }
}

// ---- IQ4_NL down rows (n_ff 640: 20 blocks of 32, 40 parts of 16 values)
inline __m512i iq4nl_pair(const uint8_t* b0) {   // two blocks' 64 values as int8 (kvalues) in part order:
    // per block part 0 (values 0-7, 16-23) then part 1 (8-15, 24-31) - Act::qp's order
    const __m128i tbl = _mm_loadu_si128((const __m128i*) kvalues_iq4nl);
    const __m128i q0 = _mm_loadu_si128((const __m128i*) (b0 + 2)), q1 = _mm_loadu_si128((const __m128i*) (b0 + 18 + 2));
    const __m128i mk = _mm_set1_epi8(0x0F);
    const __m128i lo0 = _mm_shuffle_epi8(tbl, _mm_and_si128(q0, mk)), hi0 = _mm_shuffle_epi8(tbl, _mm_and_si128(_mm_srli_epi16(q0, 4), mk));
    const __m128i lo1 = _mm_shuffle_epi8(tbl, _mm_and_si128(q1, mk)), hi1 = _mm_shuffle_epi8(tbl, _mm_and_si128(_mm_srli_epi16(q1, 4), mk));
    const __m256i b0v = _mm256_set_m128i(_mm_unpackhi_epi64(lo0, hi0), _mm_unpacklo_epi64(lo0, hi0));
    const __m256i b1v = _mm256_set_m128i(_mm_unpackhi_epi64(lo1, hi1), _mm_unpacklo_epi64(lo1, hi1));
    return _mm512_inserti64x4(_mm512_castsi256_si512(b0v), b1v, 1);
}

template <int NT>
inline void row40(const uint8_t* row, const ActH* const* y, float* res) {
    // per pair: 128-bit lane p = part 4 pr + p (4-value partial sums); four pairs reduce to one vector of 16 parts
    __m512i V[NT][12];
    for (int pr = 0; pr < 10; ++pr) {
        const __m512i w = iq4nl_pair(row + (size_t) pr * 36);
        const __mmask64 neg = _mm512_movepi8_mask(w);
        const __m512i wa = _mm512_abs_epi8(w);
        for (int t = 0; t < NT; ++t) {
            const __m512i yv = _mm512_loadu_si512((const void*) (y[t]->q + 64 * pr));
            const __m512i ys = _mm512_mask_sub_epi8(yv, neg, _mm512_setzero_si512(), yv);
            V[t][pr] = _mm512_dpbusd_epi32(_mm512_setzero_si512(), wa, ys);
        }
    }
    for (int t = 0; t < NT; ++t) V[t][10] = V[t][11] = _mm512_setzero_si512();
    // the 20 blocks' d (a gather at stride 18; the low half of each word), each twice (two parts per block)
    const __m512i gi = _mm512_mullo_epi32(_mm512_set_epi32(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0), _mm512_set1_epi32(18));
    const __m512i w16 = _mm512_i32gather_epi32(gi, (const void*) row, 1);
    const __m512 dlo = _mm512_cvtph_ps(_mm512_cvtepi32_epi16(w16));
    const __m128i w4 = _mm_i32gather_epi32((const int*) (row + 16 * 18), _mm_set_epi32(54, 36, 18, 0), 1);
    const __m512 dhi = _mm512_castps256_ps512(_mm256_castps128_ps256(_mm_cvtph_ps(_mm_cvtepi32_epi16(w4))));
    alignas(64) float dw[48];
    {
        const __m512i dup_lo = _mm512_set_epi32(7, 7, 6, 6, 5, 5, 4, 4, 3, 3, 2, 2, 1, 1, 0, 0);
        const __m512i dup_hi = _mm512_set_epi32(15, 15, 14, 14, 13, 13, 12, 12, 11, 11, 10, 10, 9, 9, 8, 8);
        _mm512_store_ps(dw, _mm512_permutexvar_ps(dup_lo, dlo));
        _mm512_store_ps(dw + 16, _mm512_permutexvar_ps(dup_hi, dlo));
        _mm512_store_ps(dw + 32, _mm512_permutexvar_ps(dup_lo, dhi));
    }
    // reduce4 of pairs 4g..4g+3: lane 4 L + j = part 4 (4 g + j) + L; back to part order
    const __m512i tr = _mm512_set_epi32(15, 11, 7, 3, 14, 10, 6, 2, 13, 9, 5, 1, 12, 8, 4, 0);
    for (int t = 0; t < NT; ++t) {
        alignas(64) int si[48];
        for (int g = 0; g < 3; ++g)
            _mm512_store_si512(si + 16 * g, _mm512_permutexvar_epi32(tr, reduce4(V[t][4 * g], V[t][4 * g + 1], V[t][4 * g + 2], V[t][4 * g + 3])));
        __m256 D[5], S[5];
        for (int q = 0; q < 5; ++q) {
            D[q] = _mm256_mul_ps(_mm256_load_ps(dw + 8 * q), _mm256_loadu_ps(y[t]->d + 8 * q));
            S[q] = _mm256_cvtepi32_ps(_mm256_load_si256((const __m256i*) (si + 8 * q)));
        }
        const __m256 z = _mm256_setzero_ps();
        __m256 a = _mm256_fmadd_ps(D[0], S[0], z);
        a = _mm256_fmadd_ps(D[4], S[4], a);
        a = _mm256_add_ps(a, _mm256_fmadd_ps(D[2], S[2], z));
        a = _mm256_add_ps(a, _mm256_add_ps(_mm256_fmadd_ps(D[1], S[1], z), _mm256_fmadd_ps(D[3], S[3], z)));
        res[t] = butterfly8(a);
    }
}
template <int NT>
void down_t(const uint8_t* w, size_t row_bytes, const ActH* const* h, float* const* out, int r0, int r1) {
    float res[NT];
    for (int r = r0; r < r1; ++r) {
        const uint8_t* row = w + (size_t) r * row_bytes;
        _mm_prefetch((const char*) row + 2048, _MM_HINT_T0);
        _mm_prefetch((const char*) row + 2048 + 64, _MM_HINT_T0);
        row40<NT>(row, h, res);
        for (int t = 0; t < NT; ++t) out[t][r] = res[t];
    }
}


// ---- Q2_0 down rows (n_ff 640: 10 blocks of 64, 20 parts of 32 values = the h q8_1 blocks)
// one block's 64 two-bit codes (0..3; the weight is code - 1) in value order
inline __m512i q2_codes(const uint8_t* blk) {
    const __m128i q = _mm_loadu_si128((const __m128i*) (blk + 2)), m3 = _mm_set1_epi8(3);
    const __m128i c0 = _mm_and_si128(q, m3), c1 = _mm_and_si128(_mm_srli_epi16(q, 2), m3);
    const __m128i c2 = _mm_and_si128(_mm_srli_epi16(q, 4), m3), c3 = _mm_and_si128(_mm_srli_epi16(q, 6), m3);
    const __m128i a0 = _mm_unpacklo_epi8(c0, c1), b0 = _mm_unpacklo_epi8(c2, c3);
    const __m128i a1 = _mm_unpackhi_epi8(c0, c1), b1 = _mm_unpackhi_epi8(c2, c3);
    const __m256i lo = _mm256_set_m128i(_mm_unpackhi_epi16(a0, b0), _mm_unpacklo_epi16(a0, b0));
    const __m256i hi = _mm256_set_m128i(_mm_unpackhi_epi16(a1, b1), _mm_unpacklo_epi16(a1, b1));
    return _mm512_inserti64x4(_mm512_castsi256_si512(lo), hi, 1);
}

template <int NT>
inline void row20(const uint8_t* row, const ActH* const* y, float* res) {
    __m512i V[NT][12];
    for (int b = 0; b < 10; ++b) {
        const __m512i c = q2_codes(row + (size_t) b * 18);
        for (int t = 0; t < NT; ++t)
            V[t][b] = _mm512_dpbusd_epi32(_mm512_setzero_si512(), c, _mm512_loadu_si512((const void*) (y[t]->q + 64 * b)));
    }
    for (int t = 0; t < NT; ++t) V[t][10] = V[t][11] = _mm512_setzero_si512();
    // the 10 blocks' d (a gather at stride 18), each twice (two parts per block)
    const __m512i gi = _mm512_mullo_epi32(_mm512_set_epi32(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0), _mm512_set1_epi32(18));
    const __m512i w16 = _mm512_mask_i32gather_epi32(_mm512_setzero_si512(), (__mmask16) 0x03FF, gi, (const void*) row, 1);
    const __m512 dd = _mm512_cvtph_ps(_mm512_cvtepi32_epi16(w16));
    const __m512 DW0 = _mm512_permutexvar_ps(_mm512_set_epi32(7, 7, 6, 6, 5, 5, 4, 4, 3, 3, 2, 2, 1, 1, 0, 0), dd);
    const __m512 DW1 = _mm512_maskz_permutexvar_ps((__mmask16) 0x000F, _mm512_set_epi32(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 9, 9, 8, 8), dd);
    // reduce4 of blocks 4g..: lane 4 L + j = block 4g+j's 16 values 16 L ..; + the neighbouring 128-bit lane gives part
    // 2(4g+j) (lane j) and 2(4g+j)+1 (lane 8+j)
    const __m512i pick = _mm512_set_epi32(0, 0, 0, 0, 0, 0, 0, 0, 11, 3, 10, 2, 9, 1, 8, 0);
    for (int t = 0; t < NT; ++t) {
        alignas(64) int si[32];
        for (int g = 0; g < 3; ++g) {
            __m512i r = reduce4(V[t][4 * g], V[t][4 * g + 1], V[t][4 * g + 2], V[t][4 * g + 3]);
            r = _mm512_add_epi32(r, _mm512_shuffle_i32x4(r, r, 0xB1));
            _mm256_store_si256((__m256i*) (si + 8 * g), _mm512_castsi512_si256(_mm512_permutexvar_epi32(pick, r)));
        }
        // the weight is code - 1: subtract each part's activation sum
        const __m512i S0i = _mm512_sub_epi32(_mm512_load_si512(si), _mm512_loadu_si512((const void*) y[t]->ysum));
        const __m512i S1i = _mm512_maskz_sub_epi32((__mmask16) 0x000F, _mm512_load_si512(si + 16), _mm512_maskz_loadu_epi32((__mmask16) 0x000F, y[t]->ysum + 16));
        const __m512 D0 = _mm512_mul_ps(DW0, _mm512_loadu_ps(y[t]->d));
        const __m512 D1 = _mm512_mul_ps(DW1, _mm512_maskz_loadu_ps((__mmask16) 0x000F, y[t]->d + 16));
        const __m512 z = _mm512_setzero_ps();
        // row_dot_20_sub16's lane l: part l, + part l + 16 (lanes < 4; +0 elsewhere), then the butterfly
        const __m512 a = _mm512_fmadd_ps(D0, _mm512_cvtepi32_ps(S0i), z);
        const __m512 b = _mm512_fmadd_ps(D1, _mm512_cvtepi32_ps(S1i), z);
        res[t] = butterfly16(_mm512_add_ps(a, b));
    }
}
template <int NT>
void down20_t(const uint8_t* w, size_t row_bytes, const ActH* const* h, float* const* out, int r0, int r1) {
    float res[NT];
    for (int r = r0; r < r1; ++r) {
        const uint8_t* row = w + (size_t) r * row_bytes;
        _mm_prefetch((const char*) row + 2048, _MM_HINT_T0);
        _mm_prefetch((const char*) row + 2048 + 64, _MM_HINT_T0);
        row20<NT>(row, h, res);
        for (int t = 0; t < NT; ++t) out[t][r] = res[t];
    }
}

// the GPU's q8_1 block (quantize_q8_1_kernel / q8_1_store): the int8 values of 32 inputs and the fp16 scale as a
// float.  amax is a max (any order); the block sum (ds.y) is not read by these formats' dots.
inline float q8_1_block(const float* xb, int8_t* q) {
    const __m512 absm = _mm512_castsi512_ps(_mm512_set1_epi32(0x7FFFFFFF));
    const __m512 v0 = _mm512_loadu_ps(xb), v1 = _mm512_loadu_ps(xb + 16);
    const float am = _mm512_reduce_max_ps(_mm512_max_ps(_mm512_and_ps(v0, absm), _mm512_and_ps(v1, absm)));
    float d = am / 127.0f;
    if (std::fabs(d) > 65504.0f) d = std::copysign(65504.0f, d);   // q8_1_finite
    if (am == 0.0f) {
        std::memset(q, 0, 32);
    } else {
        const __m512 dv = _mm512_set1_ps(d);
        for (int h = 0; h < 2; ++h) {
            const __m512 t = _mm512_div_ps(h ? v1 : v0, dv);
            // roundf: half away from zero.  t - trunc(t) is exact; |frac| >= 0.5 rounds away
            const __m512 tr = _mm512_roundscale_ps(t, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
            const __m512 fr = _mm512_and_ps(_mm512_sub_ps(t, tr), absm);
            const __mmask16 up = _mm512_cmp_ps_mask(fr, _mm512_set1_ps(0.5f), _CMP_GE_OQ);
            const __m512 one = _mm512_or_ps(_mm512_set1_ps(1.0f), _mm512_andnot_ps(absm, t));   // copysign(1, t)
            __m512 r = _mm512_mask_add_ps(tr, up, tr, one);
            r = _mm512_min_ps(_mm512_max_ps(r, _mm512_set1_ps(-127.0f)), _mm512_set1_ps(127.0f));
            _mm_storeu_si128((__m128i*) (q + 16 * h), _mm512_cvtsepi32_epi8(_mm512_cvtps_epi32(r)));
        }
    }
    return h2f(_cvtss_sh(d, _MM_FROUND_TO_NEAREST_INT));
}

}  // namespace

void quant_x(const float* x, int n, ActX& a) {
    for (int b = 0; b < n / 32; ++b) a.d8[b] = q8_1_block(x + 32 * b, a.qs + 32 * b);
}

void quant_h(const float* h, int n, int d_type, ActH& a) {
    alignas(16) int8_t q[32];
    for (int b = 0; b < n / 32; ++b) {
        const float d = q8_1_block(h + 32 * b, q);
        if (d_type == 20) {   // IQ4_NL: part order (values 0-7, 16-23, 8-15, 24-31), each scale twice
            const __m128i l = _mm_load_si128((const __m128i*) q), hh = _mm_load_si128((const __m128i*) (q + 16));
            _mm_storeu_si128((__m128i*) (a.q + 32 * b), _mm_unpacklo_epi64(l, hh));
            _mm_storeu_si128((__m128i*) (a.q + 32 * b + 16), _mm_unpackhi_epi64(l, hh));
            a.d[2 * b] = a.d[2 * b + 1] = d;
        } else {   // Q2_0: value order, the block's sum of the int8 values
            std::memcpy(a.q + 32 * b, q, 32);
            a.d[b] = d;
            int sm = 0;
            for (int i = 0; i < 32; ++i) sm += q[i];
            a.ysum[b] = sm;
        }
    }
}

bool gu_supported(int type) { return type == 18 || type == 21 || type == 22 || type == 23; }
bool down_supported(int type) { return type == 20 || type == 42; }

void gu_rows(int type, int nt, const uint8_t* blob, size_t gu_row, size_t up_off, const ActX* const* act, float* const* h, int r0, int r1) {
    switch (type) {
        case 18: gu_rows_nt<18>(nt, blob, gu_row, up_off, act, h, r0, r1); break;
        case 21: gu_rows_nt<21>(nt, blob, gu_row, up_off, act, h, r0, r1); break;
        case 22: gu_rows_nt<22>(nt, blob, gu_row, up_off, act, h, r0, r1); break;
        case 23: gu_rows_nt<23>(nt, blob, gu_row, up_off, act, h, r0, r1); break;
        default: break;
    }
}

void down_rows(int type, int nt, const uint8_t* w, size_t row_bytes, const ActH* const* h, float* const* out, int r0, int r1) {
    for (int t0 = 0; t0 < nt; t0 += 4) {
        const int k = nt - t0 < 4 ? nt - t0 : 4;
        if (type == 20) {
            switch (k) {
                case 1: down_t<1>(w, row_bytes, h + t0, out + t0, r0, r1); break;
                case 2: down_t<2>(w, row_bytes, h + t0, out + t0, r0, r1); break;
                case 3: down_t<3>(w, row_bytes, h + t0, out + t0, r0, r1); break;
                default: down_t<4>(w, row_bytes, h + t0, out + t0, r0, r1); break;
            }
        } else {
            switch (k) {
                case 1: down20_t<1>(w, row_bytes, h + t0, out + t0, r0, r1); break;
                case 2: down20_t<2>(w, row_bytes, h + t0, out + t0, r0, r1); break;
                case 3: down20_t<3>(w, row_bytes, h + t0, out + t0, r0, r1); break;
                default: down20_t<4>(w, row_bytes, h + t0, out + t0, r0, r1); break;
            }
        }
    }
}

}  // namespace strata::kernels::cpu::replica

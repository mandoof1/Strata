// src/kernels/expert_replica.cpp - prototype: a CPU replica of the GPU's native expert arithmetic (CPU/GPU parity).
//
//     build/expert_replica <shard.gguf> [layer ...]
//
// One real expert per layer, 1..4 entries (tokens) in one group, through native_expert_grouped on the GPU and through
// a scalar CPU transcription of what the GPU kernels compute: q8_1 activations (quantize_q8_1_kernel / q8_1_store),
// each 32-value part's exact integer dot and the format's integer scale step, the float (dw * d8) * sumi, the
// kernels' fixed addition trees (row_dot_80_sub16 for gate/up, row_dot_40_sub8 / row_dot_20_sub16 for down), the
// SwiGLU with strata::kernels::sexp (build with -DSTRATA_SWIGLU_SEXP) and the h quantization.  Compares the gate/up
// rows (scratch), the h blocks and the outputs bit for bit, with the accumulations as separate products and adds
// (mode 0) and as fmas (mode 1: what fp-contract could make of `s += d * sumi`).
#include "strata/artifact/gguf_reader.hpp"
#include "strata/kernels/cpu/native_expert.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"
#include "strata/kernels/iq_kernels.hpp"
#include "strata/kernels/sexp.hpp"
#include "strata/kernels/cpu/replica_avx512.hpp"

#include "ggml.h"

#include <cuda_runtime.h>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <thread>
#include <atomic>
#include <string>
#include <vector>

#pragma STDC FP_CONTRACT OFF

namespace cpu = strata::kernels::cpu;

namespace {
constexpr int64_t H = 2560, FF = 640;
constexpr int E = 7, NTMAX = 4;

uint16_t f2h(float f) { return ggml_fp32_to_fp16(f); }
float h2f(uint16_t h) { return ggml_fp16_to_fp32(h); }

// ---------------------------------------------------------------- q8_1 (36 bytes: half d, half sum, int8 qs[32])
struct Q81 { uint16_t d, s; int8_t qs[32]; };
static_assert(sizeof(Q81) == 36, "q8_1");

float fin(float v) { return std::fabs(v) > 65504.0f ? std::copysign(65504.0f, v) : v; }

void q8_1_block(const float* x, Q81& b) {
    float amax[32], sum[32];
    for (int i = 0; i < 32; ++i) { amax[i] = std::fabs(x[i]); sum[i] = x[i]; }
    for (int o = 16; o > 0; o >>= 1) {   // the warp's butterfly: every lane adds its partner's value
        float na[32], ns[32];
        for (int i = 0; i < 32; ++i) { na[i] = std::fmax(amax[i], amax[i ^ o]); ns[i] = sum[i] + sum[i ^ o]; }
        std::memcpy(amax, na, sizeof na); std::memcpy(sum, ns, sizeof ns);
    }
    const float am = amax[0];
    const float d = fin(am / 127.0f);
    for (int i = 0; i < 32; ++i) {
        float q = 0.0f;
        if (am != 0.0f) { q = std::round(x[i] / d); q = q > 127.0f ? 127.0f : q < -127.0f ? -127.0f : q; }
        b.qs[i] = (int8_t) q;
    }
    b.d = f2h(fin(d));
    b.s = f2h(fin(sum[0]));
}
void q8_1_row(const float* x, int64_t n, Q81* y) { for (int64_t i = 0; i < n / 32; ++i) q8_1_block(x + 32 * i, y[i]); }

// ---------------------------------------------------------------- the weights as integers
// ggml's to_float on a copy of the block with d = 1 and the sub-block scales set to a known value gives the signed
// grid values times a known power of two; `mul` makes them integers again.
struct Fmt { int type, qk, bytes; float mul; };
Fmt fmt_of(int t) {
    switch (t) {
        case 21: return {21, 256, 110, 1.0f};   // IQ3_S
        case 18: return {18, 256, 98, 4.0f};    // IQ3_XXS
        case 22: return {22, 256, 82, 8.0f};    // IQ2_S
        case 23: return {23, 256, 136, 1.0f};   // IQ4_XS
        case 20: return {20, 32, 18, 1.0f};     // IQ4_NL
        case 42: return {42, 64, 18, 1.0f};     // Q2_0
    }
    std::fprintf(stderr, "type %d not in the prototype\n", t);
    std::exit(2);
}
void neutral(int t, uint8_t* b) {
    b[0] = 0x00; b[1] = 0x3C;   // d = 1
    switch (t) {
        case 21: std::memset(b + 106, 0, 4); break;
        case 18: for (int i = 0; i < 8; ++i) b[2 + 64 + 4 * i + 3] &= 0x0F; break;
        case 22: std::memset(b + 74, 0, 8); break;
        case 23: b[2] = 0xAA; b[3] = 0xAA; std::memset(b + 4, 0x11, 4); break;
        default: break;
    }
}
// the integer weights of one row (n values)
void int_row(int t, const uint8_t* row, int64_t n, int8_t* w) {
    const Fmt F = fmt_of(t);
    std::vector<uint8_t> blk(F.bytes);
    std::vector<float> v(F.qk);
    const auto* tr = ggml_get_type_traits((ggml_type) t);
    for (int64_t b = 0; b < n / F.qk; ++b) {
        std::memcpy(blk.data(), row + b * F.bytes, F.bytes);
        neutral(t, blk.data());
        tr->to_float(blk.data(), v.data(), F.qk);
        for (int i = 0; i < F.qk; ++i) {
            const float q = v[i] * F.mul;
            if (q != std::nearbyint(q) || std::fabs(q) > 127) { std::fprintf(stderr, "type %d: weight %g not an int8\n", t, q); std::exit(3); }
            w[b * F.qk + i] = (int8_t) q;
        }
    }
}

// one part's value: the GPU's apply().  Part k of a row (GPU k loop), activation blocks x.
// 256-blocks (gate/up): part k is block k / 8, sub-block k % 8 (32 values, one q8_1 block).
// IQ4_NL (down, 32-blocks): part k is block k / 2, half k % 2: values 8h..8h+7 and 16+8h..16+8h+7.
// Q2_0 (down, 64-blocks): part k is q8_1 block k (block k / 2, values 32 (k % 2) ..).
float part(int t, const uint8_t* row, const int8_t* w, const Q81* x, int k) {
    const Fmt F = fmt_of(t);
    int sumi = 0;
    float dw, d8;
    if (t == 20) {
        const int b = k / 2, h = k % 2;
        const Q81& q = x[b];
        for (int i = 0; i < 8; ++i) sumi += w[b * 32 + 8 * h + i] * q.qs[8 * h + i];
        for (int i = 0; i < 8; ++i) sumi += w[b * 32 + 16 + 8 * h + i] * q.qs[16 + 8 * h + i];
        uint16_t dh; std::memcpy(&dh, row + b * F.bytes, 2);
        dw = h2f(dh); d8 = h2f(q.d);
        return (dw * d8) * (float) sumi;
    }
    if (t == 42) {
        const Q81& q = x[k];
        for (int i = 0; i < 32; ++i) sumi += w[k * 32 + i] * q.qs[i];
        uint16_t dh; std::memcpy(&dh, row + (k / 2) * F.bytes, 2);
        dw = h2f(dh); d8 = h2f(q.d);
        return (dw * d8) * (float) sumi;
    }
    const int b = k / 8, sb = k % 8;
    const Q81& q = x[k];
    const uint8_t* blk = row + b * F.bytes;
    const int8_t* wv = w + k * 32;
    uint16_t dh; std::memcpy(&dh, blk, 2);
    dw = h2f(dh); d8 = h2f(q.d);
    if (t == 22) {   // IQ2_S: two halves, two 4-bit scales
        int s0 = 0, s1 = 0;
        for (int i = 0; i < 16; ++i) s0 += wv[i] * q.qs[i];
        for (int i = 16; i < 32; ++i) s1 += wv[i] * q.qs[i];
        const int sc = blk[74 + sb];
        const int ls0 = sc & 0x0F, ls1 = sc >> 4;
        sumi = (s0 * ls0 + s1 * ls1 + (s0 + s1) / 2) / 4;
    } else {
        for (int i = 0; i < 32; ++i) sumi += wv[i] * q.qs[i];
        if (t == 21) {
            const int ls = 1 + 2 * ((blk[106 + sb / 2] >> (4 * (sb % 2))) & 0x0F);
            sumi *= ls;
        } else if (t == 18) {
            const int ls = blk[2 + 64 + 4 * sb + 3] >> 4;
            sumi = (ls * sumi + sumi / 2) / 2;
        } else if (t == 23) {
            uint16_t sh; std::memcpy(&sh, blk + 2, 2);
            const int ls = ((blk[4 + sb / 2] >> (4 * (sb % 2))) & 0x0F) | (((sh >> (2 * sb)) & 3) << 4);
            sumi *= ls - 32;
        }
    }
    return (dw * d8) * (float) sumi;
}

// s += v, as a separate add (mode 0) or as the fma the compiler may have made of s += d * sumi (mode 1)
struct Acc {
    int mode;
    float add(float s, float d, float si) const { return mode ? std::fma(d, si, s) : s + d * si; }
};
// part value as (d, sumi) so mode 1 can fuse: recompute d and sumi via a second entry point
struct PV { float d, si; };
PV part_dv(int t, const uint8_t* row, const int8_t* w, const Q81* x, int k) {
    const Fmt F = fmt_of(t);
    int sumi = 0;
    float dw, d8;
    if (t == 20) {
        const int b = k / 2, h = k % 2;
        const Q81& q = x[b];
        for (int i = 0; i < 8; ++i) sumi += w[b * 32 + 8 * h + i] * q.qs[8 * h + i];
        for (int i = 0; i < 8; ++i) sumi += w[b * 32 + 16 + 8 * h + i] * q.qs[16 + 8 * h + i];
        uint16_t dh; std::memcpy(&dh, row + b * F.bytes, 2);
        dw = h2f(dh); d8 = h2f(q.d);
        return {dw * d8, (float) sumi};
    }
    if (t == 42) {
        const Q81& q = x[k];
        for (int i = 0; i < 32; ++i) sumi += w[k * 32 + i] * q.qs[i];
        uint16_t dh; std::memcpy(&dh, row + (k / 2) * F.bytes, 2);
        dw = h2f(dh); d8 = h2f(q.d);
        return {dw * d8, (float) sumi};
    }
    const int b = k / 8, sb = k % 8;
    const Q81& q = x[k];
    const uint8_t* blk = row + b * F.bytes;
    const int8_t* wv = w + k * 32;
    uint16_t dh; std::memcpy(&dh, blk, 2);
    dw = h2f(dh); d8 = h2f(q.d);
    if (t == 22) {
        int s0 = 0, s1 = 0;
        for (int i = 0; i < 16; ++i) s0 += wv[i] * q.qs[i];
        for (int i = 16; i < 32; ++i) s1 += wv[i] * q.qs[i];
        const int sc = blk[74 + sb];
        sumi = (s0 * (sc & 0x0F) + s1 * (sc >> 4) + (s0 + s1) / 2) / 4;
    } else {
        for (int i = 0; i < 32; ++i) sumi += wv[i] * q.qs[i];
        if (t == 21) sumi *= 1 + 2 * ((blk[106 + sb / 2] >> (4 * (sb % 2))) & 0x0F);
        else if (t == 18) { const int ls = blk[2 + 64 + 4 * sb + 3] >> 4; sumi = (ls * sumi + sumi / 2) / 2; }
        else if (t == 23) {
            uint16_t sh; std::memcpy(&sh, blk + 2, 2);
            sumi *= (((blk[4 + sb / 2] >> (4 * (sb % 2))) & 0x0F) | (((sh >> (2 * sb)) & 3) << 4)) - 32;
        }
    }
    return {dw * d8, (float) sumi};
}

// row_dot_80_sub16: 16 lanes, lane t sums parts t, t+32, t+64 and (separately) t+16, t+48, then s + s16, then the
// butterfly 8, 4, 2, 1
float dot80(int t, const uint8_t* row, const int8_t* w, const Q81* x, const Acc& A) {
    float s[16];
    for (int l = 0; l < 16; ++l) {
        float a = 0.0f, b = 0.0f;
        for (int k = l; k < 80; k += 32) { const PV p = part_dv(t, row, w, x, k); a = A.add(a, p.d, p.si); }
        for (int k = l + 16; k < 64; k += 32) { const PV p = part_dv(t, row, w, x, k); b = A.add(b, p.d, p.si); }
        s[l] = a + b;
    }
    for (int o = 8; o > 0; o >>= 1) { float n[16]; for (int l = 0; l < 16; ++l) n[l] = s[l] + s[l ^ o]; std::memcpy(s, n, sizeof n); }
    return s[0];
}
// row_dot_40_sub8 (IQ4_NL down, 40 parts): lane t: parts t, t+32; then + (t+16); then + ((t+8) + (t+24)); butterfly 4, 2, 1
float dot40(int t, const uint8_t* row, const int8_t* w, const Q81* x, const Acc& A) {
    float s[8];
    for (int l = 0; l < 8; ++l) {
        float a = 0.0f;
        for (int k = l; k < 40; k += 32) { const PV p = part_dv(t, row, w, x, k); a = A.add(a, p.d, p.si); }
        PV p = part_dv(t, row, w, x, l + 16);
        const float s16 = A.add(0.0f, p.d, p.si);
        a = a + s16;
        p = part_dv(t, row, w, x, l + 8);
        const float s8 = A.add(0.0f, p.d, p.si);
        p = part_dv(t, row, w, x, l + 24);
        const float s24 = A.add(0.0f, p.d, p.si);
        s[l] = a + (s8 + s24);
    }
    for (int o = 4; o > 0; o >>= 1) { float n[8]; for (int l = 0; l < 8; ++l) n[l] = s[l] + s[l ^ o]; std::memcpy(s, n, sizeof n); }
    return s[0];
}
// row_dot_20_sub16 (Q2_0 down, 20 parts): lane t: part t, + (t < 4 ? part t+16 : 0); butterfly 8, 4, 2, 1
float dot20(int t, const uint8_t* row, const int8_t* w, const Q81* x, const Acc& A) {
    float s[16];
    for (int l = 0; l < 16; ++l) {
        PV p = part_dv(t, row, w, x, l);
        float a = A.add(0.0f, p.d, p.si);
        float b = 0.0f;
        if (l < 4) { p = part_dv(t, row, w, x, l + 16); b = A.add(0.0f, p.d, p.si); }
        s[l] = a + b;
    }
    for (int o = 8; o > 0; o >>= 1) { float n[16]; for (int l = 0; l < 16; ++l) n[l] = s[l] + s[l ^ o]; std::memcpy(s, n, sizeof n); }
    return s[0];
}

float swiglu(float g, float u) { return (g / (1.0f + strata::kernels::sexp(-g))) * u; }

size_t ndiff(const float* a, const float* b, size_t n) { size_t c = 0; for (size_t i = 0; i < n; ++i) c += std::memcmp(a + i, b + i, 4) != 0; return c; }

int check(const cpu::NativeFmt& f, const std::vector<uint8_t>& blob, int layer, cudaStream_t s) {
    int fails = 0;
    // integer weights of every row
    std::vector<int8_t> wg((size_t) FF * H), wu((size_t) FF * H), wd((size_t) H * FF);
    for (int64_t r = 0; r < FF; ++r) {
        int_row(f.gu_type, blob.data() + r * f.gu_row, H, wg.data() + r * H);
        int_row(f.gu_type, blob.data() + f.up_off + r * f.gu_row, H, wu.data() + r * H);
    }
    for (int64_t r = 0; r < H; ++r) int_row(f.d_type, blob.data() + f.down_off + r * f.d_row, FF, wd.data() + r * FF);
    // activations: normal with a few large values (the model's residual stream has some)
    std::mt19937 rng(100 + layer);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<float> x((size_t) NTMAX * H);
    for (size_t i = 0; i < x.size(); ++i) x[i] = nd(rng) * (i % 97 == 0 ? 30.0f : 1.0f);

    // GPU
    const auto L = strata::kernels::native_expert_layout(f.gu_type, f.d_type, H, FF);
    void *dblob, *dx, *dxq, *dscr;
    float* dout;
    unsigned long long* dptr;
    int32_t *dstart, *dn, *ddst, *dtok;
    const size_t scr = strata::kernels::native_expert_scratch_bytes(NTMAX, FF);
    cudaMalloc(&dblob, blob.size()); cudaMalloc(&dx, x.size() * 4); cudaMalloc(&dxq, (size_t) NTMAX * H / 32 * 36);
    cudaMalloc(&dscr, scr); cudaMalloc((void**) &dout, (size_t) NTMAX * H * 4); cudaMalloc((void**) &dptr, 8);
    cudaMalloc((void**) &dstart, 8); cudaMalloc((void**) &dn, 4); cudaMalloc((void**) &ddst, NTMAX * 4); cudaMalloc((void**) &dtok, NTMAX * 4);
    cudaMemcpy(dblob, blob.data(), blob.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(dx, x.data(), x.size() * 4, cudaMemcpyHostToDevice);
    const unsigned long long p = (unsigned long long) dblob;
    const int32_t idx[NTMAX] = {0, 1, 2, 3}, one = 1;
    cudaMemcpy(dptr, &p, 8, cudaMemcpyHostToDevice);
    cudaMemcpy(dn, &one, 4, cudaMemcpyHostToDevice);
    cudaMemcpy(ddst, idx, sizeof idx, cudaMemcpyHostToDevice);
    cudaMemcpy(dtok, idx, sizeof idx, cudaMemcpyHostToDevice);
    strata::kernels::quantize_q8_1_rows((const float*) dx, NTMAX, H, dxq, s);
    std::vector<Q81> gxq((size_t) NTMAX * H / 32);
    cudaStreamSynchronize(s);
    cudaMemcpy(gxq.data(), dxq, gxq.size() * 36, cudaMemcpyDeviceToHost);

    // CPU: x quantization
    std::vector<Q81> cxq((size_t) NTMAX * H / 32);
    for (int k = 0; k < NTMAX; ++k) q8_1_row(x.data() + k * H, H, cxq.data() + k * (H / 32));
    const bool xq_same = std::memcmp(cxq.data(), gxq.data(), cxq.size() * 36) == 0;

    // CPU: gate/up, both accumulation modes
    std::vector<float> cg[2], cu[2];
    for (int m = 0; m < 2; ++m) {
        const Acc A{m};
        cg[m].resize((size_t) NTMAX * FF); cu[m].resize((size_t) NTMAX * FF);
        for (int k = 0; k < NTMAX; ++k)
            for (int64_t r = 0; r < FF; ++r) {
                cg[m][k * FF + r] = dot80(f.gu_type, blob.data() + r * f.gu_row, wg.data() + r * H, cxq.data() + k * (H / 32), A);
                cu[m][k * FF + r] = dot80(f.gu_type, blob.data() + f.up_off + r * f.gu_row, wu.data() + r * H, cxq.data() + k * (H / 32), A);
            }
    }
    std::printf("layer %d (%s / %s): x q8_1 %s\n", layer, ggml_type_name((ggml_type) f.gu_type),
                ggml_type_name((ggml_type) f.d_type), xq_same ? "bitwise equal" : "DIFFERS");
    if (!xq_same) ++fails;

    // launches: (cap_groups, grid_groups) - one block row per possible group (the VRAM call) and strided (the PCIe call)
    const int64_t launch[4][2] = {{1, 0}, {4, 0}, {4, 1}, {4, 2}};
    for (int li = 0; li < 4; ++li)
    for (int nt = 1; nt <= NTMAX; ++nt) {
        if (li > 0 && nt != 1 && nt != NTMAX) continue;
        const int64_t capg = launch[li][0], gy = launch[li][1];
        std::printf("  [cap_groups %lld, grid_groups %lld]", (long long) capg, (long long) gy);
        const int32_t st[2] = {0, nt};
        cudaMemcpy(dstart, st, 8, cudaMemcpyHostToDevice);
        cudaMemset(dscr, 0xFF, scr);
        cudaMemset(dout, 0xFF, (size_t) NTMAX * H * 4);
        strata::kernels::native_expert_grouped(L, dptr, dstart, dn, ddst, dtok, capg, NTMAX, dxq, dscr, dout, s, gy);
        cudaStreamSynchronize(s);
        std::vector<float> gout((size_t) nt * H), gg((size_t) nt * FF), gu((size_t) nt * FF);
        cudaMemcpy(gout.data(), dout, gout.size() * 4, cudaMemcpyDeviceToHost);
        const size_t fb = (size_t) NTMAX * FF * 4, fa = (fb + 255) & ~(size_t) 255;
        cudaMemcpy(gg.data(), dscr, gg.size() * 4, cudaMemcpyDeviceToHost);
        cudaMemcpy(gu.data(), (uint8_t*) dscr + fa, gu.size() * 4, cudaMemcpyDeviceToHost);
        // h and the down rows from the CPU's own gate/up in each mode
        for (int m = 0; m < 2; ++m) {
            const Acc A{m};
            std::vector<float> h((size_t) nt * FF), out((size_t) nt * H);
            std::vector<Q81> hq((size_t) nt * FF / 32);
            for (int k = 0; k < nt; ++k) {
                for (int64_t r = 0; r < FF; ++r) h[k * FF + r] = swiglu(cg[m][k * FF + r], cu[m][k * FF + r]);
                q8_1_row(h.data() + k * FF, FF, hq.data() + k * (FF / 32));
                for (int64_t r = 0; r < H; ++r) {
                    const uint8_t* dr = blob.data() + f.down_off + r * f.d_row;
                    const int8_t* w = wd.data() + r * FF;
                    out[k * H + r] = f.d_type == 20 ? dot40(20, dr, w, hq.data() + k * (FF / 32), A)
                                                    : dot20(42, dr, w, hq.data() + k * (FF / 32), A);
                }
            }
            const size_t dg = ndiff(cg[m].data(), gg.data(), gg.size()), du = ndiff(cu[m].data(), gu.data(), gu.size());
            const size_t dout_n = ndiff(out.data(), gout.data(), gout.size());
            std::printf(" %d entr%s, %s: gate %zu/%zu up %zu/%zu out %zu/%zu differ%s\n", nt, nt == 1 ? "y" : "ies",
                        m ? "fma " : "mul+add", dg, gg.size(), du, gu.size(), dout_n, gout.size(),
                        (dg == gg.size() && du == gu.size()) ? " (scratch gate/up not written by this path)" : "");
            if (m == 0 && dout_n) {   // the first few to see the size
                int shown = 0;
                for (size_t i = 0; i < gout.size() && shown < 3; ++i)
                    if (std::memcmp(&out[i], &gout[i], 4)) { std::printf("      out[%zu] cpu %.9g gpu %.9g\n", i, out[i], gout[i]); ++shown; }
            }
        }
    }
    // the AVX-512 replica: bit for bit against the GPU, every width
    namespace rp = cpu::replica;
    if (cpu::cpu_avx512_ok() && rp::gu_supported(f.gu_type) && rp::down_supported(f.d_type)) {
        std::vector<rp::ActX> xa(NTMAX);
        std::vector<rp::ActH> ha(NTMAX);
        for (int k = 0; k < NTMAX; ++k) rp::quant_x(x.data() + k * H, (int) H, xa[k]);
        size_t xq_diff = 0;
        for (int k = 0; k < NTMAX; ++k)
            for (int64_t b = 0; b < H / 32; ++b) {
                xq_diff += std::memcmp(xa[k].qs + 32 * b, gxq[k * (H / 32) + b].qs, 32) != 0;
                xq_diff += xa[k].d8[b] != h2f(gxq[k * (H / 32) + b].d);
            }
        for (int nt = 1; nt <= NTMAX; ++nt) {
            const int32_t st[2] = {0, nt};
            cudaMemcpy(dstart, st, 8, cudaMemcpyHostToDevice);
            strata::kernels::native_expert_grouped(L, dptr, dstart, dn, ddst, dtok, 1, NTMAX, dxq, dscr, dout, s);
            cudaStreamSynchronize(s);
            std::vector<float> gout((size_t) nt * H), hh((size_t) nt * FF), out((size_t) nt * H);
            cudaMemcpy(gout.data(), dout, gout.size() * 4, cudaMemcpyDeviceToHost);
            const rp::ActX* ap[NTMAX];
            const rp::ActH* hp[NTMAX];
            float* hpp[NTMAX];
            float* op[NTMAX];
            for (int k = 0; k < nt; ++k) { ap[k] = &xa[k]; hp[k] = &ha[k]; hpp[k] = hh.data() + k * FF; op[k] = out.data() + k * H; }
            rp::gu_rows(f.gu_type, nt, blob.data(), f.gu_row, f.up_off, ap, hpp, 0, (int) FF);
            for (int k = 0; k < nt; ++k) rp::quant_h(hh.data() + k * FF, (int) FF, f.d_type, ha[k]);
            rp::down_rows(f.d_type, nt, blob.data() + f.down_off, f.d_row, hp, op, 0, (int) H);
            const size_t d = ndiff(out.data(), gout.data(), out.size());
            std::printf("  avx512 replica, %d entr%s: x q8_1 %s, out %zu/%zu differ\n", nt, nt == 1 ? "y" : "ies",
                        xq_diff ? "DIFFERS" : "equal", d, out.size());
            if (d || xq_diff) ++fails;
        }
    }
    cudaFree(dblob); cudaFree(dx); cudaFree(dxq); cudaFree(dscr); cudaFree(dout); cudaFree(dptr);
    cudaFree(dstart); cudaFree(dn); cudaFree(ddst); cudaFree(dtok);
    return fails;
}
}  // namespace

// --bench: the engine's CPU expert path (native_quant_act / native_gu_rows / native_quant_h / native_down_rows) against
// the replica, over `ne` distinct experts of one layer (more than an L3 holds), `nth` threads taking experts in turn
int bench(const strata::GgufModel& model, int layer, int nth) {
    static const int stage = std::getenv("REP_STAGE") ? std::atoi(std::getenv("REP_STAGE")) : 15;
    namespace rp = cpu::replica;
    const strata::TensorInfo* t[3] = {};
    const uint8_t* data[3] = {};
    const char* roles[3] = {"gate", "up", "down"};
    for (int r = 0; r < 3; ++r) {
        size_t at = 0;
        t[r] = model.find("blk." + std::to_string(layer) + ".ffn_" + roles[r] + "_exps.weight", &at);
        if (t[r]) data[r] = model.shard(at).tensor_data(*t[r]);
    }
    cpu::NativeFmt f;
    std::string err;
    if (!t[0] || !cpu::native_fmt((int) t[0]->type, (int) t[2]->type, H, FF, f, err)) return 1;
    if (!rp::gu_supported(f.gu_type) || !rp::down_supported(f.d_type)) { std::printf("layer %d: not in the prototype\n", layer); return 0; }
    constexpr int NE = 64;
    std::vector<std::vector<uint8_t>> blobs(NE, std::vector<uint8_t>(f.bytes));
    for (int e = 0; e < NE; ++e) {
        std::memcpy(blobs[e].data(), data[0] + (size_t) e * f.up_off, f.up_off);
        std::memcpy(blobs[e].data() + f.up_off, data[1] + (size_t) e * f.up_off, f.up_off);
        std::memcpy(blobs[e].data() + f.down_off, data[2] + (size_t) e * (f.bytes - f.down_off), f.bytes - f.down_off);
    }
    std::mt19937 rng(5);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<float> x((size_t) NTMAX * H);
    for (auto& v : x) v = nd(rng);
    std::printf("layer %d (%s / %s), %d experts (%.0f MB), %d thread%s:\n", layer, ggml_type_name((ggml_type) f.gu_type),
                ggml_type_name((ggml_type) f.d_type), NE, NE * f.bytes / 1e6, nth, nth == 1 ? "" : "s");
    for (int nt : {1, 2, 4}) {
        double us[2];
        for (int way = 0; way < 2; ++way) {
            auto work = [&](int tid, int reps) {
                std::vector<std::vector<uint8_t>> act(NTMAX, std::vector<uint8_t>(cpu::kNativeActBytes)), hq(NTMAX, std::vector<uint8_t>(cpu::kNativeHBytes));
                std::vector<rp::ActX> xa(NTMAX);
                std::vector<rp::ActH> ha(NTMAX);
                std::vector<float> ff((size_t) NTMAX * FF), out((size_t) NTMAX * H);
                const void* ap[NTMAX]; const void* hp[NTMAX]; float* fp[NTMAX]; float* op[NTMAX];
                const rp::ActX* xap[NTMAX]; const rp::ActH* hap[NTMAX];
                for (int k = 0; k < NTMAX; ++k) {
                    ap[k] = act[k].data(); hp[k] = hq[k].data(); fp[k] = ff.data() + k * FF; op[k] = out.data() + k * H;
                    xap[k] = &xa[k]; hap[k] = &ha[k];
                }
                for (int rep = 0; rep < reps; ++rep)
                    for (int e = tid; e < NE; e += nth) {
                        const uint8_t* b = blobs[e].data();
                        if (way == 0) {
                            if (stage & 1) for (int k = 0; k < nt; ++k) cpu::native_quant_act(f, x.data() + k * H, act[k].data());
                            if (stage & 2) cpu::native_gu_rows(f, b, ap, nt, fp, 0, (int) FF);
                            if (stage & 4) for (int k = 0; k < nt; ++k) cpu::native_quant_h(f, fp[k], hq[k].data());
                            if (stage & 8) cpu::native_down_rows(f, b, hp, nt, op, 0, (int) H);
                        } else {
                            if (stage & 1) for (int k = 0; k < nt; ++k) rp::quant_x(x.data() + k * H, (int) H, xa[k]);
                            if (stage & 2) rp::gu_rows(f.gu_type, nt, b, f.gu_row, f.up_off, xap, fp, 0, (int) FF);
                            if (stage & 4) for (int k = 0; k < nt; ++k) rp::quant_h(fp[k], (int) FF, f.d_type, ha[k]);
                            if (stage & 8) rp::down_rows(f.d_type, nt, b + f.down_off, f.d_row, hap, op, 0, (int) H);
                        }
                    }
            };
            auto run = [&](int reps) {
                std::vector<std::thread> th;
                for (int i = 0; i < nth; ++i) th.emplace_back(work, i, reps);
                for (auto& z : th) z.join();
            };
            run(1);   // warm
            const int reps = 4;
            const auto t0 = std::chrono::steady_clock::now();
            run(reps);
            const auto t1 = std::chrono::steady_clock::now();
            us[way] = std::chrono::duration<double, std::micro>(t1 - t0).count() / (reps * NE);
        }
        std::printf("  %d token%s: engine %.1f us / expert, replica %.1f us / expert (%+.1f%%)\n", nt, nt == 1 ? " " : "s",
                    us[0], us[1], (us[1] / us[0] - 1) * 100);
    }
    return 0;
}

int main(int argc, char** argv) {
    if (argc >= 3 && std::string(argv[1]) == "--bench") {
        const strata::GgufModel model(strata::gguf_split_paths(argv[2]));
        const int nth = argc >= 5 ? std::atoi(argv[4]) : 1;
        return bench(model, std::atoi(argv[3]), nth);
    }
    if (argc < 2) { std::fprintf(stderr, "usage: expert_replica <shard.gguf> [layer ...]\n       expert_replica --bench <shard.gguf> <layer> [threads]\n"); return 2; }
    cudaStream_t s;
    cudaStreamCreate(&s);
    const strata::GgufModel model(strata::gguf_split_paths(argv[1]));
    std::vector<int> layers;
    for (int i = 2; i < argc; ++i) layers.push_back(std::atoi(argv[i]));
    if (layers.empty()) layers = {0, 1, 2, 3, 20, 47};
    int fails = 0;
    for (int l : layers) {
        const strata::TensorInfo* t[3] = {};
        const uint8_t* data[3] = {};
        const char* roles[3] = {"gate", "up", "down"};
        for (int r = 0; r < 3; ++r) {
            size_t at = 0;
            t[r] = model.find("blk." + std::to_string(l) + ".ffn_" + roles[r] + "_exps.weight", &at);
            if (t[r]) data[r] = model.shard(at).tensor_data(*t[r]);
        }
        if (!t[0] || !t[1] || !t[2]) { std::printf("layer %d: no expert tensors\n", l); ++fails; continue; }
        cpu::NativeFmt f;
        std::string err;
        if (!cpu::native_fmt((int) t[0]->type, (int) t[2]->type, H, FF, f, err)) { std::printf("layer %d: %s\n", l, err.c_str()); ++fails; continue; }
        std::vector<uint8_t> blob(f.bytes);
        std::memcpy(blob.data(), data[0] + (size_t) E * f.up_off, f.up_off);
        std::memcpy(blob.data() + f.up_off, data[1] + (size_t) E * f.up_off, f.up_off);
        std::memcpy(blob.data() + f.down_off, data[2] + (size_t) E * (f.bytes - f.down_off), f.bytes - f.down_off);
        fails += check(f, blob, l, s);
    }
    std::printf("expert_replica: %d failures\n", fails);
    return fails ? 1 : 0;
}

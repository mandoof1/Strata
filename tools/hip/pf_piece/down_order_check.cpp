// tools/hip/pf_piece/down_order_check.cpp - does the experts' down product (Gemm::f16, T x 640 . (2560 x 640)^T, the
// hipBLASLt solution of the tuning table) give a row the same bits whatever the launch's row count and the row's place?
// STRATA_PF_PIECE merges an expert's consecutive pieces into one down launch only where this holds.
// Usage: STRATA_HIPBLASLT_TUNING=<table> down_order_check [big_T ...]   (prints the token counts that differ)
#include "strata/prefill/gemm.hpp"

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    constexpr int64_t N = 2560, K = 640;
    std::vector<int64_t> bigs;
    for (int i = 1; i < argc; ++i) bigs.push_back(std::atoll(argv[i]));
    if (bigs.empty()) bigs = {65, 300, 1000, 4096, 9000};
    int64_t TM = 0;
    for (auto b : bigs) TM = std::max(TM, b);
    hipStream_t s; hipStreamCreate(&s);
    std::mt19937 rng(7);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::vector<__half> hx((size_t) TM * K), hw((size_t) N * K);
    for (auto& v : hx) v = __float2half(nd(rng) * 0.5f);
    for (auto& v : hw) v = __float2half(nd(rng) * 0.05f);
    __half *dx, *dw; float *dy, *dz;
    hipMalloc(&dx, hx.size() * 2); hipMalloc(&dw, hw.size() * 2);
    hipMalloc(&dy, (size_t) TM * N * 4); hipMalloc(&dz, (size_t) TM * N * 4);
    hipMemcpy(dx, hx.data(), hx.size() * 2, hipMemcpyHostToDevice);
    hipMemcpy(dw, hw.data(), hw.size() * 2, hipMemcpyHostToDevice);
    strata::prefill::Gemm g;
    std::string err;
    if (!g.init(s, 16 << 20, err)) { std::fprintf(stderr, "init: %s\n", err.c_str()); return 1; }
    std::vector<float> a((size_t) TM * N), b((size_t) TM * N);
    int64_t checks = 0, bad = 0;
    for (const int64_t big : bigs) {
        g.f16((const uint16_t*) dx, (const uint16_t*) dw, dy, big, N, K);
        hipStreamSynchronize(s);
        hipMemcpy(a.data(), dy, (size_t) big * N * 4, hipMemcpyDeviceToHost);
        // sub-launches: every length 1..300, then a spread to big; offsets 0, a middle one and the end
        std::vector<int64_t> lens;
        for (int64_t l = 1; l <= std::min<int64_t>(big, 300); ++l) lens.push_back(l);
        for (int64_t l = 301; l < big; l = l * 9 / 8 + 7) lens.push_back(l);
        for (const int64_t len : lens) {
            const int64_t offs[3] = {0, (big - len) / 2, big - len};
            for (const int64_t off : offs) {
                g.f16((const uint16_t*) (dx + off * K), (const uint16_t*) dw, dz, len, N, K);
                hipStreamSynchronize(s);
                hipMemcpy(b.data(), dz, (size_t) len * N * 4, hipMemcpyDeviceToHost);
                ++checks;
                if (std::memcmp(a.data() + off * N, b.data(), (size_t) len * N * 4) != 0) {
                    ++bad;
                    std::printf("DIFF big %lld len %lld off %lld\n", (long long) big, (long long) len, (long long) off);
                }
            }
        }
    }
    if (std::getenv("OFFSCAN")) {   // which offsets keep the bits: len 64 at every offset 0..1023 of a 4096 launch
        g.f16((const uint16_t*) dx, (const uint16_t*) dw, dy, 4096, N, K);
        hipStreamSynchronize(s);
        hipMemcpy(a.data(), dy, (size_t) 4096 * N * 4, hipMemcpyDeviceToHost);
        std::printf("same at offsets:");
        for (int64_t off = 0; off < 1024; ++off) {
            g.f16((const uint16_t*) (dx + off * K), (const uint16_t*) dw, dz, 64, N, K);
            hipStreamSynchronize(s);
            hipMemcpy(b.data(), dz, (size_t) 64 * N * 4, hipMemcpyDeviceToHost);
            if (std::memcmp(a.data() + off * N, b.data(), (size_t) 64 * N * 4) == 0) std::printf(" %lld", (long long) off);
        }
        std::printf("\n");
    }
    std::printf("%lld of %lld sub-launches differ\n", (long long) bad, (long long) checks);
    return bad != 0;
}

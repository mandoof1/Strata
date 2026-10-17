// src/kernels/cpu/pool_mixed_fmt_test.cpp - local (STRATA_ALT_EXPERTS): one run_split_multi_native call whose jobs
// carry their own NativeFmt (ExpertJobMulti::fmt, a second pack's blobs) against each format's jobs in a call of
// their own, bit for bit, at several worker counts and token counts.  CPU only, random blobs with finite scales.
// Run it with and without STRATA_POOL_FUSED=1 (the fused layer, which a Q2_0 job turns off for the whole call).
#include "strata/kernels/cpu/pool.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"
#include "ggml.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace cpu = strata::kernels::cpu;

namespace {
struct Pack {
    cpu::NativeFmt f;
    std::vector<uint8_t> blobs;   // `n` blobs
    std::vector<std::vector<uint8_t>> nact;
    std::vector<cpu::ActQ> qact;
};

// random bytes, then a finite fp16 scale (2^-8 .. 2^-1) at the start of every block of each role
void fill(Pack& p, int n, std::mt19937& rng) {
    const cpu::NativeFmt& f = p.f;
    p.blobs.resize(f.bytes * (size_t) n);
    for (auto& b : p.blobs) b = (uint8_t) rng();
    const size_t gb = ggml_type_size((ggml_type) f.gu_type), db = ggml_type_size((ggml_type) f.d_type);
    for (int e = 0; e < n; ++e) {
        uint8_t* b = p.blobs.data() + (size_t) e * f.bytes;
        auto scales = [&](size_t from, size_t to, size_t step) {
            for (size_t i = from; i + 2 <= to; i += step) {
                const uint16_t h = (uint16_t) (0x1c00 + rng() % 0x800);
                std::memcpy(b + i, &h, 2);
            }
        };
        scales(0, f.down_off, gb);
        scales(f.down_off, f.bytes, db);
    }
}

void acts(Pack& p, const std::vector<std::vector<float>>& x) {
    p.nact.assign(cpu::MAXT, std::vector<uint8_t>(cpu::kNativeActBytes));
    p.qact.resize(cpu::MAXT);
    for (int t = 0; t < cpu::MAXT; ++t) {
        if (cpu::q2_native_kernels(p.f.gu_type)) cpu::act_quant_any(x[t].data(), cpu::H, p.qact[t]);
        else cpu::native_quant_act(p.f, x[t].data(), p.nact[t].data());
    }
}

void job_of(Pack& p, int e, int nt, float* out, cpu::ExpertJobMulti& j) {
    j = cpu::ExpertJobMulti();
    j.blob = p.blobs.data() + (size_t) e * p.f.bytes;
    j.nt = nt;
    for (int t = 0; t < nt; ++t) {
        j.nact[t] = p.nact[t].data();
        j.act[t] = &p.qact[t];
        j.out[t] = out + (size_t) t * cpu::H;
    }
}
}  // namespace

int main() {
    if (!cpu::cpu_avx2_ok()) { std::printf("pool_mixed_fmt_test: no AVX2 on this CPU, skipped\n"); return 77; }
    std::mt19937 rng(7);
    std::vector<std::vector<float>> x(cpu::MAXT, std::vector<float>(cpu::H));
    for (auto& v : x) for (auto& f : v) f = (float) ((int) (rng() % 20001) - 10000) * 0.0002f;
    // the IQ3_XXS pack's and the IQ2_XS pack's kinds of layers: IQ4_NL or Q2_0 down
    const int pairs[][4] = {{21, 20, 22, 42}, {18, 42, 16, 42}, {21, 20, 16, 20}, {17, 20, 29, 42}};
    int checks = 0, failures = 0;
    for (const auto& pr : pairs) {
        Pack A, B;
        std::string err;
        if (!cpu::native_fmt(pr[0], pr[1], cpu::H, cpu::FF, A.f, err) || !cpu::native_fmt(pr[2], pr[3], cpu::H, cpu::FF, B.f, err)) {
            std::printf("native_fmt: %s\n", err.c_str());
            return 2;
        }
        constexpr int NE = 12;
        fill(A, NE, rng);
        fill(B, NE, rng);
        acts(A, x);
        acts(B, x);
        for (int workers : {1, 3}) {
            cpu::ExpertPool pool(workers, false);
            for (int n : {1, 2, 5, NE}) {
                // job e: pack A when e is even, B when odd; nt cycles 1, 3, MAXT
                std::vector<float> mixed((size_t) n * cpu::MAXT * cpu::H, 0.f), sep(mixed.size(), 0.f);
                std::vector<cpu::ExpertJobMulti> jm((size_t) n), ja, jb;
                for (int e = 0; e < n; ++e) {
                    Pack& p = e % 2 == 0 ? A : B;
                    const int nt = e % 3 == 0 ? 1 : e % 3 == 1 ? 3 : cpu::MAXT;
                    job_of(p, e, nt, mixed.data() + (size_t) e * cpu::MAXT * cpu::H, jm[(size_t) e]);
                    if (e % 2 == 1) jm[(size_t) e].fmt = &B.f;
                    cpu::ExpertJobMulti js;
                    job_of(p, e, nt, sep.data() + (size_t) e * cpu::MAXT * cpu::H, js);
                    (e % 2 == 0 ? ja : jb).push_back(js);
                }
                pool.run_split_multi_native(A.f, jm.data(), n);
                pool.run_split_multi_native(A.f, ja.data(), (int) ja.size());
                pool.run_split_multi_native(B.f, jb.data(), (int) jb.size());
                ++checks;
                if (std::memcmp(mixed.data(), sep.data(), mixed.size() * sizeof(float)) != 0) {
                    if (failures < 10)
                        std::printf("FAIL types %d/%d + %d/%d workers %d n %d\n", pr[0], pr[1], pr[2], pr[3], workers, n);
                    ++failures;
                }
                bool finite = true;   // IQ1_M's scale is spread over its blocks: random bytes may make it inf
                if (pr[0] != 29 && pr[2] != 29)
                    for (float v : mixed) finite = finite && std::isfinite(v);
                if (!finite) { std::printf("FAIL: a non-finite output (types %d/%d + %d/%d)\n", pr[0], pr[1], pr[2], pr[3]); ++failures; }
            }
        }
    }
    std::printf("pool_mixed_fmt_test (STRATA_POOL_FUSED=%s): %d of %d mixed calls differ bitwise\n",
                std::getenv("STRATA_POOL_FUSED") ? std::getenv("STRATA_POOL_FUSED") : "unset", failures, checks);
    return failures ? 1 : 0;
}

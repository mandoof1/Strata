// Launch a Tensile-generated HSS kernel (Alik_Bljk, strided batched, StaggerU, WGM) with hand-packed arguments; check it
// bitwise against rocBLAS's fallback hipblasGemmEx and time both (warm).  Usage: tl <kernel.co> <name> <MT1> <DU> <WGM>
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hipblas/hipblas.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <random>
#define CK(x) do { auto ck_ = (x); if (ck_ != 0) { std::printf("ERR %d line %d\n", (int) ck_, __LINE__); std::exit(1); } } while (0)
struct __attribute__((packed)) Args {
    uint64_t sizeA, sizeB;
    float *d, *c; const void *a, *b;
    float alpha, beta;
    uint32_t strideD[2], strideC[2], strideA[2], strideB[2];
    uint32_t size0, size1, size2, sizeL;
    int32_t staggerUIter;
    uint32_t nwg0, nwg1, numFullBlocks, wgmRem1, magicWgmRem1;
};
static_assert(sizeof(Args) == 128, "kernarg layout");
int main(int argc, char** argv) {
    const char* co = argv[1]; const char* name = argv[2]; const int MT0 = 64, MT1 = std::atoi(argv[3]), DU = std::atoi(argv[4]), WGM = std::atoi(argv[5]);
    const int G = argc > 6 ? std::atoi(argv[6]) : 16;
    const int M = 1280, K = 2560;
    hipModule_t mod; hipFunction_t fn; CK(hipModuleLoad(&mod, co)); CK(hipModuleGetFunction(&fn, mod, name));
    hipblasHandle_t h; CK(hipblasCreate(&h));
    std::mt19937 rng(11); std::normal_distribution<float> dw(0.f, 0.02f), dx(0.f, 1.f);
    const int pool[] = {70, 75, 80, 85, 90, 95, 100, 105, 110, 120, 125, 135, 145, 150, 160, 170, 180, 190, 200, 215, 230, 250, 270, 300, 330, 380, 450, 600};
    std::vector<int> Ts(G); std::vector<__half*> Ws(G), Xs(G); std::vector<float*> Ys(G), Rs(G); size_t tot = 0;
    for (int e = 0; e < G; ++e) {
        Ts[e] = pool[rng() % (sizeof pool / sizeof pool[0])]; tot += Ts[e];
        std::vector<__half> hw((size_t) M * K), hx((size_t) Ts[e] * K);
        for (auto& v : hw) v = __float2half(dw(rng)); for (auto& v : hx) v = __float2half(dx(rng));
        CK(hipMalloc(&Ws[e], hw.size() * 2)); CK(hipMalloc(&Xs[e], hx.size() * 2)); CK(hipMalloc(&Ys[e], (size_t) Ts[e] * M * 4)); CK(hipMalloc(&Rs[e], (size_t) Ts[e] * M * 4));
        CK(hipMemcpy(Ws[e], hw.data(), hw.size() * 2, hipMemcpyHostToDevice)); CK(hipMemcpy(Xs[e], hx.data(), hx.size() * 2, hipMemcpyHostToDevice));
    }
    auto launch = [&](int e, hipStream_t st) {
        const int T = Ts[e]; Args a{};
        a.sizeA = (uint64_t) M * K; a.sizeB = (uint64_t) T * K;
        a.d = Ys[e]; a.c = Ys[e]; a.a = Ws[e]; a.b = Xs[e]; a.alpha = 1.f; a.beta = 0.f;
        a.strideD[0] = M; a.strideD[1] = M * T; a.strideC[0] = M; a.strideC[1] = M * T;
        a.strideA[0] = K; a.strideA[1] = M * K; a.strideB[0] = K; a.strideB[1] = T * K;
        a.size0 = M; a.size1 = T; a.size2 = 1; a.sizeL = K;
        unsigned su = 32; const int iters = K / DU; const unsigned shifted = 256 / (DU * 2);
        while (su > 1) { if (iters >= (int) (su * shifted)) break; su /= 2; }
        a.staggerUIter = (int) su - 1;
        a.nwg0 = (M + MT0 - 1) / MT0; a.nwg1 = (T + MT1 - 1) / MT1;
        a.numFullBlocks = a.nwg1 / WGM; a.wgmRem1 = a.nwg1 % WGM; if (a.wgmRem1 == 0) a.wgmRem1 = WGM;
        a.magicWgmRem1 = (uint32_t) ((1ull << 31) / a.wgmRem1 + 1);
        size_t sz = sizeof a;
        void* cfg[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, &a, HIP_LAUNCH_PARAM_BUFFER_SIZE, &sz, HIP_LAUNCH_PARAM_END};
        CK(hipModuleLaunchKernel(fn, a.nwg0, a.nwg1, 1, 256, 1, 1, 0, st, nullptr, cfg));
    };
    const float al = 1.f, be = 0.f;
    auto roc = [&] { for (int e = 0; e < G; ++e) hipblasGemmEx(h, HIPBLAS_OP_T, HIPBLAS_OP_N, M, Ts[e], K, &al, Ws[e], HIP_R_16F, K, Xs[e], HIP_R_16F, K, &be, Rs[e], HIP_R_32F, M, HIPBLAS_COMPUTE_32F, HIPBLAS_GEMM_DEFAULT); };
    std::vector<hipStream_t> ss(8); for (auto& s : ss) CK(hipStreamCreateWithFlags(&s, hipStreamNonBlocking));
    hipEvent_t start, done[8]; CK(hipEventCreateWithFlags(&start, hipEventDisableTiming)); for (auto& d : done) CK(hipEventCreateWithFlags(&d, hipEventDisableTiming));
    auto multi = [&](int S) {   // the group's experts round-robin over S streams, forked from and joined to stream 0
        hipEventRecord(start, 0);
        for (int s = 0; s < S; ++s) hipStreamWaitEvent(ss[s], start, 0);
        for (int e = 0; e < G; ++e) launch(e, ss[e % S]);
        for (int s = 0; s < S; ++s) { hipEventRecord(done[s], ss[s]); hipStreamWaitEvent(0, done[s], 0); }
    };
    // bitwise once
    roc(); multi(4); CK(hipDeviceSynchronize());
    int bad = 0; for (int e = 0; e < G; ++e) { std::vector<float> y((size_t) Ts[e] * M), r(y.size());
        CK(hipMemcpy(y.data(), Ys[e], y.size() * 4, hipMemcpyDeviceToHost)); CK(hipMemcpy(r.data(), Rs[e], r.size() * 4, hipMemcpyDeviceToHost)); bad += !!std::memcmp(y.data(), r.data(), y.size() * 4); }
    std::printf("G=%d experts, %zu tokens: %d experts differ\n", G, tot, bad);
    auto timeit = [&](auto f) { auto t0 = std::chrono::steady_clock::now(); while (std::chrono::steady_clock::now() - t0 < std::chrono::milliseconds(800)) { f(); hipDeviceSynchronize(); }
        hipEvent_t e0, e1; hipEventCreate(&e0); hipEventCreate(&e1); float ms; hipEventRecord(e0, 0); for (int i = 0; i < 20; ++i) f(); hipEventRecord(e1, 0); hipEventSynchronize(e1); hipEventElapsedTime(&ms, e0, e1); return ms / 20; };
    const double fl = 2.0 * M * K * tot;
    const float tr = timeit(roc); std::printf("rocBLAS per expert  %7.1f us %5.1f TF\n", tr * 1000, fl / (tr * 1e-3) / 1e12);
    for (int S : {1, 2, 4, 8}) { const float t = timeit([&] { multi(S); }); std::printf("tensile %d stream(s) %7.1f us %5.1f TF  x%.2f\n", S, t * 1000, fl / (t * 1e-3) / 1e12, tr / t); }
}

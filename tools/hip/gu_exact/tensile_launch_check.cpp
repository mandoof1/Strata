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
    const int M = 1280, K = 2560, TM = 4100;
    hipModule_t mod; hipFunction_t fn; CK(hipModuleLoad(&mod, co)); CK(hipModuleGetFunction(&fn, mod, name));
    hipblasHandle_t h; CK(hipblasCreate(&h));
    std::mt19937 rng(5); std::normal_distribution<float> dw(0.f, 0.02f), dx(0.f, 1.f);
    std::vector<__half> hw((size_t) M * K), hx((size_t) TM * K);
    for (auto& v : hw) v = __float2half(dw(rng)); for (auto& v : hx) v = __float2half(dx(rng));
    __half *W, *X; float *Y, *R;
    CK(hipMalloc(&W, hw.size() * 2)); CK(hipMalloc(&X, hx.size() * 2)); CK(hipMalloc(&Y, (size_t) TM * M * 4)); CK(hipMalloc(&R, (size_t) TM * M * 4));
    CK(hipMemcpy(W, hw.data(), hw.size() * 2, hipMemcpyHostToDevice)); CK(hipMemcpy(X, hx.data(), hx.size() * 2, hipMemcpyHostToDevice));
    auto launch = [&](int T, float* out) {
        Args a{};
        a.sizeA = (uint64_t) M * K; a.sizeB = (uint64_t) T * K;
        a.d = out; a.c = out; a.a = W; a.b = X; a.alpha = 1.f; a.beta = 0.f;
        a.strideD[0] = M; a.strideD[1] = M * T; a.strideC[0] = M; a.strideC[1] = M * T;
        a.strideA[0] = K; a.strideA[1] = M * K; a.strideB[0] = K; a.strideB[1] = T * K;
        a.size0 = M; a.size1 = T; a.size2 = 1; a.sizeL = K;
        // Tensile's staggerUIter: StaggerU 32, stride 256 bytes = (256 / (DU * 2)) DU-iterations
        unsigned su = 32; const int iters = K / DU; const unsigned shifted = 256 / (DU * 2);
        while (su > 1) { if (iters >= (int) (su * shifted)) break; su /= 2; }
        a.staggerUIter = (int) su - 1;
        a.nwg0 = (M + MT0 - 1) / MT0; a.nwg1 = (T + MT1 - 1) / MT1;
        a.numFullBlocks = a.nwg1 / WGM; a.wgmRem1 = a.nwg1 % WGM; if (a.wgmRem1 == 0) a.wgmRem1 = WGM;
        a.magicWgmRem1 = (uint32_t) ((1ull << 31) / a.wgmRem1 + 1);
        size_t sz = sizeof a;
        void* cfg[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, &a, HIP_LAUNCH_PARAM_BUFFER_SIZE, &sz, HIP_LAUNCH_PARAM_END};
        CK(hipModuleLaunchKernel(fn, a.nwg0, a.nwg1, 1, 256, 1, 1, 0, 0, nullptr, cfg));
    };
    std::vector<float> y((size_t) TM * M), r((size_t) TM * M);
    int bad = 0, n = 0;
    for (int T = 65; T <= TM; T += (T < 600 ? 1 : 29)) {
        const float al = 1.f, be = 0.f;
        CK(hipblasGemmEx(h, HIPBLAS_OP_T, HIPBLAS_OP_N, M, T, K, &al, W, HIP_R_16F, K, X, HIP_R_16F, K, &be, R, HIP_R_32F, M, HIPBLAS_COMPUTE_32F, HIPBLAS_GEMM_DEFAULT));
        launch(T, Y); CK(hipDeviceSynchronize());
        CK(hipMemcpy(y.data(), Y, (size_t) T * M * 4, hipMemcpyDeviceToHost)); CK(hipMemcpy(r.data(), R, (size_t) T * M * 4, hipMemcpyDeviceToHost));
        ++n; if (std::memcmp(y.data(), r.data(), (size_t) T * M * 4)) { if (bad++ < 5) std::printf("T=%d differs\n", T); }
    }
    std::printf("%s: %d token counts in 65..%d, %d differ\n", name, n, TM, bad);
    for (int T : {80, 128, 200, 320, 512, 1024}) {
        const float al = 1.f, be = 0.f; hipEvent_t e0, e1; hipEventCreate(&e0); hipEventCreate(&e1); float ms1, ms2;
        auto roc = [&] { hipblasGemmEx(h, HIPBLAS_OP_T, HIPBLAS_OP_N, M, T, K, &al, W, HIP_R_16F, K, X, HIP_R_16F, K, &be, R, HIP_R_32F, M, HIPBLAS_COMPUTE_32F, HIPBLAS_GEMM_DEFAULT); };
        auto t0 = std::chrono::steady_clock::now(); while (std::chrono::steady_clock::now() - t0 < std::chrono::milliseconds(800)) { roc(); launch(T, Y); hipDeviceSynchronize(); }
        hipEventRecord(e0); for (int i = 0; i < 50; ++i) roc(); hipEventRecord(e1); hipEventSynchronize(e1); hipEventElapsedTime(&ms1, e0, e1);
        hipEventRecord(e0); for (int i = 0; i < 50; ++i) launch(T, Y); hipEventRecord(e1); hipEventSynchronize(e1); hipEventElapsedTime(&ms2, e0, e1);
        const double fl = 2.0 * M * K * T;
        std::printf("T=%4d  rocBLAS fallback %6.1f us (%5.1f TF)   tensile %6.1f us (%5.1f TF)  x%.2f\n", T, ms1 * 20, fl / (ms1 / 50 * 1e-3) / 1e12, ms2 * 20, fl / (ms2 / 50 * 1e-3) / 1e12, ms1 / ms2);
    }
}

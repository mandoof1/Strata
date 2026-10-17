// Replica of rocBLAS's HSS MT64x32x8 GSU1 kernel arithmetic (N=1280, K=2560): out[t][n] = sequential FMA chain over k
// starting at ((n / 64) % 16) * 128 and wrapping.  Bitwise check against hipblasGemmEx for T = 1..1100, plus timing.
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hipblas/hipblas.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <random>
#define CK(x) do { auto e = (x); if (e != 0) { std::printf("ERR %d line %d\n", (int) e, __LINE__); std::exit(1); } } while (0)
__global__ void naive(const __half* __restrict__ W, const __half* __restrict__ X, float* __restrict__ Y, int N, int T, int K) {
    const int n = blockIdx.x * blockDim.x + threadIdx.x, t = blockIdx.y;
    if (n >= N || t >= T) return;
    const __half* w = W + (size_t) n * K; const __half* x = X + (size_t) t * K;
    const int start = ((n / 64) % 16) * 128;
    float acc = 0.f;
    for (int i = 0; i < K; ++i) { int k = start + i; if (k >= K) k -= K; acc = __builtin_fmaf(__half2float(w[k]), __half2float(x[k]), acc); }
    Y[(size_t) t * N + n] = acc;
}
int main() {
    const int N = 1280, K = 2560, TM = 1100;
    hipblasHandle_t h; CK(hipblasCreate(&h));
    std::mt19937 rng(3); std::normal_distribution<float> dw(0.f, 0.02f), dx(0.f, 1.f);
    std::vector<__half> hw((size_t) N * K), hx((size_t) TM * K);
    for (auto& v : hw) v = __float2half(dw(rng)); for (auto& v : hx) v = __float2half(dx(rng));
    __half *W, *X; float *Y, *R;
    CK(hipMalloc(&W, hw.size() * 2)); CK(hipMalloc(&X, hx.size() * 2)); CK(hipMalloc(&Y, (size_t) TM * N * 4)); CK(hipMalloc(&R, (size_t) TM * N * 4));
    CK(hipMemcpy(W, hw.data(), hw.size() * 2, hipMemcpyHostToDevice)); CK(hipMemcpy(X, hx.data(), hx.size() * 2, hipMemcpyHostToDevice));
    std::vector<float> y((size_t) TM * N), r((size_t) TM * N);
    int first = -1, last_bad = -1; std::string runs;
    for (int T = 1; T <= TM; ++T) {
        const float a = 1.f, b = 0.f;
        CK(hipblasGemmEx(h, HIPBLAS_OP_T, HIPBLAS_OP_N, N, T, K, &a, W, HIP_R_16F, K, X, HIP_R_16F, K, &b, Y, HIP_R_32F, N, HIPBLAS_COMPUTE_32F, HIPBLAS_GEMM_DEFAULT));
        hipLaunchKernelGGL(naive, dim3(N / 128, T), dim3(128), 0, 0, W, X, R, N, T, K);
        CK(hipDeviceSynchronize());
        CK(hipMemcpy(y.data(), Y, (size_t) T * N * 4, hipMemcpyDeviceToHost)); CK(hipMemcpy(r.data(), R, (size_t) T * N * 4, hipMemcpyDeviceToHost));
        const bool same = std::memcmp(y.data(), r.data(), (size_t) T * N * 4) == 0;
        if (same && first < 0) first = T;
        if (!same) last_bad = T;
        if (T % 100 == 0 || (same != (std::memcmp(y.data(), r.data(), 4) == 0))) {}
        runs += same ? '1' : '0';
    }
    // print the T ranges where the replica is bitwise
    char c = 0; int s = 1;
    for (int T = 1; T <= TM + 1; ++T) { char v = T <= TM ? runs[T - 1] : 'x'; if (v != c) { if (c) std::printf("T %d-%d: %s\n", s, T - 1, c == '1' ? "BITWISE" : "differs"); c = v; s = T; } }
    // timing at T = 320 and 1024
    for (int T : {128, 320, 1024}) {
        hipEvent_t e0, e1; hipEventCreate(&e0); hipEventCreate(&e1); const float a = 1.f, b = 0.f; float ms1, ms2;
        hipEventRecord(e0); for (int i = 0; i < 20; ++i) hipblasGemmEx(h, HIPBLAS_OP_T, HIPBLAS_OP_N, N, T, K, &a, W, HIP_R_16F, K, X, HIP_R_16F, K, &b, Y, HIP_R_32F, N, HIPBLAS_COMPUTE_32F, HIPBLAS_GEMM_DEFAULT);
        hipEventRecord(e1); hipEventSynchronize(e1); hipEventElapsedTime(&ms1, e0, e1);
        hipEventRecord(e0); for (int i = 0; i < 20; ++i) hipLaunchKernelGGL(naive, dim3(N / 128, T), dim3(128), 0, 0, W, X, R, N, T, K);
        hipEventRecord(e1); hipEventSynchronize(e1); hipEventElapsedTime(&ms2, e0, e1);
        std::printf("T=%d rocBLAS %.1f us  naive replica %.1f us  (%.1f TFLOPS rocBLAS)\n", T, ms1 * 50, ms2 * 50, 2.0 * N * K * T / (ms1 / 20 * 1e-3) / 1e12);
    }
}

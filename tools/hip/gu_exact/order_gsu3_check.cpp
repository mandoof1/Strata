// Replica of rocBLAS's HSS MT128x32x16 GSU3 arithmetic (T < 65): split g in {0,1,2} = DepthU-16 iterations
// g, g+3, ... (53 or 54 of them), each a sequential FMA chain starting at list position ((n/128) % 4) * 8 (wrapping),
// result = (p0 + p1) + p2.  Bitwise check vs hipblasGemmEx for T = 1..64.
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hipblas/hipblas.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <random>
#define CK(x) do { auto ck_ = (x); if (ck_ != 0) { std::printf("ERR %d line %d\n", (int) ck_, __LINE__); std::exit(1); } } while (0)
__global__ void naive3(const __half* W, const __half* X, float* Y, int N, int T, int K) {
    const int n = blockIdx.x * blockDim.x + threadIdx.x, t = blockIdx.y;
    if (n >= N || t >= T) return;
    const __half* w = W + (size_t) n * K; const __half* x = X + (size_t) t * K;
    const int IT = K / 16, rot = ((n / 128) % 4) * 8;
    float p[3];
    for (int g = 0; g < 3; ++g) {
        const int len = (IT - g + 2) / 3;   // iterations g, g+3, ... < IT
        float acc = 0.f;
        for (int i = 0; i < len; ++i) {
            const int it = g + 3 * ((rot + i) % len);
            for (int k = it * 16; k < it * 16 + 16; ++k) acc = __builtin_fmaf(__half2float(w[k]), __half2float(x[k]), acc);
        }
        p[g] = acc;
    }
    Y[(size_t) t * N + n] = (p[0] + p[1]) + p[2];
}
int main() {
    const int N = 1280, K = 2560, TM = 64;
    hipblasHandle_t h; CK(hipblasCreate(&h));
    std::mt19937 rng(9); std::normal_distribution<float> dw(0.f, 0.02f), dx(0.f, 1.f);
    std::vector<__half> hw((size_t) N * K), hx((size_t) TM * K);
    for (auto& v : hw) v = __float2half(dw(rng)); for (auto& v : hx) v = __float2half(dx(rng));
    __half *W, *X; float *Y, *R;
    CK(hipMalloc(&W, hw.size() * 2)); CK(hipMalloc(&X, hx.size() * 2)); CK(hipMalloc(&Y, (size_t) TM * N * 4)); CK(hipMalloc(&R, (size_t) TM * N * 4));
    CK(hipMemcpy(W, hw.data(), hw.size() * 2, hipMemcpyHostToDevice)); CK(hipMemcpy(X, hx.data(), hx.size() * 2, hipMemcpyHostToDevice));
    std::vector<float> y((size_t) TM * N), r((size_t) TM * N);
    for (int T = 1; T <= TM; ++T) {
        const float a = 1.f, b = 0.f;
        CK(hipblasGemmEx(h, HIPBLAS_OP_T, HIPBLAS_OP_N, N, T, K, &a, W, HIP_R_16F, K, X, HIP_R_16F, K, &b, Y, HIP_R_32F, N, HIPBLAS_COMPUTE_32F, HIPBLAS_GEMM_DEFAULT));
        hipLaunchKernelGGL(naive3, dim3(N / 128, T), dim3(128), 0, 0, W, X, R, N, T, K); CK(hipDeviceSynchronize());
        CK(hipMemcpy(y.data(), Y, (size_t) T * N * 4, hipMemcpyDeviceToHost)); CK(hipMemcpy(r.data(), R, (size_t) T * N * 4, hipMemcpyDeviceToHost));
        size_t same = 0; for (size_t i = 0; i < (size_t) T * N; ++i) same += !std::memcmp(&y[i], &r[i], 4);
        std::printf("T=%2d %s (%.4f)\n", T, same == (size_t) T * N ? "BITWISE" : "differs", (double) same / (T * N));
    }
}

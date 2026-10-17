// Bitwise check of a Tensile client dump (A: M x K fp16, B: N x K fp16, D: N x M fp32) against rocBLAS's fallback
// hipblasGemmEx on the same A and B.  Usage: cmpdump <dir> M N K
#include <hip/hip_runtime.h>
#include <hipblas/hipblas.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <string>
static std::vector<char> rd(const std::string& p) { FILE* f = std::fopen(p.c_str(), "rb"); std::fseek(f, 0, SEEK_END); long n = std::ftell(f); std::fseek(f, 0, SEEK_SET); std::vector<char> v(n); std::fread(v.data(), 1, n, f); std::fclose(f); return v; }
int main(int argc, char** argv) {
    const std::string d = argv[1]; const int M = std::atoi(argv[2]), N = std::atoi(argv[3]), K = std::atoi(argv[4]);
    auto A = rd(d + "/tensor_A.bin"), B = rd(d + "/tensor_B.bin"), D = rd(d + "/tensor_D.bin");
    if (A.size() != (size_t) M * K * 2 || B.size() != (size_t) N * K * 2 || D.size() != (size_t) N * M * 4) { std::printf("size mismatch %zu %zu %zu\n", A.size(), B.size(), D.size()); return 1; }
    void *a, *b; float* y; hipMalloc(&a, A.size()); hipMalloc(&b, B.size()); hipMalloc(&y, D.size());
    hipMemcpy(a, A.data(), A.size(), hipMemcpyHostToDevice); hipMemcpy(b, B.data(), B.size(), hipMemcpyHostToDevice);
    hipblasHandle_t h; hipblasCreate(&h); const float al = 1.f, be = 0.f;
    hipblasGemmEx(h, HIPBLAS_OP_T, HIPBLAS_OP_N, M, N, K, &al, a, HIP_R_16F, K, b, HIP_R_16F, K, &be, y, HIP_R_32F, M, HIPBLAS_COMPUTE_32F, HIPBLAS_GEMM_DEFAULT);
    std::vector<float> r((size_t) N * M); hipMemcpy(r.data(), y, D.size(), hipMemcpyDeviceToHost);
    const float* t = (const float*) D.data(); size_t same = 0, finite = 0;
    for (size_t i = 0; i < r.size(); ++i) { same += !std::memcmp(&r[i], &t[i], 4); finite += std::isfinite(t[i]); }
    std::printf("bitwise equal %zu / %zu (%.4f), finite %zu\n", same, r.size(), (double) same / r.size(), finite);
    for (int i = 0; i < 3; ++i) std::printf("  [%d] rocBLAS %.9g tensile %.9g\n", i, r[i], t[i]);
}

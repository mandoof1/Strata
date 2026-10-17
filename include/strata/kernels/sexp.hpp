// include/strata/kernels/sexp.hpp - exp(x) in FP32 from operations every IEEE platform rounds the same way
// (STRATA_SWIGLU_SEXP: the routed experts' SwiGLU, so a CPU row can match a GPU row bit for bit).
//
// __expf on AMD is v_exp_f32(x * log2 e), and v_exp_f32 is not correctly rounded (on gfx1100 1 ulp off for ~4% of
// inputs, in a way no CPU reproduces), while the rounded product x * log2 e loses ~|x| / 2 ulp on its own. This one is
// Cody-Waite: k = rint(x log2 e), r = x - k ln 2 in two fmas, a degree-7 Horner polynomial for e^r (|r| <= 0.347) in
// fmas, then ldexp. Only explicit fma, rint and ldexp - each correctly rounded on both sides - so nothing depends on
// the compiler contracting a product into a sum. Max error ~1 ulp (tools/hip/sexp checks it and the GPU against the
// host for every float).
#pragma once

#include <cmath>

#if defined(__CUDACC__) || defined(__HIPCC__)
#define STRATA_SEXP_HD __host__ __device__ __forceinline__
#else
#define STRATA_SEXP_HD inline
#endif

namespace strata::kernels {

STRATA_SEXP_HD float sexp(const float x) {
    if (!(x < 88.72284f)) return x != x ? x : INFINITY;   // e^x > FLT_MAX (and NaN)
    if (x < -103.97208f) return 0.0f;                     // below half the smallest subnormal
    const float k = rintf(x * 1.44269504f);
    float r = fmaf(k, -0.693145751953125f, x);            // ln 2 high part: 15 bits, k * it is exact
    r = fmaf(k, -1.42860677e-06f, r);                     // ln 2 low part
    float p = 1.98412698e-04f;                            // 1/5040
    p = fmaf(p, r, 1.38888889e-03f);                      // 1/720
    p = fmaf(p, r, 8.33333333e-03f);                      // 1/120
    p = fmaf(p, r, 4.16666667e-02f);                      // 1/24
    p = fmaf(p, r, 1.66666667e-01f);                      // 1/6
    p = fmaf(p, r, 0.5f);
    p = fmaf(p, r, 1.0f);
    p = fmaf(p, r, 1.0f);
    return ldexpf(p, (int) k);
}

}  // namespace strata::kernels

#undef STRATA_SEXP_HD

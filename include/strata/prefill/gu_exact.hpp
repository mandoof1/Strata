// include/strata/prefill/gu_exact.hpp - the experts' gate/up product for many experts in one launch, bit for bit what
// rocBLAS's FP16 fallback gives each expert alone (STRATA_PF_GU_EXACT, HIP gfx1100).
//
// The prompt path's gate/up product (Y[T, 1280] = X[T, 2560] . W[1280, 2560]^T, FP16 in, FP32 out) has no hipBLASLt
// calibration, so it runs on rocBLAS's fallback Tensile kernels, two of them:
// - from 65 tokens Cijk_Alik_Bljk_HSS_BH_MT64x32x8 (GSU1, StaggerU 32 / 256 bytes): every output is one FMA chain over k
//   in order, starting at k = ((n / 64) % 16) * 128 and wrapping;
// - 2..64 tokens Cijk_Alik_Bljk_HSS_BH_MT128x32x16 (GSU3) + Cijk_S_PostGSU3: split g of 3 takes the 16-wide k steps
//   g, g+3, g+6, ... (53 or 54 of them), each split one FMA chain starting at its step ((n / 128) % 4) * 8 and wrapping,
//   and the result is (p0 + p1) + p2.
// (1 token: a GEMV, left to rocBLAS.)  These kernels compute the same chains for every output of a group of experts, so
// the products are the same bits (checked against Gemm::f16 at start-up; any difference turns them off).
// On gfx1100 the >= 65-token experts run on a Tensile-generated kernel instead (gu_tensile_gfx1100.s: the same
// order, MacroTile 64x128, ~2x the fallback's speed), one launch per expert over 8 streams; STRATA_PF_GU_TENSILE=0
// keeps them on gu_exact's own kernel.
#pragma once

#include <cstdint>

namespace strata::prefill {

constexpr int kGuExactMax = 64;          // experts in one launch
constexpr int64_t kGuExactMinT = 2;      // 1 token: rocBLAS's GEMV
constexpr int64_t kGuExactSplitT = 65;   // below: the 3-split chains (GSU3)
constexpr int64_t kGuExactMaxT = 16384;
/// FP32 workspace the 3-split experts of one group need: 3 partials of (64 rows x 1280) per expert
constexpr int64_t kGuExactWsFloats = (int64_t) 3 * kGuExactMax * 64 * 1280;

struct GuExactGroup {
    const uint16_t* x[kGuExactMax];   // FP16 activations, row stride 2560
    const uint16_t* w[kGuExactMax];   // FP16 gate/up weights [1280, 2560]
    float* y[kGuExactMax];            // FP32 out, row stride 1280
    int t[kGuExactMax];               // tokens of each expert
    int b0[kGuExactMax + 1];          // filled by the launcher
    int row0[kGuExactMax + 1];        // (the 3-split experts' rows in the workspace, filled by the launcher)
    int n = 0;
};

/// Y_e = X_e . W_e^T for every expert of the group: one launch for the experts from 65 tokens, one plus a reduction for
/// those below (`ws`: kGuExactWsFloats floats).  HIP; false elsewhere or on a launch error.
bool gu_exact_run(GuExactGroup& g, float* ws, void* stream);

/// True when this device runs the >= 65-token experts on the Tensile kernel (gfx1100, loaded, not turned off).
bool gu_exact_tensile();

}  // namespace strata::prefill

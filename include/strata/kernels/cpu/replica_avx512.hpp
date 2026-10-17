// src/kernels/cpu/replica_avx512.hpp - STRATA_CPU_PARITY: the GPU's native expert arithmetic on the CPU (AVX-512),
// bit for bit (see replica_avx512.cpp).  n_embd 2560, n_ff 640.
#pragma once

#include <cstddef>
#include <cstdint>

namespace strata::kernels::cpu::replica {

/// A token's gate/up activations as the GPU quantizes them (q8_1): the int8 values in order, each block's fp16 scale
/// as a float.  2,880 bytes: fits a kNativeActBytes slot.
struct ActX {
    int8_t qs[2560];
    float d8[80];
};
/// A token's down activations (q8_1 of h, 640 values).  IQ4_NL: the values in part order (per 32: 0-7, 16-23, 8-15,
/// 24-31), d = each scale twice; Q2_0: value order, d = the scales, ysum = each block's sum.  880 bytes: fits a
/// kNativeHBytes slot.
struct ActH {
    int8_t q[640];
    float d[40];
    int32_t ysum[20];
};
void quant_x(const float* x, int n, ActX& a);
void quant_h(const float* h, int n, int d_type, ActH& a);

bool gu_supported(int type);     ///< IQ3_XXS, IQ3_S, IQ2_S, IQ4_XS
bool down_supported(int type);   ///< IQ4_NL, Q2_0
/// h = SwiGLU(gate, up) (sexp) for rows [r0, r1) of nt tokens: the GPU's bits
void gu_rows(int type, int nt, const uint8_t* blob, size_t gu_row, size_t up_off, const ActX* const* act, float* const* h,
             int r0, int r1);
/// the down rows [r0, r1) of nt tokens: the GPU's bits
void down_rows(int type, int nt, const uint8_t* w, size_t row_bytes, const ActH* const* h, float* const* out, int r0, int r1);

}  // namespace strata::kernels::cpu::replica

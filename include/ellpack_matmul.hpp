// include/ellpack_matvec.hpp
#pragma once
#include <cstdint>
#include <immintrin.h>
#include "utils.hpp"
#include "ellpack_encoder.hpp"

void sorted_ellpack_matmul_microtx(
    const SortedEllpack& E,  // must have E.KB == KB
    const float* X,          // [B x N], row-major
    uint32_t B,
    const float* bias,       // [M] or nullptr
    float* Y                 // [M x B], row-major
);

void ellpack_matmul_batchmajor(
    const Ellpack& E,
    const float*   X,        // [B x N] row-major
    uint32_t       B,        // batch size
    const float*   bias,     // [M] or nullptr
    float*         Y         // [M x B] row-major
);

/// ellpack_matmul: Y = E × X  + bias
/// - E: m×n ELLPACK representation
/// - X: n×C input matrix (col‑major or row‑major, see notes)
/// - C: number of columns in X
/// - bias: length‑m vector to add to each output column (nullptr ⇒ zero init)
/// - Y: output buffer, size m×C (row‑major, column j starts at Y + j*m)
template<bool USE_MASK, bool FUSE_RELU>
void ellpack_matmul_fused(
    const Ellpack&    E,
    const float*      X,
    uint32_t          C,
    const float*      bias,
    float*            Y
);

inline void ellpack_matmul(
    const Ellpack &E,
    const float*   X,
    uint32_t       C,
    const float*   bias,
    float*         Y)
{
    bool use_avx512   = supports_avx512();
    uint32_t simd_w   = use_avx512 ? 16u : 8u;
    bool use_mask     = (C % simd_w) != 0;

    if (use_avx512) {
        if (use_mask)
            ellpack_matmul_fused<true,  false>(E, X, C, bias, Y);
        else
            ellpack_matmul_fused<false, false>(E, X, C, bias, Y);
    } else {
        // scalar fallback
        ellpack_matmul_fused<true, false>(E, X, C, bias, Y);
    }
}
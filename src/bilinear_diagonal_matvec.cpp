#include "bilinear_diagonal_matvec.hpp"
#include <cstring>     // std::memcpy
#include <algorithm>   // std::fill
#include <immintrin.h> // AVX2 intrinsics
#include <omp.h>

void quasi_dense_matvec_mt(
    const QuasiDense& Q,
    const XtDense&   X,
    const float*     bias,
    float*           y,
    int              threads
) {
    const uint32_t m = Q.m;
    const uint32_t r = Q.r;

    // Initialize output y: copy bias or zero
    if (bias) {
        std::memcpy(y, bias, size_t(m) * sizeof(float));
    } else {
        std::fill(y, y + size_t(m), 0.0f);
    }

    // Determine thread count
    const int num_threads = (threads > 0 ? threads : omp_get_max_threads());

    // Row‑wise vectorized dot of packed Wd and Xt
    #pragma omp parallel for num_threads(num_threads)
    for (uint32_t i = 0; i < m; ++i) {
        const float* wrow = Q.Wd.data() + size_t(i) * r;
        const float* xrow = X.Xt.data() + size_t(i) * r;
        uint32_t j = 0;
        __m256 acc_vec = _mm256_setzero_ps();

        // Vectorized loop, 8 floats per iteration
        for (; j + 8 <= r; j += 8) {
            __m256 w = _mm256_loadu_ps(wrow + j);
            __m256 x = _mm256_loadu_ps(xrow + j);
            acc_vec = _mm256_fmadd_ps(w, x, acc_vec);
        }

        // Horizontal add of acc_vec
        __m128 hi = _mm256_extractf128_ps(acc_vec, 1);
        __m128 lo = _mm256_castps256_ps128(acc_vec);
        __m128 sum128 = _mm_add_ps(lo, hi);
        sum128 = _mm_hadd_ps(sum128, sum128);
        sum128 = _mm_hadd_ps(sum128, sum128);
        float acc = _mm_cvtss_f32(sum128);

        // Remainder scalar cleanup
        for (; j < r; ++j) {
            acc += wrow[j] * xrow[j];
        }

        y[i] += acc;
    }
}
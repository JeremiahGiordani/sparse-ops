#include "bilinear_diagonal_matvec.hpp"
#include <cstring>     // std::memcpy
#include <algorithm>   // std::fill
#include <immintrin.h> // AVX2/AVX512 intrinsics
#include <omp.h>

/// Detect at runtime whether AVX512F is supported.
static inline bool supports_avx512() {
#if defined(__GNUC__)
    return __builtin_cpu_supports("avx512f");
#else
    return false;
#endif
}

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

    // Detect AVX512 availability once
    const bool use512 = supports_avx512();

    if (use512) {
        // 16-wide AVX-512 code path
        #pragma omp parallel for num_threads(num_threads)
        for (uint32_t i = 0; i < m; ++i) {
            const float* wrow = Q.Wd.data()  + size_t(i) * r;
            const float* xrow = X.Xt.data()  + size_t(i) * r;
            uint32_t j = 0;
            __m512 acc_vec = _mm512_setzero_ps();

            // vectorized 16 floats per iteration
            for (; j + 16 <= r; j += 16) {
                __m512 w = _mm512_loadu_ps(wrow + j);
                __m512 x = _mm512_loadu_ps(xrow + j);
                acc_vec = _mm512_fmadd_ps(w, x, acc_vec);
            }

            // horizontal reduction of 16-wide acc_vec
            __m256 lo256 = _mm512_castps512_ps256(acc_vec);
            __m256 hi256 = _mm512_extractf32x8_ps(acc_vec, 1);
            __m256 sum256 = _mm256_add_ps(lo256, hi256);
            __m128 lo128 = _mm256_castps256_ps128(sum256);
            __m128 hi128 = _mm256_extractf128_ps(sum256, 1);
            __m128 sum128 = _mm_add_ps(lo128, hi128);
            sum128 = _mm_hadd_ps(sum128, sum128);
            sum128 = _mm_hadd_ps(sum128, sum128);
            float acc = _mm_cvtss_f32(sum128);

            // remainder scalar cleanup
            for (; j < r; ++j) {
                acc += wrow[j] * xrow[j];
            }
            y[i] += acc;
        }
    } else {
        // 8-wide AVX2 code path
        #pragma omp parallel for num_threads(num_threads)
        for (uint32_t i = 0; i < m; ++i) {
            const float* wrow = Q.Wd.data()  + size_t(i) * r;
            const float* xrow = X.Xt.data()  + size_t(i) * r;
            uint32_t j = 0;
            __m256 acc_vec = _mm256_setzero_ps();

            // vectorized 8 floats per iteration
            for (; j + 8 <= r; j += 8) {
                __m256 w = _mm256_loadu_ps(wrow + j);
                __m256 x = _mm256_loadu_ps(xrow + j);
                acc_vec = _mm256_fmadd_ps(w, x, acc_vec);
            }

            // horizontal add of acc_vec
            __m128 hi = _mm256_extractf128_ps(acc_vec, 1);
            __m128 lo = _mm256_castps256_ps128(acc_vec);
            __m128 sum128 = _mm_add_ps(lo, hi);
            sum128 = _mm_hadd_ps(sum128, sum128);
            sum128 = _mm_hadd_ps(sum128, sum128);
            float acc = _mm_cvtss_f32(sum128);

            // remainder scalar cleanup
            for (; j < r; ++j) {
                acc += wrow[j] * xrow[j];
            }
            y[i] += acc;
        }
    }
}

// src/bilinear_diagonal_matvec.cpp (updated for AlignedBuffer format)
#include "bilinear_diagonal_matvec.hpp"
#include <cstring>       // memcpy
#include <algorithm>     // fill
#include <immintrin.h>   // AVX2/AVX-512 intrinsics + prefetch + maskload
#include <cstdint>
#include <omp.h>

/// Detect at runtime whether AVX512F is supported
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

    // Choose vector width at runtime
    const bool use512 = supports_avx512();

    if (use512) {
        #pragma omp parallel for num_threads(num_threads)
        for (uint32_t i = 0; i < m; ++i) {
            // AlignedBuffer provides .ptr
            const float* wrow = Q.Wd.ptr + size_t(i) * r;
            const float* xrow = X.Xt.ptr + size_t(i) * r;
            // Prefetch next row
            if (i + 1 < m) {
                _mm_prefetch((const char*)(wrow + r), _MM_HINT_T0);
                _mm_prefetch((const char*)(xrow + r), _MM_HINT_T0);
            }

            // 1) compute how many floats until the next 64‑byte boundary
            uintptr_t addr = (uintptr_t)(wrow);
            size_t mis    = addr & 63;                   // misalignment in bytes
            size_t head   = mis ? (64 - mis)/sizeof(float) : 0;
            if (head > r) head = r;                      // guard if r<head

            uint32_t j = 0;
            float acc = 0.0f;

            // 2) scalar or unaligned loop for the head
            for (; j < head; ++j) {
                acc += wrow[j] * xrow[j];
            }

            // 3) now (wrow+j) is 64‑byte aligned; do as many full AVX‑512 loads as you can
            __m512 acc_vec = _mm512_setzero_ps();
            for (; j + 16 <= r; j += 16) {
                __m512 wv = _mm512_load_ps(wrow + j);     // now safe
                __m512 xv = _mm512_load_ps(xrow + j);
                acc_vec = _mm512_fmadd_ps(wv, xv, acc_vec);
            }

            // 4) masked tail (unaligned) for the final rem = r−j elements
            uint32_t rem = r - j;
            if (rem) {
                __mmask16 m = (__mmask16(1) << rem) - 1;
                __m512 wv = _mm512_maskz_loadu_ps(m, wrow + j);
                __m512 xv = _mm512_maskz_loadu_ps(m, xrow + j);
                acc_vec  = _mm512_fmadd_ps(wv, xv, acc_vec);
            }

            // 5) horizontal‑reduce acc_vec into a scalar
            __m256 lo256 = _mm512_castps512_ps256(acc_vec);
            __m256 hi256 = _mm512_extractf32x8_ps(acc_vec, 1);
            __m256 sum256 = _mm256_add_ps(lo256, hi256);
            __m128 lo128  = _mm256_castps256_ps128(sum256);
            __m128 hi128  = _mm256_extractf128_ps(sum256, 1);
            __m128 sum128 = _mm_add_ps(lo128, hi128);
            sum128 = _mm_hadd_ps(sum128, sum128);
            sum128 = _mm_hadd_ps(sum128, sum128);
            acc += _mm_cvtss_f32(sum128);

            // 6) Store
            y[i] += acc;
        }
    } else {
        #pragma omp parallel for num_threads(num_threads)
        for (uint32_t i = 0; i < m; ++i) {
            const float* wrow = Q.Wd.ptr + size_t(i) * r;
            const float* xrow = X.Xt.ptr + size_t(i) * r;
            if (i + 1 < m) {
                _mm_prefetch((const char*)(wrow + r), _MM_HINT_T0);
                _mm_prefetch((const char*)(xrow + r), _MM_HINT_T0);
            }

            uint32_t j = 0;
            __m256 acc_vec = _mm256_setzero_ps();
            // Main vector loop, 8-wide
            for (; j + 8 <= r; j += 8) {
                __m256 w = _mm256_load_ps(wrow + j);
                __m256 x = _mm256_load_ps(xrow + j);
                acc_vec = _mm256_fmadd_ps(w, x, acc_vec);
            }
            // Masked remainder via maskload
            uint32_t rem = r - j;
            if (rem) {
                int32_t mask_arr[8];
                for (uint32_t k = 0; k < 8; ++k) {
                    mask_arr[k] = (k < (int)rem) ? -1 : 0;
                }
                __m256i maskvec = _mm256_loadu_si256((const __m256i*)mask_arr);
                __m256 w = _mm256_maskload_ps(wrow + j, maskvec);
                __m256 x = _mm256_maskload_ps(xrow + j, maskvec);
                acc_vec = _mm256_fmadd_ps(w, x, acc_vec);
            }
            // Horizontal reduction of acc_vec
            __m128 hi = _mm256_extractf128_ps(acc_vec, 1);
            __m128 lo = _mm256_castps256_ps128(acc_vec);
            __m128 sum128 = _mm_add_ps(lo, hi);
            sum128 = _mm_hadd_ps(sum128, sum128);
            sum128 = _mm_hadd_ps(sum128, sum128);
            float acc = _mm_cvtss_f32(sum128);

            y[i] += acc;
        }
    }
}

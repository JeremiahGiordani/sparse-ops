// src/bilinear_diagonal_matvec.cpp (updated for AlignedBuffer format)
#include "bilinear_diagonal_matvec.hpp"
#include "quasi_dense_encoder.hpp"
#include <cstring>       // memcpy
#include <algorithm>     // fill
#include <immintrin.h>   // AVX2/AVX-512 intrinsics + prefetch + maskload
#include <cstdint>
#include <cstdlib>       // getenv, atoi
#include <omp.h>

/// Detect at runtime whether AVX512F is supported
static inline bool supports_avx512() {
#if defined(__GNUC__)
    return __builtin_cpu_supports("avx512f");
#else
    return false;
#endif
}

void quasi_dense_matvec(
    const QuasiDense& Q,
    const float*      x,
    const float*     bias,
    float*           y
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
    const char* env = std::getenv("OMP_NUM_THREADS");
    int num_threads = env ? std::atoi(env) : omp_get_max_threads();

    // Choose vector width at runtime
    const bool use512 = supports_avx512();

    if (use512) {
        #pragma omp parallel for num_threads(num_threads) schedule(static)
        for (uint32_t i = 0; i < m; ++i) {
            size_t base = size_t(i) * Q.r;
            // 1) gather into the row‐buffer that was preallocated in Q.Xt
            const uint32_t* idxRow = Q.idx.data() + base;
            float*           xrow  = Q.Xt.ptr  + base;
            for (uint32_t j = 0; j < Q.r; ++j) {
                xrow[j] = x[ idxRow[j] ];
            }

            const float* wrow = Q.Wd.ptr + base;
            uint32_t      r    = Q.r;
            uint32_t count     = Q.nnz[i];   
            uint32_t      j    = 0;
            __m512        accv = _mm512_setzero_ps();

            // prefetch the start of the next row’s weight block
            if (i + 1 < m) {
                const float* next_wrow = wrow + r;
                _mm_prefetch((const char*)(next_wrow),      _MM_HINT_T0);
                _mm_prefetch((const char*)(next_wrow + r),  _MM_HINT_T0);  // also prefetch next xrow if desired

                const float* next_xrow = Q.Xt.ptr + (base + r);
                _mm_prefetch((const char*)(next_xrow),     _MM_HINT_T0);
                _mm_prefetch((const char*)(next_xrow + r), _MM_HINT_T0);
                
            }
            
            for (; j + 16 <= count; j += 16) {
                __m512 wv = _mm512_loadu_ps(wrow + j);
                __m512 xv = _mm512_loadu_ps(xrow + j);
                accv = _mm512_fmadd_ps(wv, xv, accv);
            }

            // 4) masked tail (unaligned) for the final rem = r−j elements
            uint32_t rem = count - j;
            if (rem) {
                __mmask16 m = (__mmask16(1) << rem) - 1;
                __m512 wv = _mm512_maskz_loadu_ps(m, wrow + j);
                __m512 xv = _mm512_maskz_loadu_ps(m, xrow + j);
                accv  = _mm512_fmadd_ps(wv, xv, accv);
            }

            // 5) horizontal‑reduce accv into a scalar
            __m256 lo256 = _mm512_castps512_ps256(accv);
            __m256 hi256 = _mm512_extractf32x8_ps(accv, 1);
            __m256 sum256 = _mm256_add_ps(lo256, hi256);
            __m128 lo128  = _mm256_castps256_ps128(sum256);
            __m128 hi128  = _mm256_extractf128_ps(sum256, 1);
            __m128 sum128 = _mm_add_ps(lo128, hi128);
            sum128 = _mm_hadd_ps(sum128, sum128);
            sum128 = _mm_hadd_ps(sum128, sum128);
            float acc = _mm_cvtss_f32(sum128);

            // 6) Store
            y[i] += acc;
        }
    } else {
        #pragma omp parallel for num_threads(num_threads) schedule(static)
        for (uint32_t i = 0; i < m; ++i) {
            const float* wrow = Q.Wd.ptr + size_t(i) * r;
            size_t base = size_t(i) * Q.r;
            const uint32_t* idxRow = Q.idx.data() + base;
            float* xrow  = Q.Xt.ptr  + base;
            for (uint32_t j = 0; j < Q.r; ++j) {
                xrow[j] = x[ idxRow[j] ];
            }

            if (i + 1 < m) {
                _mm_prefetch((const char*)(wrow + r), _MM_HINT_T0);
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
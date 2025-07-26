// src/bilinear_diagonal_matvec.cpp (updated for AlignedBuffer format)
#include "bilinear_diagonal_matvec.hpp"
#include "quasi_dense_encoder.hpp"
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
    const float*      x,
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
            uint32_t      j    = 0;
            __m512        accv = _mm512_setzero_ps();

            // prefetch the start of the next row’s weight block
            if (i + 1 < m) {
                const float* next_wrow = wrow + r;
                _mm_prefetch((const char*)(next_wrow),      _MM_HINT_T0);
                _mm_prefetch((const char*)(next_wrow + r),  _MM_HINT_T0);  // also prefetch next xrow if desired
            }
            
            for (; j + 16 <= r; j += 16) {
                __m512 wv = _mm512_loadu_ps(wrow + j);
                __m512 xv = _mm512_load_ps(xrow + j);
                accv = _mm512_fmadd_ps(wv, xv, accv);
            }

            // 4) masked tail (unaligned) for the final rem = r−j elements
            uint32_t rem = r - j;
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


void quasi_dense_matvec_gather(
    const QuasiDense& Q,
    const float*      x,     // length n
    const float*      bias,  // length m or nullptr
    float*            y,     // length m
    int               threads
) {
    const uint32_t m = Q.m, r = Q.r;

    // Initialize output y: copy bias or zero
    if (bias) {
        std::memcpy(y, bias, size_t(m) * sizeof(float));
    } else {
        std::fill(y, y + size_t(m), 0.0f);
    }

    // Determine thread count
    const int num_threads = (threads > 0 ? threads : omp_get_max_threads());

    // decide AVX‑512 vs AVX2…
    #pragma omp parallel for num_threads(num_threads)
    for (uint32_t i = 0; i < m; ++i) {
        const float* wrow = Q.Wd.ptr + size_t(i)*r;
        const uint32_t* idx = Q.idx.data() + size_t(i)*r;
        float acc = bias ? bias[i] : 0.0f;

        uint32_t j = 0;
    #if defined(__AVX512F__)
        __m512 accv = _mm512_setzero_ps();
        // process 16 at a time
        for (; j + 16 <= r; j += 16) {
        // load 16 weights
        __m512 wv = _mm512_loadu_ps(wrow + j);
        // gather 16 inputs from x
        __m512i idxv = _mm512_loadu_si512((__m512i*)(idx + j));
        __m512 xv  = _mm512_i32gather_ps(idxv, x, 4);
        accv = _mm512_fmadd_ps(wv, xv, accv);
        }
        // horizontal reduce accv → acc
        __m256 lo256 = _mm512_castps512_ps256(accv);
        __m256 hi256 = _mm512_extractf32x8_ps(accv,1);
        __m256 sum256 = _mm256_add_ps(lo256, hi256);
        __m128 lo128 = _mm256_castps256_ps128(sum256);
        __m128 hi128 = _mm256_extractf128_ps(sum256,1);
        __m128 sum128 = _mm_add_ps(lo128, hi128);
        sum128 = _mm_hadd_ps(sum128, sum128);
        sum128 = _mm_hadd_ps(sum128, sum128);
        acc += _mm_cvtss_f32(sum128);

        // tail with masked gather
        uint32_t rem = r - j;
        if (rem) {
        __mmask16 msk = ( (__mmask16)1 << rem ) - 1;
        __m512i idxv = _mm512_maskz_loadu_epi32(msk, (idx+j));
        __m512 wv   = _mm512_maskz_loadu_ps(msk, wrow+j);
        __m512 xv   = _mm512_i32gather_ps(idxv, x, 4);
        __m512 tv   = _mm512_fmadd_ps(wv, xv, _mm512_setzero_ps());
        // reduce tv same as above, accumulate into acc
        __m256 lo = _mm512_castps512_ps256(tv);
        __m256 hi = _mm512_extractf32x8_ps(tv,1);
        __m256 s2 = _mm256_add_ps(lo, hi);
        __m128 l2 = _mm256_castps256_ps128(s2);
        __m128 h2 = _mm256_extractf128_ps(s2,1);
        __m128 st = _mm_add_ps(l2,h2);
        st = _mm_hadd_ps(st,st);
        st = _mm_hadd_ps(st,st);
        acc += _mm_cvtss_f32(st);
        }
    #else
        __m256 accv = _mm256_setzero_ps();
        for (; j + 8 <= r; j += 8) {
        __m256 wv = _mm256_loadu_ps(wrow + j);
        __m256i idxv = _mm256_loadu_si256((__m256i*)(idx+j));
        __m256 xv  = _mm256_i32gather_ps(x, idxv, 4);
        accv = _mm256_fmadd_ps(wv, xv, accv);
        }
        // horizontal reduce accv → acc …
        __m128 lo = _mm256_castps256_ps128(accv);
        __m128 hi = _mm256_extractf128_ps(accv,1);
        __m128 sum = _mm_add_ps(lo,hi);
        sum = _mm_hadd_ps(sum,sum);
        sum = _mm_hadd_ps(sum,sum);
        acc += _mm_cvtss_f32(sum);

        // tail with scalar gather (or maskload approach) …
        for (; j < r; ++j) {
        acc += wrow[j] * x[ idx[j] ];
        }
    #endif

        y[i] = acc;
    }
}



void quasi_dense_matvec_mt(
    const QuasiDense& Q,
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
            const float* xrow = Q.Xt.ptr + size_t(i) * r;
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
            const float* xrow = Q.Xt.ptr + size_t(i) * r;
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
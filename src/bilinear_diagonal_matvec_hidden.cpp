#include "bilinear_diagonal_matvec.hpp"
#include <cstring>       // memset
#include <algorithm>     // fill
#include <immintrin.h>   // AVX2/AVX512 intrinsics, gather
#include <cstdint>
#include <omp.h>

static inline bool supports_avx512() {
#if defined(__GNUC__)
    return __builtin_cpu_supports("avx512f");
#else
    return false;
#endif
}

void quasi_dense_matvec_hidden_mt(
    const QuasiDense& Q,
    const QuasiDense& Q_next,
    const float*      x,
    const float*      bias,
    float*            yXt,
    int               threads
) {
    const uint32_t m   = Q.m;
    const uint32_t r   = Q.r;
    const int      nth = (threads > 0 ? threads : omp_get_max_threads());
    const bool     use512 = supports_avx512();

    // zero next‑layer buffer once
    std::memset(yXt, 0, size_t(Q_next.m) * Q_next.r * sizeof(float));

    #pragma omp parallel for num_threads(nth) schedule(static)
    for (uint32_t i = 0; i < m; ++i) {
        // compute dot(Q.Wd[i,:], x[idx]) + bias[i]
        const float*    wrow   = Q.Wd.ptr     + size_t(i) * r;
        const uint32_t* idxRow = Q.idx.data() + size_t(i) * r;
        float            acc    = bias ? bias[i] : 0.0f;
        uint32_t         j      = 0;

        if (use512) {
            __m512 accv = _mm512_setzero_ps();
            for (; j + 16 <= r; j += 16) {
                __m512 wv = _mm512_loadu_ps(wrow + j);
                __m512i iv = _mm512_loadu_si512((__m512i*)(idxRow + j));
                __m512 xv = _mm512_i32gather_ps(iv, x, 4);
                accv = _mm512_fmadd_ps(wv, xv, accv);
            }
            uint32_t rem = r - j;
            if (rem) {
                __mmask16 msk = ((__mmask16)1 << rem) - 1;
                __m512 wv = _mm512_maskz_loadu_ps(msk, wrow + j);
                __m512i iv = _mm512_maskz_loadu_epi32(msk, (idxRow + j));
                __m512 xv = _mm512_i32gather_ps(iv, x, 4);
                accv = _mm512_fmadd_ps(wv, xv, accv);
            }
            // horizontal reduce
            __m256 lo256 = _mm512_castps512_ps256(accv);
            __m256 hi256 = _mm512_extractf32x8_ps(accv, 1);
            __m256 sum256 = _mm256_add_ps(lo256, hi256);
            __m128 lo128 = _mm256_castps256_ps128(sum256);
            __m128 hi128 = _mm256_extractf128_ps(sum256, 1);
            __m128 sum128 = _mm_add_ps(lo128, hi128);
            sum128 = _mm_hadd_ps(sum128, sum128);
            sum128 = _mm_hadd_ps(sum128, sum128);
            acc += _mm_cvtss_f32(sum128);
        } else {
            __m256 accv = _mm256_setzero_ps();
            for (; j + 8 <= r; j += 8) {
                __m256 wv = _mm256_loadu_ps(wrow + j);
                __m256i iv = _mm256_loadu_si256((__m256i*)(idxRow + j));
                __m256 xv = _mm256_i32gather_ps(x, iv, 4);
                accv = _mm256_fmadd_ps(wv, xv, accv);
            }
            __m128 lo = _mm256_castps256_ps128(accv);
            __m128 hi = _mm256_extractf128_ps(accv, 1);
            __m128 sum = _mm_add_ps(lo, hi);
            sum = _mm_hadd_ps(sum, sum);
            sum = _mm_hadd_ps(sum, sum);
            acc += _mm_cvtss_f32(sum);
        }

        // scatter acc into Q_next's Xt buffer
        uint32_t start = Q_next.rev_off[i];
        uint32_t end   = Q_next.rev_off[i + 1];
        for (uint32_t p = start; p < end; ++p) {
            yXt[ Q_next.rev_pos[p] ] = acc;
        }
    }
}




void quasi_dense_matvec_hidden_mt(
    const QuasiDense& Q,
    const QuasiDense& Q_next,
    const XtDense&    X,
    const float*      bias,
    float*            yXt,
    int               threads
) {
    const uint32_t m    = Q.m;
    const uint32_t r    = Q.r;
    const int      nth  = (threads>0 ? threads : omp_get_max_threads());
    const bool     use512 = supports_avx512();

    // zero next-layer buffer
    std::memset(yXt, 0, size_t(Q_next.m) * Q_next.r * sizeof(float));

    #pragma omp parallel for num_threads(nth) schedule(static)
    for (uint32_t i = 0; i < m; ++i) {
        size_t base     = size_t(i) * r;
        const float* wrow   = Q.Wd.ptr   + base;
        const float* xrow   = X.Xt.ptr    + base;
        uint32_t     nnz_i  = Q.nnz[i];
        float        acc    = bias ? bias[i] : 0.0f;
        uint32_t     j      = 0;

        if (use512) {
            // head to 64B align
            uintptr_t addr = (uintptr_t)(wrow);
            size_t mis = addr & 63;
            size_t head = mis ? (64-mis)/sizeof(float) : 0;
            if (head > nnz_i) head = nnz_i;
            for (; j < head; ++j) acc += wrow[j] * xrow[j];

            // vector body
            __m512 accv = _mm512_setzero_ps();
            for (; j + 16 <= nnz_i; j += 16) {
                __m512 wv = _mm512_load_ps(wrow + j);
                __m512 xv = _mm512_load_ps(xrow + j);
                accv = _mm512_fmadd_ps(wv, xv, accv);
            }
            // tail for remaining nnz
            uint32_t rem = nnz_i - j;
            if (rem) {
                __mmask16 msk = ((__mmask16)1 << rem) - 1;
                __m512 wv = _mm512_maskz_loadu_ps(msk, wrow + j);
                __m512 xv = _mm512_maskz_loadu_ps(msk, xrow + j);
                accv = _mm512_fmadd_ps(wv, xv, accv);
            }
            // horizontal reduce
            __m256 lo = _mm512_castps512_ps256(accv);
            __m256 hi = _mm512_extractf32x8_ps(accv,1);
            __m256 sum2 = _mm256_add_ps(lo, hi);
            __m128 lo2 = _mm256_castps256_ps128(sum2);
            __m128 hi2 = _mm256_extractf128_ps(sum2,1);
            __m128 sum1 = _mm_add_ps(lo2, hi2);
            sum1 = _mm_hadd_ps(sum1,sum1);
            sum1 = _mm_hadd_ps(sum1,sum1);
            acc += _mm_cvtss_f32(sum1);
        } else {
            __m256 accv = _mm256_setzero_ps();
            for (; j + 8 <= nnz_i; j += 8) {
                __m256 wv = _mm256_loadu_ps(wrow + j);
                __m256 xv = _mm256_loadu_ps(xrow + j);
                accv = _mm256_fmadd_ps(wv, xv, accv);
            }
            __m128 lo = _mm256_castps256_ps128(accv);
            __m128 hi = _mm256_extractf128_ps(accv,1);
            __m128 sum1 = _mm_add_ps(lo, hi);
            sum1 = _mm_hadd_ps(sum1,sum1);
            sum1 = _mm_hadd_ps(sum1,sum1);
            acc += _mm_cvtss_f32(sum1);
            // scalar tail
            for (; j < nnz_i; ++j) acc += wrow[j] * xrow[j];
        }

        // scatter into next-layer Xt
        uint32_t start = Q_next.rev_off[i];
        uint32_t end   = Q_next.rev_off[i+1];
        for (uint32_t p = start; p < end; ++p) {
            yXt[ Q_next.rev_pos[p] ] = acc;
        }
    }
}
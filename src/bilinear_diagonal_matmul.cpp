#include "bilinear_diagonal_matvec.hpp"
#include <cstring>  // memcpy, memset
#include <immintrin.h>
#include <omp.h>
#include <cstdlib>

static inline bool supports_avx512() {
#if defined(__GNUC__)
    return __builtin_cpu_supports("avx512f");
#else
    return false;
#endif
}

void quasi_dense_matmul(
    const QuasiDense& Q,
    const float*      X,      // [K × C], row-major
    uint32_t          C,
    const float*      bias,   // [M]
    float*            Y      // [M × C], row-major
) {
    const uint32_t m   = Q.m;
    const uint32_t r   = Q.r;
    const char* env = std::getenv("OMP_NUM_THREADS");
    int nth = env ? std::atoi(env) : omp_get_max_threads();
    const bool     use_avx512 = supports_avx512();

    #pragma omp parallel for num_threads(nth) schedule(static)
    for (uint32_t i = 0; i < m; ++i) {
        float* yrow = Y + i * C;
        size_t base = size_t(i) * r;

        // Initialize output row with bias or zeros
        if (bias) {
            for (uint32_t c = 0; c < C; ++c)
                yrow[c] = bias[i];
        } else {
            for (uint32_t c = 0; c < C; ++c)
                yrow[c] = 0.0f;
        }

        // Iterate through non-zeros of sparse row
        for (uint32_t j = 0; j < Q.nnz[i]; ++j) {
            uint32_t k = Q.idx[base + j];         // column in sparse matrix
            float    w = Q.Wd.ptr[base + j];      // weight
            const float* xrow = X + k * C;        // row in dense matrix X

            if (use_avx512) {
                constexpr int VLEN = 16;
                __m512 wvec = _mm512_set1_ps(w);

                uint32_t c = 0;
                for (; c + VLEN <= C; c += VLEN) {
                    __m512 yv = _mm512_loadu_ps(yrow + c);
                    __m512 xv = _mm512_loadu_ps(xrow + c);
                    yv = _mm512_fmadd_ps(wvec, xv, yv);
                    _mm512_storeu_ps(yrow + c, yv);
                }

                // Tail loop
                for (; c < C; ++c)
                    yrow[c] += w * xrow[c];

            } else {
                constexpr int VLEN = 8;
                __m256 wvec = _mm256_set1_ps(w);

                uint32_t c = 0;
                for (; c + VLEN <= C; c += VLEN) {
                    __m256 yv = _mm256_loadu_ps(yrow + c);
                    __m256 xv = _mm256_loadu_ps(xrow + c);
                    yv = _mm256_fmadd_ps(wvec, xv, yv);
                    _mm256_storeu_ps(yrow + c, yv);
                }

                // Tail loop
                for (; c < C; ++c)
                    yrow[c] += w * xrow[c];
            }
        }
    }
}
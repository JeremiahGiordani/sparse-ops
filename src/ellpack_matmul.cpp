#include "ellpack_matvec.hpp"
#include <cstring>  // memcpy, memset
#include <immintrin.h>
#include <omp.h>
#include <cstdlib>
#include <algorithm>  // std::fill, std::max
#include <cstdint>  // uint32_t, uint64_t

static inline bool supports_avx512() {
#if defined(__GNUC__)
    return __builtin_cpu_supports("avx512f");
#else
    return false;
#endif
}

void ellpack_matmul(
    const Ellpack&    E,
    const float*      X,      // [N × C], row-major
    uint32_t          C,
    const float*      bias,   // [M]
    float*            Y       // [M × C], row-major
) {
    const uint32_t m = E.m;
    const uint32_t r = E.r;
    const char* env = std::getenv("OMP_NUM_THREADS");
    int nth       = env ? std::atoi(env) : omp_get_max_threads();
    const bool use_avx512 = supports_avx512();
    const uint32_t simd_width = use_avx512 ? 16u : 8u;

    #pragma omp parallel for num_threads(nth) schedule(static)
    for (uint32_t i = 0; i < m; ++i) {
        float* yrow = Y + size_t(i) * C;
        size_t base = size_t(i) * r;
        uint32_t count = E.nnz[i];

        // 1) init output row
        if (bias) {
            float bi = bias[i];
            for (uint32_t c = 0; c < C; ++c) yrow[c] = bi;
        } else {
            for (uint32_t c = 0; c < C; ++c) yrow[c] = 0.0f;
        }

        // 2) dispatch AVX‑512 vs scalar
        if (use_avx512) {
            for (uint32_t cb = 0; cb < C; cb += simd_width) {
                // how many cols in this vector block?
                uint32_t block_cols = std::min(simd_width, C - cb);
                __mmask16 mask = (__mmask16(1) << block_cols) - 1;

                // load existing y-block
                __m512 yv = _mm512_maskz_loadu_ps(mask, yrow + cb);

                // accumulate each NNZ into the block
                for (uint32_t j = 0; j < count; ++j) {
                    float    wj   = E.Wd.ptr[base + j];
                    uint32_t col  = E.idx [base + j];
                    const float* xblk = X + size_t(col) * C + cb;

                    __m512 wv = _mm512_set1_ps(wj);
                    __m512 xv = _mm512_maskz_loadu_ps(mask, xblk);
                    yv = _mm512_fmadd_ps(wv, xv, yv);
                }

                // store back the updated y-block
                _mm512_mask_storeu_ps(yrow + cb, mask, yv);
            }

        } else {
            // simple scalar fallback
            for (uint32_t j = 0; j < count; ++j) {
                float    wj  = E.Wd.ptr[base + j];
                uint32_t col = E.idx [base + j];
                const float* xrow = X + size_t(col) * C;
                for (uint32_t c = 0; c < C; ++c) {
                    yrow[c] += wj * xrow[c];
                }
            }
        }
    }
}

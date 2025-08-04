#include "utils.hpp"  
#include "ellpack_matvec.hpp"
#include <cstring>  // memcpy, memset
#include <immintrin.h>
#include <omp.h>
#include <cstdlib>
#include <algorithm>  // std::fill, std::max
#include <cstdint>  // uint32_t, uint64_t



template<bool USE_MASK, bool FUSE_RELU>
void ellpack_matmul_fused(
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
                __mmask16 mask = 0xFFFF;
                if constexpr (USE_MASK) {
                    uint32_t block_cols = std::min(simd_width, C - cb);
                    mask = (__mmask16(1) << block_cols) - 1;
                }

                // load existing y-block
                __m512 yv = USE_MASK
                    ? _mm512_maskz_loadu_ps(mask, yrow + cb)
                    : _mm512_loadu_ps(yrow + cb);

                // accumulate each NNZ into the block
                for (uint32_t j = 0; j < count; ++j) {
                    float    wj   = E.Wd.ptr[base + j];
                    uint32_t col  = E.idx [base + j];
                    const float* xblk = X + size_t(col) * C + cb;

                    __m512 wv = _mm512_set1_ps(wj);
                    __m512 xv = USE_MASK
                        ? _mm512_maskz_loadu_ps(mask, xblk)
                        : _mm512_loadu_ps(xblk);
                    yv = _mm512_fmadd_ps(wv, xv, yv);
                }

                if constexpr(FUSE_RELU) {
                    yv = _mm512_max_ps(yv, _mm512_setzero_ps());
                }

                // store back the updated y-block
                if constexpr (USE_MASK) {
                    _mm512_mask_storeu_ps(yrow + cb, mask, yv);
                } else {
                    _mm512_storeu_ps(yrow + cb, yv);
                }
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



template<bool USE_MASK, bool FUSE_RELU>
void ellpack_matmul_fused_outer(
    const Ellpack& ET,      // transposed: ET.m=N, ET.n=M
    const float*   X,       // [B × N] row-major
    uint32_t       B,       // batch size
    const float*   bias,    // length=M or nullptr
    float*         Y        // [M × B] row-major
) {
    const uint32_t N = ET.m;
    const uint32_t M = ET.n;
    const uint32_t r = ET.r;
    bool use_avx512   = supports_avx512();
    uint32_t W        = use_avx512 ? 16 : 8;

    // 1) Init Y from bias (or zero)
    #pragma omp parallel for
    for (uint32_t i = 0; i < M; ++i) {
        float* yrow = Y + size_t(i)*B;
        if (bias) {
            float bi = bias[i];
            for (uint32_t b = 0; b < B; ++b) yrow[b] = bi;
        } else {
            for (uint32_t b = 0; b < B; ++b) yrow[b] = 0.0f;
        }
    }

    // 2) Outer‐product over features j
    for (uint32_t j = 0; j < N; ++j) {
        // pointer to X[:,j]
        const float* xcol = X + size_t(j)*B;
        size_t        base = size_t(j)*r;
        uint32_t      cnt  = ET.nnz[j];

        // vectorize across the batch dimension
        for (uint32_t bb = 0; bb < B; bb += W) {
            uint32_t block = std::min(W, B - bb);
            __mmask16 mask = USE_MASK
                ? (__mmask16(1) << block) - 1
                : (__mmask16)0xFFFF;

            // load X[bb:bb+block, j]
            __m512 xv = use_avx512
                ? (USE_MASK
                    ? _mm512_maskz_loadu_ps(mask, xcol + bb)
                    : _mm512_loadu_ps(xcol + bb))
                : _mm512_loadu_ps(xcol + bb);

            // for each nonzero (j → i)
            for (uint32_t t = 0; t < cnt; ++t) {
                uint32_t i = ET.idx[base + t];
                float    w = ET.Wd.ptr[base + t];

                float*       yrow = Y + size_t(i)*B;
                __m512 yv = use_avx512
                    ? (USE_MASK
                        ? _mm512_maskz_loadu_ps(mask, yrow + bb)
                        : _mm512_loadu_ps(yrow + bb))
                    : _mm512_loadu_ps(yrow + bb);

                __m512 wv = _mm512_set1_ps(w);
                yv = _mm512_fmadd_ps(wv, xv, yv);

                if constexpr (FUSE_RELU) {
                    yv = _mm512_max_ps(yv, _mm512_setzero_ps());
                }

                if (use_avx512) {
                    if (USE_MASK) {
                        _mm512_mask_storeu_ps(yrow + bb, mask, yv);
                    } else {
                        _mm512_storeu_ps(yrow + bb, yv);
                    }
                } else {
                    // scalar fallback write‐back
                    float tmp[16];
                    _mm512_storeu_ps(tmp, yv);
                    for (uint32_t b = 0; b < block; ++b) {
                        yrow[bb + b] = tmp[b];
                    }
                }
            }
        }
    }
}


// Tell the linker to generate code for instantiations:
template void ellpack_matmul_fused<false, false>(
    const Ellpack&, const float*, uint32_t, const float*, float*);
template void ellpack_matmul_fused<false, true>(
    const Ellpack&, const float*, uint32_t, const float*, float*);
template void ellpack_matmul_fused<true, false>(
    const Ellpack&, const float*, uint32_t, const float*, float*);
template void ellpack_matmul_fused<true, true>(
    const Ellpack&, const float*, uint32_t, const float*, float*);

// Tell the linker to generate code for instantiations:
template void ellpack_matmul_fused_outer<false, false>(
    const Ellpack&, const float*, uint32_t, const float*, float*);
template void ellpack_matmul_fused_outer<false, true>(
    const Ellpack&, const float*, uint32_t, const float*, float*);
template void ellpack_matmul_fused_outer<true, false>(
    const Ellpack&, const float*, uint32_t, const float*, float*);
template void ellpack_matmul_fused_outer<true, true>(
    const Ellpack&, const float*, uint32_t, const float*, float*);
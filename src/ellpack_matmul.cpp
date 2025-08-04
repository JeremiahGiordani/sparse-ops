#include "utils.hpp"  
#include "ellpack_matvec.hpp"
#include <cstring>  // memcpy, memset
#include <immintrin.h>
#include <omp.h>
#include <cstdlib>
#include <algorithm>  // std::fill, std::max
#include <cstdint>  // uint32_t, uint64_t
#include <array> 



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

void ellpack_matmul_outer(
    const Ellpack&    E,
    const float*      X,      // [N × C], row-major
    uint32_t          C,
    const float*      bias,   // [M]
    float*            Y       // [M × C], row-major
) {
    const uint32_t m = E.m;
    const uint32_t r = E.r;
    const uint32_t n = E.n;
    const char* env = std::getenv("OMP_NUM_THREADS");
    int nth       = env ? std::atoi(env) : omp_get_max_threads();
    const bool use_avx512 = supports_avx512();
    const uint32_t simd_width = use_avx512 ? 16u : 8u;

    float* X_packed = new float[n * C];

    #pragma omp parallel for
    for (uint32_t col = 0; col < n; ++col) {
        for (uint32_t b = 0; b < C; ++b) {
            X_packed[col * C + b] = X[b * n + col];  // X_input[b][col]
        }
    }

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
                uint32_t block_cols = std::min(simd_width, C - cb);
                mask = (__mmask16(1) << block_cols) - 1;

                // load existing y-block
                __m512 yv = _mm512_maskz_loadu_ps(mask, yrow + cb);

                // accumulate each NNZ into the block
                for (uint32_t j = 0; j < count; ++j) {
                    float    wj   = E.Wd.ptr[base + j];
                    uint32_t col  = E.idx [base + j];
                    const float* xblk = X_packed + size_t(col) * C + cb;

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


void ellpack_matmul_tiled(
    const Ellpack &E,        // this E is *row*→features: E.m = M, E.n = N
    const float   *X,        // [B × N], row‐major
    uint32_t       B,
    const float   *bias,     // [M] or nullptr
    float         *Y         // [M × B], row‐major
) {
    const uint32_t M      = E.m;
    const uint32_t N      = E.n;
    const uint32_t r      = E.r;
    const bool     use512 = supports_avx512();
    const uint32_t simd   = use512 ? 16u : 8u;
    const uint32_t blocks = (B + simd - 1) / simd;

    // build a per‐block index vector for gathers once
    std::array<int32_t,16> idx32;
    for (uint32_t j = 0; j < 16; ++j) {
      idx32[j] = 4 * int32_t(j * N);  
      // byte‐offset = (b0 + j)*N * sizeof(float), we'll add b0 below
    }

    // threading
    const char* env = std::getenv("OMP_NUM_THREADS");
    int         nth = env ? std::atoi(env) : omp_get_max_threads();

    #pragma omp parallel for num_threads(nth) schedule(static)
    for (uint32_t row = 0; row < M; ++row) {
      size_t      base = size_t(row) * r;
      const float bval = bias ? bias[row] : 0.0f;

      for (uint32_t bi = 0; bi < blocks; ++bi) {
        uint32_t b0        = bi * simd;
        uint32_t block_sz  = std::min(simd, B - b0);
        __mmask16 mask     = (__mmask16(1) << block_sz) - 1;

        // start accumulator with bias
        __m512 acc = _mm512_set1_ps(bval);

        // for each nonzero in this output row
        for (uint32_t j = 0; j < E.nnz[row]; ++j) {
            float    w     = E.Wd.ptr[base + j];
            uint32_t feat  = E.idx [base + j];


            // adjust the index vector for this block
            __m512i indices = _mm512_add_epi32(
                _mm512_set1_epi32(int32_t(b0) * int32_t(N*4)),
                _mm512_loadu_si512(idx32.data())
            );
            __m512 xv = _mm512_i32gather_ps(indices, X, 1);
            acc = _mm512_fmadd_ps(_mm512_set1_ps(w), xv, acc);

        }

        // **one** masked store per block
        float* outp = Y + size_t(row) * B + b0;
        _mm512_mask_storeu_ps(outp, mask, acc);
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

// // Tell the linker to generate code for instantiations:
// template void ellpack_matmul_fused_outer<false, false>(
//     const Ellpack&, const float*, uint32_t, const float*, float*);
// template void ellpack_matmul_fused_outer<false, true>(
//     const Ellpack&, const float*, uint32_t, const float*, float*);
// template void ellpack_matmul_fused_outer<true, false>(
//     const Ellpack&, const float*, uint32_t, const float*, float*);
// template void ellpack_matmul_fused_outer<true, true>(
//     const Ellpack&, const float*, uint32_t, const float*, float*);
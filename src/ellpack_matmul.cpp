#include "utils.hpp"  
#include "ellpack_matvec.hpp"
#include <cstring>  // memcpy, memset
#include <immintrin.h>
#include <omp.h>
#include <cstdlib>
#include <algorithm>  // std::fill, std::max
#include <cstdint>  // uint32_t, uint64_t

static void pack_BxN_to_NxB(
    const float *src,      // [B × N] row‐major
    float       *dst,      // [N × B] row‐major (i.e. [N rows][B cols])
    size_t       B,
    size_t       N
) {
    constexpr size_t Tile = 64;  // tune to your L1 size
    for (size_t i0 = 0; i0 < N; i0 += Tile) {
      size_t i_max = std::min(N, i0 + Tile);
      for (size_t j0 = 0; j0 < B; j0 += Tile) {
        size_t j_max = std::min(B, j0 + Tile);
        for (size_t i = i0; i < i_max; ++i) {
          const float *prow = src + i;        // &src[0*N + i]
          float       *drow = dst + i*B;      // &dst[i*B + 0]
          for (size_t j = j0; j < j_max; ++j) {
            drow[j] = prow[j*N];               // src[j*N + i]
          }
        }
      }
    }
}


template<bool USE_MASK, bool FUSE_RELU>
void ellpack_matmul_fused(
    const Ellpack& E,
    const float*   X_ml,   // [B × N]
    uint32_t       B,      // batch size
    const float*   bias,   // [M] or nullptr
    float*         Y       // [M × B]
) {
    const uint32_t M          = E.m;
    const uint32_t N          = E.n;
    const uint32_t r          = E.r;
    const char*    env        = std::getenv("OMP_NUM_THREADS");
    int            nth        = env ? std::atoi(env) : omp_get_max_threads();
    const bool     use_avx512 = supports_avx512();
    const uint32_t simd_width = use_avx512 ? 16u : 8u;

    // 1) Pack the batch-first input [B×N] → [N×B] once
    std::vector<float> pack;
    pack.resize(size_t(N) * B);
    pack_BxN_to_NxB(X_ml, pack.data(), B, N);
    const float* X = pack.data();  // now [N rows][B cols]

    // 2) Parallel over output rows
    #pragma omp parallel for num_threads(nth) schedule(static)
    for (uint32_t i = 0; i < M; ++i) {
        float*  yrow  = Y + size_t(i) * B;
        size_t  base  = size_t(i) * r;
        uint32_t count = E.nnz[i];

        // 2a) Initialize Y row
        if (bias) {
            float bi = bias[i];
            for (uint32_t b = 0; b < B; ++b) {
                yrow[b] = bi;
            }
        } else {
            for (uint32_t b = 0; b < B; ++b) {
                yrow[b] = 0.0f;
            }
        }

        // 2b) SIMD path
        if (use_avx512) {
            for (uint32_t bb = 0; bb < B; bb += simd_width) {
                uint32_t block = std::min(simd_width, B - bb);
                __mmask16 mask = USE_MASK
                    ? (__mmask16(1) << block) - 1
                    : ( (__mmask16)0xFFFF );

                // load existing y-block
                __m512 yv = USE_MASK
                    ? _mm512_maskz_loadu_ps(mask, yrow + bb)
                    : _mm512_loadu_ps(yrow + bb);

                // accumulate each non-zero
                for (uint32_t j = 0; j < count; ++j) {
                    float    wj   = E.Wd.ptr[base + j];
                    uint32_t col  = E.idx[base + j];
                    const float* xblk = X + size_t(col) * B + bb;

                    __m512 wv = _mm512_set1_ps(wj);
                    __m512 xv = USE_MASK
                        ? _mm512_maskz_loadu_ps(mask, xblk)
                        : _mm512_loadu_ps(xblk);
                    yv = _mm512_fmadd_ps(wv, xv, yv);
                }

                if constexpr (FUSE_RELU) {
                    yv = _mm512_max_ps(yv, _mm512_setzero_ps());
                }

                // store back
                if (USE_MASK) {
                    _mm512_mask_storeu_ps(yrow + bb, mask, yv);
                } else {
                    _mm512_storeu_ps(yrow + bb, yv);
                }
            }

        } else {
            // scalar fallback
            for (uint32_t j = 0; j < count; ++j) {
                float    wj  = E.Wd.ptr[base + j];
                uint32_t col = E.idx[base + j];
                for (uint32_t b = 0; b < B; ++b) {
                    yrow[b] += wj * X[b * N + col];
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
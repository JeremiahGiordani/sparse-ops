#include <cbcoo_matmul.hpp>
#include <immintrin.h>
#include <omp.h>

template<bool FUSE_RELU, bool Y_BxM>
void cbcoo_spmm_stream(
    const CBCOO&  E,
    const float*  X,      // [B x N], row-major
    uint32_t      B,
    const float*  bias,   // [M] or nullptr
    float*        Y       // [ (Y_BxM? B×M : M×B) ], row-major
) {
    const uint32_t M = E.m, N = E.n, KB = E.KB, NB = E.NB;

    // Parallelize over batch rows; each thread gets its own y_local
    #pragma omp parallel
    {
        // L1/L2-local accumulator
        AlignedBuffer ybuf(M);
        float* y_local = ybuf.ptr;

        #pragma omp for schedule(static)
        for (uint32_t b = 0; b < B; ++b) {
            // init y_local
            if (bias) {
                // broadcast bias
                for (uint32_t i = 0; i < M; ++i) y_local[i] = bias[i];
            } else {
                // zero
                std::fill_n(y_local, M, 0.0f);
            }

            const float* xrow = X + size_t(b) * N;

            // walk column blocks
            for (uint32_t cb = 0, k0 = 0; cb < NB; ++cb, k0 += KB) {
                const CBCOOBlock& blk = E.blocks[cb];
                const uint32_t k_extent = std::min<uint32_t>(KB, N - k0);

                // contiguous load of small tile of x
                float x_tile[64]; // KB<=64 reasonable
                for (uint32_t kr = 0; kr < k_extent; ++kr) x_tile[kr] = xrow[k0 + kr];

                // for each k_rel, apply outer-product into y_local
                for (uint32_t kr = 0; kr < k_extent; ++kr) {
                    const float xk = x_tile[kr];
                    if (xk == 0.0f) continue; // cheap skip

                    const uint32_t beg = blk.koffs[kr];
                    const uint32_t end = blk.koffs[kr+1];

                    // vectorized chunks of 16 (AVX-512); tail scalar
                    uint32_t p = beg;
                    for (; p + 16 <= end; p += 16) {
                        // load 16 rows and weights
                        __m512i ridx = _mm512_loadu_si512((const __m512i*)&blk.rows[p]);
                        __m512  wv   = _mm512_loadu_ps(&blk.val.ptr[p]);
                        // gather y_local
                        __m512  yv   = _mm512_i32gather_ps(ridx, y_local, 4);
                        // y += w * xk
                        yv = _mm512_fmadd_ps(wv, _mm512_set1_ps(xk), yv);
                        // scatter back
                        _mm512_i32scatter_ps(y_local, ridx, yv, 4);
                    }
                    for (; p < end; ++p) {
                        y_local[ blk.rows[p] ] += blk.val.ptr[p] * xk;
                    }
                }
            }

            if constexpr (FUSE_RELU) {
                for (uint32_t i = 0; i < M; ++i) y_local[i] = std::max(y_local[i], 0.0f);
            }

            // write out y_local → Y
            if constexpr (Y_BxM) {
                float* ydst = Y + size_t(b) * M;
                std::memcpy(ydst, y_local, M * sizeof(float));
            } else {
                // Y as [M x B] row-major: strided column write
                for (uint32_t i = 0; i < M; ++i) Y[size_t(i)*B + b] = y_local[i];
            }
        }
    }
}

template void cbcoo_spmm_stream<false, false>(
    const CBCOO&, const float*, uint32_t, const float*, float*);
template void cbcoo_spmm_stream<false, true>(
    const CBCOO&, const float*, uint32_t, const float*, float*);
template void cbcoo_spmm_stream<true, false>(
    const CBCOO&, const float*, uint32_t, const float*, float*);
template void cbcoo_spmm_stream<true, true>(
    const CBCOO&, const float*, uint32_t, const float*, float*);

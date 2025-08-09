#include "conv_kernels.hpp"
#include "utils.hpp"
#include <immintrin.h>
#include <omp.h>
#include <algorithm>

// Fortran/B-fast offset: (B,C,H,W) with B contiguous
static inline size_t off4_BCHW(uint32_t B, uint32_t C, uint32_t H,
                               uint32_t c, uint32_t h, uint32_t w) {
    return size_t(B) * ( c + size_t(C) * ( h + size_t(H) * w ) );
}

void conv2d_rbm_fmajor_implicit(
    const ConvAttr&  c,
    const float*     src,    // (B,C,H,W)_F
    uint32_t         B,
    float*           dst     // (B,Cout,Hout,Wout)_F
) {

    const bool use_avx512 = supports_avx512();
    const uint32_t V = use_avx512 ? 16u : 8u;
    const __m512 Z = _mm512_setzero_ps();

    const uint32_t C   = c.Cin;
    const uint32_t H   = c.H_in;
    const uint32_t W   = c.W_in;
    const uint32_t Ho  = c.H_out;
    const uint32_t Wo  = c.W_out;
    const uint32_t K   = c.Cin * c.kH * c.kW;

    const char* env = std::getenv("OMP_NUM_THREADS");
    int nth       = env ? std::atoi(env) : omp_get_max_threads();

    // Parallel over spatial and RBM blocks
    #pragma omp parallel for num_threads(nth) schedule(static)
    for (uint32_t ph = 0; ph < Ho; ++ph) {
        for (uint32_t pw = 0; pw < Wo; ++pw) {

            for (const auto& blk : c.rbm.blocks) {
                const uint32_t Ct = blk.Ct;

                for (uint32_t cb = 0; cb < B; cb += V) {
                    const uint32_t lanes = std::min(V, B - cb);
                    if (use_avx512){
                        const __mmask16 mask = (__mmask16(1) << lanes) - 1;
                        __m512 acc[16]; // Ct<=16 for Ct_max=8/16
                        // seed accumulators with bias (if any)
                        for (uint32_t r = 0; r < Ct; ++r) {
                            float bi = (c.bias_ptr ? c.bias_ptr[blk.M0 + r] : 0.0f);
                            __m512 b = _mm512_set1_ps(bi);
                            acc[r] = _mm512_maskz_mov_ps(mask, b);   // zero lanes outside 'mask'
                        }

                        // Iterate union columns once
                        for (uint32_t u = 0; u < blk.krel.size(); ++u) {
                            const uint32_t k = blk.krel[u];        // 0..K-1
                            const auto& km = c.kmap[k];            // {cin, dh, dw}
                            const int ih = int(ph) * int(c.stride_h) + km.dh;
                            const int iw = int(pw) * int(c.stride_w) + km.dw;
                            if (!( (unsigned)ih < (unsigned)H && (unsigned)iw < (unsigned)W )) {
                                continue; // out-of-bounds pad
                            }
                            const size_t xoff = off4_BCHW(B, C, H, km.cin, (uint32_t)ih, (uint32_t)iw);
                            const float* xvec = src + xoff + cb;
                            const __m512 xv = _mm512_mask_loadu_ps(Z, mask, xvec);

                            const uint32_t beg = blk.colptr[u], end = blk.colptr[u+1];
                            for (uint32_t t = beg; t < end; ++t) {
                                const uint16_t r = blk.pairs[t].row;
                                const float    w = blk.pairs[t].w;
                                acc[r] = _mm512_fmadd_ps(_mm512_set1_ps(w), xv, acc[r]);
                            }
                        }

                        // Optional fused ReLU
                        if (c.fuse_relu) {
                            for (uint32_t r = 0; r < Ct; ++r) {
                                acc[r] = _mm512_max_ps(acc[r], Z);
                            }
                        }

                        // Store (B,Cout,Ho,Wo)_F
                        for (uint32_t r = 0; r < Ct; ++r) {
                            float* ydst = dst + off4_BCHW(B, c.Cout, Ho, blk.M0 + r, ph, pw) + cb;
                            _mm512_mask_storeu_ps(ydst, mask, acc[r]);
                        }
                    }
                    else{
                        // TODO: AVX2/scalar fallback if needed
                        (void)lanes; (void)blk; (void)ph; (void)pw;
                    }
                    
                } // batch tiles
            } // blocks
        } // pw
    } // ph
}

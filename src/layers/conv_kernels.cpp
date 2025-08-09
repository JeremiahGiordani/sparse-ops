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

static inline uint32_t choose_patch_tile(uint32_t K, uint32_t B, uint32_t P) {
    // Target ~64 KiB working set for X_tile = K * (B*P_t) * 4 bytes
    size_t target = 512 * 1024;
    size_t Pt = target / (size_t(K) * B * sizeof(float));
    Pt = std::clamp<size_t>(Pt - Pt%8, 16, P);
    return (uint32_t)Pt;
}

static inline size_t offNCHW(uint32_t C, uint32_t H, uint32_t W,
                             uint32_t c, uint32_t h, uint32_t w) {
  return ( size_t(c) * H + h ) * W + w;
}

static inline uint32_t choose_tile(uint32_t K, uint32_t P) {
  // Aim ~256–512 KiB X_tile: K * Pt * 4 bytes
  size_t target = 512 * 1024;
  size_t Pt = target / (std::max<size_t>(1, size_t(K)) * sizeof(float));
  Pt = std::min<size_t>(Pt, P);
  if (Pt < 64) Pt = std::min<size_t>(P, 64);
  Pt -= (Pt % 8);
  if (!Pt) Pt = std::min<size_t>(P, 8);
  return (uint32_t)Pt;
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

    // Parallel over spatial and RBM blocks
    #pragma omp parallel for collapse(2) schedule(static)
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


void conv2d_tiled_im2col_fmajor(
    const ConvAttr& c,
    const float*    src,   // (B,C,H,W)_F
    uint32_t        B,
    float*          dst    // (B,Cout,Ho,Wo)_F
) {
    const uint32_t C   = c.Cin;
    const uint32_t H   = c.H_in;
    const uint32_t W   = c.W_in;
    const uint32_t Ho  = c.H_out;
    const uint32_t Wo  = c.W_out;
    const uint32_t M   = c.Cout;
    const uint32_t K   = c.Cin * c.kH * c.kW;
    const uint32_t P   = Ho * Wo;

    // Choose tile
    const uint32_t Pt = choose_patch_tile(K, B, P);
    const uint32_t simd_w = supports_avx512() ? 16u : 8u;
    // We'll always call the masked variant of ELLPACK; it's robust for any C_tile.
    const bool fuse_relu = c.fuse_relu;

    // Per-thread scratch
    #pragma omp parallel
    {
        AlignedBuffer X_tile(size_t(K) * size_t(Pt) * B);  // K x C_tile
        AlignedBuffer Y_tile(size_t(M) * size_t(Pt) * B);  // M x C_tile

        #pragma omp for schedule(static)
        for (uint32_t p0 = 0; p0 < P; p0 += Pt) {
            const uint32_t plen = std::min(Pt, P - p0);
            const uint32_t C_tile = plen * B;

            // 1) Pack X_tile: K rows, each has C_tile contiguous columns
            for (uint32_t k = 0; k < K; ++k) {
                float* xrow = X_tile.ptr + size_t(k) * C_tile;
                // walk patches in this tile
                for (uint32_t t = 0; t < plen; ++t) {
                    const uint32_t p = p0 + t;
                    const size_t off = c.patch_indices[size_t(p) * K + k];
                    float* dstp = xrow + size_t(t) * B;
                    if (off == c.sentinel_off) {
                        std::memset(dstp, 0, B * sizeof(float));
                    } else {
                        const float* srcp = src + off * B;
                        uint32_t rem = B;
                        float* d = dstp;
                        const float* s = srcp;
                        while (rem >= 16) {
                            _mm512_storeu_ps(d, _mm512_loadu_ps(s));
                            d += 16; s += 16; rem -= 16;
                        }
                        if (rem) {
                            __mmask16 mask = (__mmask16(1) << rem) - 1;
                            _mm512_mask_storeu_ps(d, mask, _mm512_maskz_loadu_ps(mask, s));
                        }
                    }
                }
            }

            // 2) GEMM: (M x K) * (K x C_tile) -> (M x C_tile)
            if (fuse_relu) {
                ellpack_matmul_fused<true, true>(c.E, X_tile.ptr, C_tile, c.bias_ptr, Y_tile.ptr);
            } else {
                ellpack_matmul_fused<true, false>(c.E, X_tile.ptr, C_tile, c.bias_ptr, Y_tile.ptr);
            }

            // 3) Scatter Y_tile into (B,Cout,Ho,Wo)_F; copy B-contiguous chunks
            for (uint32_t m = 0; m < M; ++m) {
                const float* yrow = Y_tile.ptr + size_t(m) * C_tile;
                for (uint32_t t = 0; t < plen; ++t) {
                    const uint32_t p = p0 + t;
                    const uint32_t ph = p / Wo;
                    const uint32_t pw = p % Wo;
                    float* ydst = dst + off4_BCHW(B, M, Ho, m, ph, pw);
                    // copy B contiguous values from yrow[(t*B) .. (t*B+B)]
                    std::memcpy(ydst, yrow + size_t(t) * B, B * sizeof(float));
                }
            }
        } // tiles
    } // omp parallel
}


void conv2d_tiled_im2col_cmajor(
    const ConvAttr& c,
    const float*    src,   // (B,C,H,W) C-order
    uint32_t        B,
    float*          dst    // (B,Cout,Ho,Wo) C-order
) {
  const uint32_t C   = c.Cin,   H  = c.H_in,  W  = c.W_in;
  const uint32_t M   = c.Cout,  Ho = c.H_out, Wo = c.W_out;
  const uint32_t K   = c.Cin * c.kH * c.kW;
  const uint32_t P   = Ho * Wo;

  const uint32_t Pt  = choose_tile(K, P);
  const bool fuse_relu = c.fuse_relu;

  #pragma omp parallel
  {
    AlignedBuffer X_tile(size_t(K) * Pt);   // K x Pt
    AlignedBuffer Y_tile(size_t(M) * Pt);   // M x Pt

    #pragma omp for schedule(static)
    for (uint32_t b = 0; b < B; ++b) {
      const float* src_b = src + size_t(b) * C * H * W;
      float*       dst_b = dst + size_t(b) * M * Ho * Wo;

      for (uint32_t p0 = 0; p0 < P; p0 += Pt) {
        const uint32_t plen  = std::min(Pt, P - p0);
        const uint32_t Ctile = plen; // GEMM columns

        // 1) Pack K x plen (each column is one patch)
        for (uint32_t k = 0; k < K; ++k) {
          float* xrow = X_tile.ptr + size_t(k) * plen;
          for (uint32_t t = 0; t < plen; ++t) {
            const uint32_t p = p0 + t;
            const size_t offC = c.patch_indices[size_t(p) * K + k];
            xrow[t] = (offC == c.sentinel_off) ? 0.0f : src_b[offC];
          }
        }

        // 2) GEMM: (M x K) * (K x Ctile) -> (M x Ctile)
        if (fuse_relu) {
          ellpack_matmul_fused<true, true>(c.E, X_tile.ptr, Ctile, c.bias_ptr, Y_tile.ptr);
        } else {
          ellpack_matmul_fused<true, false>(c.E, X_tile.ptr, Ctile, c.bias_ptr, Y_tile.ptr);
        }

        // 3) Store into (C-order) (Cout,Ho,Wo) — contiguous along P dimension
        for (uint32_t m = 0; m < M; ++m) {
          const float* yrow = Y_tile.ptr + size_t(m) * plen;
          float* ydst = dst_b + size_t(m) * Ho * Wo + p0;
          std::memcpy(ydst, yrow, plen * sizeof(float)); // big, contiguous chunk
        }
      } // tiles
    } // batch
  } // omp
}
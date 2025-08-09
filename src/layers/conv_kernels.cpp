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

static inline uint32_t choose_Tw(uint32_t Wo, uint32_t /*B*/, uint32_t /*K*/) {
    // Start with 512 and clamp — good on typical L2 sizes; adjust if needed
    uint32_t Tw = 512u;
    if (Tw > Wo) Tw = Wo;
    if (Tw < 64u) Tw = std::min<uint32_t>(Wo, 64u);
    // keep it multiple of 8 for nicer tails
    Tw -= (Tw % 8u);
    if (Tw == 0u) Tw = std::min<uint32_t>(Wo, 8u);
    return Tw;
}

static inline size_t offNCHW(uint32_t C, uint32_t H, uint32_t W,
                             uint32_t c, uint32_t h, uint32_t w) {
  return ( size_t(c) * H + h ) * W + w;
}

inline void nchw_to_pf(const float* srcNCHW, uint32_t B, uint32_t C, uint32_t H, uint32_t W,
                       float* dstPF) {
    // dstPF shape [C, P], P=B*H*W; columns contiguous
    for (uint32_t c = 0; c < C; ++c) {
        float* row = dstPF + size_t(c) * (size_t(B)*H*W);
        for (uint32_t h = 0; h < H; ++h) {
            for (uint32_t w = 0; w < W; ++w) {
                const float* src_hw = srcNCHW + ( (size_t(c)*H + h) * W + w ) + 0;
                float* dst_col = row + ( (size_t(h)*W + w) * B );
                // copy B values for this (c,h,w) across batch
                std::memcpy(dst_col, src_hw, B * sizeof(float));
            }
        }
        // Note: srcNCHW pointer must be advanced by B per scalar; if your NCHW is standard,
        // the B dimension is the leading dimension; adapt indices if needed.
    }
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
    const uint32_t Th = 1;                      // one output row per tile (simple & fast)
    const uint32_t Tw = std::min<uint32_t>(Wo, 512);

    #pragma omp for schedule(static)
    for (uint32_t b = 0; b < B; ++b) {
    const float* src_b = src + size_t(b) * C * H * W;
    float*       dst_b = dst + size_t(b) * M * Ho * Wo;

    for (uint32_t ph0 = 0; ph0 < Ho; ph0 += Th) {
        const uint32_t rows = std::min<uint32_t>(Th, Ho - ph0);

        for (uint32_t pw0 = 0; pw0 < Wo; pw0 += Tw) {
        const uint32_t cols  = std::min<uint32_t>(Tw, Wo - pw0);
        const uint32_t plen  = rows * cols;             // columns in X_tile/Y_tile for this tile
        const uint32_t Ctile = plen;

        // 1) Pack K x plen (columns enumerate patches in this tile row-major over (ph, pw))
        //    Column index mapping: t = r*cols + c, where r in [0..rows), c in [0..cols)
        for (uint32_t k = 0; k < K; ++k) {
            float* xrow = X_tile.ptr + size_t(k) * plen;

            const auto& km = c.kmap[k]; // {cin, dh, dw}

            for (uint32_t r = 0; r < rows; ++r) {
            const uint32_t ph = ph0 + r;
            const int ih = int(ph) * int(c.stride_h) + km.dh;

            // Entire row out of bounds vertically? fill zeros for the whole 'cols'
            if ((unsigned)ih >= (unsigned)H) {
                std::memset(xrow + size_t(r) * cols, 0, cols * sizeof(float));
                continue;
            }

            // Base input w for the leftmost patch in this tile, for this k
            const int iw_base = int(pw0) * int(c.stride_w) + km.dw;

            if (c.stride_w == 1) {
                // Contiguous case: iw increases by 1 across the tile → one memcpy + zero head/tail
                int left_zeros  = std::max(0, -iw_base);
                int right_zeros = std::max(0, (iw_base + int(cols)) - int(W));
                int mid         = int(cols) - left_zeros - right_zeros;

                float* dst_row = xrow + size_t(r) * cols;

                if (left_zeros)  std::memset(dst_row, 0, size_t(left_zeros) * sizeof(float));
                if (mid > 0) {
                const uint32_t iw_start = uint32_t(iw_base + left_zeros);
                const float* src_row = src_b + ( size_t(km.cin) * H + uint32_t(ih) ) * W + iw_start;
                std::memcpy(dst_row + left_zeros, src_row, size_t(mid) * sizeof(float));
                }
                if (right_zeros) std::memset(dst_row + (cols - right_zeros), 0, size_t(right_zeros) * sizeof(float));
            } else {
                // Strided case (e.g., sW=2): copy elementwise (rare, small layers)
                float* dst_row = xrow + size_t(r) * cols;
                int iw = iw_base;
                for (uint32_t ccol = 0; ccol < cols; ++ccol, ++iw) {
                if ((unsigned)iw < (unsigned)W) {
                    dst_row[ccol] = src_b[( size_t(km.cin) * H + uint32_t(ih) ) * W + uint32_t(iw)];
                } else {
                    dst_row[ccol] = 0.0f;
                }
                }
            }
            } // rows (Th)
        } // k

        // 2) GEMM: (M x K) * (K x Ctile) -> (M x Ctile)
        if (fuse_relu) {
            ellpack_matmul_fused<true, true>(c.E, X_tile.ptr, Ctile, c.bias_ptr, Y_tile.ptr);
        } else {
            ellpack_matmul_fused<true, false>(c.E, X_tile.ptr, Ctile, c.bias_ptr, Y_tile.ptr);
        }

        // 3) Store back: each (m, r) slice is contiguous over 'cols'
        for (uint32_t m = 0; m < M; ++m) {
            const float* yrow = Y_tile.ptr + size_t(m) * plen;
            for (uint32_t r = 0; r < rows; ++r) {
            const uint32_t ph = ph0 + r;
            float* ydst = dst_b + ( size_t(m) * Ho + ph ) * Wo + pw0;
            const float* src_row = yrow + size_t(r) * cols;
            std::memcpy(ydst, src_row, size_t(cols) * sizeof(float));
            }
        }

        } // pw0
    } // ph0
    } // batch
  } // omp
}


void conv2d_patchmajor_tiled(
    const ConvAttr& c,
    const float*    src_pf,   // [Cin, B*H*W], PF columns contiguous
    uint32_t        B,
    float*          dst_pf    // [Cout, B*H_out*W_out], PF
) {
    const uint32_t Cin  = c.Cin;
    const uint32_t Cout = c.Cout;
    const uint32_t H    = c.H_in;
    const uint32_t W    = c.W_in;
    const uint32_t Ho   = c.H_out;
    const uint32_t Wo   = c.W_out;

    const uint32_t kH   = c.kH;
    const uint32_t kW   = c.kW;
    const uint32_t sH   = c.stride_h;
    const uint32_t sW   = c.stride_w;
    const uint32_t dH   = c.dil_h;
    const uint32_t dW   = c.dil_w;
    const uint32_t padH = c.pad_h;
    const uint32_t padW = c.pad_w;

    const uint32_t K    = Cin * kH * kW;             // flattened kernel length
    const uint32_t Pin  = B * H * W;                 // columns in PF input per channel
    const uint32_t Pout = B * Ho * Wo;               // columns in PF output per channel

    const bool fuse_relu = c.fuse_relu;

    // Choose width tile
    const uint32_t Tw    = choose_Tw(Wo, B, K);
    const size_t   maxCtile = size_t(Tw) * B;        // columns per tile (cols * B)

    // Parallel over output tiles (each tile writes a disjoint [cols*B] range)
    #pragma omp parallel
    {
        // Per-thread scratch: X_tile [K × (cols*B)], Y_tile [Cout × (cols*B)]
        AlignedBuffer X_tile(size_t(K)    * maxCtile);
        AlignedBuffer Y_tile(size_t(Cout) * maxCtile);

        // Iterate tiles: output row ph0 (Th=1) and width blocks [pw0..pw0+cols)
        #pragma omp for collapse(2) schedule(static)
        for (uint32_t ph0 = 0; ph0 < Ho; ++ph0) {
            for (uint32_t pw0 = 0; pw0 < Wo; pw0 += Tw) {
                const uint32_t cols   = std::min<uint32_t>(Tw, Wo - pw0);
                const uint32_t C_tile = cols * B;

                // 1) Pack X_tile: K rows, each has C_tile contiguous columns
                //    Each k row corresponds to (cin, kh, kw)
                //    For stride_w==1 we do one big memcpy with head/tail zeros.
                uint32_t k = 0;
                for (uint32_t cin = 0; cin < Cin; ++cin) {
                    // PF row base for this input channel
                    const float* src_row = src_pf + size_t(cin) * Pin;

                    for (uint32_t kh = 0; kh < kH; ++kh) {
                        const int ih = int(ph0) * int(sH) + (int(kh) * int(dH) - int(padH));

                        for (uint32_t kw = 0; kw < kW; ++kw, ++k) {
                            float* xrow = X_tile.ptr + size_t(k) * C_tile;

                            // Vertical OOB → all zeros for this row
                            if ((unsigned)ih >= (unsigned)H) {
                                std::memset(xrow, 0, size_t(C_tile) * sizeof(float));
                                continue;
                            }

                            const int iw_base = int(pw0) * int(sW) + (int(kw) * int(dW) - int(padW));

                            if (sW == 1) {
                                // Contiguous across width: columns are [(ph0, pw0 .. pw0+cols-1), all B]
                                // Head zeros if iw_base < 0, tail zeros if iw_base+cols > W
                                const int left0  = std::max(0, -iw_base);
                                const int right0 = std::max(0, (iw_base + int(cols)) - int(W));
                                const int mid    = int(cols) - left0 - right0;

                                // layout inside xrow: [cols] blocks of size B (each block contiguous)
                                // Zero left head
                                if (left0) {
                                    std::memset(xrow, 0, size_t(left0) * B * sizeof(float));
                                }
                                // Mid memcpy
                                if (mid > 0) {
                                    const uint32_t iw_start = uint32_t(iw_base + left0);
                                    const size_t col0 = ( size_t(ih) * W + iw_start ) * B;
                                    std::memcpy(xrow + size_t(left0) * B,
                                                src_row + col0,
                                                size_t(mid) * B * sizeof(float));
                                }
                                // Zero right tail
                                if (right0) {
                                    std::memset(xrow + size_t(cols - right0) * B,
                                                0, size_t(right0) * B * sizeof(float));
                                }
                            } else {
                                // Strided width: columns spaced; copy per column block of B
                                float* dstp = xrow;
                                int iw = iw_base;
                                for (uint32_t ccol = 0; ccol < cols; ++ccol, ++iw, dstp += B) {
                                    if ((unsigned)iw < (unsigned)W) {
                                        const size_t col0 = ( size_t(ih) * W + uint32_t(iw) ) * B;
                                        std::memcpy(dstp, src_row + col0, B * sizeof(float));
                                    } else {
                                        std::memset(dstp, 0, B * sizeof(float));
                                    }
                                }
                            }
                        } // kw
                    } // kh
                } // cin

                // 2) GEMM via ELLPACK: (Cout × K) * (K × C_tile) = (Cout × C_tile)
                if (fuse_relu) {
                    ellpack_matmul_fused<true, true >(c.E, X_tile.ptr, C_tile, c.bias_ptr, Y_tile.ptr);
                } else {
                    ellpack_matmul_fused<true, false>(c.E, X_tile.ptr, C_tile, c.bias_ptr, Y_tile.ptr);
                }

                // 3) Store to PF output: one big memcpy per output channel (row)
                //    PF row base for output channel m is dst_pf + m * Pout
                const size_t col_out0 = ( size_t(ph0) * Wo + pw0 ) * B; // starting column in PF
                for (uint32_t m = 0; m < Cout; ++m) {
                    float*       dst_row = dst_pf + size_t(m) * Pout + col_out0;
                    const float* yrow    = Y_tile.ptr + size_t(m) * C_tile;
                    std::memcpy(dst_row, yrow, size_t(C_tile) * sizeof(float));
                }
            } // pw0
        } // ph0
    } // omp parallel
}
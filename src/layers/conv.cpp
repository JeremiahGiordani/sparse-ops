#include "sparse_onnx.hpp"
#include "iostream"

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vector>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <stdexcept>
#include <omp.h>
#include <immintrin.h>

RunResult SparseOnnxModel::applyConv(
    const ConvAttr &c,
    const float    *src,  // FORTRAN layout: (Cin*H_in*W_in) × B
    uint32_t        B)
const {

    size_t rows_out   = size_t(c.kernel_dims[0]) * c.H_out * c.W_out;
    size_t total_elems= rows_out * B;
    void*  raw_out    = nullptr;
    posix_memalign(&raw_out, 64, total_elems * sizeof(float));
    float* dst_all    = static_cast<float*>(raw_out);
    std::fill_n(dst_all, total_elems, 0.0f);

    // im2col dims
    size_t patch_size  = size_t(c.kernel_dims[1])
                       * c.kernel_dims[2]
                       * c.kernel_dims[3];      // K
    size_t in_features = size_t(c.kernel_dims[1]) 
                       * c.H_in 
                       * c.W_in;
    size_t num_patches = size_t(c.H_out) * c.W_out; // N

    std::vector<float> col_buf(patch_size * num_patches);

    // Fortran strides for src[B, Cin, H_in, W_in]
    size_t stride_b = 1;
    size_t stride_c = size_t(B) * stride_b;
    size_t stride_h = size_t(c.kernel_dims[1]) * stride_c; // Cin * B
    size_t stride_w = size_t(c.H_in) * stride_h;           // H_in * Cin * B

    for (uint32_t b = 0; b < B; ++b) {
        float* dst_b = dst_all + size_t(b) * rows_out;

        // --- BUILD col_buf IN ROW-MAJOR ORDER (K rows × N cols) ---
        for (size_t p = 0; p < patch_size; ++p) {
            for (size_t n = 0; n < num_patches; ++n) {
                // pick up the offset for (row p, col n):
                size_t idx_fortran = n * patch_size + p;
                size_t off = c.patch_indices[idx_fortran];

                float val = 0.0f;
                if (off != in_features) {
                    // decode C-order off -> (ic, ih, iw)
                    int ic = int(off / (c.H_in * c.W_in));
                    int rem= int(off % (c.H_in * c.W_in));
                    int ih = rem / c.W_in;
                    int iw = rem % c.W_in;
                    // Fortran‐style linear index into src:
                    size_t fin = size_t(b)*stride_b
                               + size_t(ic)*stride_c
                               + size_t(ih)*stride_h
                               + size_t(iw)*stride_w;
                    val = src[fin];
                }

                // place into col_buf row-major:
                col_buf[p * num_patches + n] = val;
            }
        }

        // --- SPARSE GEMM (now with row-major col_buf) ---
        ellpack_matmul(
          c.E,
          col_buf.data(),
          static_cast<uint32_t>(num_patches),
          c.bias_ptr,
          dst_b
        );
    }

    return { dst_all, uint32_t(rows_out), true };
}

std::vector<KMap> build_kmap(uint32_t Cin, uint32_t kH, uint32_t kW,
                                    uint32_t pad_h, uint32_t pad_w,
                                    uint32_t dil_h=1, uint32_t dil_w=1) {
    std::vector<KMap> map;
    map.reserve(size_t(Cin) * kH * kW);
    for (uint32_t c = 0; c < Cin; ++c) {
        for (uint32_t kh = 0; kh < kH; ++kh) {
            for (uint32_t kw = 0; kw < kW; ++kw) {
                KMap m;
                m.cin = c;
                m.dh  = int32_t(-int32_t(pad_h) + int32_t(kh) * int32_t(dil_h));
                m.dw  = int32_t(-int32_t(pad_w) + int32_t(kw) * int32_t(dil_w));
                map.push_back(m);
            }
        }
    }
    return map;
}

EllpackW encode_ellpack_from_weight(const float* W,
                                           uint32_t Cout, uint32_t Cin,
                                           uint32_t kH, uint32_t kW)
{
    const uint32_t K = Cin * kH * kW;

    // pass 1: count nnz per output channel (row)
    std::vector<uint32_t> counts(Cout, 0);
    uint32_t rmax = 0;

    for (uint32_t co = 0; co < Cout; ++co) {
        uint32_t cnt = 0;
        // flatten in the same order as kmap (cin-major, then kh, kw)
        for (uint32_t c = 0; c < Cin; ++c) {
            for (uint32_t kh = 0; kh < kH; ++kh) {
                for (uint32_t kw = 0; kw < kW; ++kw) {
                    // weight layout from PyTorch: [Cout, Cin, kH, kW]
                    size_t off = size_t(co)*Cin*kH*kW + size_t(c)*kH*kW + size_t(kh)*kW + kw;
                    float v = W[off];
                    if (v != 0.0f) ++cnt;
                }
            }
        }
        counts[co] = cnt;
        rmax = std::max(rmax, cnt);
    }

    EllpackW E(Cout, K, rmax);
    E.nnz = counts;
    std::fill_n(E.Wd.ptr, size_t(Cout)*rmax, 0.0f);

    // pass 2: fill
    for (uint32_t co = 0; co < Cout; ++co) {
        size_t base = size_t(co) * rmax;
        uint32_t pos = 0;

        uint32_t k = 0;
        for (uint32_t c = 0; c < Cin; ++c) {
            for (uint32_t kh = 0; kh < kH; ++kh) {
                for (uint32_t kw = 0; kw < kW; ++kw, ++k) {
                    size_t off = size_t(co)*Cin*kH*kW + size_t(c)*kH*kW + size_t(kh)*kW + kw;
                    float v = W[off];
                    if (v != 0.0f) {
                        E.Wd.ptr[base + pos] = v;
                        E.idx    [base + pos] = k; // index into KMap order
                        ++pos;
                    }
                }
            }
        }
        // remaining [pos..rmax) stay zero
    }
    return E;
}



// Performs a single 2D convolution via im2col + ELLPACK sparse GEMM (padding=1, stride=1, no bias).
// weight: pointer to weights of shape (Cout, Cin, kH, kW) in standard C-order
// input:  pointer to input tensor of shape (Cin*H*W, B) in Fortran-style (column-major)
// output: pointer to output tensor of shape (B, Cout, H, W) in C-order
// B: batch size, Cin: input channels, H,W: height & width of input
// Cout: output channels, kH,kW: kernel height & width
void conv2d_implicit_im2col_fmajor(
    const ConvPlan& P,
    const float*    X,
    uint32_t        B,
    uint32_t        H, uint32_t W,
    uint32_t        Hout, uint32_t Wout,
    const float*    bias,   // nullptr if no bias
    float*          Y
) {
    const uint32_t Cin = P.Cin;
    const uint32_t Cout = P.Cout;
    const uint32_t sh = P.stride_h, sw = P.stride_w;
    const auto& E = P.W;
    const auto& kmap = P.kmap;

    const bool avx512 = supports_avx512();
    const uint32_t V = avx512 ? 16u : 8u;

    // Parallel over spatial tiles and output channels
    #pragma omp parallel for collapse(2) schedule(static)
    for (uint32_t ho = 0; ho < Hout; ++ho) {
        for (uint32_t wo = 0; wo < Wout; ++wo) {

            auto y_tile_base = [&](uint32_t co)->float* {
                // Fortran (B,Cout,Hout,Wout): index(b,c,ho,wo) = b + B*(c + Cout*(ho + Hout*wo))
                size_t base = size_t(B) * ( co + size_t(Cout)*( ho + size_t(Hout)*wo ) );
                return Y + base;
            };

            for (uint32_t co = 0; co < Cout; ++co) {
                const size_t row_base = size_t(co) * E.r;
                const uint32_t cnt = E.nnz[co];

                for (uint32_t cb = 0; cb < B; cb += V) {
                    const uint32_t lanes = std::min(V, B - cb);
                    const __mmask16 mask = (__mmask16(1) << lanes) - 1;

                    float* yptr = y_tile_base(co) + cb;

                    if (avx512) {
                        __m512 yv = bias ? _mm512_set1_ps(bias[co]) : _mm512_setzero_ps();

                        for (uint32_t j = 0; j < cnt; ++j) {
                            const float w = E.Wd.ptr[row_base + j];
                            const uint32_t k = E.idx[row_base + j];
                            const auto km = kmap[k];

                            const int32_t hin = int32_t(ho)*int32_t(sh) + km.dh;
                            const int32_t win = int32_t(wo)*int32_t(sw) + km.dw;
                            if ((unsigned)hin >= H || (unsigned)win >= W) continue;

                            // X offset: (B,Cin,H,W)_F → b + B*(cin + Cin*(hin + H*win))
                            const size_t xoff = size_t(B) * ( km.cin + size_t(Cin)*( hin + size_t(H)*win ) );
                            const float* xblk = X + xoff + cb; // contiguous across B

                            __m512 xv = _mm512_mask_loadu_ps(_mm512_setzero_ps(), mask, xblk);
                            yv = _mm512_fmadd_ps(_mm512_set1_ps(w), xv, yv);
                        }

                        _mm512_mask_storeu_ps(yptr, mask, yv);
                    } else {
                        // AVX2/scalar fallback
                        if (bias) {
                            for (uint32_t l = 0; l < lanes; ++l) yptr[l] = bias[co];
                        } else {
                            for (uint32_t l = 0; l < lanes; ++l) yptr[l] = 0.0f;
                        }

                        for (uint32_t j = 0; j < cnt; ++j) {
                            const float w = E.Wd.ptr[row_base + j];
                            const uint32_t k = E.idx[row_base + j];
                            const auto km = kmap[k];

                            const int32_t hin = int32_t(ho)*int32_t(sh) + km.dh;
                            const int32_t win = int32_t(wo)*int32_t(sw) + km.dw;
                            if ((unsigned)hin >= H || (unsigned)win >= W) continue;

                            const size_t xoff = size_t(B) * ( km.cin + size_t(Cin)*( hin + size_t(H)*win ) );
                            const float* xblk = X + xoff + cb;
                            for (uint32_t l = 0; l < lanes; ++l)
                                yptr[l] += w * xblk[l];
                        }
                    } // end AVX512/else
                } // batch tiles
            } // co
        } // wo
    } // ho
}


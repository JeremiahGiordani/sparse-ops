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

template<bool FUSE_RELU>
static void conv2d_implicit_im2col_fmajor(
    const ConvAttr& P,
    const float*    X,
    uint32_t        B,
    float*          Y)
{
    const auto& E = P.E;
    const auto& kmap = P.kmap;

    const uint32_t Cin   = P.Cin;
    const uint32_t Cout  = P.Cout;
    const uint32_t H     = P.H_in;
    const uint32_t W     = P.W_in;
    const uint32_t Hout  = P.H_out;
    const uint32_t Wout  = P.W_out;
    const uint32_t sh    = P.stride_h;
    const uint32_t sw    = P.stride_w;

    const bool avx512 = supports_avx512();
    const uint32_t V  = avx512 ? 16u : 8u;

    // Y Fortran index(b,c,ho,wo) = b + B*(c + Cout*(ho + Hout*wo))
    auto y_tile_base = [&](uint32_t co, uint32_t ho, uint32_t wo) -> float* {
        size_t base = size_t(B) * ( co + size_t(Cout) * ( ho + size_t(Hout) * wo ) );
        return Y + base;
    };

    #pragma omp parallel for collapse(2) schedule(static)
    for (uint32_t ho = 0; ho < Hout; ++ho) {
        for (uint32_t wo = 0; wo < Wout; ++wo) {
            for (uint32_t co = 0; co < Cout; ++co) {
                const size_t row_base = size_t(co) * E.r;
                const uint32_t cnt = E.nnz[co];

                for (uint32_t cb = 0; cb < B; cb += V) {
                    const uint32_t lanes = std::min(V, B - cb);
                    const __mmask16 mask = (__mmask16(1) << lanes) - 1;
                    float* yptr = y_tile_base(co, ho, wo) + cb;

                    if (avx512) {
                        __m512 yv = P.bias_ptr ? _mm512_set1_ps(P.bias_ptr[co]) : _mm512_setzero_ps();

                        for (uint32_t j = 0; j < cnt; ++j) {
                            const float w = E.Wd.ptr[row_base + j];
                            const uint32_t k = E.idx[row_base + j];
                            const KMap km = kmap[k];

                            const int32_t hin = int32_t(ho)*int32_t(sh) + km.dh;
                            const int32_t win = int32_t(wo)*int32_t(sw) + km.dw;
                            if ((unsigned)hin >= H || (unsigned)win >= W) continue;

                            // X Fortran offset: b + B*(cin + Cin*(hin + H*win))
                            const size_t xoff = size_t(B) * ( km.cin + size_t(Cin) * ( hin + size_t(H)*win ) );
                            const float* xblk = X + xoff + cb;

                            __m512 xv = _mm512_mask_loadu_ps(_mm512_setzero_ps(), mask, xblk);
                            yv = _mm512_fmadd_ps(_mm512_set1_ps(w), xv, yv);
                        }
                        
                        if constexpr (FUSE_RELU) yv = _mm512_max_ps(yv, _mm512_setzero_ps());
                        _mm512_mask_storeu_ps(yptr, mask, yv);
                    } else {
                        // AVX2/scalar fallback
                        if (P.bias_ptr) {
                            for (uint32_t l = 0; l < lanes; ++l) yptr[l] = P.bias_ptr[co];
                        } else {
                            for (uint32_t l = 0; l < lanes; ++l) yptr[l] = 0.0f;
                        }

                        for (uint32_t j = 0; j < cnt; ++j) {
                            const float w = E.Wd.ptr[row_base + j];
                            const uint32_t k = E.idx[row_base + j];
                            const KMap km = kmap[k];

                            const int32_t hin = int32_t(ho)*int32_t(sh) + km.dh;
                            const int32_t win = int32_t(wo)*int32_t(sw) + km.dw;
                            if ((unsigned)hin >= H || (unsigned)win >= W) continue;

                            const size_t xoff = size_t(B) * ( km.cin + size_t(Cin) * ( hin + size_t(H)*win ) );
                            const float* xblk = X + xoff + cb;

                            for (uint32_t l = 0; l < lanes; ++l) yptr[l] += w * xblk[l];
                        }
                        if constexpr (FUSE_RELU) {
                            for (uint32_t l = 0; l < lanes; ++l) yptr[l] = std::max(yptr[l], 0.0f);
                        }
                    }
                } // batch tiles
            } // co
        } // wo
    } // ho
}

static float* alloc_aligned(size_t elems) {
    void* p = nullptr;
    size_t bytes = elems * sizeof(float);
    if (posix_memalign(&p, 64, (bytes + 63) & ~size_t(63)) != 0 || !p)
        throw std::bad_alloc();
    return static_cast<float*>(p);
}

RunResult SparseOnnxModel::applyConv(const ConvAttr& c, const float* src, uint32_t B) const {
    const size_t elems = size_t(B) * c.Cout * c.H_out * c.W_out;
    void* raw = nullptr;
    posix_memalign(&raw, 64, elems*sizeof(float));
    float* out = reinterpret_cast<float*>(raw);

    if (c.fuse_relu) conv2d_implicit_im2col_fmajor<true >(c, src, B, out);
    else             conv2d_implicit_im2col_fmajor<false>(c, src, B, out);

    return { out, c.Cout * c.H_out * c.W_out, /*owned=*/true };
}

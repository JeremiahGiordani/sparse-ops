#include "sparse_onnx.hpp"

// src/layers/pool.cpp (or wherever you keep layer impls)
#include "sparse_onnx.hpp"
#include <algorithm>
#include <limits>
#include <immintrin.h>
#include <omp.h>

static inline float* alloc_aligned(size_t elems) {
    void* p=nullptr; size_t bytes = elems*sizeof(float);
    if (posix_memalign(&p, 64, (bytes+63)&~size_t(63))!=0 || !p) throw std::bad_alloc();
    return (float*)p;
}

// Fortran offset helpers (B-fast)
static inline size_t off4_BCHW(uint32_t B, uint32_t C, uint32_t H,
                               uint32_t c, uint32_t h, uint32_t w) {
    // b + B*(c + C*(h + H*w))  -- caller adds +b
    return size_t(B) * ( c + size_t(C) * ( h + size_t(H)*w ) );
}

static inline size_t off2_BC(uint32_t B, uint32_t c) {
    // b + B*c  -- caller adds +b
    return size_t(B) * c;
}

RunResult SparseOnnxModel::applyMaxPool(
    const PoolAttr &p,
    const float    *src,     // (B,C,H,W)_F
    uint32_t        /*features*/, // unused; we have p.C/H/W
    uint32_t        B,
    float*          out_buf
) const {
    const uint32_t C = p.C, H = p.H, W = p.W;
    const uint32_t Ho = p.H_out, Wo = p.W_out;
    const size_t out_elems = size_t(B) * C * Ho * Wo;

    float* dst = out_buf ? out_buf : alloc_aligned(out_elems);
    const bool owned = (out_buf == nullptr);

    const bool avx512 = supports_avx512();
    const uint32_t V = avx512 ? 16u : 8u;
    const float ninf = -std::numeric_limits<float>::infinity();

    // Parallel over (ho,wo) and channel
    #pragma omp parallel for collapse(2) schedule(static)
    for (uint32_t ho = 0; ho < Ho; ++ho) {
        for (uint32_t wo = 0; wo < Wo; ++wo) {
            const int h_start = int(ho)*p.sH - p.padH0;
            const int w_start = int(wo)*p.sW - p.padW0;

            for (uint32_t c = 0; c < C; ++c) {
                float* ybase = dst + off4_BCHW(B, C, Ho, c, ho, wo);

                for (uint32_t cb = 0; cb < B; cb += V) {
                    const uint32_t lanes = std::min(V, B - cb);
                    const __mmask16 mask = (__mmask16(1) << lanes) - 1;

                    if (avx512) {
                        __m512 yv = _mm512_set1_ps(ninf);

                        for (int kh = 0; kh < p.kH; ++kh) {
                            const int hin = h_start + kh;
                            if ((unsigned)hin >= H) continue;
                            for (int kw = 0; kw < p.kW; ++kw) {
                                const int win = w_start + kw;
                                if ((unsigned)win >= W) continue;

                                const float* xblk = src + off4_BCHW(B, C, H, c, (uint32_t)hin, (uint32_t)win) + cb;
                                __m512 xv = _mm512_mask_loadu_ps(_mm512_set1_ps(ninf), mask, xblk);
                                yv = _mm512_max_ps(yv, xv);
                            }
                        }
                        _mm512_mask_storeu_ps(ybase + cb, mask, yv);
                    } else {
                        // scalar/AVX2-ish fallback
                        float ytmp[16];
                        for (uint32_t l = 0; l < lanes; ++l) ytmp[l] = ninf;

                        for (int kh = 0; kh < p.kH; ++kh) {
                            const int hin = h_start + kh;
                            if ((unsigned)hin >= H) continue;
                            for (int kw = 0; kw < p.kW; ++kw) {
                                const int win = w_start + kw;
                                if ((unsigned)win >= W) continue;

                                const float* xblk = src + off4_BCHW(B, C, H, c, (uint32_t)hin, (uint32_t)win) + cb;
                                for (uint32_t l = 0; l < lanes; ++l)
                                    ytmp[l] = std::max(ytmp[l], xblk[l]);
                            }
                        }
                        std::memcpy(ybase + cb, ytmp, lanes*sizeof(float));
                    }
                } // batch tiles
            } // c
        } // wo
    } // ho

    return { dst, C * Ho * Wo, owned };
}

RunResult SparseOnnxModel::applyGlobalAveragePool(
    const PoolAttr &p,
    const float    *src,     // (B,C,H,W)_F
    uint32_t        /*features*/,
    uint32_t        B,
    float*          out_buf
) const {
    const uint32_t C = p.C, H = p.H, W = p.W;
    const size_t out_elems = size_t(B) * C;    // (B,C)_F

    float* dst = out_buf ? out_buf : alloc_aligned(out_elems);
    const bool owned = (out_buf == nullptr);

    const bool avx512 = supports_avx512();
    const uint32_t V = avx512 ? 16u : 8u;
    const float scale = 1.0f / float(H * W);

    #pragma omp parallel for schedule(static)
    for (uint32_t c = 0; c < C; ++c) {
        float* ybase = dst + off2_BC(B, c);

        for (uint32_t cb = 0; cb < B; cb += V) {
            const uint32_t lanes = std::min(V, B - cb);
            const __mmask16 mask = (__mmask16(1) << lanes) - 1;

            if (avx512) {
                __m512 sumv = _mm512_setzero_ps();

                for (uint32_t h = 0; h < H; ++h) {
                    for (uint32_t w = 0; w < W; ++w) {
                        const float* xblk = src + off4_BCHW(B, C, H, c, h, w) + cb;
                        __m512 xv = _mm512_mask_loadu_ps(_mm512_setzero_ps(), mask, xblk);
                        sumv = _mm512_add_ps(sumv, xv);
                    }
                }
                __m512 yv = _mm512_mul_ps(sumv, _mm512_set1_ps(scale));
                _mm512_mask_storeu_ps(ybase + cb, mask, yv);
            } else {
                float sum[16] = {0};
                for (uint32_t h = 0; h < H; ++h) {
                    for (uint32_t w = 0; w < W; ++w) {
                        const float* xblk = src + off4_BCHW(B, C, H, c, h, w) + cb;
                        for (uint32_t l = 0; l < lanes; ++l) sum[l] += xblk[l];
                    }
                }
                for (uint32_t l = 0; l < lanes; ++l) ybase[cb + l] = sum[l] * scale;
            }
        } // batch tiles
    } // c

    return { dst, C, owned };
}


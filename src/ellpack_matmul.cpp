#include "utils.hpp"  
#include "ellpack_matvec.hpp"
#include <cstring>  // memcpy, memset
#include <immintrin.h>
#include <omp.h>
#include <cstdlib>
#include <algorithm>  // std::fill, std::max
#include <cstdint>  // uint32_t, uint64_t

struct ScratchBlock {
    float* buf;
    ScratchBlock(uint32_t KB, uint32_t V) {
        size_t bytes = size_t(KB) * V * sizeof(float);
        void* p = nullptr;
        posix_memalign(&p, 64, bytes);
        buf = reinterpret_cast<float*>(p);
    }
    ~ScratchBlock() { free(buf); }
};

void sorted_ellpack_matmul_microtx(
    const SortedEllpack& E,  // must have E.KB == KB
    const float* X,          // [B x N], row-major
    uint32_t B,
    const float* bias,       // [M] or nullptr
    float* Y                 // [M x B], row-major
) {
    const uint32_t m  = E.m;
    const uint32_t n  = E.n;
    const uint32_t NB = E.NB;
    const uint32_t KB = E.KB;

    const bool use_avx512 = true; // assume AVX-512 fast path; add feature detect if needed
    const uint32_t V = 16;        // batch lanes per tile (fits AVX-512)

    // 0) Initialize Y with bias (once per batch tile)
    for (uint32_t cb = 0; cb < B; cb += V) {
        const uint32_t lanes = std::min(V, B - cb);
        const __mmask16 mask = (__mmask16(1u) << lanes) - 1u;

        #pragma omp parallel for schedule(static)
        for (uint32_t i = 0; i < m; ++i) {
            float* yptr = Y + size_t(i) * B + cb;
            if (bias) {
                __m512 yv = _mm512_set1_ps(bias[i]);
                _mm512_mask_storeu_ps(yptr, mask, yv);
            } else {
                // zero init
                _mm512_mask_storeu_ps(yptr, mask, _mm512_setzero_ps());
            }
        }

        // 1) Iterate column blocks (k0..k0+KB)
        for (uint32_t b = 0, k0 = 0; b < NB; ++b, k0 += KB) {
            const uint32_t k_extent = std::min<uint32_t>(KB, n - k0);

            // 1a) Micro-transpose: X[cb:cb+lanes, k0:k0+k_extent] -> scratch[k_rel][lane]
            ScratchBlock scratch(KB, V); // small (e.g., 16*16*4 = 1KB)
            float* sbuf = scratch.buf;
            // Scalar micro-transpose is surprisingly fast at this size; replace with AVX512 shuffles if needed.
            for (uint32_t lane = 0; lane < lanes; ++lane) {
                const float* xrow = X + size_t(cb + lane) * n + k0;
                for (uint32_t k = 0; k < k_extent; ++k) {
                    sbuf[size_t(k) * V + lane] = xrow[k];
                }
                // pad remaining [k_extent..KB) if last block (zeros)
                for (uint32_t k = k_extent; k < KB; ++k) {
                    sbuf[size_t(k) * V + lane] = 0.0f;
                }
            }
            // If lanes < V, zero the tail lanes to keep masked loads trivial
            for (uint32_t k = 0; k < KB; ++k) {
                for (uint32_t lane = lanes; lane < V; ++lane) {
                    sbuf[size_t(k) * V + lane] = 0.0f;
                }
            }

            // 1b) Accumulate this block into Y. Each thread walks different rows.
            #pragma omp parallel for schedule(static)
            for (uint32_t i = 0; i < m; ++i) {
                const size_t ptr_base = size_t(i) * (size_t(NB) + 1);
                const uint32_t off = E.rowblk_ptr[ptr_base + b];
                const uint32_t end = E.rowblk_ptr[ptr_base + b + 1];
                if (off == end) continue; // this row has no nnz in this block

                float* yptr = Y + size_t(i) * B + cb;
                __m512 yv = _mm512_mask_loadu_ps(_mm512_setzero_ps(), ((__mmask16(1u)<<lanes)-1u), yptr);

                // For all entries in this (row, block) slice: FMA with the already-transposed slab
                for (uint32_t p = off; p < end; ++p) {
                    const float    w  = E.Wd.ptr[p];
                    const uint16_t kr = E.krel[p]; // 0..KB-1
                    const float*   xk = sbuf + size_t(kr) * V; // contiguous V-lane vector
                    __m512 xv = _mm512_loadu_ps(xk);
                    yv = _mm512_fmadd_ps(_mm512_set1_ps(w), xv, yv);
                }

                _mm512_mask_storeu_ps(yptr, ((__mmask16(1u)<<lanes)-1u), yv);
            }
        }
    }
}


void ellpack_matmul_batchmajor(
    const Ellpack& E,
    const float*   X,        // [B x N] row-major
    uint32_t       B,        // batch size
    const float*   bias,     // [M] or nullptr
    float*         Y         // [M x B] row-major
) {
    const uint32_t m = E.m;
    const uint32_t N = E.n;
    const uint32_t r = E.r;

    const bool use_avx512 = supports_avx512();
    const uint32_t simd_width = use_avx512 ? 16u : 8u; // we only implement AVX-512 fast path below

    const char* env = std::getenv("OMP_NUM_THREADS");
    int nth = env ? std::atoi(env) : omp_get_max_threads();

    // AVX-512 lanes vector [0..15]
    const __m512i lanes = _mm512_set_epi32(15,14,13,12,11,10,9,8,7,6,5,4,3,2,1,0);

    #pragma omp parallel for num_threads(nth) schedule(static)
    for (uint32_t i = 0; i < m; ++i) {
        const size_t base = size_t(i) * r;
        const uint32_t count = E.nnz[i];

        // Process Y in tiles across the batch dimension.
        for (uint32_t cb = 0; cb < B; cb += simd_width) {
            // tail handling
            uint32_t lane_cols = std::min(simd_width, B - cb);
            __mmask16 mask = (__mmask16(1) << lane_cols) - 1;

            float* yptr = Y + size_t(i) * B + cb;

            // Initialize y-block
            if (use_avx512) {
                __m512 yv = _mm512_setzero_ps();
                if (bias) {
                    yv = _mm512_set1_ps(bias[i]);
                }

                // Build the per-batch-tile gather base offsets once:
                // idx_cb[lane] = ((cb + lane) * N) * 4 (byte offsets)
                const int strideN4 = int(N) * 4;
                const int base4 = int(cb) * int(N) * 4;
                __m512i idx_cb = _mm512_add_epi32(
                    _mm512_set1_epi32(base4),
                    _mm512_mullo_epi32(lanes, _mm512_set1_epi32(strideN4))
                );

                // Loop over the nnz in this output row
                for (uint32_t j = 0; j < count; ++j) {
                    const float    wj  = E.Wd.ptr[base + j];
                    const uint32_t col = E.idx[base + j];        // 0..N-1
                    const __m512i offs = _mm512_add_epi32(idx_cb, _mm512_set1_epi32(int(col) * 4));

                    // Gather X[cb:cb+lanes, col] with a mask for tail
                    __m512 xv = _mm512_mask_i32gather_ps(_mm512_setzero_ps(), mask, offs, X, 1);
                    yv = _mm512_fmadd_ps(_mm512_set1_ps(wj), xv, yv);
                }


                // Store once per batch tile
                _mm512_mask_storeu_ps(yptr, mask, yv);
            } else {
                // Scalar fallback
                // yptr[c] = bias[i] or 0
                if (bias) {
                    std::fill(yptr, yptr + lane_cols, bias[i]);
                } else {
                    std::fill(yptr, yptr + lane_cols, 0.0f);
                }
                for (uint32_t j = 0; j < count; ++j) {
                    const float    wj  = E.Wd.ptr[base + j];
                    const uint32_t col = E.idx[base + j];
                    // Add wj * X[cb:cb+lane_cols, col]
                    for (uint32_t lane = 0; lane < lane_cols; ++lane) {
                        yptr[lane] += wj * X[size_t(cb + lane) * N + col];
                    }
                }
            }
        }
    }
}


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


// Tell the linker to generate code for instantiations:
template void ellpack_matmul_fused<false, false>(
    const Ellpack&, const float*, uint32_t, const float*, float*);
template void ellpack_matmul_fused<false, true>(
    const Ellpack&, const float*, uint32_t, const float*, float*);
template void ellpack_matmul_fused<true, false>(
    const Ellpack&, const float*, uint32_t, const float*, float*);
template void ellpack_matmul_fused<true, true>(
    const Ellpack&, const float*, uint32_t, const float*, float*);
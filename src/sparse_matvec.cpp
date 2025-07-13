#include "bcoo16_encoder.hpp"
#include <immintrin.h>
#include <cstring>

// ──────────────────────────────────────────────────────────────
// 1.  Extracted inner kernel (now reusable)
inline void KernelBody(const BCOO16Block* blk,
                       size_t            nBlk,
                       const float*      x,
                       float*            y)
{
    const __m512i idxOff = _mm512_set_epi32(
        15,14,13,12,11,10,9,8,7,6,5,4,3,2,1,0);

    for (size_t i = 0; i < nBlk; ++i) {
        __mmask16 m = blk[i].bitmask;
        if (!m) continue;

        __m512 vals = _mm512_maskz_loadu_ps(m, blk[i].values);
        __m512i idx = _mm512_add_epi32(idxOff,
                          _mm512_set1_epi32(blk[i].first_col));
        __m512 xv   = _mm512_mask_i32gather_ps(_mm512_setzero_ps(), m, idx, x, 4);

        float acc = _mm512_reduce_add_ps(_mm512_mul_ps(vals, xv));
        y[blk[i].row_id] += acc;
    }
}

// ──────────────────────────────────────────────────────────────
// 2.  Original single-thread wrapper (calls KernelBody)
void sparse_matvec_avx512(const BCOO16& A,
                          const float*  x,
                          const float*  b,
                          float*        y,
                          size_t        M)
{
    std::memcpy(y, b, M * sizeof(float));           // copy bias
    KernelBody(A.blocks.data(), A.blocks.size(), x, y);
}

#include "sparse_matvec.hpp"
#include <immintrin.h>
#include <cstddef>
#include <cstdint>
#include <cstring>  // for std::memcpy, std::memset
#include <stdexcept>
#include <iostream>

void sparse_matvec_avx512(const BCOO16& A, const float* x, const float* b, float* y, size_t M) {
#if !defined(__AVX512F__)
    throw std::runtime_error("AVX-512 not supported on this system.");
#else
    // const size_t num_blocks = A.row_id.size();

    // if (A.first_col.size() != num_blocks || A.bitmask.size() != num_blocks ||
    //     A.values.size() != num_blocks * 16) {
    //     throw std::runtime_error("BCOO16 structure is malformed.");
    // }

    // 1. Initialize output y
    if (b) {
        std::memcpy(y, b, M * sizeof(float));
    } else {
        std::memset(y, 0, M * sizeof(float));
    }

    // const float* val_ptr = A.values.data();
    // const uint32_t* row_ids = A.row_id.data();
    // const uint32_t* col_starts = A.first_col.data();
    // const uint16_t* bitmasks = A.bitmask.data();

    const __m512i idx_offset = _mm512_set_epi32(
        15,14,13,12,11,10,9,8,7,6,5,4,3,2,1,0);

    for (const auto& blk : A.blocks) {
        __mmask16 mask = blk.bitmask;
        if (!mask) continue;

        __m512 val_vec = _mm512_maskz_loadu_ps(mask, blk.values);
        __m512i col_vec = _mm512_add_epi32(
                              _mm512_set1_epi32(blk.first_col), idx_offset);
        __m512 x_vec = _mm512_mask_i32gather_ps(
                           _mm512_setzero_ps(), mask, col_vec, x, 4);

        float sum = _mm512_reduce_add_ps(_mm512_mul_ps(val_vec, x_vec));
        y[blk.row_id] += sum;
    }
#endif
}

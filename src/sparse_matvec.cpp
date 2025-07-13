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
    const size_t num_blocks = A.row_id.size();

    if (A.first_col.size() != num_blocks || A.bitmask.size() != num_blocks ||
        A.values.size() != num_blocks * 16) {
        throw std::runtime_error("BCOO16 structure is malformed.");
    }

    // 1. Initialize output y
    if (b) {
        std::memcpy(y, b, M * sizeof(float));
    } else {
        std::memset(y, 0, M * sizeof(float));
    }

    const float* val_ptr = A.values.data();
    const uint32_t* row_ids = A.row_id.data();
    const uint32_t* col_starts = A.first_col.data();
    const uint16_t* bitmasks = A.bitmask.data();

    // 2. Prepare a vector of [0, 1, ..., 15] for column offsets
    const __m512i idx_offset = _mm512_set_epi32(
        15, 14, 13, 12, 11, 10, 9, 8,
         7,  6,  5,  4,  3,  2, 1, 0
    );

    for (size_t i = 0; i < num_blocks; ++i, val_ptr += 16) {
        __mmask16 lane_mask = static_cast<__mmask16>(bitmasks[i]);
        if (!lane_mask) continue;

        uint32_t row = row_ids[i];
        if (row >= M) {
            throw std::runtime_error("Row index out of bounds.");
        }

        uint32_t col_base = col_starts[i];
        __m512i col_idx = _mm512_add_epi32(_mm512_set1_epi32(col_base), idx_offset);

        // 3. Masked load for values
        __m512 val_vec = _mm512_maskz_loadu_ps(lane_mask, val_ptr);

        // 4. Masked gather from input vector
        __m512 x_vec = _mm512_mask_i32gather_ps(
            _mm512_setzero_ps(),  // src
            lane_mask,
            col_idx,
            x,
            sizeof(float)
        );

        // 5. Fused multiply-add: val * x
        __m512 prod = _mm512_mul_ps(val_vec, x_vec);

        // 6. Horizontal sum across all elements
        float block_sum = _mm512_reduce_add_ps(prod);  // requires ICX+; see below for fallback

        // 7. Accumulate result into y
        y[row] += block_sum;
    }
#endif
}

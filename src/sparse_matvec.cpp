#include "sparse_matvec.hpp"
#include <immintrin.h>
#include <cstddef>
#include <cstdint>
#include <vector>
#include <cstring>  // for std::memset
#include <cassert>
#include <stdexcept>
#include <iostream>
#include <bitset>

// Perform y = A @ x + b, where A is in BCOO-16 format
void sparse_matvec_avx512(const BCOO16& A, const float* x, const float* b, float* y, size_t M) {
    std::memset(y, 0, M * sizeof(float));

    const size_t num_blocks = A.row_id.size();
    const float* values = A.values.data();
    const uint32_t* row_ids = A.row_id.data();
    const uint32_t* col_starts = A.first_col.data();
    const uint16_t* bitmasks = A.bitmask.data();

    size_t value_index = 0;

    std::vector<bool> bias_added(M, false); 

    for (size_t i = 0; i < num_blocks; ++i) {
        const uint32_t row = row_ids[i];
        const uint32_t col_base = col_starts[i];
        const uint16_t mask = bitmasks[i];

        if (mask == 0) {
            continue;  // Skip this block entirely
        }

        if (row >= M) {
            throw std::runtime_error("row index out of bounds");
        }

        // Count set bits in mask
        int active_lanes = _mm_popcnt_u32(mask);

        if (value_index + active_lanes > A.values.size()) {
            throw std::runtime_error("Not enough values remaining in value buffer");
        }

        // Load actual values for this block
        alignas(64) float val_block[16] = {0.0f};
        int vi = value_index;
        for (int j = 0; j < 16; ++j) {
            if ((mask >> j) & 1) {
                val_block[j] = values[vi++];
            }
        }

        __mmask16 lane_mask = static_cast<__mmask16>(mask);
        __m512 val_vec = _mm512_load_ps(val_block);

        // Build gather indices
        alignas(64) int32_t indices[16];
        for (int j = 0; j < 16; ++j) {
            indices[j] = static_cast<int32_t>(col_base + j);
        }
        __m512i index_vec = _mm512_load_epi32(indices);

        // Gather from x: masked
        __m512 x_vec = _mm512_mask_i32gather_ps(_mm512_setzero_ps(), lane_mask, index_vec, x, sizeof(float));

        // Multiply and reduce
        __m512 prod = _mm512_mul_ps(val_vec, x_vec);
        float partial[16];
        _mm512_storeu_ps(partial, prod);

        float acc = 0.0f;
        for (int j = 0; j < 16; ++j) {
            if ((mask >> j) & 1) {
                acc += partial[j];
            }
        }

        // Add bias if provided
        if (b != nullptr && !bias_added[row]) {
            acc += b[row];
            bias_added[row] = true;
        }

        y[row] += acc;


        value_index += active_lanes;
    }

}


#include "bcoo16_encoder.hpp"
#include <vector>
#include <cstddef>
#include <algorithm>
#include <cstdint>
#include <cassert>

// --- Encode dense matrix to BCOO-16 ---
BCOO16 encode_to_bcoo16(const std::vector<std::vector<float>>& dense_matrix) {
    BCOO16 bcoo16;

    bcoo16.original_num_rows = dense_matrix.size();
    bcoo16.original_num_cols = dense_matrix.empty() ? 0 : dense_matrix[0].size();

    for (size_t i = 0; i < dense_matrix.size(); ++i) {
        size_t row_len = dense_matrix[i].size();
        for (size_t j = 0; j < row_len; j += 16) {
            uint16_t mask = 0;
            for (int k = 0; k < 16 && (j + k) < row_len; ++k) {
                if (dense_matrix[i][j + k] != 0.0f) {
                    mask |= (1 << k);
                }
            }

            if (mask != 0) {
                bcoo16.row_id.push_back(i);
                bcoo16.first_col.push_back(j);
                bcoo16.bitmask.push_back(mask);

                for (int k = 0; k < 16 && (j + k) < row_len; ++k) {
                    if (mask & (1 << k)) {
                        bcoo16.values.push_back(dense_matrix[i][j + k]);
                    }
                }
            }
        }
    }

    assert(bcoo16.values.size() == bcoo16.row_id.size() * 16);
    return bcoo16;
}


std::vector<std::vector<float>> decode_from_bcoo16(const BCOO16& bcoo16) {
    assert(bcoo16.values.size() == bcoo16.row_id.size() * 16);
    size_t rows = bcoo16.original_num_rows;
    size_t cols = bcoo16.original_num_cols;
    std::vector<std::vector<float>> dense(rows, std::vector<float>(cols, 0.0f));

    size_t value_index = 0;
    for (size_t i = 0; i < bcoo16.row_id.size(); ++i) {
        int row = bcoo16.row_id[i];
        int col_start = bcoo16.first_col[i];
        uint16_t mask = bcoo16.bitmask[i];

        for (int j = 0; j < 16; ++j) {
            if ((mask & (1 << j)) && (col_start + j) < cols) {
                dense[row][col_start + j] = bcoo16.values[value_index++];
            }
        }
    }

    return dense;
}
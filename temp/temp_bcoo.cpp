// bcoo16_encoder.cpp
// Implementation of BCOO-16 encoder and decoder for sparse matrices.

#include "bcoo16_encoder.hpp"
#include <vector>
#include <cstddef>
#include <algorithm>
#include <cstdint>

// Function to encode a dense matrix into BCOO-16 format
BCOO16 encode_to_bcoo16(const std::vector<std::vector<float>>& dense_matrix) {
    BCOO16 bcoo16;
    for (size_t i = 0; i < dense_matrix.size(); ++i) {
        for (size_t j = 0; j < dense_matrix[i].size(); j += 16) {
            uint16_t mask = 0;
            std::vector<float> vals;
            for (int k = 0; k < 16 && j + k < dense_matrix[i].size(); ++k) {
                float val = dense_matrix[i][j + k];
                if (val != 0) {
                    mask |= (1 << k);
                    vals.push_back(val);
                }
            }
            if (mask != 0) {
                bcoo16.row_id.push_back(i);
                bcoo16.first_col.push_back(j);
                bcoo16.bitmask.push_back(mask);
                // Add 16 values, 0 if not set
                for (int k = 0; k < 16; ++k) {
                    if (mask & (1 << k)) {
                        bcoo16.values.push_back(dense_matrix[i][j + k]);
                    } else {
                        bcoo16.values.push_back(0.0f);
                    }
                }
            }
        }
    }
    return bcoo16;
}

// Function to decode a BCOO-16 format back to a dense matrix
std::vector<std::vector<float>> decode_from_bcoo16(const BCOO16& bcoo16) {
    std::vector<std::vector<float>> dense_matrix;
    // Determine the size of the dense matrix
    size_t num_rows = *std::max_element(bcoo16.row_id.begin(), bcoo16.row_id.end()) + 1;
    size_t max_col = 0;
    for (size_t i = 0; i < bcoo16.first_col.size(); ++i) {
        max_col = std::max(max_col, bcoo16.first_col[i] + 15);
    }
    size_t num_cols = max_col + 1;
    dense_matrix.resize(num_rows, std::vector<float>(num_cols, 0.0f));

    // Populate the dense matrix using BCOO-16 data
    for (size_t i = 0; i < bcoo16.row_id.size(); ++i) {
        size_t row = bcoo16.row_id[i];
        size_t start_col = bcoo16.first_col[i];
        uint16_t mask = bcoo16.bitmask[i];
        for (int k = 0; k < 16; ++k) {
            if (mask & (1 << k)) {
            if (start_col + k >= dense_matrix[row].size()) continue;  // prevent out-of-bounds
            dense_matrix[row][start_col + k] = bcoo16.values[i * 16 + k];
        }
    }
    return dense_matrix;
}

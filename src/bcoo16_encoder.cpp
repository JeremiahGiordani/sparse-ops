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
        for (size_t j = 0; j < dense_matrix[i].size(); ++j) {
            if (dense_matrix[i][j] != 0) {
                bcoo16.row_id.push_back(i);
                bcoo16.first_col.push_back(j);
                bcoo16.values.push_back(dense_matrix[i][j]);
                bcoo16.bitmask.push_back(1); // Simplified bitmask for demonstration
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
    size_t num_cols = *std::max_element(bcoo16.first_col.begin(), bcoo16.first_col.end()) + 1;
    dense_matrix.resize(num_rows, std::vector<float>(num_cols, 0.0f));

    // Populate the dense matrix using BCOO-16 data
    for (size_t i = 0; i < bcoo16.row_id.size(); ++i) {
        dense_matrix[bcoo16.row_id[i]][bcoo16.first_col[i]] = bcoo16.values[i];
    }
    return dense_matrix;
}

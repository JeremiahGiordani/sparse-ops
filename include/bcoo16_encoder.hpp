// bcoo16_encoder.hpp
// Header file for BCOO-16 encoder and decoder.

#ifndef BCOO16_ENCODER_HPP
#define BCOO16_ENCODER_HPP

#include <vector>
#include <cstdint>
#include <cstddef>

// Structure to represent BCOO-16 format
struct BCOO16 {
    std::vector<uint32_t> row_id;
    std::vector<uint32_t> first_col;
    std::vector<float> values;
    std::vector<uint16_t> bitmask;

    // Store original matrix shape
    size_t original_num_rows = 0;
    size_t original_num_cols = 0;
};

// Function prototypes
BCOO16 encode_to_bcoo16(const std::vector<std::vector<float>>& dense_matrix);
std::vector<std::vector<float>> decode_from_bcoo16(const BCOO16& bcoo16);

#endif // BCOO16_ENCODER_HPP

#pragma once
#include <vector>
#include <cstddef>
#include <cstdint>

// ─────────────────────────────────────────────────────────────────────────────
// Cache-aligned 64-B block (now defined in the header so it's a complete type)
struct BCOO16Block {        // ← replace old definition
    uint32_t row_id;
    uint16_t first_col;
    uint16_t bitmask;
    uint32_t val_off;       // NEW
};
struct BCOO16 {
    uint32_t original_num_rows, original_num_cols;
    std::vector<BCOO16Block> blocks;
    std::vector<float>       values;    // NEW: contiguous nnz buffer
};


// Encoder / decoder declarations
BCOO16 encode_to_bcoo16(const std::vector<std::vector<float>>& dense);
std::vector<std::vector<float>> decode_from_bcoo16(const BCOO16& bcoo);

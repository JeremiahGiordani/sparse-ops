#pragma once
#include <vector>
#include <cstddef>
#include <cstdint>

// ─────────────────────────────────────────────────────────────────────────────
// Cache-aligned 64-B block (now defined in the header so it's a complete type)
struct alignas(64) BCOO16Block {
    uint32_t row_id;
    uint32_t first_col;
    uint16_t bitmask;
    uint16_t _pad;          // keeps the struct 64-byte aligned
    float    values[16];    // always 16 floats, zeros where mask bit = 0
};

// Container for all blocks + original dims
struct BCOO16 {
    std::size_t original_num_rows{};
    std::size_t original_num_cols{};
    std::vector<BCOO16Block> blocks;   // AoS
};

// Encoder / decoder declarations
BCOO16 encode_to_bcoo16(const std::vector<std::vector<float>>& dense);
std::vector<std::vector<float>> decode_from_bcoo16(const BCOO16& bcoo);

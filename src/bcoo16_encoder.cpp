#include "bcoo16_encoder.hpp"
#include <vector>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <algorithm>   // std::all_of
#include <cassert>

// ─────────────────────────────────────────────────────────────────────────────
// Helper: verify matrix is rectangular
static void ensure_rectangular(const std::vector<std::vector<float>>& mat) {
    if (mat.empty()) return;
    const size_t cols = mat[0].size();
    bool ok = std::all_of(mat.begin(), mat.end(),
                          [cols](const std::vector<float>& row) {
                              return row.size() == cols;
                          });
    if (!ok)
        throw std::runtime_error("encode_to_bcoo16: input matrix is jagged");
}

// ─────────────────────────────────────────────────────────────────────────────
// Encode dense matrix  →  BCOO-16   (padded 16-value blocks)
BCOO16 encode_to_bcoo16(const std::vector<std::vector<float>>& dense_matrix)
{
    ensure_rectangular(dense_matrix);

    BCOO16 out;
    out.original_num_rows = dense_matrix.size();
    out.original_num_cols = dense_matrix.empty() ? 0 : dense_matrix[0].size();

    const size_t rows = out.original_num_rows;
    const size_t cols = out.original_num_cols;

    for (size_t r = 0; r < rows; ++r) {
        const auto& row = dense_matrix[r];

        // Walk across the row in tiles of 16 columns
        for (size_t base_c = 0; base_c < cols; base_c += 16) {

            uint16_t mask = 0;
            float     vals[16] = {0.0f};

            // Build bitmask & padded value array
            for (int lane = 0; lane < 16; ++lane) {
                size_t c = base_c + lane;
                if (c >= cols) break;            // row tail shorter than 16
                float v = row[c];
                vals[lane] = v;
                if (v != 0.0f) mask |= (1u << lane);
            }

            // Skip fully-zero blocks
            if (mask == 0) continue;

            // Append block meta-data
            out.row_id.push_back(static_cast<uint32_t>(r));
            out.first_col.push_back(static_cast<uint32_t>(base_c));
            out.bitmask.push_back(mask);

            // Append the **full 16-value vector** (padded zeros stay)
            out.values.insert(out.values.end(), vals, vals + 16);
        }
    }

    // Structural invariants
    assert(out.row_id.size() == out.first_col.size());
    assert(out.row_id.size() == out.bitmask.size());
    assert(out.values.size() == out.row_id.size() * 16);

    return out;
}

// ─────────────────────────────────────────────────────────────────────────────
// Decode BCOO-16  →  dense matrix  (for testing / correctness)
std::vector<std::vector<float>> decode_from_bcoo16(const BCOO16& bcoo16)
{
    // Invariants must hold (caught earlier in encoder, but double-check)
    if (bcoo16.values.size() != bcoo16.row_id.size() * 16)
        throw std::runtime_error("decode_from_bcoo16: malformed BCOO16");

    const size_t rows = bcoo16.original_num_rows;
    const size_t cols = bcoo16.original_num_cols;
    std::vector<std::vector<float>> dense(rows, std::vector<float>(cols, 0.0f));

    const float* val_ptr = bcoo16.values.data();

    for (size_t blk = 0; blk < bcoo16.row_id.size(); ++blk, val_ptr += 16) {
        uint32_t   r     = bcoo16.row_id[blk];
        uint32_t   c0    = bcoo16.first_col[blk];
        uint16_t   mask  = bcoo16.bitmask[blk];

        for (int lane = 0; lane < 16; ++lane) {
            if (!((mask >> lane) & 1u)) continue;          // skip zero lanes
            size_t c = c0 + static_cast<size_t>(lane);
            if (c < cols && r < rows)
                dense[r][c] = val_ptr[lane];
        }
    }
    return dense;
}

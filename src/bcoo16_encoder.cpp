#include "bcoo16_encoder.hpp"
#include <vector>
#include <cstddef>
#include <cstdint>
#include <algorithm>
#include <stdexcept>
#include <cassert>

namespace {

static void ensure_rectangular(const std::vector<std::vector<float>>& mat)
{
    if (mat.empty()) return;
    const size_t cols = mat[0].size();
    bool ok = std::all_of(mat.begin(), mat.end(),
                          [cols](const std::vector<float>& row) {
                              return row.size() == cols;
                          });
    if (!ok)
        throw std::runtime_error("encode_to_bcoo16: input matrix is jagged");
}

} // unnamed namespace

//──────────────────────────────────────────────────────────────────────────────
// Encode dense → vector<BCOO16Block>
BCOO16 encode_to_bcoo16(const std::vector<std::vector<float>>& dense_matrix)
{
    ensure_rectangular(dense_matrix);

    BCOO16 out;
    out.original_num_rows = dense_matrix.size();
    out.original_num_cols = dense_matrix.empty() ? 0 : dense_matrix[0].size();

    const size_t rows = out.original_num_rows;
    const size_t cols = out.original_num_cols;

    out.blocks.reserve(rows * (cols + 15) / 16);   // heuristic

    for (size_t r = 0; r < rows; ++r) {
        const auto& row = dense_matrix[r];

        for (size_t base_c = 0; base_c < cols; base_c += 16) {

            uint16_t mask = 0;
            float    vals[16] = {0.0f};

            for (int lane = 0; lane < 16; ++lane) {
                size_t c = base_c + lane;
                if (c >= cols) break;      // tail of last tile
                float v = row[c];
                vals[lane] = v;
                if (v != 0.0f) mask |= (1u << lane);
            }

            if (mask == 0) continue;       // skip all-zero block

            BCOO16Block blk{};
            blk.row_id    = static_cast<uint32_t>(r);
            blk.first_col = static_cast<uint32_t>(base_c);
            blk.bitmask   = mask;
            std::copy(vals, vals + 16, blk.values);

            out.blocks.emplace_back(std::move(blk));
        }
    }

    return out;
}

//──────────────────────────────────────────────────────────────────────────────
// Decode vector<BCOO16Block> → dense (for tests / debug)
std::vector<std::vector<float>> decode_from_bcoo16(const BCOO16& bcoo16)
{
    const size_t rows = bcoo16.original_num_rows;
    const size_t cols = bcoo16.original_num_cols;
    std::vector<std::vector<float>> dense(rows, std::vector<float>(cols, 0.0f));

    for (const auto& blk : bcoo16.blocks) {
        uint32_t   r    = blk.row_id;
        uint32_t   c0   = blk.first_col;
        uint16_t   mask = blk.bitmask;

        for (int lane = 0; lane < 16; ++lane) {
            if (!((mask >> lane) & 1u)) continue;
            size_t c = c0 + static_cast<size_t>(lane);
            if (r < rows && c < cols)
                dense[r][c] = blk.values[lane];
        }
    }
    return dense;
}

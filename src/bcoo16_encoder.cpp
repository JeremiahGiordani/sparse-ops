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
BCOO16 encode_to_bcoo16(const std::vector<std::vector<float>>& dense)
{
    BCOO16 C;
    C.original_num_rows = dense.size();
    C.original_num_cols = dense.empty() ? 0 : dense[0].size();

    for (size_t r = 0; r < dense.size(); ++r) {
        const auto& row = dense[r];
        for (size_t c = 0; c < row.size(); c += 16) {

            uint16_t mask = 0;
            for (int k = 0; k < 16 && c + k < row.size(); ++k)
                if (row[c + k] != 0.0f) mask |= (1u << k);

            if (!mask) continue;          // totally empty 16‑tile

            /* ---------- build block header BEFORE pushing values ---------- */
            BCOO16Block blk;
            blk.row_id    = r;
            blk.first_col = c;
            blk.bitmask   = mask;
            blk.val_off   = static_cast<uint32_t>(C.values.size()); // <<<<<

            /* push block header */
            C.blocks.emplace_back(blk);

            /* push its live values in lane order */
            for (int k = 0; k < 16 && c + k < row.size(); ++k)
                if (mask & (1u << k))
                    C.values.push_back(row[c + k]);
        }
    }
    return C;
}


//──────────────────────────────────────────────────────────────────────────────
// Decode vector<BCOO16Block> → dense (for tests / debug)
std::vector<std::vector<float>> decode_from_bcoo16(const BCOO16& b)
{
    std::vector<std::vector<float>> dense(
        b.original_num_rows,
        std::vector<float>(b.original_num_cols, 0.0f));

    for (const auto& blk : b.blocks) {
        const float* v = b.values.data() + blk.val_off;
        int idx = 0;
        for (int lane = 0; lane < 16; ++lane)
            if (blk.bitmask & (1u << lane))
                dense[blk.row_id][blk.first_col+lane] = v[idx++];
    }
    return dense;
}

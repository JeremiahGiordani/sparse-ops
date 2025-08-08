#include "cbcoo_encoder.hpp"
#include <algorithm>
#include <immintrin.h>

CBCOO convert_to_cbcoo(const float* W, uint32_t m, uint32_t n, uint32_t KB) {
    const uint32_t NB = (n + KB - 1) / KB;
    CBCOO E{m, n, KB, NB, {}};
    E.blocks.resize(NB);

    // First pass: counts per (block, k_rel)
    std::vector<uint32_t> counts; counts.reserve(NB * KB);
    counts.assign(NB * KB, 0u);
    size_t total = 0;
    for (uint32_t i = 0; i < m; ++i) {
        const float* row = W + size_t(i) * n;
        for (uint32_t j = 0; j < n; ++j) if (row[j] != 0.0f) {
            uint32_t b = j / KB, kr = j % KB;
            counts[b*KB + kr]++; total++;
        }
    }

    // Allocate per block & prefix-sum koffs
    for (uint32_t b = 0; b < NB; ++b) {
        CBCOOBlock blk;
        blk.koffs.resize(KB + 1);
        uint32_t run = 0;
        for (uint32_t kr = 0; kr < KB; ++kr) {
            blk.koffs[kr] = run;
            run += counts[b*KB + kr];
        }
        blk.koffs[KB] = run;
        blk.rows.resize(run);
        blk.val  = AlignedBuffer(run);
        E.blocks[b] = std::move(blk);
    }

    // Reset write cursors
    std::vector<uint32_t> wptr = counts;
    for (uint32_t b = 0; b < NB; ++b) for (uint32_t kr = 0; kr < KB; ++kr) wptr[b*KB + kr] = 0u;

    // Fill rows/vals grouped by (block, k_rel) (already sorted by j)
    for (uint32_t i = 0; i < m; ++i) {
        const float* row = W + size_t(i) * n;
        for (uint32_t j = 0; j < n; ++j) {
            float v = row[j]; if (!v) continue;
            uint32_t b = j / KB, kr = j % KB;
            CBCOOBlock& blk = E.blocks[b];
            uint32_t pos = blk.koffs[kr] + wptr[b*KB + kr]++;
            blk.rows[pos] = i;
            blk.val.ptr[pos] = v;
        }
    }
    return E;
}
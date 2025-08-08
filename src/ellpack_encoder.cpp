// src/ellpack_encoder.cpp
#include "ellpack_encoder.hpp"
#include <algorithm>
#include <immintrin.h>

Ellpack convert_to_ellpack(const float* W, uint32_t m, uint32_t n) {
    // 1) Count non-zeros per row and track the maximum
    std::vector<uint32_t> rowCounts(m);
    uint32_t r0 = 0;
    for (uint32_t i = 0; i < m; ++i) {
        const float* row = W + size_t(i) * n;
        uint32_t cnt = 0;
        for (uint32_t j = 0; j < n; ++j)
            if (row[j] != 0.0f) ++cnt;
        rowCounts[i] = cnt;
        r0 = std::max(r0, cnt);
    }

    // 2) Construct Ellpack with stride = max nnz
    Ellpack E(m, n, r0);
    // 3) Copy rowCounts into E.nnz
    E.nnz = std::move(rowCounts);

    // 4) Zero‐initialize the packed‐value buffer
    std::fill_n(E.Wd.ptr, size_t(m) * r0, 0.0f);

    // 5) Pack values and their column‐indices
    for (uint32_t i = 0; i < m; ++i) {
        const float* row = W + size_t(i) * n;
        size_t base     = size_t(i) * r0;
        uint32_t pos    = 0;
        for (uint32_t j = 0; j < n; ++j) {
            float v = row[j];
            if (v != 0.0f) {
                E.Wd.ptr   [base + pos] = v;
                E.idx      [base + pos] = j;
                ++pos;
            }
        }
        // any positions [pos..r0) remain zero
    }

    return E;
}

void decode_from_ellpack(const Ellpack& E, float* W_out) {
    // zero‐initialize output
    std::fill(W_out, W_out + size_t(E.m) * E.n, 0.0f);

    // scatter each non‑zero back to its column
    for (uint32_t i = 0; i < E.m; ++i) {
        size_t base_e = size_t(i) * E.r;
        size_t base_w = size_t(i) * E.n;
        for (uint32_t j = 0; j < E.nnz[i]; ++j) {
            float     v = E.Wd.ptr[base_e + j];
            uint32_t  c = E.idx [base_e + j];
            W_out[base_w + c] = v;
        }
    }
}


SortedEllpack convert_to_sorted_ellpack(const float* W, uint32_t m, uint32_t n, uint32_t KB) {
    const uint32_t NB = (n + KB - 1) / KB;

    // 1) Per-(row,block) counts
    std::vector<uint32_t> counts(size_t(m) * NB, 0);
    size_t total_nnz = 0;
    for (uint32_t i = 0; i < m; ++i) {
        const float* row = W + size_t(i) * n;
        for (uint32_t j = 0; j < n; ++j) {
            float v = row[j];
            if (v != 0.0f) {
                uint32_t b = j / KB;
                counts[size_t(i)*NB + b] += 1;
                total_nnz += 1;
            }
        }
    }

    // 2) Build rowblk_ptr (CSR over (row,block))
    SortedEllpack E(m, n, KB, total_nnz);
    // rowblk_ptr[i*(NB+1) + b] is start offset; last entry is end
    uint64_t running = 0;
    for (uint32_t i = 0; i < m; ++i) {
        size_t base = size_t(i) * (NB + 1);
        E.rowblk_ptr[base] = uint32_t(running);
        for (uint32_t b = 0; b < NB; ++b) {
            running += counts[size_t(i)*NB + b];
            E.rowblk_ptr[base + b + 1] = uint32_t(running);
        }
    }

    // 3) Prepare write cursors per (row,block)
    std::vector<uint32_t> write_ptr = E.rowblk_ptr; // copy; will advance as we fill

    // 4) Fill payload, emitting entries grouped by (row, block)
    for (uint32_t i = 0; i < m; ++i) {
        const float* row = W + size_t(i) * n;
        // Optional: keep a small per-block staging of (col, val) to sort by col within block
        // For simplicity, we emit in natural col order which already arrives sorted.
        for (uint32_t j = 0; j < n; ++j) {
            float v = row[j];
            if (v != 0.0f) {
                uint32_t b    = j / KB;
                uint16_t krel = uint16_t(j - b * KB); // j % KB
                uint32_t pos  = write_ptr[size_t(i)*(NB+1) + b]++;
                E.Wd.ptr[pos] = v;
                E.krel[pos]   = krel;
            }
        }
    }

    return E;
}
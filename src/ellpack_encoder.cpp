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
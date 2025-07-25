// src/quasi_dense_encoder.cpp
#include "quasi_dense_encoder.hpp"
#include <algorithm>

QuasiDense convert_to_quasi_dense(const float* W, uint32_t m, uint32_t n) {
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

    // 2) Construct QuasiDense with stride = max nnz
    QuasiDense Q(m, n, r0);
    // 3) Copy rowCounts into Q.nnz
    Q.nnz = std::move(rowCounts);

    // 4) Zero‐initialize the packed‐value buffer
    std::fill_n(Q.Wd.ptr, size_t(m) * r0, 0.0f);

    // 5) Pack values and their column‐indices
    for (uint32_t i = 0; i < m; ++i) {
        const float* row = W + size_t(i) * n;
        size_t base     = size_t(i) * r0;
        uint32_t pos    = 0;
        for (uint32_t j = 0; j < n; ++j) {
            float v = row[j];
            if (v != 0.0f) {
                Q.Wd.ptr   [base + pos] = v;
                Q.idx      [base + pos] = j;
                ++pos;
            }
        }
        // any positions [pos..r0) remain zero
    }
    return Q;
}

void decode_from_quasi_dense(const QuasiDense& Q, float* W_out) {
    // zero‐initialize output
    std::fill(W_out, W_out + size_t(Q.m) * Q.n, 0.0f);

    // scatter each non‑zero back to its column
    for (uint32_t i = 0; i < Q.m; ++i) {
        size_t base_q = size_t(i) * Q.r;
        size_t base_w = size_t(i) * Q.n;
        for (uint32_t j = 0; j < Q.nnz[i]; ++j) {
            float     v = Q.Wd.ptr[base_q + j];
            uint32_t  c = Q.idx [base_q + j];
            W_out[base_w + c] = v;
        }
    }
}

XtDense transform_input(const QuasiDense& Q, const float* x) {
    XtDense X(Q.m, Q.r);
    for (uint32_t i = 0; i < Q.m; ++i) {
        size_t base = size_t(i)*Q.r;
        for (uint32_t j = 0; j < Q.r; ++j) {
            X.Xt.ptr[base+j] = x[ Q.idx[base+j] ];
        }
    }
    return X;
}


void copy_input_to_Xt(const QuasiDense& Q, const float* x) {
    // gather only the true non‑zeros per row, leave the rest untouched
    for (uint32_t i = 0; i < Q.m; ++i) {
        size_t      base   = size_t(i) * Q.r;
        uint32_t    len    = Q.nnz[i];
        const auto* idxRow = Q.idx.data() + base;
        float*      xrow   = Q.Xt.ptr     + base;

        for (uint32_t j = 0; j < len; ++j) {
            xrow[j] = x[ idxRow[j] ];
        }
        // zero‑pad the remainder so your dot sees 0s:
        std::fill(xrow + len, xrow + base + Q.r, 0.0f);
    }
}
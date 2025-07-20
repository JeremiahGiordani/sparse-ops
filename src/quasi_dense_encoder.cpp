// src/quasi_dense_encoder.cpp

#include "quasi_dense_encoder.hpp"
#include <algorithm>  // std::fill, std::max

QuasiDense convert_to_quasi_dense(const float* W, uint32_t m, uint32_t n) {
    // Phase 1: compute non-zero counts per row, find max r
    std::vector<uint32_t> nnz(m);
    uint32_t r = 0;
    for (uint32_t i = 0; i < m; ++i) {
        uint32_t count = 0;
        const float* row = W + size_t(i) * n;
        for (uint32_t j = 0; j < n; ++j) {
            if (row[j] != 0.0f) {
                ++count;
            }
        }
        nnz[i] = count;
        r = std::max(r, count);
    }

    // Allocate QuasiDense
    QuasiDense Q{m, n, r, {}, {}};
    Q.Wd.assign(size_t(m) * r, 0.0f);
    Q.idx.assign(size_t(m) * r, 0);

    // Phase 2: pack values and indices
    for (uint32_t i = 0; i < m; ++i) {
        const float* row = W + size_t(i) * n;
        size_t base = size_t(i) * r;
        uint32_t pos = 0;
        for (uint32_t j = 0; j < n; ++j) {
            float v = row[j];
            if (v != 0.0f) {
                Q.Wd[base + pos] = v;
                Q.idx[base + pos] = j;
                ++pos;
            }
        }
        // remaining slots stay zero/padded
    }

    return Q;
}

void decode_from_quasi_dense(const QuasiDense& Q, float* W_out) {
    // Zero-out output
    std::fill(W_out, W_out + size_t(Q.m) * Q.n, 0.0f);

    // Scatter packed values back to original positions
    for (uint32_t i = 0; i < Q.m; ++i) {
        size_t base_q = size_t(i) * Q.r;
        size_t base_w = size_t(i) * Q.n;
        for (uint32_t j = 0; j < Q.r; ++j) {
            float v = Q.Wd[base_q + j];
            if (v != 0.0f) {
                uint32_t col = Q.idx[base_q + j];
                W_out[base_w + col] = v;
            }
        }
    }
}

XtDense transform_input(const QuasiDense& Q, const float* x) {
    XtDense X{Q.m, Q.r, {}};
    X.Xt.assign(size_t(Q.m) * Q.r, 0.0f);
    for (uint32_t i = 0; i < Q.m; ++i) {
        size_t base = size_t(i) * Q.r;
        for (uint32_t j = 0; j < Q.r; ++j) {
            uint32_t col = Q.idx[base + j];
            X.Xt[base + j] = x[col];
        }
    }
    return X;
}

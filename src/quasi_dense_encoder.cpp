// src/quasi_dense_encoder.cpp
#include "quasi_dense_encoder.hpp"
#include <algorithm>

QuasiDense convert_to_quasi_dense(const float* W, uint32_t m, uint32_t n) {
    // compute r
    std::vector<uint32_t> nnz(m);
    uint32_t r = 0;
    for (uint32_t i = 0; i < m; ++i) {
        uint32_t c = 0;
        const float* row = W + size_t(i)*n;
        for (uint32_t j = 0; j < n; ++j) if (row[j]!=0.0f) ++c;
        nnz[i]=c; r = std::max(r,c);
    }
    
    QuasiDense Q(m,n,r);
    
    std::fill_n(Q.Wd.ptr, size_t(m) * r, 0.0f);
    // pack values & indices
    for (uint32_t i = 0; i < m; ++i) {
        const float* row = W + size_t(i)*n;
        size_t base = size_t(i)*r;
        uint32_t pos = 0;
        for (uint32_t j = 0; j < n; ++j) {
            float v = row[j];
            if (v!=0.0f) {
                Q.Wd.ptr[base+pos] = v;
                Q.idx  [base+pos] = j;
                ++pos;
            }
        }
    }
    return Q;
}

void decode_from_quasi_dense(const QuasiDense& Q, float* W_out) {
    std::fill(W_out, W_out + size_t(Q.m)*Q.n, 0.0f);
    
    for (uint32_t i = 0; i < Q.m; ++i) {
        size_t base_q = size_t(i)*Q.r;
        size_t base_w = size_t(i)*Q.n;
        for (uint32_t j = 0; j < Q.r; ++j) {
            float v = Q.Wd.ptr[base_q+j];
            if (v!=0.0f) {
                W_out[base_w + Q.idx[base_q+j]] = v;
            }
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

// src/bilinear_diagonal_matvec.cpp
#include "bilinear_diagonal_matvec.hpp"
#include <cstring>     // std::memcpy
#include <algorithm>   // std::fill
#include <omp.h>

void quasi_dense_matvec_mt(
    const QuasiDense& Q,
    const XtDense&   X,
    const float*     bias,
    float*           y,
    int              threads
) {
    const uint32_t m = Q.m;
    const uint32_t r = Q.r;

    // Initialize output y: copy bias or zero
    if (bias) {
        std::memcpy(y, bias, size_t(m) * sizeof(float));
    } else {
        std::fill(y, y + size_t(m), 0.0f);
    }

    // Determine thread count
    const int num_threads = (threads > 0 ? threads : omp_get_max_threads());

    // Row‑wise dot of packed Wd and Xt
    #pragma omp parallel for num_threads(num_threads)
    for (uint32_t i = 0; i < m; ++i) {
        const float* wrow = Q.Wd.data()  + size_t(i) * r;
        const float* xrow = X.Xt.data()  + size_t(i) * r;
        float acc = 0.0f;
        for (uint32_t j = 0; j < r; ++j) {
            acc += wrow[j] * xrow[j];
        }
        y[i] += acc;
    }
}
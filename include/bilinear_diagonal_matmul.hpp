// include/bilinear_diagonal_matvec.hpp
#pragma once
#include <cstdint>
#include "quasi_dense_encoder.hpp"

/// quasi_dense_matmul_mt: Y = Q × X  + bias
/// - Q: m×n quasi‑dense
/// - X: n×C input matrix (col‑major or row‑major, see notes)
/// - C: number of columns in X
/// - bias: length‑m vector to add to each output column (nullptr ⇒ zero init)
/// - Y: output buffer, size m×C (row‑major, column j starts at Y + j*m)
/// - threads: OpenMP threads
void quasi_dense_matmul_mt(
    const QuasiDense& Q,
    const float*      X,
    uint32_t          C,
    const float*      bias,
    float*            Y,
    int               threads = 0
);

// include/ellpack_matvec.hpp
#pragma once
#include <cstdint>
#include "ellpack_encoder.hpp"

/// ellpack_matmul: Y = E × X  + bias
/// - E: m×n ELLPACK representation
/// - X: n×C input matrix (col‑major or row‑major, see notes)
/// - C: number of columns in X
/// - bias: length‑m vector to add to each output column (nullptr ⇒ zero init)
/// - Y: output buffer, size m×C (row‑major, column j starts at Y + j*m)
/// - threads: OpenMP threads
void ellpack_matmul(
    const Ellpack&    E,
    const float*      X,
    uint32_t          C,
    const float*      bias,
    float*            Y
);

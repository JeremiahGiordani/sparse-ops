// include/bilinear_diagonal_matvec.hpp
#pragma once

#include <cstdint>
#include "quasi_dense_encoder.hpp"

/// Perform quasi‑dense mat‑vec: y_i = bias_i + dot( Q.Wd[i, :], X.Xt[i, :] )
/// - Q: quasi‑dense representation (m×r)
/// - X: transformed input (m×r)
/// - bias: length m (if nullptr, y is zero‑initialized)
/// - y: output buffer length m
/// - threads: number of OpenMP threads (0 = automatic)
void quasi_dense_matvec_mt(
    const QuasiDense& Q,
    const XtDense&   X,
    const float*     bias,
    float*           y,
    int              threads = 0
);
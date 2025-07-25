// include/bilinear_diagonal_matvec.hpp
#pragma once

#include <cstdint>
#include "quasi_dense_encoder.hpp"

/// Perform quasi‑dense mat‑vec: y_i = bias_i + dot( Q.Wd[i, :], X.Xt[i, :] )
/// - Q: quasi‑dense representation (m×r)
/// - x: Input tensor
/// - bias: length m (if nullptr, y is zero‑initialized)
/// - y: output buffer length m
/// - threads: number of OpenMP threads (0 = automatic)
void quasi_dense_matvec_mt(
    const QuasiDense& Q,
    const float*      x,
    const float*      bias,
    float*            y,
    int               threads = 0
);

/// Perform quasi‑dense mat‑vec: y_i = bias_i + dot( Q.Wd[i, :], X.Xt[i, :] )
/// - Q: quasi‑dense representation (m×r)
/// - bias: length m (if nullptr, y is zero‑initialized)
/// - y: output buffer length m
/// - threads: number of OpenMP threads (0 = automatic)
void quasi_dense_matvec_mt(
    const QuasiDense& Q,
    const float*     bias,
    float*           y,
    int              threads = 0
);


/// Perform quasi‑dense mat‑vec on the fly: y_i = bias_i + dot( Q.Wd[i, :], x )
/// - Q: quasi‑dense representation (m×r)
/// - x: input vector (length n)
/// - bias: length m (if nullptr, y is zero‑initialized)
/// - y: output buffer length m
/// - threads: number of OpenMP threads (0 = automatic) 
void quasi_dense_matvec_gather(
    const QuasiDense& Q,
    const float*      x,     // length n
    const float*      bias,  // length m or nullptr
    float*            y,     // length m
    int               threads = 0
);


/// Hidden‑layer fused quasi‑dense mat‑vec:
/// - Q: quasi‑dense representation (m×r)
/// - Q_next: next layer's quasi‑dense representation (m×r)
/// - x: Input tensor
/// - bias: length m (if nullptr, y is zero‑initialized)
/// - yXt: output buffer length Q_next.m * Q_next.r
/// - threads: number of OpenMP threads (0 = automatic)
/// - Scatters each y[i] into yXt at positions specified by Q_next.rev_off/pos
/// - yXt must point to an array of length Q_next.m * Q_next.r
void quasi_dense_matvec_hidden_mt(
    const QuasiDense& Q,
    const QuasiDense& Q_next,
    const float*      x,
    const float*      bias,
    int               threads = 0
);


/// Hidden‑layer fused quasi‑dense mat‑vec:
/// - Q: quasi‑dense representation (m×r)
/// - Q_next: next layer's quasi‑dense representation (m×r)
/// - bias: length m (if nullptr, y is zero‑initialized)
/// - yXt: output buffer length Q_next.m * Q_next.r
/// - threads: number of OpenMP threads (0 = automatic)
/// - Scatters each y[i] into yXt at positions specified by Q_next.rev_off/pos
/// - yXt must point to an array of length Q_next.m * Q_next.r
void quasi_dense_matvec_hidden_mt(
    const QuasiDense& Q,
    const QuasiDense& Q_next,
    const float*      bias,
    int               threads
);
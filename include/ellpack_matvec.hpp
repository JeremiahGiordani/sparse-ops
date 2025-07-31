// include/bilinear_diagonal_matvec.hpp
#pragma once

#include <cstdint>
#include "ellpack_encoder.hpp"

/// Perform Ellpack mat‑vec: y_i = bias_i + dot( E.Wd[i, :], X.Xt[i, :] )
/// - E: Ellpack representation (m×r)
/// - x: Input tensor
/// - bias: length m (if nullptr, y is zero‑initialized)
/// - y: output buffer length m
void ellpack_matvec(
    const Ellpack&    E,
    const float*      x,
    const float*      bias,
    float*            y
);

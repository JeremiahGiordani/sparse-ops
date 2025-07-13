#pragma once

#include "bcoo16_encoder.hpp"
#include <cstddef>

void sparse_matvec_avx512(
    const BCOO16& A,
    const float* x,
    const float* b,
    float* y,
    size_t M);

#pragma once
#include "bcoo16_encoder.hpp"

// Forward declaration visible to bindings.cpp
void sparse_matvec_avx512_mt(const BCOO16& A,
                             const float*  x,
                             const float*  b,
                             float*        y,
                             int           num_threads = 0);

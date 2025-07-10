#pragma once
#include <cstddef>

void dense_matvec(const float* A, const float* x, const float* b, float* y, size_t M, size_t K);

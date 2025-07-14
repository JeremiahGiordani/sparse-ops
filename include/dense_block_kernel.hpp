#pragma once
extern "C" void dense_block_kernel(const float* A,
                                   const float* x,
                                   const float* b,
                                   float*       y,
                                   size_t M, size_t K);
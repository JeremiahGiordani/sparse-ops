/* dense_block_kernel.cpp
 * Pure dense y = A·x + b     (for baseline benchmarking)
 * Uses the same AVX-512 micro-kernel already proven fast.
 */
#include "dense_matvec.hpp"          // your existing file
#include <cstddef>

extern "C"
void dense_block_kernel(const float* A,
                        const float* x,
                        const float* b,
                        float*       y,
                        size_t       M,
                        size_t       K)
{
    dense_matvec(A, x, b, y, M, K);   // call your tuned routine verbatim
}

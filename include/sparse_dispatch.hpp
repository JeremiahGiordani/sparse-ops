#pragma once
#include "bcoo16_encoder.hpp"

/* ------------------------------------------------------------------
   Kernel function type

   blocks : pointer to BCOO16Block headers
   nBlocks: number of blocks
   values : contiguous array of all non‑zeros (b.val_off indexes here)
   x      : dense input vector
   y      : output vector (accumulated: y[row] += …)
------------------------------------------------------------------ */
using KernelFn = void (*)(const BCOO16Block* blocks,
                          size_t             nBlocks,
                          const float*       values,
                          const float*       x,
                          float*             y);

/* Return a pointer to a JIT‑compiled kernel that matches matrix A */
KernelFn get_spmv_kernel(const BCOO16& A);


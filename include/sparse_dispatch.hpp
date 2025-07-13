#pragma once
#include "bcoo16_encoder.hpp"
#include "jit_cache.hpp"     // KernelFn typedef

// Return a function pointer to the JIT-generated SpMV kernel
KernelFn get_spmv_kernel(const BCOO16& A);

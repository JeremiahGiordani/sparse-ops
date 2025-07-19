#include "sparse_dispatch.hpp"
#include "codegen/spmv_template.hpp"
#include "jit_cache.hpp"

#include <sstream>

/* ------------------------------------------------------------------
   Build or retrieve a specialised kernel for matrix A
------------------------------------------------------------------ */
KernelFn get_spmv_kernel(const BCOO16& A)
{
    /* ---- cache key: rows|cols|blocks|nnz|ISA ---- */
    std::ostringstream key;
    key << A.original_num_rows << '|'
        << A.original_num_cols << '|'
        << A.blocks.size()     << '|'
        << A.values.size()     << '|'
        << (__builtin_cpu_supports("avx512f") ? "avx512" : "avx2");

    /* ---- generate C++ source ---- */
    std::string cpp = generate_spmv_cpp(
        A, "spmv_kernel",
        __builtin_cpu_supports("avx512f"));

    /* ---- compile or fetch from cache ---- */
    return get_or_build_kernel(key.str(), cpp, "spmv_kernel");
}

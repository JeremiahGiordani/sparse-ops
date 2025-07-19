#pragma once
#include <string>
#include <functional>
#include "sparse_dispatch.hpp"

// Forward-declare block type so we don’t need the whole encoder header here.
struct BCOO16Block;

/*  Return a function pointer for a compiled kernel.
 *
 *  If an .so for `key` exists in ~/.cache/sparseops, it is dlopened and the
 *  symbol `func_name` is returned. Otherwise `cpp_src` is compiled with clang
 *  (plus `clang_flags`), stored in the cache, and then dlopened.
 *
 *  Throws std::runtime_error on any failure.
 */
KernelFn get_or_build_kernel(const std::string& cache_key,
                             const std::string& cpp_source,
                             const std::string& exported_symbol);

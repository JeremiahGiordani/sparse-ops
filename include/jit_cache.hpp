#pragma once
#include <string>
#include <functional>

// Forward-declare block type so we don’t need the whole encoder header here.
struct BCOO16Block;

/*  Signature of every generated SpMV kernel
 *
 *  Parameters:
 *    blocks   – pointer to contiguous array of BCOO16Block
 *    nBlocks  – number of blocks
 *    x        – dense input vector  (length = matrix.cols)
 *    y        – output vector, bias already copied in (length = matrix.rows)
 */
using KernelFn =
    void (*)(const BCOO16Block* blocks,
             std::size_t        nBlocks,
             const float*       x,
             float*             y);

/*  Return a function pointer for a compiled kernel.
 *
 *  If an .so for `key` exists in ~/.cache/sparseops, it is dlopened and the
 *  symbol `func_name` is returned. Otherwise `cpp_src` is compiled with clang
 *  (plus `clang_flags`), stored in the cache, and then dlopened.
 *
 *  Throws std::runtime_error on any failure.
 */
KernelFn get_or_build_kernel(const std::string& key,
                             const std::string& cpp_src,
                             const std::string& func_name,
                             const std::string& clang_flags = "-O3 -march=native");

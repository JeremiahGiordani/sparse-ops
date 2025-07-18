#include "bcoo16_encoder.hpp"
#include "jit_cache.hpp"           // <-- need get_or_build_kernel
#include "codegen/spmv_template.hpp"
#include "sparse_dispatch.hpp"
#include "sparse_preproc.hpp" 
#include <omp.h>
#include <cstring>
#include <vector>
#include <immintrin.h>

// row_ptr builder unchanged …

static std::vector<size_t> make_row_ptr(const BCOO16& A)
{
    size_t M = A.original_num_rows;
    std::vector<size_t> rp(M + 1, 0);
    size_t blkIdx = 0;
    for (size_t r = 0; r < M; ++r) {
        while (blkIdx < A.blocks.size() && A.blocks[blkIdx].row_id == r)
            ++blkIdx;
        rp[r + 1] = blkIdx;
    }
    return rp;
}

void sparse_matvec_avx512_mt(const BCOO16& A,
                             const float*  x,
                             const float*  b,
                             float*        y,
                             int threads)
{
    size_t M = A.original_num_rows;
    auto row_ptr = make_row_ptr(A);
    auto fn      = get_spmv_kernel(A);             // JIT’d micro-kernel

    if (threads > 0)  omp_set_num_threads(threads);

    /* bias once, serial */
    if (b) std::memcpy(y, b, M*sizeof(float));
    else   std::memset(y, 0, M*sizeof(float));

#pragma omp parallel for schedule(static)
    for (size_t r = 0; r < M; ++r)
    {
        size_t blk0 = row_ptr[r];
        size_t blk1 = row_ptr[r+1];
        fn(A.blocks.data() + blk0,           // pointer to row’s blocks
           blk1 - blk0,                      // #blocks in this row
           x,
           y);                               // kernel adds into y in-place
    }
}

#include "bcoo16_encoder.hpp"
#include "jit_cache.hpp"           // <-- need get_or_build_kernel
#include "codegen/spmv_template.hpp"
#include "sparse_dispatch.hpp"
#include <omp.h>
#include <cstring>
#include <vector>

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
                             int           num_threads /*0 = OMP default*/)
{
    size_t M = A.original_num_rows;
    auto row_ptr = make_row_ptr(A);

    // -------- get (or compile) the same kernel used by the single-thread path
    auto fn = get_spmv_kernel(A);   // returns KernelFn

    if (num_threads > 0) omp_set_num_threads(num_threads);

#pragma omp parallel
    {
        int tid  = omp_get_thread_num();
        int T    = omp_get_num_threads();
        size_t rows_per_thr = (M + T - 1) / T;
        size_t r0 = tid * rows_per_thr;
        size_t r1 = std::min(M, r0 + rows_per_thr);

        // copy bias slice
        std::memcpy(y + r0, b + r0, (r1 - r0) * sizeof(float));

        // block slice for this thread
        size_t blk0 = row_ptr[r0];
        size_t blk1 = row_ptr[r1];

        fn(A.blocks.data() + blk0,      // pointer to first block
           blk1 - blk0,                 // number of blocks
           x,
           y);                          // *shared* y; no race because rows disjoint
    }
}

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
                             int           num_threads /*0 = OMP default*/)
{
    size_t M = A.original_num_rows;
    auto row_ptr = make_row_ptr(A);

    // -------- get (or compile) the same kernel used by the single-thread path

    if (num_threads > 0) omp_set_num_threads(num_threads);

    #pragma omp parallel for schedule(static)
    for (size_t row = 0; row < M; ++row)
    {
        float acc = b ? b[row] : 0.0f;        // bias once

        // process all blocks of this row
        size_t blk0 = row_ptr[row];
        size_t blk1 = row_ptr[row+1];

        /* load-reuse loop identical to fused dense kernel */
        for (size_t bi = blk0; bi < blk1; ) {
            int base = A.blocks[bi].first_col & ~15;
            __m512 xv = _mm512_loadu_ps(x + base);

            __m512 acc_vec = _mm512_setzero_ps();
            while (bi < blk1 && (A.blocks[bi].first_col & ~15) == base) {
                __m512 v = _mm512_loadu_ps(A.blocks[bi].values);
                acc_vec  = _mm512_fmadd_ps(v, xv, acc_vec);
                ++bi;
            }
            acc += _mm512_reduce_add_ps(acc_vec);
        }
        y[row] = acc;
    }

}

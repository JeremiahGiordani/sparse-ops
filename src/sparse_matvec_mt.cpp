#include "sparse_matvec_mt.hpp"
#include "sparse_dispatch.hpp"
#include <omp.h>
#include <vector>
#include <cstring>

/* ------------------------------------------------------------------ */
/* Partition blocks, not rows --------------------------------------- */
void sparse_matvec_avx512_mt(const BCOO16& A,
                             const float*  x,
                             const float*  b,
                             float*        y,
                             int           threads)
{
    const size_t M   = A.original_num_rows;
    const size_t nB  = A.blocks.size();

    if (threads > 0) omp_set_num_threads(threads);
    const int T = threads>0 ? threads : omp_get_max_threads();

    /* y scratch space per thread (avoid false sharing) */
    std::vector<std::vector<float>> ypriv(T, std::vector<float>(M, 0.0f));
    if (b) {
        /* copy bias once into each private buffer */
        for (int t = 0; t < T; ++t)
            std::memcpy(ypriv[t].data(), b, M*sizeof(float));
    }

    /* one kernel for all threads */
    KernelFn kernel = get_spmv_kernel(A);

#pragma omp parallel
    {
        const int  tid   = omp_get_thread_num();
        const int  T     = omp_get_num_threads();
        const size_t nB  = A.blocks.size();

        /* thread‑local replica of x (first touch ⇒ NUMA‑local) */
        static thread_local std::vector<float> x_local;
        if (x_local.empty()) {
            x_local.assign(A.original_num_cols, 0.0f);
            std::memcpy(x_local.data(), x,
                        A.original_num_cols * sizeof(float));
        }
        const float* xT = x_local.data();          // NUMA‑local pointer

        /* block range for this thread */
        size_t blk0 = ( nB * tid     ) / T;
        size_t blk1 = ( nB * (tid+1) ) / T;

        kernel(A.blocks.data() + blk0,
            blk1 - blk0,
            A.values.data(),
            xT,                                  // NUMA‑local x
            ypriv[tid].data());
    } /* omp parallel */
}

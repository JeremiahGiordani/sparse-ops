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
                             int threads)
{
    const size_t M = A.original_num_rows;

    /* --------- per‑thread private y --------- */
    const int T = (threads > 0) ? threads : omp_get_max_threads();
    std::vector<std::vector<float>> ypriv(T, std::vector<float>(M, 0.0f));
    if (b)
        for (int t = 0; t < T; ++t)
            std::memcpy(ypriv[t].data(), b, M * sizeof(float));

    KernelFn kernel = get_spmv_kernel(A);

    /* --------- parallel block processing --------- */
#pragma omp parallel num_threads(threads)
    {
        const int  tid  = omp_get_thread_num();
        const int  Ttot = omp_get_num_threads();

        size_t blk0 = (A.blocks.size() * tid    ) / Ttot;
        size_t blk1 = (A.blocks.size() * (tid+1)) / Ttot;

        /* NUMA‑local x replica ---------------------------------------------- */
        static thread_local std::vector<float> x_local;
        /* allocate or resize to current K */
        if (x_local.size() != A.original_num_cols)
            x_local.resize(A.original_num_cols);

        /* always copy the caller‑supplied x */
        std::memcpy(x_local.data(), x,
                    A.original_num_cols * sizeof(float));
        kernel(A.blocks.data() + blk0,
               blk1 - blk0,
               A.values.data(),
               x_local.data(),
               ypriv[tid].data());
    } /* omp parallel */

    /* --------- reduction: y = bias + Σ ypriv --------- */
    if (b)
        std::memcpy(y, b, M * sizeof(float));
    else
        std::memset(y, 0, M * sizeof(float));

    for (int t = 0; t < T; ++t) {
        const float* src = ypriv[t].data();
        for (size_t i = 0; i < M; ++i)
            y[i] += src[i] - (b ? b[i] : 0.0f);   // add dot‑products only
    }
}

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
    const size_t M  = A.original_num_rows;
    const size_t nB = A.blocks.size();

    if (threads > 0)  omp_set_num_threads(threads);
    const int T = threads>0 ? threads : omp_get_max_threads();

    /* ---------- single‑thread fast path ---------- */
    if (T == 1) {
        if (b) std::memcpy(y, b, M*sizeof(float));
        else   std::memset(y, 0, M*sizeof(float));

        get_spmv_kernel(A)(A.blocks.data(), nB,
                           A.values.data(), x, y);
        return;
    }

    /* ---------- multi‑thread path ---------- */
    /* public y starts with the bias ---------------- */
    if (b) std::memcpy(y, b, M*sizeof(float));
    else   std::memset(y, 0, M*sizeof(float));

    /* private buffers start at zero ---------------- */
    std::vector<std::vector<float>> ypriv(T, std::vector<float>(M, 0.0f));

    KernelFn kernel = get_spmv_kernel(A);

#pragma omp parallel
    {
        int tid       = omp_get_thread_num();
        size_t blk0   = (nB *  tid      ) / T;
        size_t blk1   = (nB * (tid + 1)) / T;

        kernel(A.blocks.data() + blk0,
               blk1 - blk0,
               A.values.data(),
               x,
               ypriv[tid].data());
    }

    /* -------- reduction y += Σ ypriv[t] -------- */
    for (int t = 0; t < T; ++t)
        for (size_t r = 0; r < M; ++r)
            y[r] += ypriv[t][r];
}



// benchmark/bench_matvec.cpp
//
// Implements timing hooks for MKL’s CSR sparse‐matrix–vector multiply
// vs. our QuasiDense backend’s matvec.

#include <chrono>
#include <vector>
#include <algorithm>
#include <iostream>

// MKL Sparse BLAS
#include <mkl_spblas.h>
#include <mkl.h>

#include <omp.h>

// Include data generator (produces BenchmarkData)
#include "data_gen.hpp"

// QuasiDense encoder & matvec API
#include "quasi_dense_encoder.hpp"
#include "bilinear_diagonal_matvec.hpp"

/// Run MKL CSR matvec [y = A·x] for `runs` repetitions, return median latency (μs)
double benchmark_mkl_matvec(BenchmarkData &data,
                            const std::vector<float> &x,
                            std::vector<float> &y,
                            int runs)
{
    // 1) Build MKL CSR handle
    sparse_matrix_t A;
    mkl_sparse_s_create_csr(&A,
        SPARSE_INDEX_BASE_ZERO,
        data.M, data.N,
        /* row_start */ data.row_ptr.data(),
        /* row_end   */ data.row_ptr.data() + 1,
        /* col_idx   */ data.col_ind.data(),
        /* values    */ data.values.data());
    mkl_sparse_optimize(A);

    // 2) Descriptor for a general matrix
    matrix_descr descr;
    descr.type = SPARSE_MATRIX_TYPE_GENERAL;
    descr.mode = SPARSE_FILL_MODE_FULL;
    descr.diag = SPARSE_DIAG_NON_UNIT;

    const float alpha = 1.0f, beta = 0.0f;

    // 3) Warm‑up (spawn threads, page‑fault code pages, etc.)
    mkl_sparse_s_mv(SPARSE_OPERATION_NON_TRANSPOSE,
                    alpha, A, descr,
                    x.data(), beta, y.data());

    // 4) Time loop
    std::vector<double> times;
    times.reserve(runs);
    for (int i = 0; i < runs; ++i) {
        auto t0 = std::chrono::high_resolution_clock::now();
        mkl_sparse_s_mv(SPARSE_OPERATION_NON_TRANSPOSE,
                        alpha, A, descr,
                        x.data(), beta, y.data());
        auto t1 = std::chrono::high_resolution_clock::now();
        times.push_back(
            std::chrono::duration<double, std::micro>(t1 - t0).count()
        );
    }

    mkl_sparse_destroy(A);

    // 5) Compute & return median
    std::nth_element(times.begin(),
                     times.begin() + times.size()/2,
                     times.end());
    return times[times.size()/2];
}

/// Run QuasiDense matvec [y = Q·x] for `runs` repetitions, return median latency (μs)
double benchmark_quasi_matvec(BenchmarkData &data,
                              const std::vector<float> &x,
                              std::vector<float> &y,
                              int runs)
{
    {
    const auto &Q = data.Q;
    if (Q.m * Q.r != Q.idx.size()) {
        std::cerr << "Unexpected idx buffer size! "
                << "Q.m*Q.r=" << Q.m*Q.r
                << " but idx.size()=" << Q.idx.size() << "\n";
        std::abort();
    }
    for (uint32_t i = 0; i < Q.m * Q.r; ++i) {
        if (Q.idx[i] >= Q.n) {
        std::cerr << "Out‑of‑bounds index at packed pos " << i
                    << ": idx=" << Q.idx[i] << ", but n=" << Q.n << "\n";
        std::abort();
        }
    }
    }
    // 1) Warm‑up
    quasi_dense_matvec(data.Q, x.data(), nullptr, y.data());

    // 2) Time loop
    std::vector<double> times;
    times.reserve(runs);
    for (int i = 0; i < runs; ++i) {
        auto t0 = std::chrono::high_resolution_clock::now();
        quasi_dense_matvec(data.Q, x.data(), nullptr, y.data());
        auto t1 = std::chrono::high_resolution_clock::now();
        times.push_back(
            std::chrono::duration<double, std::micro>(t1 - t0).count()
        );
    }

    // 3) Compute & return median
    std::nth_element(times.begin(),
                     times.begin() + times.size()/2,
                     times.end());
    return times[times.size()/2];
}

/**
 * Benchmark MKL dense mat‑vec [y = Wdense * x] via cblas_sgemv
 * @param data  contains Wdense (size M*N)
 * @param x     input vector (size  N)
 * @param y     output vector (size  M)
 * @param runs  number of repetitions
 * @returns     median latency in microseconds
 */
double benchmark_mkl_dense_matvec(const BenchmarkData &data,
                                  const std::vector<float> &x,
                                  std::vector<float> &y,
                                  int runs)
{
    const int  M     = int(data.M);
    const int  N     = int(data.N);
    const float alpha = 1.0f, beta = 0.0f;

    // Warm‑up
    cblas_sgemv(CblasRowMajor,
                CblasNoTrans,
                M, N,
                alpha,
                data.Wdense.data(),    // A: row-major M×N
                N,                     // lda
                x.data(), 1,           // incx
                beta,
                y.data(), 1);          // incy

    // Time loop
    std::vector<double> times;
    times.reserve(runs);
    for (int i = 0; i < runs; ++i) {
        auto t0 = std::chrono::high_resolution_clock::now();
        cblas_sgemv(CblasRowMajor,
                    CblasNoTrans,
                    M, N,
                    alpha,
                    data.Wdense.data(),
                    N,
                    x.data(), 1,
                    beta,
                    y.data(), 1);
        auto t1 = std::chrono::high_resolution_clock::now();
        times.push_back(
            std::chrono::duration<double, std::micro>(t1 - t0).count()
        );
    }

    // median
    std::nth_element(times.begin(),
                     times.begin() + times.size()/2,
                     times.end());
    return times[times.size()/2];
}


/**
 * Benchmark a “standard” OpenMP CSR mat–vec:
 *    y[i] = sum_j W[row_ptr[i] .. row_ptr[i+1]-1] * x[col_ind[j]]
 *
 * @param data  contains CSR arrays: row_ptr[M+1], col_ind[nnz], values[nnz]
 * @param x     input vector (size N)
 * @param y     output vector (size M)
 * @param runs  number of repetitions
 * @returns     median latency in microseconds
 */
double benchmark_openmp_csr_matvec(const BenchmarkData &data,
                                   const std::vector<float> &x,
                                   std::vector<float> &y,
                                   int runs)
{
    const int M = int(data.M);
    const auto &rowp = data.row_ptr;
    const auto &cols = data.col_ind;
    const auto &vals = data.values;

    // Warm‑up once
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < M; ++i) {
        float acc = 0.0f;
        for (MKL_INT idx = rowp[i]; idx < rowp[i+1]; ++idx) {
            acc += vals[idx] * x[ cols[idx] ];
        }
        y[i] = acc;
    }

    // Time loop
    std::vector<double> times;
    times.reserve(runs);
    for (int rep = 0; rep < runs; ++rep) {
        auto t0 = std::chrono::high_resolution_clock::now();

        #pragma omp parallel for schedule(static)
        for (int i = 0; i < M; ++i) {
            float acc = 0.0f;
            for (MKL_INT idx = rowp[i]; idx < rowp[i+1]; ++idx) {
                acc += vals[idx] * x[ cols[idx] ];
            }
            y[i] = acc;
        }

        auto t1 = std::chrono::high_resolution_clock::now();
        times.push_back(
            std::chrono::duration<double, std::micro>(t1 - t0).count()
        );
    }

    // Return median
    std::nth_element(times.begin(),
                     times.begin() + times.size()/2,
                     times.end());
    return times[times.size()/2];
}
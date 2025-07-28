// benchmark/matmul/bench_matmul.cpp

#include <chrono>
#include <vector>
#include <algorithm>
#include <iostream>

#include <mkl_spblas.h>   // MKL sparse SpMM
#include <mkl.h>          // MKL_INT, CBLAS, threading control

#include "data_gen.hpp"                  // common/generate_data, BenchmarkData
#include "ellpack_matmul.hpp"  // ellpack_matmul

/// Benchmark MKL SpMM: Y = A*X, where A is CSR sparse, X is dense.
/// Returns median runtime in microseconds.
double benchmark_mkl_spmm(const BenchmarkData &data,
                          const std::vector<float> &X,
                          std::vector<float> &Y,
                          int C,
                          int runs)
{
    // 1) Build the MKL CSR handle for A
    sparse_matrix_t A;
    mkl_sparse_s_create_csr(&A,
        SPARSE_INDEX_BASE_ZERO,
        data.M, data.N,
        const_cast<MKL_INT*>(data.row_ptr.data()),
        const_cast<MKL_INT*>(data.row_ptr.data() + 1),
        const_cast<MKL_INT*>(data.col_ind.data()),
        const_cast<float*>  (data.values.data()));

    // 2) Give MKL a hint that we'll do row‑major SpMM
    matrix_descr descr;
    descr.type = SPARSE_MATRIX_TYPE_GENERAL;
    descr.mode = SPARSE_FILL_MODE_FULL;
    descr.diag = SPARSE_DIAG_NON_UNIT;
    mkl_sparse_set_mm_hint(
        A,
        SPARSE_OPERATION_NON_TRANSPOSE,
        descr,
        SPARSE_LAYOUT_ROW_MAJOR,  // use sparse_layout_t, not CBLAS_LAYOUT
        /*dense_columns=*/ C,
        /*expected_calls=*/ runs
    );
    mkl_sparse_optimize(A);

    const float alpha = 1.0f;
    const float beta  = 0.0f;

    // 3) Warm‑up, so threads spawn and pages fault in
    mkl_sparse_s_mm(
        SPARSE_OPERATION_NON_TRANSPOSE,
        alpha,
        A, descr,
        SPARSE_LAYOUT_ROW_MAJOR,
        X.data(), /*ldb=*/C, /*nrhs=*/C,
        beta,
        Y.data(), /*ldc=*/C
    );

    // 4) Timed runs
    std::vector<double> times;
    times.reserve(runs);
    for (int i = 0; i < runs; ++i) {
        auto t0 = std::chrono::high_resolution_clock::now();
        mkl_sparse_s_mm(
            SPARSE_OPERATION_NON_TRANSPOSE,
            alpha,
            A, descr,
            SPARSE_LAYOUT_ROW_MAJOR,
            X.data(), C, C,
            beta,
            Y.data(), C
        );
        auto t1 = std::chrono::high_resolution_clock::now();
        times.push_back(
            std::chrono::duration<double, std::micro>(t1 - t0).count()
        );
    }

    // 5) Clean up and return median
    mkl_sparse_destroy(A);

    auto mid = times.begin() + times.size()/2;
    std::nth_element(times.begin(), mid, times.end());
    return *mid;
}

/// Benchmark our Ellpack mat‑mul: Y = E * X.
/// Returns median runtime in microseconds.
double benchmark_ellpack_matmul(const BenchmarkData &data,
                                const std::vector<float> &X,
                                std::vector<float> &Y,
                                int C,
                              int runs)
{
    // 1) Warm‑up
    ellpack_matmul(data.E, X.data(), C, nullptr, Y.data());

    // 2) Timed loop
    std::vector<double> times;
    times.reserve(runs);
    for (int i = 0; i < runs; ++i) {
        auto t0 = std::chrono::high_resolution_clock::now();
        ellpack_matmul(data.E, X.data(), C, nullptr, Y.data());
        auto t1 = std::chrono::high_resolution_clock::now();
        times.push_back(
            std::chrono::duration<double, std::micro>(t1 - t0).count()
        );
    }

    // 3) Return median
    std::nth_element(times.begin(),
                     times.begin() + times.size()/2,
                     times.end());
    return times[times.size()/2];
}

/// Benchmark dense GEMM via BLAS: Y = Wdense × X.
/// Returns median runtime in microseconds.
double benchmark_mkl_gemm(const BenchmarkData &data,
                          const std::vector<float> &X,
                          std::vector<float> &Y,
                          int C,
                          int runs)
{
    const int M = int(data.M);
    const int N = int(data.N);
    const float alpha = 1.0f, beta = 0.0f;

    // Warm‑up
    cblas_sgemm(
        CblasRowMajor,
        CblasNoTrans, CblasNoTrans,
        M, C, N,
        alpha,
        data.Wdense.data(), N,
        X.data(),           C,
        beta,
        Y.data(),           C
    );

    // Timed loop
    std::vector<double> times;
    times.reserve(runs);
    for (int i = 0; i < runs; ++i) {
        auto t0 = std::chrono::high_resolution_clock::now();
        cblas_sgemm(
            CblasRowMajor,
            CblasNoTrans, CblasNoTrans,
            M, C, N,
            alpha,
            data.Wdense.data(), N,
            X.data(),           C,
            beta,
            Y.data(),           C
        );
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


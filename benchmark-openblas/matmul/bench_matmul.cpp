// benchmark/matmul/bench_matmul.cpp

#include <chrono>
#include <vector>
#include <algorithm>
#include <iostream>

#include <cblas.h>      // CBLAS, threading control

#include "data_gen.hpp"                  // common/generate_data, BenchmarkData
#include "ellpack_matmul.hpp"  // ellpack_matmul

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
double benchmark_openblas_gemm(const BenchmarkData &data,
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


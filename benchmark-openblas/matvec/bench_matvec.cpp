// benchmark/bench_matvec.cpp
//
// Implements timing hooks for Openblas matvec multiply
// vs. our Ellpack backend’s matvec.

#include <chrono>
#include <vector>
#include <algorithm>
#include <iostream>

// OpenBLAS
#include <cblas.h>

#include <omp.h>

// Include data generator (produces BenchmarkData)
#include "data_gen.hpp"

// Ellpack encoder & matvec API
#include "ellpack_encoder.hpp"
#include "ellpack_matvec.hpp"

/// Run Ellpack matvec [y = E·x] for `runs` repetitions, return median latency (μs)
double benchmark_ellpack_matvec(BenchmarkData &data,
                              const std::vector<float> &x,
                              std::vector<float> &y,
                              int runs)
{
    {
    const auto &E = data.E;
    if (E.m * E.r != E.idx.size()) {
        std::cerr << "Unexpected idx buffer size! "
                << "E.m*E.r=" << E.m*E.r
                << " but idx.size()=" << E.idx.size() << "\n";
        std::abort();
    }
    for (uint32_t i = 0; i < E.m * E.r; ++i) {
        if (E.idx[i] >= E.n) {
        std::cerr << "Out‑of‑bounds index at packed pos " << i
                    << ": idx=" << E.idx[i] << ", but n=" << E.n << "\n";
        std::abort();
        }
    }
    }
    // 1) Warm‑up
    ellpack_matvec(data.E, x.data(), nullptr, y.data());

    // 2) Time loop
    std::vector<double> times;
    times.reserve(runs);
    for (int i = 0; i < runs; ++i) {
        auto t0 = std::chrono::high_resolution_clock::now();
        ellpack_matvec(data.E, x.data(), nullptr, y.data());
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
 * Benchmark OpenBLAS dense mat‑vec [y = Wdense * x] via cblas_sgemv
 * @param data  contains Wdense (size M*N)
 * @param x     input vector (size  N)
 * @param y     output vector (size  M)
 * @param runs  number of repetitions
 * @returns     median latency in microseconds
 */
double benchmark_openblas_matvec(const BenchmarkData &data,
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


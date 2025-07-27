// benchmark/main.cpp
//
// Entry point for benchmarking MKL CSR sparse‐matrix–vector multiply
// vs. our QuasiDense backend’s matvec.
//
// Usage:
//   ./benchmark \
//     --M 1000 \
//     --N 1000 \
//     --sparsity 0.9 \
//     --runs 100 \
//     --threads 4 \
//     --seed 42

#include <iostream>
#include <vector>
#include <random>
#include <string>
#include <cstdint>

// MKL and OpenMP for thread control
#include <mkl.h>
#include <omp.h>
#include "data_gen.hpp"  // For BenchmarkData and generate_data

// Forward declarations from data_gen.cpp and bench_matvec.cpp
// (Assuming you compile those separately and link them in CMake)


double benchmark_mkl_matvec(BenchmarkData &data,
                            const std::vector<float> &x,
                            std::vector<float> &y,
                            int runs);

double benchmark_quasi_matvec(BenchmarkData &data,
                              const std::vector<float> &x,
                              std::vector<float> &y,
                              int runs);

double benchmark_mkl_dense_matvec(const BenchmarkData &data,
                                   const std::vector<float> &x,
                                   std::vector<float> &y,
                                   int runs);




int main(int argc, char** argv) {
    // Default parameters
    int64_t M         = 1000;
    int64_t N         = 1000;
    double  sparsity  = 0.9;
    int     runs      = 100;
    int     threads   = 1;
    int   omp_threads = 1;
    uint64_t seed     = 42;
    bool irregular = false;

    // Simple CLI parsing
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--M" && i+1 < argc) {
            M = std::stoll(argv[++i]);
        } else if (arg == "--N" && i+1 < argc) {
            N = std::stoll(argv[++i]);
        } else if (arg == "--sparsity" && i+1 < argc) {
            sparsity = std::stod(argv[++i]);
        } else if (arg == "--runs" && i+1 < argc) {
            runs = std::stoi(argv[++i]);
        } else if (arg == "--threads" && i+1 < argc) {
            threads = std::stoi(argv[++i]);
        } else if (arg == "--omp-threads" && i+1 < argc) {
            omp_threads = std::stoi(argv[++i]);
        } else if (arg == "--seed" && i+1 < argc) {
            seed = std::stoull(argv[++i]);
        } else if (arg == "--irregular" && i+1 < argc) {
            irregular = std::stoi(argv[++i]);
        } else {
            std::cerr << "Unknown or incomplete arg: " << arg << "\n"
                      << "Usage: " << argv[0]
                      << " [--M rows] [--N cols] [--sparsity p]"
                         " [--runs r] [--threads t] [--seed s] [--irregular 0|1]\n";
            return 1;
        }
    }

    // Fix thread counts
    mkl_set_num_threads(threads);
    omp_set_num_threads(omp_threads);
    mkl_set_num_threads_local(threads);
    mkl_set_dynamic(0);

    std::cout
        << "=== Sparse MatVec Benchmark ===\n"
        << "Matrix dims:     " << M << "×" << N << "\n"
        << "Sparsity:        " << sparsity << "\n"
        << "Repetitions:     " << runs << "\n"
        << "Threads:         " << threads << "\n"
        << "OpenMP threads:  " << omp_threads << "\n"
        << "RNG seed:        " << seed << "\n\n";

    // 1) Generate the data (CSR + QuasiDense)
    auto data = generate_data(M, N, sparsity, seed, irregular);

    // 2) Create a random input vector x
    std::vector<float> x(N);
    {
        std::mt19937_64 rng(seed);
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        for (auto &v : x) v = dist(rng);
    }

    // 3) Allocate output vector y
    std::vector<float> y(M);

    // 4) Benchmark MKL
    double t_mkl = benchmark_mkl_matvec(data, x, y, runs);
    std::cout << "MKL sparse matvec:   " << t_mkl   << " µs\n";

    // 5) Benchmark QuasiDense
    double t_quasi = benchmark_quasi_matvec(data, x, y, runs);
    std::cout << "QuasiDense matvec:   " << t_quasi << " µs\n";

    // 6) Benchmark MKL dense matvec
    double t_mkl_dense = benchmark_mkl_dense_matvec(data, x, y, runs);
    std::cout << "MKL dense matvec:    " << t_mkl_dense << " µs\n";

    return 0;
}

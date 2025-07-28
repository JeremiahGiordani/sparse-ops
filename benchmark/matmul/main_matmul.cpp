// benchmark/matmul/main_matmul.cpp
//
// Entry point for benchmarking MKL SpMM (sparse×dense) and
// Ellpack mat‑mul vs. a dense BLAS GEMM.
//
// Usage:
//   ./sparse_matmul_bench \
//     --M 1000 \
//     --N 1000 \
//     --C 128  \
//     --sparsity 0.9 \
//     --runs 100 \
//     --mkl-threads 4 \
//     --omp-threads 4 \
//     --seed 42 \
//     --irregular 0

#include <iostream>
#include <vector>
#include <random>
#include <string>
#include <cstdint>

#include <mkl.h>
#include <omp.h>

#include "data_gen.hpp"  // BenchmarkData, generate_data

// Forward declarations from bench_matmul.cpp
double benchmark_mkl_spmm(const BenchmarkData &data,
                          const std::vector<float> &X,
                          std::vector<float> &Y,
                          int C,
                          int runs);

double benchmark_ellpack_matmul(const BenchmarkData &data,
                              const std::vector<float> &X,
                              std::vector<float> &Y,
                              int C,
                              int runs);

double benchmark_mkl_gemm(const BenchmarkData &data,
                                   const std::vector<float> &X,
                                   std::vector<float> &Y,
                                   int C,
                                   int runs);


int main(int argc, char** argv) {
    // default parameters
    int64_t M        = 1000;
    int64_t N        = 1000;
    int     C        = 128;
    double  sparsity = 0.9;
    int     runs     = 100;
    int     mkl_threads  = 1;
    int     omp_threads = 1;
    uint64_t seed    = 42;
    bool    irregular = false;

    // parse CLI
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--M" && i+1 < argc) {
            M = std::stoll(argv[++i]);
        } else if (arg == "--N" && i+1 < argc) {
            N = std::stoll(argv[++i]);
        } else if (arg == "--C" && i+1 < argc) {
            C = std::stoi(argv[++i]);
        } else if (arg == "--sparsity" && i+1 < argc) {
            sparsity = std::stod(argv[++i]);
        } else if (arg == "--runs" && i+1 < argc) {
            runs = std::stoi(argv[++i]);
        } else if (arg == "--mkl-threads" && i+1 < argc) {
            mkl_threads = std::stoi(argv[++i]);
        } else if (arg == "--omp-threads" && i+1 < argc) {
            omp_threads = std::stoi(argv[++i]);
        } else if (arg == "--seed" && i+1 < argc) {
            seed = std::stoull(argv[++i]);
        } else if (arg == "--irregular" && i+1 < argc) {
            irregular = (std::stoi(argv[++i]) != 0);
        } else {
            std::cerr << "Unknown or incomplete arg: " << arg << "\n"
                      << "Usage: " << argv[0]
                      << " --M <rows> --N <cols> --C <channels>"
                         " --sparsity <p> --runs <r>"
                         " --mkl-threads <t> --omp-threads <o>"
                         " --seed <s> --irregular <0|1>\n";
            return 1;
        }
    }

    // set threading
    mkl_set_dynamic(0);
    mkl_set_num_threads(mkl_threads);
    omp_set_num_threads(omp_threads);

    std::cout
        << "=== Sparse MatMul Benchmark ===\n"
        << "Matrix dims:     " << M << "×" << N << "\n"
        << "Dense cols C:    " << C << "\n"
        << "Sparsity:        " << sparsity << "\n"
        << "Repetitions:     " << runs << "\n"
        << "MKL threads:     " << mkl_threads << "\n"
        << "OpenMP threads:  " << omp_threads << "\n"
        << "RNG seed:        " << seed << "\n"
        << "Irregular last row: " << irregular << "\n\n";

    // generate shared data
    auto data = generate_data(M, N, sparsity, seed, irregular);

    // make random dense input X (N × C)
    std::vector<float> X;
    X.reserve(size_t(N) * C);
    {
        std::mt19937_64 rng(seed);
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        for (int64_t i = 0; i < N * C; ++i) {
            X.push_back(dist(rng));
        }
    }

    // output buffer Y (M × C)
    std::vector<float> Y(size_t(M) * C);

    // 1) MKL SpMM
    double t_mkl_spmm = benchmark_mkl_spmm(data, X, Y, C, runs);
    std::cout << "MKL sparse×dense  : " << t_mkl_spmm << " µs\n";

    // 2) Dense BLAS GEMM
    double t_blas_mm = benchmark_mkl_gemm(data, X, Y, C, runs);
    std::cout << "BLAS dense GEMM   : " << t_blas_mm << " µs\n";

    // 3) Ellpack sparse×dense
    double t_ellpack_mm = benchmark_ellpack_matmul(data, X, Y, C, runs);
    std::cout << "Ellpack matmul    : " << t_ellpack_mm << " µs\n";

    return 0;
}

// benchmark/data_gen.hpp
#pragma once

#include <cstdint>
#include <vector>
#include <random>
#include <mkl.h>                 // for MKL_INT
#include "../include/quasi_dense_encoder.hpp"  // for QuasiDense

/// Holds both the CSR data for MKL and the QuasiDense object
struct BenchmarkData {
    int64_t               M, N;        // dims
    std::vector<MKL_INT>  row_ptr;     // size M+1
    std::vector<MKL_INT>  col_ind;     // size = #nonzeros
    std::vector<float>    values;      // size = #nonzeros
    QuasiDense            Q;           // QuasiDense handle
    std::vector<float>    Wdense;
};

/**
 * @brief Generate a random M×N matrix with given sparsity, then
 *        build both:
 *          1) CSR arrays (row_ptr, col_ind, values)
 *          2) QuasiDense Q = convert_to_quasi_dense(...)
 *
 * @param M            Number of rows
 * @param N            Number of cols
 * @param sparsity     Fraction of entries to zero out (0.0→dense, 1.0→all zero)
 * @param seed         RNG seed (default = random_device())
 */
BenchmarkData generate_data(int64_t M,
                            int64_t N,
                            double  sparsity,
                            uint64_t seed = std::random_device{}());

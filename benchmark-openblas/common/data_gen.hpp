// benchmark/data_gen.hpp
#pragma once

#include <cstdint>
#include <vector>
#include <random>
#include "ellpack_encoder.hpp"  // for Ellpack

/// Holds both the vector and the Ellpack object
struct BenchmarkData {
    int64_t               M, N;        // dims
    std::vector<int>  row_ptr;     // size M+1
    std::vector<int>  col_ind;     // size = #nonzeros
    std::vector<float>    values;      // size = #nonzeros
    Ellpack               E;           // Ellpack handle
    std::vector<float>    Wdense;
};

/**
 * @brief Generate a random M×N matrix with given sparsity, then
 *        build both:
 *          1) CSR arrays (row_ptr, col_ind, values)
 *          2) Ellpack E = convert_to_ellpack(...)
 *
 * @param M            Number of rows
 * @param N            Number of cols
 * @param sparsity     Fraction of entries to zero out (0.0→dense, 1.0→all zero)
 * @param seed         RNG seed (default = random_device())
 * @param irregular    If true, the last row is always dense
 */
BenchmarkData generate_data(int64_t M,
                            int64_t N,
                            double  sparsity,
                            uint64_t seed = std::random_device{}(),
                            bool    irregular = false);

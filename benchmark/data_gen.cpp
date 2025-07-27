// benchmark/data_gen.cpp
#include "data_gen.hpp"
#include <vector>
#include <random>
#include <cstdint>
#include <mkl.h>  

// include your QuasiDense encoder header
#include "quasi_dense_encoder.hpp"

/**
 * @brief Generate a random M×N matrix with given sparsity, then
 *        build both:
 *          1) CSR arrays (row_ptr, col_ind, values)
 *          2) QuasiDense Q = convert_to_quasi_dense(W, M, N)
 *
 * @param M            Number of rows
 * @param N            Number of cols
 * @param sparsity     Fraction of entries to set to zero (0.0→dense, 1.0→all zero)
 * @param seed         RNG seed (default = random_device())
 * @param irregular    If true, the last row is always dense
 */
BenchmarkData generate_data(int64_t M,
                            int64_t N,
                            double  sparsity,
                            uint64_t seed,
                            bool     irregular)
{
    // 1) Build a flat dense matrix W (row-major)
    std::vector<float> W;
    W.reserve(M * N);

    std::mt19937_64                     rng(seed);
    std::normal_distribution<float>     val_dist(0.0f, 1.0f);
    std::uniform_real_distribution<>    zero_coin(0.0, 1.0);

    double keep_prob = 1.0 - sparsity;
    // If irregular, ensure the last row is always dense
    if (irregular) {
        for (int64_t i = 0; i < M; ++i) {
            for (int64_t j = 0; j < N; ++j) {
                if (i == M - 1) {
                    // last row: always keep a random value
                    W.push_back(val_dist(rng));
                } else {
                    if (zero_coin(rng) < keep_prob) {
                        W.push_back(val_dist(rng));
                    } else {
                        W.push_back(0.0f);
                    }
                }
            }
        }
    } else {
        for (int64_t i = 0; i < M * N; ++i) {
            if (zero_coin(rng) < keep_prob) {
                W.push_back(val_dist(rng));
            } else {
                W.push_back(0.0f);
            }
        }
    }

    
    

    // 2) Build CSR from W
    std::vector<MKL_INT>   row_ptr(M + 1, 0);
    std::vector<MKL_INT>   col_ind;
    std::vector<float> values;
    col_ind.reserve(int((1.0 - sparsity) * M * N));
    values.reserve(col_ind.capacity());

    for (int64_t i = 0; i < M; ++i) {
        int nnz_row = 0;
        for (int64_t j = 0; j < N; ++j) {
            float v = W[i * N + j];
            if (v != 0.0f) {
                col_ind.push_back(int(j));
                values.push_back(v);
                ++nnz_row;
            }
        }
        row_ptr[i + 1] = row_ptr[i] + nnz_row;
    }

    // 3) Build QuasiDense using our encoder
    //    Assumes you have a free function:
    //      QuasiDense convert_to_quasi_dense(const std::vector<float>& W,
    //                                        uint32_t m, uint32_t n);
    QuasiDense Q = convert_to_quasi_dense(W.data(), uint32_t(M), uint32_t(N));

    return BenchmarkData{M, N, std::move(row_ptr),
                         std::move(col_ind), std::move(values),
                         std::move(Q), std::move(W)};
}

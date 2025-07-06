// sparse_matvec.cpp

#include "sparse_matvec.hpp"
#include <vector>
#include <cstring>  // for memset
#include <omp.h>

void sparse_matvec(const float* input, int in_dim,
                   const float* values, const int* indices, const int* indptr, int out_dim,
                   const float* bias,
                   float* output) {

    std::memset(output, 0, sizeof(float) * out_dim);

    #pragma omp parallel for
    for (int row = 0; row < out_dim; ++row) {
        float sum = bias[row];
        for (int idx = indptr[row]; idx < indptr[row + 1]; ++idx) {
            int col = indices[idx];
            sum += values[idx] * input[col];
        }
        output[row] = sum;
    }
}
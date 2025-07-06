#pragma once

extern "C" {
void sparse_matvec(const float* input, int in_dim,
                   const float* values, const int* indices, const int* indptr, int out_dim,
                   const float* bias,
                   float* output);
}
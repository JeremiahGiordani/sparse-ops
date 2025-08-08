#pragma once
#include <cstdint>
#include <immintrin.h>
#include <cstring>
#include "utils.hpp"
#include "cbcoo_encoder.hpp"

template<bool FUSE_RELU=false, bool Y_BxM=true>
void cbcoo_spmm_stream(
    const CBCOO&  E,
    const float*  X,      // [B x N], row-major
    uint32_t      B,
    const float*  bias,   // [M] or nullptr
    float*        Y       // [ (Y_BxM? B×M : M×B) ], row-major
);

inline void cbcoo_spmm(
    const CBCOO   &E,
    const float*   X,
    uint32_t       C,
    const float*   bias,
    float*         Y)
{


    cbcoo_spmm_stream<false,  false>(E, X, C, bias, Y);

}
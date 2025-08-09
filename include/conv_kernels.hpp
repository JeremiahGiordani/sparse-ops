// include/conv_kernels.hpp
#pragma once
#include "rbm_conv.hpp"
#include "sparse_onnx.hpp"

void conv2d_rbm_fmajor_implicit(
    const ConvAttr&        c,
    const float*           src,
    uint32_t               B,
    float*                 dst
);

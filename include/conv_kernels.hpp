// include/conv_kernels.hpp
#pragma once
#include "rbm_conv.hpp"
#include "sparse_onnx.hpp"
#include "ellpack_matmul.hpp"


void conv2d_rbm_fmajor_implicit(
    const ConvAttr&        c,
    const float*           src,
    uint32_t               B,
    float*                 dst
);


void conv2d_tiled_im2col_fmajor(
    const ConvAttr& c,
    const float*    src,   // (B,C,H,W) Fortran (B contiguous)
    uint32_t        B,
    float*          dst    // (B,Cout,H_out,W_out) Fortran
);


void conv2d_tiled_im2col_cmajor(
    const ConvAttr& c,
    const float*    src,  // (B,C,H,W), C-order (NCHW)
    uint32_t        B,
    float*          dst   // (B,Cout,H_out,W_out), C-order
);

void conv2d_patchmajor_tiled(
    const ConvAttr& c,
    const float*    src_pf,  // PF input
    uint32_t        B,
    float*          dst_pf   // PF output
);
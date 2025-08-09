// include/rbm_conv.hpp
#pragma once
#include <vector>
#include <cstdint>
#include "aligned_buffer.hpp"
#include "ellpack_encoder.hpp" // for Ellpack (we use E.idx / E.Wd / E.nnz)

struct RBMColPair {
    uint16_t row;   // row-id within the block [0..Ct)
    float    w;     // weight
};

struct RBMBlock {
    uint32_t M0;       // starting out-channel index for this block
    uint32_t Ct;       // rows in this block (<= Ct_max)
    std::vector<uint32_t> krel;     // union column indices (length U)
    std::vector<uint32_t> colptr;   // U+1; pairs slice per union column
    std::vector<RBMColPair> pairs;  // flattened pairs for all union columns
    std::vector<float> bias;        // length Ct (optional; empty if no bias)
};

struct RBMPlan {
    uint32_t M;          // Cout
    uint32_t N;          // Cin*kH*kW
    uint32_t Ct_max;     // block height used during build
    std::vector<RBMBlock> blocks;
};

/// Build RBM from an existing ELLPACK (per-row) encoding.
/// Assumes each row's indices in E.idx are already in ascending k_rel order (they are in your convert_to_ellpack).
RBMPlan build_rbm_from_ellpack(const Ellpack& E, const float* bias_or_null, uint32_t Ct_max=8);

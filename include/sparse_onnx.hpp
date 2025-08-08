// include/sparse_onnx.hpp
#pragma once

#include <string>
#include <cstdint>
#include <memory>
#include <vector>
#include <variant>
#include <unordered_map>
#include <array> 
#include <cmath> 
#include <algorithm>
#include <cstring>
#include <limits>
#include <cstdlib>

#include "ellpack_matmul.hpp"       // ellpack_matmul
#include "ellpack_encoder.hpp"   // for Ellpack
#include "activations.hpp"          // relu_inplace, sigmoid_inplace, tanh_inplace

/// \file sparse_onnx.hpp
/// C++ interface for loading an ONNX model and running inference
/// via pre-encoded ELLPACK sparse kernels, with fixed (static) batch size
/// inferred from the model’s input shape.

/// Supported layer types in the ONNX graph.
enum class LayerType {  
  MatMul,      // matmul or fused matmul+relu  
  Activation,  // relu/sigmoid/tanh  
  Elementwise, // add  
  Pool,        // maxpool/globalavgpool  
  Reshape,     // flatten  
  Conv         // convolution  
};  

enum class LayerOp {  
  MatMul, MatMulRelu,  
  Relu, Sigmoid, Tanh,  
  Add,  
  MaxPool, GlobalAveragePool,  
  Flatten,  
  Conv  
};

// 1) Define each payload type
struct MatMulAttr {
  Ellpack    E;
  float*     bias_ptr;    // length = E.m
};


struct ConvAttr {
    // The ELLPACK handle for the *im2col* weight matrix, of size:
    //   rows = Cout
    //   cols = Cin * kH * kW
    Ellpack              E;
    float*               bias_ptr;
    std::array<int,4>    kernel_dims; // dimensions: {Cout, Cin, kH, kW}
    std::array<int,4>    pads; // {padH_begin, padW_begin, padH_end, padW_end}
    std::array<int,2>    strides; // {sH, sW}
    std::array<int,2>    dilations; // {dH, dW}
    int                  group;
    int                  H_in, W_in;    // spatial dims of the *input*
    int                  H_out, W_out;  // computed from pads/strides
    std::vector<size_t>  patch_indices; // length = Cin*kH*kW * (H_out*W_out)
};

struct PoolAttr {
  std::vector<int>   kernel_shape;  // {kH, kW}
  std::vector<int>   pads;          // same convention
  std::vector<int>   strides;       // {sH, sW}
  bool               is_global;     // true for GlobalAveragePool
};

struct FlattenAttr {
  int axis;  // e.g. 1
};

struct AddAttr { /* no payload */ };
struct ActAttr { /* no payload */ };

// 2) Combine into one variant
using LayerAttr = std::variant<
    MatMulAttr, ConvAttr, PoolAttr, FlattenAttr, AddAttr, ActAttr>;

struct Layer {
  LayerType type;
  LayerOp   op;
  LayerAttr attr;
  std::vector<std::string> inputs;   // graph inputs to this node
  std::vector<std::string> outputs;
};


struct RunResult {
    float*   data;  ///< pointer to the freshly‐allocated output buffer
    uint32_t features;  ///< number of rows of that buffer
    bool owned;
};


/// A simple ONNX-backed sparse inference engine that
/// — Parses a fixed-batch ONNX model at load time,
/// — Extracts and ELLPACK-encodes all weights,
/// — Allocates one output buffer per MatMul layer (m × batch_dim),
/// — Runs inference with zero allocations and no ping-pong copies.
class SparseOnnxModel {
public:
    /// Load and preprocess the ONNX model at \p onnx_path:
    /// 1) Parse the graph and infer a fixed batch dimension from the first
    ///    non-initializer input’s shape (must be static).
    /// 2) Extract all weight & bias initializers and convert weights to ELLPACK.
    /// 3) Build the execution plan and allocate one scratch buffer per MatMul.
    explicit SparseOnnxModel(const std::string& onnx_path);

    /// Run inference:
    ///  - \p input: pointer to float array of shape [input_dim × batch_dim]
    ///  - \p B: must equal the batch_dim inferred at load time
    ///  - \p output: pointer to float array of shape [output_dim × batch_dim]
    ///
    /// Both input and output are row-major contiguous.
    void run(const float* input, uint32_t B, float* output) const;

    /// Number of rows (m) in the final output (i.e., output_dim).
    std::vector<size_t> output_shape() const {
        auto &int_dims = shape_map_.at(output_name_);    // vector<int>
        std::vector<size_t> dims;
        dims.reserve(int_dims.size());
        for (int d : int_dims) dims.push_back(static_cast<size_t>(d));
        return dims;
    }

private:

    std::vector<Layer>                            layers_;      ///< Execution sequence
    std::unique_ptr<float[]>                      bias_data_;   ///< All biases packed contiguously
    uint32_t                                      batch_dim_;   ///< Current batch size
    uint32_t                                      in_features_; ///< Input feature size
    std::vector<int> input_shape_; 
    std::unordered_map<std::string, std::vector<int>> shape_map_;
    mutable std::unique_ptr<float[]>   flatten_buf_;   // only for the very first flatten
    mutable bool                       flattened_ = false;


    uint32_t                                      max_rows_;    ///< Max rows (m) across all MatMul layers
    uint32_t                                      output_rows_; ///< Rows of the final (last MatMul) layer
    size_t                                        last_matmul_idx_;
    uint32_t                                      simd_w;
    bool                                          use_mask;
    std::string                                   input_name_;
    std::string                                   output_name_;
    // per‐op helpers, return a freshly‐allocated buffer of shape [rows×C]
    RunResult applyMatMul            (const MatMulAttr&   , const float* src, uint32_t B, float* out_buf = nullptr) const;
    RunResult applyMatMulRelu        (const MatMulAttr&   , const float* src, uint32_t B, float* out_buf = nullptr) const;
    RunResult applyAdd               (const AddAttr&      , const float* in_A, const float* in_B, uint32_t features, uint32_t B, float* out_buf = nullptr) const;
    RunResult applyRelu              (const ActAttr&      , const float* src, uint32_t features, uint32_t B, float* out_buf = nullptr) const;
    RunResult applySigmoid           (const ActAttr&      , const float* src, uint32_t features, uint32_t B, float* out_buf = nullptr) const;
    RunResult applyTanh              (const ActAttr&      , const float* src, uint32_t features, uint32_t B, float* out_buf = nullptr) const;
    RunResult applyMaxPool           (const PoolAttr&     , const float* src, uint32_t features, uint32_t B, float* out_buf = nullptr) const;
    RunResult applyGlobalAveragePool (const PoolAttr&     , const float* src, uint32_t features, uint32_t B, float* out_buf = nullptr) const;
    RunResult applyFlatten           (const FlattenAttr&  , const float* src, uint32_t features, uint32_t B, float* out_buf = nullptr) const;
    RunResult applyConv              (const ConvAttr&     , const float* src, uint32_t B) const;
};

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
#include <immintrin.h>
#include <unordered_set>

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
  MatMul,  
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
  bool       fuse_relu = false;
};

struct KMap {
  uint32_t cin;
  int32_t  dh;
  int32_t  dw;
};


struct ConvAttr {
  // ELLPACK over K = Cin*kH*kW (rows = Cout)
  Ellpack              E;
  float*               bias_ptr = nullptr;

  // kernel + conv params
  uint32_t             Cin = 0, Cout = 0, kH = 0, kW = 0;
  uint32_t             stride_h = 1, stride_w = 1;
  uint32_t             pad_h = 0, pad_w = 0;
  uint32_t             dil_h = 1, dil_w = 1;
  uint32_t             group = 1; // (only 1 supported for now)

  // spatial geometry
  uint32_t             H_in = 0, W_in = 0;
  uint32_t             H_out = 0, W_out = 0;

  // precomputed map from k∈[0..K) → (cin, dh, dw)
  std::vector<KMap>    kmap;

  bool                 fuse_relu = false;
};
struct PoolAttr {
  // hyperparams
  int kH{1}, kW{1};
  int sH{1}, sW{1};
  int padH0{0}, padW0{0}, padH1{0}, padW1{0};
  bool is_global{false};

  // geometry (input/output)
  uint32_t C{0}, H{0}, W{0};
  uint32_t H_out{0}, W_out{0};
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
    std::unordered_map<std::string, std::vector<int>> flatten_src_shape_;
    std::unordered_map<std::string, std::string> name_alias_;
    // per‐op helpers, return a freshly‐allocated buffer of shape [rows×C]
    RunResult applyMatMul            (const MatMulAttr&   , const float* src, uint32_t B, float* out_buf = nullptr) const;
    RunResult applyAdd               (const AddAttr&      , const float* in_A, const float* in_B, uint32_t features, uint32_t B, float* out_buf = nullptr) const;
    RunResult applyRelu              (const ActAttr&      , const float* src, uint32_t features, uint32_t B, float* out_buf = nullptr) const;
    RunResult applySigmoid           (const ActAttr&      , const float* src, uint32_t features, uint32_t B, float* out_buf = nullptr) const;
    RunResult applyTanh              (const ActAttr&      , const float* src, uint32_t features, uint32_t B, float* out_buf = nullptr) const;
    RunResult applyMaxPool           (const PoolAttr&     , const float* src, uint32_t features, uint32_t B, float* out_buf = nullptr) const;
    RunResult applyGlobalAveragePool (const PoolAttr&     , const float* src, uint32_t features, uint32_t B, float* out_buf = nullptr) const;
    RunResult applyFlatten           (const FlattenAttr&  , const float* src, uint32_t features, uint32_t B, float* out_buf = nullptr) const;
    RunResult applyConv              (const ConvAttr&     , const float* src, uint32_t B) const;
};

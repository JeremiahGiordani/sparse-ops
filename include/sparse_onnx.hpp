// include/sparse_onnx.hpp
#pragma once

#include <string>
#include <cstdint>
#include <memory>
#include <vector>

#include "ellpack_encoder.hpp"   // for Ellpack

/// \file sparse_onnx.hpp
/// C++ interface for loading an ONNX model and running inference
/// via pre-encoded ELLPACK sparse kernels, with fixed (static) batch size
/// inferred from the model’s input shape.

/// Supported layer types in the ONNX graph.
enum class LayerType {
    MatMul,
    Relu,
    Sigmoid,
    Tanh
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
    ///  - \p C: must equal the batch_dim inferred at load time
    ///  - \p output: pointer to float array of shape [output_dim × batch_dim]
    ///
    /// Both input and output are row-major contiguous.
    void run(const float* input, uint32_t C, float* output) const;

    /// Number of rows (m) in the final output (i.e., output_dim).
    uint32_t output_rows() const { return output_rows_; }

private:
    struct Layer {
        LayerType       type;      ///< MatMul or activation
        Ellpack         E;         ///< Pre-encoded sparse weight handle
        float*          bias_ptr;  ///< Pointer into bias_data_ (length = E.m)
    };

    std::vector<Layer>                          layers_;      ///< Execution sequence
    std::unique_ptr<float[]>                    bias_data_;   ///< All biases packed contiguously
    std::vector<std::unique_ptr<float[]>>       layer_bufs_;  ///< One [m × batch_dim] buffer per MatMul
    uint32_t                                     batch_dim_;   ///< Fixed batch size (from ONNX input)
    uint32_t                                     max_rows_;    ///< Max rows (m) across all MatMul layers
    uint32_t                                     output_rows_; ///< Rows of the final (last MatMul) layer
};

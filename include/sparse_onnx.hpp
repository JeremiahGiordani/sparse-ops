// include/sparse_onnx.hpp
#pragma once

#include <string>
#include <cstdint>
#include <memory>
#include <vector>

#include "ellpack_encoder.hpp"   // for Ellpack

/// \file sparse_onnx.hpp
/// C++ interface for loading an ONNX model and running inference
/// via pre-encoded ELLPACK sparse kernels.

/// Supported layer types in the ONNX graph.
enum class LayerType {
    MatMul,
    Relu,
    Sigmoid,
    Tanh
};

/// A simple ONNX-backed sparse inference engine that
/// offloads all weight encoding to load time and uses
/// ELLPACK kernels + SIMD activations at runtime.
class SparseOnnxModel {
public:
    /// Parse the ONNX file at \p onnx_path, extract every
    /// weight & bias initializer, convert weights to ELLPACK,
    /// and build the execution plan.
    explicit SparseOnnxModel(const std::string& onnx_path);

    /// Run inference:
    ///   - \p input: pointer to float array of shape [input_dim * C]
    ///   - \p C: number of columns per activation (batch size or channel count)
    ///   - \p output: pointer to float array of shape [output_dim * C]
    ///
    /// Both input and output are row-major contiguous.
    void run(const float* input, uint32_t C, float* output) const;

    /// Returns the number of rows (m) in the final output.
    uint32_t output_rows() const { return output_rows_; }

private:
    struct Layer {
        LayerType       type;      ///< Operation type (MatMul or activation)
        Ellpack         E;         ///< Pre-encoded sparse weight handle
        float*          bias_ptr;  ///< Pointer into bias_data_ (length = E.m)
    };

    std::vector<Layer>           layers_;      ///< Sequence of all layers
    std::unique_ptr<float[]>     bias_data_;   ///< Contiguous storage for all biases
    mutable std::unique_ptr<float[]> buf1_, buf2_; ///< Ping-pong workspaces
    mutable size_t               buf_cap_ = 0;
    uint32_t                     max_rows_;    ///< Max rows (m) across all layers
    uint32_t                     output_rows_; ///< Rows of the final layer
};

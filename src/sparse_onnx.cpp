// src/sparse_onnx.cpp

#include "sparse_onnx.hpp"

#include <onnx.pb.h>          // ONNX protobuf definitions
#include <fstream>
#include <unordered_map>
#include <stdexcept>
#include <vector>
#include <algorithm>
#include <cstring>

#include "ellpack_encoder.hpp"      // convert_to_ellpack, Ellpack
#include "ellpack_matmul.hpp"       // ellpack_matmul
#include "activations.hpp"          // relu_inplace, sigmoid_inplace, tanh_inplace

namespace {

/// Helper to parse a TensorProto of floats into a flat std::vector<float>.
static void parseTensor(
    const onnx::TensorProto &tp,
    std::vector<float>     &out
) {
    // compute total size
    size_t size = 1;
    for (int i = 0; i < tp.dims_size(); ++i) {
        size *= static_cast<size_t>(tp.dims(i));
    }
    out.resize(size);

    if (tp.has_raw_data() && !tp.raw_data().empty()) {
        // raw_data is a string of bytes
        const std::string &raw = tp.raw_data();
        std::memcpy(out.data(), raw.data(), size * sizeof(float));
    } else if (tp.float_data_size() > 0) {
        // fallback to float_data field
        for (int i = 0; i < tp.float_data_size(); ++i) {
            out[i] = tp.float_data(i);
        }
    } else {
        throw std::runtime_error("ONNX tensor has no weight data");
    }
}

} // anonymous namespace

SparseOnnxModel::SparseOnnxModel(const std::string &onnx_path) {
    // 1) Load ONNX model from file
    onnx::ModelProto model;
    {
        std::ifstream in(onnx_path, std::ios::binary);
        if (!in) {
            throw std::runtime_error("Failed to open ONNX file: " + onnx_path);
        }
        if (!model.ParseFromIstream(&in)) {
            throw std::runtime_error("Failed to parse ONNX model: " + onnx_path);
        }
    }

    const auto &graph = model.graph();

    // 2) Build a map from initializer name -> TensorProto*
    std::unordered_map<std::string, const onnx::TensorProto*> init_map;
    init_map.reserve(graph.initializer_size());
    for (const auto &init : graph.initializer()) {
        init_map[init.name()] = &init;
    }

    // Temporary storage for biases before we pack into one buffer
    struct TempLayer {
        LayerType            type;
        Ellpack              E;
        std::vector<float>   bias;    // empty if none
    };
    std::vector<TempLayer> temp_layers;
    temp_layers.reserve(graph.node_size());

    // Track maximum rows (m) across all MatMul/Gemm layers
    max_rows_ = 0;

    // 3) Walk the graph and build each layer
    for (const auto &node : graph.node()) {
        const std::string &op = node.op_type();
        if (op == "MatMul" || op == "Gemm") {
            // weight is input(1)
            auto itW = init_map.find(node.input(1));
            if (itW == init_map.end()) {
                throw std::runtime_error("Weight initializer '"
                                         + node.input(1)
                                         + "' not found");
            }
            const auto *W_tp = itW->second;

            // parse weight tensor into a flat vector
            std::vector<float> W_data;
            parseTensor(*W_tp, W_data);

            // dims should be [M, N]
            if (W_tp->dims_size() != 2) {
                throw std::runtime_error("Weight tensor must be 2D");
            }
            uint32_t M = static_cast<uint32_t>(W_tp->dims(0));
            uint32_t N = static_cast<uint32_t>(W_tp->dims(1));

            // convert to ELLPACK once, at load time
            Ellpack E = convert_to_ellpack(W_data.data(), M, N);

            // parse optional bias for Gemm
            std::vector<float> bias_vec;
            if (op == "Gemm" && node.input_size() > 2) {
                auto itB = init_map.find(node.input(2));
                if (itB != init_map.end()) {
                    parseTensor(*itB->second, bias_vec);
                    // bias_vec.size() should equal M
                }
            }

            temp_layers.push_back({LayerType::MatMul,
                                   std::move(E),
                                   std::move(bias_vec)});
            max_rows_ = std::max(max_rows_, M);

        } else if (op == "Relu" || op == "Sigmoid" || op == "Tanh") {
            LayerType t = (op == "Relu" ? LayerType::Relu
                          : op == "Sigmoid" ? LayerType::Sigmoid
                          : LayerType::Tanh);
            // no E or bias needed
            temp_layers.push_back({t, Ellpack(0u, 0u, 0u), {}});
        } else {
            throw std::runtime_error("Unsupported ONNX op: " + op);
        }
    }

    // 4) Pack all biases into one contiguous buffer
    size_t total_bias = 0;
    for (auto &tl : temp_layers) {
        total_bias += tl.bias.size();
    }
    bias_data_ = std::unique_ptr<float[]>(new float[total_bias]);
    float *bias_ptr = bias_data_.get();

    // Build final layers_ vector
    layers_.reserve(temp_layers.size());
    size_t bias_offset = 0;
    for (auto &tl : temp_layers) {
        float *ptr = nullptr;
        if (!tl.bias.empty()) {
            ptr = bias_ptr + bias_offset;
            std::memcpy(ptr,
                        tl.bias.data(),
                        tl.bias.size() * sizeof(float));
            bias_offset += tl.bias.size();
        }
        layers_.push_back({ tl.type,
                            std::move(tl.E),
                            ptr });
    }

    // 5) Record output dimension (rows of final layer)
    if (layers_.empty()) {
        throw std::runtime_error("ONNX model contains no supported layers");
    }
    // Walk backwards until we find a MatMul layer
    for (auto it = layers_.rbegin(); it != layers_.rend(); ++it) {
        if (it->type == LayerType::MatMul) {
            output_rows_ = it->E.m;
            break;
        }
    }
    if (output_rows_ == 0) {
        throw std::runtime_error(
        "Could not determine output dimension: no MatMul layer found");
    }
}

void SparseOnnxModel::run(
    const float *input,
    uint32_t      C,
    float       *output
) const {
    // Determine initial input size (n × C)
    uint32_t n = layers_.front().E.n;
    std::vector<float> cur_buf(input, input + size_t(n)*C);
    std::vector<float> next_buf;  // only used for matmuls

    // Process each layer in sequence
    for (const auto &L : layers_) {
        if (L.type == LayerType::MatMul) {
            // Sparse mat-mul: [n×C] → [m×C]
            uint32_t m = L.E.m;
            next_buf.resize(size_t(m) * C);

            ellpack_matmul(
                L.E,
                cur_buf.data(),
                C,
                L.bias_ptr,
                next_buf.data()
            );

            // Move result into cur_buf
            cur_buf.swap(next_buf);
            n = m;  // now next layer sees input of size m×C

        } else {
            // Activation: apply directly to cur_buf (size = n×C)
            size_t len = cur_buf.size();
            switch (L.type) {
              case LayerType::Relu:
                relu_inplace(cur_buf.data(), len);
                break;
              case LayerType::Sigmoid:
                sigmoid_inplace(cur_buf.data(), len);
                break;
              case LayerType::Tanh:
                tanh_inplace(cur_buf.data(), len);
                break;
              default:
                break;
            }
            // no buffer swap, shape stays (n×C)
        }
    }

    // Copy the final cur_buf (size = output_rows × C) into user output
    std::memcpy(output,
                cur_buf.data(),
                size_t(output_rows_) * C * sizeof(float));
}

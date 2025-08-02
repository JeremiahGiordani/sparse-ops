// src/sparse_onnx.cpp

#include "sparse_onnx.hpp"

#include <onnx.pb.h>          // ONNX protobuf definitions
#include <fstream>
#include <unordered_map>
#include <stdexcept>
#include <vector>
#include <algorithm>
#include <cstring>
#include <cstddef>
#include <limits>
#include <omp.h>

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

    // 2) Build name→initializer map
    std::unordered_map<std::string, const onnx::TensorProto*> init_map;
    init_map.reserve(graph.initializer_size());
    for (const auto &init : graph.initializer()) {
        init_map[init.name()] = &init;
    }

    // 3) Infer fixed batch dimension from first non-initializer input
    batch_dim_ = 1;
    for (const auto &vi : graph.input()) {
        if (init_map.count(vi.name())) {
            // this input is actually an initializer (constant), skip
            continue;
        }
        if (!vi.has_type() || !vi.type().has_tensor_type()) {
            throw std::runtime_error("Input '" + vi.name() +
                "' has no tensor_type");
        }
        const auto &shape = vi.type().tensor_type().shape();
        if (shape.dim_size() != 2) {
            throw std::runtime_error("Expected 2D input (batch × features), got "
                + std::to_string(shape.dim_size()) + "D for '" + vi.name() + "'");
        }
        const auto &dim0 = shape.dim(0);
        if (!dim0.has_dim_value()) {
            throw std::runtime_error("Dynamic batch dimension not supported");
        }
        batch_dim_ = static_cast<uint32_t>(dim0.dim_value());
        break;
    }

    // Temporary staging for (type, ELLPACK, bias_vector)
    struct TempLayer {
        LayerType          type;
        Ellpack            E;
        std::vector<float> bias;  // may be empty
    };
    std::vector<TempLayer> temp_layers;
    temp_layers.reserve(graph.node_size());
    max_rows_ = 0;

    // 4) Walk the graph and record layers
    for (const auto &node : graph.node()) {
        const std::string &op = node.op_type();
        if (op == "MatMul" || op == "Gemm") {
            // weight initializer → input(1)
            auto itW = init_map.find(node.input(1));
            if (itW == init_map.end()) {
                throw std::runtime_error("Weight initializer '" +
                    node.input(1) + "' not found");
            }
            const onnx::TensorProto* W_tp = itW->second;

            // parse weight into vector
            std::vector<float> W_data;
            parseTensor(*W_tp, W_data);

            if (W_tp->dims_size() != 2) {
                throw std::runtime_error("Weight tensor must be 2D");
            }
            uint32_t M = static_cast<uint32_t>(W_tp->dims(0));
            uint32_t N = static_cast<uint32_t>(W_tp->dims(1));

            // ELLPACK encode
            Ellpack E = convert_to_ellpack(W_data.data(), M, N);

            // optional bias for Gemm
            std::vector<float> bias_vec;
            if (op == "Gemm" && node.input_size() > 2) {
                auto itB = init_map.find(node.input(2));
                if (itB != init_map.end()) {
                    parseTensor(*itB->second, bias_vec);
                }
            }

            temp_layers.push_back({LayerType::MatMul,
                                   std::move(E),
                                   std::move(bias_vec)});
            max_rows_ = std::max(max_rows_, M);

        } else if (op == "Relu" || op == "Sigmoid" || op == "Tanh") {
            LayerType t = (op == "Relu"    ? LayerType::Relu
                          : op == "Sigmoid"? LayerType::Sigmoid
                                            : LayerType::Tanh);
            temp_layers.push_back({t, Ellpack(0u,0u,0u), {}});
        } else {
            throw std::runtime_error("Unsupported ONNX op: " + op);
        }
    }

    // 5) Pack all biases into one buffer
    size_t total_bias = 0;
    for (auto &tl : temp_layers) {
        total_bias += tl.bias.size();
    }
    bias_data_.reset(new float[total_bias]);
    float *bias_ptr = bias_data_.get();

    layers_.reserve(temp_layers.size());
    size_t bias_off = 0;
    for (auto &tl : temp_layers) {
        float* ptr = nullptr;
        if (!tl.bias.empty()) {
            ptr = bias_ptr + bias_off;
            std::memcpy(ptr, tl.bias.data(), tl.bias.size()*sizeof(float));
            bias_off += tl.bias.size();
        }
        layers_.push_back({tl.type, std::move(tl.E), ptr});
    }

    // 6) Determine output_rows_ = m of last MatMul
    output_rows_ = 0;
    for (auto it = layers_.rbegin(); it != layers_.rend(); ++it) {
        if (it->type == LayerType::MatMul) {
            output_rows_ = it->E.m;
            break;
        }
    }
    if (output_rows_ == 0) {
        throw std::runtime_error("No MatMul layer found for output dimension");
    }

    // 7) Allocate one scratch buffer per MatMul layer
    resize_buffers(batch_dim_);
}

void SparseOnnxModel::resize_buffers(uint32_t new_C) const {
    // Update batch size
    batch_dim_ = new_C;

    // Compute per-layer offsets into one big scratch array
    offsets_.clear();
    offsets_.reserve(layers_.size());
    size_t total_floats = 0;

    for (const auto &L : layers_) {
        if (L.type == LayerType::MatMul) {
            offsets_.push_back(total_floats);
            total_floats += size_t(L.E.m) * batch_dim_;
        } else {
            // activations don’t get their own scratch buffer
            offsets_.push_back(SIZE_MAX);
        }
    }

    // (Re)allocate the arena
    arena_buf_.reset(new float[total_floats]);
}


void SparseOnnxModel::run(
    const float *input,
    uint32_t      C,
    float       *output
) const {
    // Ensure the batch size matches the fixed dimension we inferred
    if (C != batch_dim_) {
        resize_buffers(C);
    }
    omp_set_schedule(omp_sched_static, 1);

    // 'src' always points to the current activation buffer
    const float* src = input;

    for (size_t i = 0; i < layers_.size(); ++i) {
        const Layer &L = layers_[i];

        if (L.type == LayerType::MatMul) {
            // MatMul: write into the pre‐allocated buffer for this layer
            bool is_last = (L.E.m == output_rows_);
            float *dst;
            if (is_last) {
                dst = output;           // write directly into user's buffer
            } else {
                size_t off = offsets_[i];
                dst = arena_buf_.get() + off;
            }

            ellpack_matmul(
                L.E,            // the ELLPACK handle
                src,            // input [n × C]
                C,              // batch size
                L.bias_ptr,     // length = E.m
                dst             // writes [m × C]
            );

            // the next layer reads from here
            src = dst;

        } else {
            // Activation: apply in-place on 'src' (size = prev_m × C)
            uint32_t n = (i == 0
                          ? layers_[0].E.n        // input_dim for first layer
                          : layers_[i-1].E.m);    // rows of previous MatMul
            size_t len = static_cast<size_t>(n) * C;
            switch (L.type) {
              case LayerType::Relu:
                relu_inplace(const_cast<float*>(src), len);
                break;
              case LayerType::Sigmoid:
                sigmoid_inplace(const_cast<float*>(src), len);
                break;
              case LayerType::Tanh:
                tanh_inplace(const_cast<float*>(src), len);
                break;
              default:
                break;
            }
            // 'src' remains the same buffer for the next layer
        }
    }
}
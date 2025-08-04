// src/sparse_onnx.cpp

#include "sparse_onnx.hpp"

#include <onnx.pb.h>          // ONNX protobuf definitions
#include <fstream>
#include <unordered_map>
#include <stdexcept>
#include <cstddef>
#include <omp.h>
#include <cstdlib>


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
    // 1) Load ONNX model
    onnx::ModelProto model;
    {
        std::ifstream in(onnx_path, std::ios::binary);
        if (!in || !model.ParseFromIstream(&in)) {
            throw std::runtime_error("Failed to open/parse ONNX file: " + onnx_path);
        }
    }
    const auto &graph = model.graph();

    // 2) Build name→initializer map
    std::unordered_map<std::string, const onnx::TensorProto*> init_map;
    init_map.reserve(graph.initializer_size());
    for (const auto &init : graph.initializer()) {
        init_map[init.name()] = &init;
    }

    // Find the real model input (skip initializers)
    input_name_.clear();
    for (const auto &vi : graph.input()) {
        const std::string &nm = vi.name();
        if (init_map.count(nm)) {
            continue;
        }
        input_name_ = nm;
        break;
    }
    if (input_name_.empty()) {
        throw std::runtime_error("No non-initializer input found in ONNX graph");
    }

    // Record the (single) graph output
    if (graph.output_size() != 1) {
        throw std::runtime_error("Expected exactly one graph output");
    }
    output_name_ = graph.output(0).name();

    // 3) Infer fixed batch_dim_ from first non‐initializer input
    batch_dim_ = 1;
    for (const auto &vi : graph.input()) {
        if (init_map.count(vi.name())) continue;  // skip constant inputs
        auto &shape = vi.type().tensor_type().shape();
        if (shape.dim_size()!=2 || !shape.dim(0).has_dim_value())
            throw std::runtime_error("Expected static 2D input");
        batch_dim_    = static_cast<uint32_t>(shape.dim(0).dim_value());
        in_features_  = static_cast<uint32_t>(shape.dim(1).dim_value());
        break;
    }

    // 4) Stage 1: Collect all bias vectors so we can pack them into one big buffer
    struct BiasPlan { size_t layer_idx; std::vector<float> data; };
    std::vector<BiasPlan> bias_plans;
    bias_plans.reserve(graph.node_size());

    // 5) Walk the graph & build layers_
    layers_.reserve(graph.node_size());
    for (size_t idx = 0; idx < graph.node_size(); ++idx) {
        const auto &node = graph.node(idx);
        const auto &op   = node.op_type();

        // ——— Fuse MatMul/Gemm → Relu ———
        if ((op=="MatMul" || op=="Gemm")
            && idx+1<graph.node_size()
            && graph.node(idx+1).op_type()=="Relu")
        {
            // --- parse weight initializer ---
            auto itW = init_map.find(node.input(1));
            if (itW==init_map.end())
                throw std::runtime_error("Weight '" + node.input(1) + "' not found");
            auto *W_tp = itW->second;

            // --- unpack weight data & dims ---
            std::vector<float> W_data;
            parseTensor(*W_tp, W_data);
            uint32_t M = W_tp->dims(0), N = W_tp->dims(1);

            // --- encode to ELLPACK ---
            Ellpack E = convert_to_ellpack(W_data.data(), M, N);

            // --- optional bias (only for Gemm) ---
            std::vector<float> bdata;
            if (op=="Gemm" && node.input_size()>2) {
                if (auto itB = init_map.find(node.input(2)); itB != init_map.end()) {
                    parseTensor(*itB->second, bdata);
                }
            }

            // --- record layer + bias plan ---
            MatMulAttr matmul_attr{ std::move(E), nullptr };

            // build the fused layer
            layers_.push_back({
                LayerType::MatMul,               // category
                LayerOp::MatMulRelu,             // specific op
                std::move(matmul_attr),          // payload
                /* inputs: only the activation tensor */ 
                { node.input(0) },
                /* outputs: the Relu’s single output */
                { graph.node(idx+1).output(0) }
            });
            bias_plans.push_back({ layers_.size()-1, std::move(bdata) });

            ++idx;  // skip the Relu node
            continue;
        }

        // ——— Plain MatMul / Gemm ———
        if (op=="MatMul" || op=="Gemm") {
            auto itW = init_map.find(node.input(1));
            if (itW==init_map.end())
                throw std::runtime_error("Weight '" + node.input(1) + "' not found");
            auto *W_tp = itW->second;

            std::vector<float> W_data;
            parseTensor(*W_tp, W_data);
            uint32_t M = W_tp->dims(0), N = W_tp->dims(1);
            Ellpack E = convert_to_ellpack(W_data.data(), M, N);

            std::vector<float> bdata;
            if (op=="Gemm" && node.input_size()>2) {
                if (auto itB = init_map.find(node.input(2)); itB != init_map.end()) {
                    parseTensor(*itB->second, bdata);
                }
            }

            MatMulAttr matmul_attr{ std::move(E), nullptr };
            layers_.push_back({
                LayerType::MatMul,
                LayerOp::MatMul,
                std::move(matmul_attr),
                /* inputs: only the activation tensor */
                { node.input(0) },
                /* outputs: this node’s output */
                { node.output(0) }
            });
            bias_plans.push_back({ layers_.size()-1, std::move(bdata) });
            continue;
        }

        // ——— Elementwise Add ———
        if (op=="Add") {
            layers_.push_back({
                LayerType::Elementwise,
                LayerOp::Add,
                AddAttr{},                // no payload
                /* both inputs to add */
                { node.input(0), node.input(1) },
                /* single output */
                { node.output(0) }
            });
            continue;
        }

        // ——— Activations ———
        if (op=="Relu" || op=="Sigmoid" || op=="Tanh") {
            LayerOp lop = (op=="Relu"    ? LayerOp::Relu
                           : op=="Sigmoid"? LayerOp::Sigmoid
                                          : LayerOp::Tanh);
            layers_.push_back({
                LayerType::Activation,
                lop,
                ActAttr{},                // no payload
                /* activation input */
                { node.input(0) },
                /* activation output */
                { node.output(0) }
            });
            continue;
        }

        // ——— Pooling ———
        if (op=="MaxPool" || op=="GlobalAveragePool") {
            onnx::NodeProto const &n = node;
            PoolAttr p;
            p.is_global = (op=="GlobalAveragePool");
            // read attributes…
            for (auto &attr : n.attribute()) {
                if      (attr.name()=="kernel_shape") p.kernel_shape = {attr.ints().Get(0), attr.ints().Get(1)};
                else if (attr.name()=="strides")      p.strides      = {attr.ints().Get(0), attr.ints().Get(1)};
                else if (attr.name()=="pads")         p.pads         = {attr.ints().Get(0), attr.ints().Get(1),
                                                                          attr.ints().Get(2), attr.ints().Get(3)};
            }
            LayerOp lop = (op=="MaxPool"
                         ? LayerOp::MaxPool
                         : LayerOp::GlobalAveragePool);
            layers_.push_back({
                LayerType::Pool,
                lop,
                std::move(p),
                /* pooling input */
                { node.input(0) },
                /* pooling output */
                { node.output(0) }
            });
            continue;
        }

        // ——— Flatten ———
        if (op=="Flatten") {
            int axis = 1;
            for (auto &a : node.attribute()) {
                if (a.name()=="axis") { axis = static_cast<int>(a.i()); break; }
            }
            layers_.push_back({
                LayerType::Reshape,
                LayerOp::Flatten,
                FlattenAttr{axis},
                /* flatten input */
                { node.input(0) },
                /* flatten output */
                { node.output(0) }
            });
            continue;
        }

        if (op == "Conv") {
            // 1) Load & parse the raw 4D kernel
            auto itW = init_map.find(node.input(1));
            if (itW == init_map.end()) {
                throw std::runtime_error("Conv weight initializer not found");
            }
            const onnx::TensorProto* W_tp = itW->second;
            std::vector<float> raw_kernel;
            parseTensor(*W_tp, raw_kernel);  // length = Cout*Cin*kH*kW

            // 2) Extract dimensions
            std::array<int,4> dims = {
                int(W_tp->dims(0)),  // Cout
                int(W_tp->dims(1)),  // Cin
                int(W_tp->dims(2)),  // kH
                int(W_tp->dims(3))   // kW
            };
            int Cout = dims[0], Cin = dims[1], kH = dims[2], kW = dims[3];

            // 3) Build the dense weight matrix of shape [Cout × (Cin*kH*kW)]
            size_t M = size_t(Cout);
            size_t N = size_t(Cin) * kH * kW;
            std::vector<float> weight_mat;
            weight_mat.reserve(M*N);
            // raw_kernel is already in [Cout][Cin][kH][kW] row‐major order,
            // so a straight copy yields the desired 2D layout:
            weight_mat.insert(weight_mat.end(),
                            raw_kernel.begin(),
                            raw_kernel.end());

            // 4) Encode into ELLPACK
            Ellpack E = convert_to_ellpack(weight_mat.data(), static_cast<uint32_t>(M), static_cast<uint32_t>(N));

            // 5) Optional bias
            std::vector<float> bias_vec;
            if (node.input_size() > 2) {
                if (auto itB = init_map.find(node.input(2)); itB != init_map.end()) {
                    parseTensor(*itB->second, bias_vec);
                }
            }

            // 6) Create the ConvAttr and push the layer
            ConvAttr c {
                std::move(E),          // ELLPACK handle
                nullptr,               // bias_ptr
                dims,                  // kernel_dims
                {0,0,0,0},             // pads
                {1,1},                 // strides
                {1,1},                 // dilations
                1                      // group
            };

            // Overwrite defaults from the node’s attributes
            for (const auto &A : node.attribute()) {
                const std::string &n = A.name();
                if (n == "pads" && A.ints_size() == 4) {
                    c.pads = {
                        int(A.ints(0)), int(A.ints(1)),
                        int(A.ints(2)), int(A.ints(3))
                    };
                }
                else if (n == "strides" && A.ints_size() == 2) {
                    c.strides = {
                        int(A.ints(0)), int(A.ints(1))
                    };
                }
                else if (n == "dilations" && A.ints_size() == 2) {
                    c.dilations = {
                        int(A.ints(0)), int(A.ints(1))
                    };
                }
                else if (n == "group") {
                    c.group = static_cast<int>(A.i());
                }
            }

            layers_.push_back({
                LayerType::Conv,
                LayerOp::Conv,
                std::move(c),
                /* inputs  */ { node.input(0) },
                /* outputs */ { node.output(0) }
            });

            // 7) Schedule its bias for packing
            bias_plans.push_back({ layers_.size()-1, std::move(bias_vec) });

            continue;
        }

        throw std::runtime_error("Unsupported ONNX op: " + op);
    }

    // Sum up every bias vector’s length
    size_t total_bias = 0;
    for (auto &bp : bias_plans) {
        total_bias += bp.data.size();
    }

    // Allocate one contiguous block
    bias_data_.reset(new float[total_bias]);
    float* bptr = bias_data_.get();

    // Copy each bias vector into the block and update the Layer’s payload
    for (auto &bp : bias_plans) {
        size_t idx = bp.layer_idx;
        size_t len = bp.data.size();

        // memcpy into the big buffer
        std::memcpy(bptr, bp.data.data(), len * sizeof(float));

        // Point the payload’s bias_ptr at this slice
        Layer &L = layers_[idx];
        if (std::holds_alternative<MatMulAttr>(L.attr)) {
            auto &ma = std::get<MatMulAttr>(L.attr);
            ma.bias_ptr = bptr;
        }
        else if (std::holds_alternative<ConvAttr>(L.attr)) {
            auto &ca = std::get<ConvAttr>(L.attr);
            ca.bias_ptr = bptr;
        }

        bptr += len;  // advance to next free slot
    }

    // Last MatMul layer index
    last_matmul_idx_ = 0;
    for (size_t i = 0; i < layers_.size(); ++i) {
        if (layers_[i].type == LayerType::MatMul) {
            last_matmul_idx_ = i;
        }
    }

    // Number of output rows = m of the final MatMul
    output_rows_ = 0;
    for (auto it = layers_.rbegin(); it != layers_.rend(); ++it) {
        if (it->type == LayerType::MatMul) {
            auto &ma = std::get<MatMulAttr>(it->attr);
            output_rows_ = ma.E.m;
            break;
        }
    }
    if (output_rows_ == 0) {
        throw std::runtime_error("No MatMul layer found for output dimension");
    }

    bool use_avx512   = supports_avx512();
    simd_w   = use_avx512 ? 16u : 8u;
    use_mask     = (batch_dim_ % simd_w) != 0;
}

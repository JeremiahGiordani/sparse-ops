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

// Reorder columns of W (M x N) *in place into* newW, mapping row-major (W-fast) to Fortran (C-fast).
static void reorder_fc_columns_for_fortran_flatten(
    const std::vector<float>& W_in, // M*N
    uint32_t M, uint32_t C, uint32_t H, uint32_t Wd, // (Wd = width)
    std::vector<float>& W_out       // M*N (result)
) {
    const uint32_t N = C * H * Wd;
    W_out.assign(M * size_t(N), 0.0f);

    for (uint32_t c = 0; c < C; ++c) {
        for (uint32_t h = 0; h < H; ++h) {
            for (uint32_t w = 0; w < Wd; ++w) {
                const uint32_t rm = c*H*Wd + h*Wd + w;          // PyTorch flatten order
                const uint32_t cm = c + C*(h + H*w);            // our Fortran flatten order
                const size_t    src_col = rm;
                const size_t    dst_col = cm;
                // copy column rm -> cm
                for (uint32_t m = 0; m < M; ++m) {
                    W_out[m*size_t(N) + dst_col] = W_in[m*size_t(N) + src_col];
                }
            }
        }
    }
}

static void transpose_rc(const std::vector<float>& A, uint32_t R, uint32_t C,
                         std::vector<float>& AT) {
    AT.resize(size_t(R) * C);
    for (uint32_t r = 0; r < R; ++r)
        for (uint32_t c = 0; c < C; ++c)
            AT[size_t(c)*R + r] = A[size_t(r)*C + c];
}

}


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
    auto resolve = [this](const std::string& nm) -> std::string {
        auto it = name_alias_.find(nm);
        if (it == name_alias_.end()) return nm;
        // find root
        std::string root = it->second;
        while (true) {
            auto jt = name_alias_.find(root);
            if (jt == name_alias_.end()) break;
            root = jt->second;
        }
        // path compression
        if (root != it->second) {
            // store canonical directly for next time
            name_alias_[nm] = root;
        }
        return root;
    };
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
        input_name_ = resolve(nm);
        break;
    }
    if (input_name_.empty()) {
        throw std::runtime_error("No non-initializer input found in ONNX graph");
    }

    // Record the (single) graph output
    if (graph.output_size() != 1) {
        throw std::runtime_error("Expected exactly one graph output");
    }
    output_name_ = resolve(graph.output(0).name());

    // 3) Infer fixed batch_dim_ from first non‐initializer input
    batch_dim_ = 1;
    for (const auto &vi : graph.input()) {
        if (init_map.count(vi.name())) continue;  // skip constant inputs
        auto &shape_proto = vi.type().tensor_type().shape();
        input_shape_.resize(shape_proto.dim_size());
        for (int i = 0; i < shape_proto.dim_size(); ++i) {
            if (!shape_proto.dim(i).has_dim_value())
                throw std::runtime_error("Dynamic dims not supported");
            input_shape_[i] = shape_proto.dim(i).dim_value();
        }
        
        if (input_shape_.size() == 2) {
            // 2D input: [batch × features]
            in_features_ = static_cast<uint32_t>(input_shape_[1]);
        }
        else if (input_shape_.size() > 2) {
            // ND input: flatten all but the batch dim
            uint32_t prod = 1;
            for (size_t d = 1; d < input_shape_.size(); ++d) {
                prod *= static_cast<uint32_t>(input_shape_[d]);
            }
            in_features_ = prod;
        }
        else {
            throw std::runtime_error("Input tensor must have rank ≥ 2");
        }
        batch_dim_ = static_cast<uint32_t>(input_shape_[0]);
        shape_map_[input_name_] = input_shape_;  // record full shape if you need it later
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
        const std::string &out = node.output(0);

        if (op=="MatMul" || op=="Gemm") {
            auto itW = init_map.find(resolve(node.input(1)));
            if (itW==init_map.end())
                throw std::runtime_error("Weight '" + resolve(node.input(1)) + "' not found");
            const onnx::TensorProto* W_tp = itW->second;

            std::vector<float> W_raw;
            parseTensor(*W_tp, W_raw);
            if (W_tp->dims_size() != 2) throw std::runtime_error("FC weight must be 2D");

            uint32_t d0 = (uint32_t)W_tp->dims(0);
            uint32_t d1 = (uint32_t)W_tp->dims(1);

            // Normalize to MxN = (out_features x in_features)
            uint32_t M = 0, N = 0;
            std::vector<float> W_MxN;  // row-major (M x N)

            if (op=="MatMul") {
                // ONNX MatMul: input is (B x N), weight is (N x M) -> output (B x M)
                N = d0; M = d1;
                transpose_rc(W_raw, /*R=*/N, /*C=*/M, /*out*/W_MxN); // -> (M x N)
            } else { // Gemm
                // Defaults: PyTorch tends to export with transB=1 and B=(M x N)
                int64_t transB = 1; // ONNX default is 0, but PyTorch sets 1 for Linear
                for (const auto& A : node.attribute()) {
                    if (A.name()=="transB") transB = A.i();
                }
                if (transB == 1) {
                    // W is already (M x N)
                    M = d0; N = d1;
                    W_MxN = std::move(W_raw);
                } else {
                    // W is (N x M) -> transpose to (M x N)
                    N = d0; M = d1;
                    transpose_rc(W_raw, /*R=*/N, /*C=*/M, /*out*/W_MxN);
                }
            }

            // If input is output of a Flatten over 4D (B,C,H,W), permute columns once
            const std::string x_name = resolve(node.input(0));
            auto fit = flatten_src_shape_.find(x_name);
            if (fit != flatten_src_shape_.end()) {
                const auto& fsrc = fit->second; // expect {B,C,H,W}
                if (fsrc.size()==4) {
                    const uint32_t C = (uint32_t)fsrc[1];
                    const uint32_t H = (uint32_t)fsrc[2];
                    const uint32_t Wd= (uint32_t)fsrc[3];
                    if (C*H*Wd == N) {
                        std::vector<float> W_perm;
                        reorder_fc_columns_for_fortran_flatten(W_MxN, M, C, H, Wd, W_perm);
                        W_MxN.swap(W_perm);
                    }
                }
            }

            // Encode to ELLPACK (M x N)
            Ellpack E = convert_to_ellpack(W_MxN.data(), M, N);

            // Bias (Gemm may have it; MatMul typically not)
            std::vector<float> bdata;
            if (op=="Gemm" && node.input_size()>2) {
                if (auto itB = init_map.find(resolve(node.input(2))); itB != init_map.end()) {
                    parseTensor(*itB->second, bdata);
                }
            }

            MatMulAttr ma{ std::move(E), /*bias_ptr*/nullptr, /*fuse_relu*/false };
            // If you use the “bias packing after fusion” approach, stash raw bias here instead.

            layers_.push_back({
                LayerType::MatMul,
                LayerOp::MatMul,
                std::move(ma),
                { x_name },
                { node.output(0) }
            });

            // If you’re still doing your earlier bias packing: only push if non-empty
            if (!bdata.empty()) {
                bias_plans.push_back({ layers_.size()-1, std::move(bdata) });
            }

            shape_map_[out] = { int(batch_dim_), int(M) };
            continue;
        }

        // ——— Elementwise Add ———
        if (op=="Add") {
            const std::string a0 = resolve(node.input(0));
            const std::string a1 = resolve(node.input(1));
            layers_.push_back({
                LayerType::Elementwise,
                LayerOp::Add,
                AddAttr{},                // no payload
                /* both inputs to add */
                { a0, a1 },
                /* single output */
                { node.output(0) }
            });
            shape_map_[out] = shape_map_.at(a0);
            continue;
        }

        // ——— Activations ———
        if (op=="Relu" || op=="Sigmoid" || op=="Tanh") {
            LayerOp lop = (op=="Relu"    ? LayerOp::Relu
                           : op=="Sigmoid"? LayerOp::Sigmoid
                                          : LayerOp::Tanh);
            const std::string x_name = resolve(node.input(0));
            layers_.push_back({
                LayerType::Activation,
                lop,
                ActAttr{},                // no payload
                /* activation input */
                { x_name },
                /* activation output */
                { node.output(0) }
            });
            shape_map_[out] = shape_map_.at(x_name);
            continue;
        }

        // ——— Pooling ———
        if (op=="MaxPool" || op=="GlobalAveragePool") {
            onnx::NodeProto const &n = node;
            PoolAttr p;
            p.is_global = (op=="GlobalAveragePool");

            // Defaults
            p.kH = p.kW = 1;
            p.sH = p.sW = 1;
            p.padH0 = p.padH1 = p.padW0 = p.padW1 = 0;

            for (const auto &attr : n.attribute()) {
                const auto &nm = attr.name();
                if      (nm=="kernel_shape" && attr.ints_size()==2) { p.kH = int(attr.ints(0)); p.kW = int(attr.ints(1)); }
                else if (nm=="strides"      && attr.ints_size()==2) { p.sH = int(attr.ints(0)); p.sW = int(attr.ints(1)); }
                else if (nm=="pads"         && attr.ints_size()==4) {
                    p.padH0 = int(attr.ints(0)); p.padW0 = int(attr.ints(1));
                    p.padH1 = int(attr.ints(2)); p.padW1 = int(attr.ints(3));
                }
                // NOTE: ignoring dilations/ceil_mode; can add if you need
            }

            const std::string x_name = resolve(node.input(0));
            const auto in_shape = shape_map_.at(x_name); // expect {B,C,H,W}
            if (in_shape.size() != 4) throw std::runtime_error("Pool input must be 4D");
            const int B  = in_shape[0];
            p.C = (uint32_t)in_shape[1];
            p.H = (uint32_t)in_shape[2];
            p.W = (uint32_t)in_shape[3];

            if (p.is_global) {
                p.H_out = 1; p.W_out = 1;
                shape_map_[out] = { B, int(p.C) };           // we expose (B,C) (Fortran) for GAP
            } else {
                p.H_out = (uint32_t)((int(p.H) + p.padH0 + p.padH1 - p.kH) / p.sH + 1);
                p.W_out = (uint32_t)((int(p.W) + p.padW0 + p.padW1 - p.kW) / p.sW + 1);
                shape_map_[out] = { B, int(p.C), int(p.H_out), int(p.W_out) };
            }

            layers_.push_back({
                LayerType::Pool,
                (op=="MaxPool" ? LayerOp::MaxPool : LayerOp::GlobalAveragePool),
                std::move(p),
                { x_name },
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
            const std::string x_name = resolve(node.input(0));
            
            layers_.push_back({
                LayerType::Reshape,
                LayerOp::Flatten,
                FlattenAttr{axis},
                { x_name },
                { node.output(0) }
            });

            auto in_shape = shape_map_.at(x_name);  // expect {B,C,H,W}
            // Remember the *input* shape so FC can realign its columns
            flatten_src_shape_[ node.output(0) ] = in_shape;

            std::vector<int> flat = { in_shape[0] };
            int prod = 1;
            for (int i = axis; i < (int)in_shape.size(); ++i) prod *= in_shape[i];
            flat.push_back(prod);
            shape_map_[out] = flat;
            continue;
        }

        if (op == "Conv") {
            // 1) Load raw kernel (Cout, Cin, kH, kW)
            auto itW = init_map.find(resolve(node.input(1)));
            if (itW == init_map.end()) throw std::runtime_error("Conv weight initializer not found");
            const onnx::TensorProto* W_tp = itW->second;

            std::vector<float> raw_kernel;
            parseTensor(*W_tp, raw_kernel);

            const int Cout = int(W_tp->dims(0));
            const int Cin  = int(W_tp->dims(1));
            const int kH   = int(W_tp->dims(2));
            const int kW   = int(W_tp->dims(3));

            // 2) Flatten weights to 2D [Cout × (Cin*kH*kW)] in cin-major, then kh, then kw
            //    (This matches the KMap order we’ll build below.)
            const uint32_t K = uint32_t(Cin) * kH * kW;
            std::vector<float> weight_mat; weight_mat.reserve(size_t(Cout) * K);
            // raw_kernel is [Cout][Cin][kH][kW] row-major already in that nesting order,
            // so a straight copy preserves the (cin,kh,kw) inner iteration.
            weight_mat.insert(weight_mat.end(), raw_kernel.begin(), raw_kernel.end());

            // 3) ELLPACK encode (rows=Cout, cols=K)
            Ellpack E = convert_to_ellpack(weight_mat.data(),
                                        static_cast<uint32_t>(Cout),
                                        K);

            // 4) Read conv attributes
            uint32_t pad_h=0, pad_w=0, stride_h=1, stride_w=1, dil_h=1, dil_w=1, group=1;
            for (const auto &A : node.attribute()) {
                const std::string &n = A.name();
                if (n == "pads"     && A.ints_size() == 4) { pad_h = A.ints(0); pad_w = A.ints(1); /* end pads ignored */ }
                else if (n == "strides"   && A.ints_size() == 2) { stride_h = A.ints(0); stride_w = A.ints(1); }
                else if (n == "dilations" && A.ints_size() == 2) { dil_h = A.ints(0); dil_w = A.ints(1); }
                else if (n == "group") { group = static_cast<uint32_t>(A.i()); }
            }
            if (group != 1) throw std::runtime_error("Grouped conv not supported (yet)");

            // 5) Input geometry
            const auto in_shape = shape_map_.at(resolve(node.input(0))); // {B, Cin, H_in, W_in}
            if (in_shape.size() != 4) throw std::runtime_error("Conv input must be 4D");
            const uint32_t B    = static_cast<uint32_t>(in_shape[0]);
            const uint32_t Cin_ = static_cast<uint32_t>(in_shape[1]);
            const uint32_t H_in = static_cast<uint32_t>(in_shape[2]);
            const uint32_t W_in = static_cast<uint32_t>(in_shape[3]);
            if (Cin_ != static_cast<uint32_t>(Cin)) throw std::runtime_error("Conv Cin mismatch");

            // 6) Output geometry
            const uint32_t H_out = (H_in + 2*pad_h - (uint32_t(dil_h)*(kH-1) + 1)) / stride_h + 1;
            const uint32_t W_out = (W_in + 2*pad_w - (uint32_t(dil_w)*(kW-1) + 1)) / stride_w + 1;

            // 7) Optional bias
            std::vector<float> bias_vec;
            if (node.input_size() > 2) {
                if (auto itB = init_map.find(resolve(node.input(2))); itB != init_map.end()) {
                    parseTensor(*itB->second, bias_vec);
                }
            }

            // 8) Build kmap (cin-major, then kh, kw) with pre-padded offsets
            std::vector<KMap> kmap; kmap.reserve(K);
            for (uint32_t c = 0; c < (uint32_t)Cin; ++c) {
                for (uint32_t kh = 0; kh < (uint32_t)kH; ++kh) {
                    for (uint32_t kw = 0; kw < (uint32_t)kW; ++kw) {
                        kmap.push_back(KMap{
                            /*cin=*/c,
                            /*dh =*/ int32_t(-int(pad_h) + int(kh)*int(dil_h)),
                            /*dw =*/ int32_t(-int(pad_w) + int(kw)*int(dil_w))
                        });
                    }
                }
            }

            // 9) Assemble ConvAttr
            ConvAttr cattr{
                std::move(E),         // E
                /*bias_ptr*/ nullptr, // bias_ptr

                // geometry/params
                static_cast<uint32_t>(Cin),
                static_cast<uint32_t>(Cout),
                static_cast<uint32_t>(kH),
                static_cast<uint32_t>(kW),
                stride_h, stride_w,
                pad_h, pad_w,
                dil_h, dil_w,
                group,
                static_cast<uint32_t>(H_in),
                static_cast<uint32_t>(W_in),
                static_cast<uint32_t>(H_out),
                static_cast<uint32_t>(W_out),

                // kmap
                std::move(kmap)
            };

            cattr.use_rbm = (Cout >= 32 && K >= 64);
            if (cattr.use_rbm) {
                cattr.rbm = build_rbm_from_ellpack(cattr.E, cattr.bias_ptr, /*Ct_max=*/8);
            }

            // 10) Push layer + record output shape ({B, Cout, H_out, W_out})
            const std::string x_name = resolve(node.input(0));
            layers_.push_back({
                LayerType::Conv,
                LayerOp::Conv,
                std::move(cattr),
                { x_name },
                { node.output(0) }
            });
            shape_map_[ node.output(0) ] = { int(B), Cout, int(H_out), int(W_out) };

            // stash bias for packing later
            if (!bias_vec.empty()) {
                bias_plans.push_back({ layers_.size() - 1, std::move(bias_vec) });
            }
            continue;
        }

        if (op == "Identity") {
            const std::string in  = resolve(node.input(0));
            const std::string out = node.output(0);

            // Record alias and propagate shape immediately
            name_alias_[out] = in;
            // shape of 'out' is same as 'in'
            auto it = shape_map_.find(in);
            if (it != shape_map_.end()) {
                shape_map_[out] = it->second;
            }
            continue; // do not emit a Layer
        }


        throw std::runtime_error("Unsupported ONNX op: " + op);
    }

    // Sum up every bias vector’s length
    size_t total_bias = 0;
    for (auto &bp : bias_plans) total_bias += bp.data.size();

    bias_data_.reset(total_bias ? new float[total_bias] : nullptr);
    float* bptr = bias_data_.get();

    for (auto &bp : bias_plans) {
        const size_t idx = bp.layer_idx;
        const size_t len = bp.data.size();

        Layer &L = layers_[idx];

        if (len == 0) {
            // No bias for this layer: make sure ptr stays null
            if (std::holds_alternative<MatMulAttr>(L.attr)) {
                std::get<MatMulAttr>(L.attr).bias_ptr = nullptr;
            } else if (std::holds_alternative<ConvAttr>(L.attr)) {
                std::get<ConvAttr>(L.attr).bias_ptr = nullptr;
            }
            continue;
        }

        // Copy and set pointer
        std::memcpy(bptr, bp.data.data(), len * sizeof(float));
        if (std::holds_alternative<MatMulAttr>(L.attr)) {
            std::get<MatMulAttr>(L.attr).bias_ptr = bptr;
        } else if (std::holds_alternative<ConvAttr>(L.attr)) {
            std::get<ConvAttr>(L.attr).bias_ptr = bptr;
        }
        bptr += len;
    }

    auto fuse_use_counts = [&](const std::vector<Layer>& Ls){
        std::unordered_map<std::string,int> use;
        for (auto& L : Ls) for (auto& in : L.inputs) use[in]++; 
        return use;
    };

    auto try_fuse_relu_after = [&](size_t i, std::unordered_map<std::string,int>& use)->bool {
        if (i+1 >= layers_.size()) return false;
        Layer& A = layers_[i];
        Layer& B = layers_[i+1];

        // Producer must be MatMul or Conv
        if (!(A.op == LayerOp::MatMul || A.op == LayerOp::Conv)) return false;
        // Consumer must be Relu with a single input and single output
        if (!(B.op == LayerOp::Relu && B.inputs.size()==1 && B.outputs.size()==1)) return false;
        // And the Relu must take exactly A's output as its only input
        if (B.inputs[0] != A.outputs[0]) return false;
        // And A's output must not be used anywhere else
        if (use[A.outputs[0]] != 1) return false;

        // Fuse: set flag, redirect A's output name, erase B
        if (A.op == LayerOp::MatMul) {
            auto& ma = std::get<MatMulAttr>(A.attr);
            ma.fuse_relu = true;
        } else { // Conv
            auto& ca = std::get<ConvAttr>(A.attr);
            ca.fuse_relu = true;
        }
        // Decrement use-count for the input we’re consuming
        use[B.inputs[0]]--;
        // A now produces B's former output tensor
        A.outputs[0] = B.outputs[0];

        // Propagate shape_map (Relu is shape-preserving)
        shape_map_[A.outputs[0]] = shape_map_.at(B.outputs[0]);

        // Drop B
        layers_.erase(layers_.begin() + (i+1));
        return true;
    };

    auto use = fuse_use_counts(layers_);
    for (size_t i = 0; i < layers_.size(); ++i) {
        // Greedy local fusion: A (MatMul|Conv) + Relu
        if (try_fuse_relu_after(i, use)) {
            // Don't advance i; we might be able to fuse chains in future
        }
    }


    // Last MatMul layer index
    last_matmul_idx_ = 0;
    for (size_t i = 0; i < layers_.size(); ++i) {
        if (layers_[i].type == LayerType::MatMul) {
            last_matmul_idx_ = i;
        }
    }

    // Number of output rows = m of the final MatMul

    bool use_avx512   = supports_avx512();
    simd_w   = use_avx512 ? 16u : 8u;
    use_mask     = (batch_dim_ % simd_w) != 0;
}

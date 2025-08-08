#include "sparse_onnx.hpp"

void SparseOnnxModel::run(
    const float* input,
    uint32_t      B,
    float*       output
) const {
    const float* input_ptr  = input;
    float*       output_ptr = output;
    std::unordered_map<std::string,bool> owned;
    owned[input_name_] = false;   // we don’t free the user’s input

    // 1) Compute how many times each tensor is consumed
    std::unordered_map<std::string,int> refcount;
    for (const auto &L : layers_) {
        for (const auto &iname : L.inputs) {
            refcount[iname]++;
        }
    }

    // 2) Prepare buffers and shape map, seed with the model input
    std::unordered_map<std::string,float*>   buf;
    std::unordered_map<std::string,uint32_t> features_map;
    buf[input_name_]       = const_cast<float*>(input);
    features_map[input_name_] = in_features_;

    // 3) Execute each layer in sequence
    for (const auto &L : layers_) {
        // Gather input pointers and their row counts
        std::vector<float*>   ins;
        std::vector<uint32_t> in_features;
        ins.reserve(L.inputs.size());
        in_features.reserve(L.inputs.size());
        for (const auto &iname : L.inputs) {
            ins .push_back(buf.at(iname));
            in_features.push_back(features_map.at(iname));
        }
        bool is_final = std::find(
          L.outputs.begin(),
          L.outputs.end(),
          output_name_) != L.outputs.end();

        // Dispatch to the appropriate helper
        RunResult R;
        switch (L.op) {
          case LayerOp::MatMul: {
            auto &m = std::get<MatMulAttr>(L.attr);
            R = applyMatMul(m, ins[0], B, is_final ? output_ptr : nullptr);
            break;
          }
          case LayerOp::Add: {
            R = applyAdd(
                std::get<AddAttr>(L.attr),
                ins[0], ins[1],
                in_features[0], B);
            break;
          }
          case LayerOp::Relu: {
            R = applyRelu(
                std::get<ActAttr>(L.attr),
                ins[0], in_features[0], B);
            break;
          }
          case LayerOp::Sigmoid: {
            R = applySigmoid(
                std::get<ActAttr>(L.attr),
                ins[0], in_features[0], B);
            break;
          }
          case LayerOp::Tanh: {
            R = applyTanh(
                std::get<ActAttr>(L.attr),
                ins[0], in_features[0], B);
            break;
          }
          case LayerOp::MaxPool: {
            R = applyMaxPool(
                std::get<PoolAttr>(L.attr),
                ins[0], in_features[0], B);
            break;
          }
          case LayerOp::GlobalAveragePool: {
            R = applyGlobalAveragePool(
                std::get<PoolAttr>(L.attr),
                ins[0], in_features[0], B);
            break;
          }
          case LayerOp::Flatten: {
            R = applyFlatten(
                std::get<FlattenAttr>(L.attr),
                ins[0], in_features[0], B);
            break;
          }
          case LayerOp::Conv: {
            auto &c = std::get<ConvAttr>(L.attr);
            R = applyConv(c, ins[0], B);
            break;
          }
          default:
            throw std::logic_error("Unhandled LayerOp in run(): " +
                                   std::to_string(int(L.op)));
        }

        // 4) Register outputs under their tensor names
        for (const auto &oname : L.outputs) {
            buf[oname]      = R.data;
            features_map[oname] = R.features;
            owned[oname] = R.owned;
        }

        std::unordered_set<float*> out_ptrs;
        out_ptrs.reserve(L.outputs.size());
        for (const auto &oname : L.outputs) out_ptrs.insert(buf[oname]);

        // 5) Free any inputs no longer needed
        for (auto &iname : L.inputs) {
            if (--refcount[iname] == 0 && iname != input_name_) {
                float* iptr = buf[iname];
                bool aliased = out_ptrs.count(iptr) != 0;
                if (owned[iname] && !aliased) {
                    free(iptr);
                }
                buf.erase(iname);
                features_map.erase(iname);
                owned.erase(iname);
            }
      }
    }
    {
        auto it = buf.find(output_name_);
        if (it != buf.end()) {
            float* final_buf = it->second;
            if (final_buf != output_ptr) {
                // The internal layout for both MatMul and Conv is already the target Fortran layout.
                const auto dims = shape_map_.at(output_name_); // e.g., {B,M} or {B,Cout,Hout,Wout}
                size_t total = 1;
                for (int d : dims) total *= size_t(d);
                std::memcpy(output_ptr, final_buf, total * sizeof(float));
            }
        }
    }
}
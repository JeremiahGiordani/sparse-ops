#include "sparse_onnx.hpp"

RunResult SparseOnnxModel::applyFlatten(
    const FlattenAttr &/*f*/,
    const float       *src,
    uint32_t           features,
    uint32_t           B,
    float*            out_buf
) const {
    size_t tot = size_t(features) * B;
    bool   owned = true;
    void *raw = nullptr;
    posix_memalign(&raw, 64, tot * sizeof(float));
    float *dst = reinterpret_cast<float*>(raw);
    std::memcpy(dst, src, tot * sizeof(float));
    return { dst, tot, owned };  // new row count is rows*C
}
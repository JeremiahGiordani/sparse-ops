#include "sparse_onnx.hpp"

RunResult SparseOnnxModel::applyMaxPool(
    const PoolAttr &/*p*/,
    const float    *src,
    uint32_t        rows,
    uint32_t        C,
    float*            out_buf
) const {
    // Stub: just copy input back
    size_t tot = size_t(rows) * C;
    bool   owned = true;
    void *raw = nullptr;
    posix_memalign(&raw, 64, tot * sizeof(float));
    float *dst = reinterpret_cast<float*>(raw);
    std::memcpy(dst, src, tot * sizeof(float));
    return { dst, rows, owned };
}

RunResult SparseOnnxModel::applyGlobalAveragePool(
    const PoolAttr &/*p*/,
    const float    *src,
    uint32_t        rows,
    uint32_t        C,
    float*            out_buf
) const {
    // Stub: copy input
    size_t tot = size_t(rows) * C;
    bool   owned = true;
    void *raw = nullptr;
    posix_memalign(&raw, 64, tot * sizeof(float));
    float *dst = reinterpret_cast<float*>(raw);
    std::memcpy(dst, src, tot * sizeof(float));
    return { dst, rows, owned };
}

#include "sparse_onnx.hpp"

RunResult SparseOnnxModel::applyRelu(
    const ActAttr  &/*a*/,
    const float    *src,
    uint32_t        rows,
    uint32_t        C,
    float*            out_buf
) const {
    size_t tot = size_t(rows) * C;
    bool   owned = true;
    void *raw = nullptr;
    if (posix_memalign(&raw, 64, tot * sizeof(float)) != 0) {
        throw std::bad_alloc();
    }
    float *dst = reinterpret_cast<float*>(raw);
    for (size_t i = 0; i < tot; ++i) {
        float v = src[i];
        dst[i] = v > 0.0f ? v : 0.0f;
    }
    return { dst, rows, owned };
}

RunResult SparseOnnxModel::applySigmoid(
    const ActAttr  &/*a*/,
    const float    *src,
    uint32_t        rows,
    uint32_t        C,
    float*            out_buf
) const {
    size_t tot = size_t(rows) * C;
    bool   owned = true;
    void *raw = nullptr;
    if (posix_memalign(&raw, 64, tot * sizeof(float)) != 0) {
        throw std::bad_alloc();
    }
    float *dst = reinterpret_cast<float*>(raw);
    for (size_t i = 0; i < tot; ++i) {
        dst[i] = 1.0f / (1.0f + std::exp(-src[i]));
    }
    return { dst, rows, owned };
}

RunResult SparseOnnxModel::applyTanh(
    const ActAttr  &/*a*/,
    const float    *src,
    uint32_t        rows,
    uint32_t        C,
    float*            out_buf
) const {
    size_t tot = size_t(rows) * C;
    bool   owned = true;
    void *raw = nullptr;
    if (posix_memalign(&raw, 64, tot * sizeof(float)) != 0) {
        throw std::bad_alloc();
    }
    float *dst = reinterpret_cast<float*>(raw);
    for (size_t i = 0; i < tot; ++i) {
        dst[i] = std::tanh(src[i]);
    }
    return { dst, rows, owned };
}
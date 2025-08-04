#include "sparse_onnx.hpp"

RunResult SparseOnnxModel::applyAdd(
    const AddAttr   &/*a*/,
    const float     *in_A,
    const float     *in_B,
    uint32_t         features,
    uint32_t         B,
    float*            out_buf
) const {
    size_t tot = size_t(features) * B;
    bool   owned = true;
    void *raw = nullptr;
    if (posix_memalign(&raw, 64, tot * sizeof(float)) != 0) {
        throw std::bad_alloc();
    }
    float *dst = reinterpret_cast<float*>(raw);
    for (size_t i = 0; i < tot; ++i) {
        dst[i] = in_A[i] + in_B[i];
    }
    return { dst, features, owned };
}
#include "sparse_onnx.hpp"

RunResult SparseOnnxModel::applyAdd(
    const AddAttr   &/*a*/,
    const float     *A,
    const float     *B,
    uint32_t         rows,
    uint32_t         C,
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
        dst[i] = A[i] + B[i];
    }
    return { dst, rows, owned };
}
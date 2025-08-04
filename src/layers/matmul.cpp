#include "sparse_onnx.hpp"

RunResult SparseOnnxModel::applyMatMul(
    const MatMulAttr &m,
    const float      *src,
    uint32_t          B,
    float*            out_buf
) const {
    // Number of output rows
    uint32_t M = m.E.m;
    // Total elements = M rows × C columns
    size_t   elems = size_t(M) * B;

    void *raw = nullptr;
    float* dst;
    bool   owned = false;
    if (out_buf) {
        dst   = out_buf;
        owned = false;
    } else {
        posix_memalign(&raw, 64, elems*sizeof(float));
        dst   = reinterpret_cast<float*>(raw);
        owned = true;
    }

    // Perform the sparse matmul (no ReLU fusion)
    // This calls ellpack_matmul_fused<false,false> under the hood
    if (use_mask) {
        ellpack_matmul_fused<true,  false>(m.E, src, B, m.bias_ptr, dst);
    } else {
        ellpack_matmul_fused<false, false>(m.E, src, B, m.bias_ptr, dst);
    }

    // Return both the buffer pointer and the row count
    return { dst, M, owned };
}

RunResult SparseOnnxModel::applyMatMulRelu(
    const MatMulAttr &m,
    const float      *src,
    uint32_t          B,
    float*            out_buf
) const {
    // Number of output rows
    uint32_t M = m.E.m;
    // Total elements = M rows × C columns
    size_t   elems = size_t(M) * B;

    // Allocate a 64‐byte‐aligned buffer for the output
    void *raw = nullptr;
    float* dst;
    bool   owned = false;
    if (out_buf) {
        dst   = out_buf;
        owned = false;
    } else {
        posix_memalign(&raw, 64, elems*sizeof(float));
        dst   = reinterpret_cast<float*>(raw);
        owned = true;
    }

    // Perform the sparse matmul (no ReLU fusion)
    // This calls ellpack_matmul_fused<false,false> under the hood
    if (use_mask) {
        ellpack_matmul_fused<true,  true>(m.E, src, B, m.bias_ptr, dst);
    } else {
        ellpack_matmul_fused<false, true>(m.E, src, B, m.bias_ptr, dst);
    }

    // Return both the buffer pointer and the row count
    return { dst, M, owned };
}
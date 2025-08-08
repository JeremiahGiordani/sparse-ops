#include "sparse_onnx.hpp"

RunResult SparseOnnxModel::applyMatMul(
    const MatMulAttr &m,
    const float      *src,
    uint32_t          B,
    float*            out_buf) const
{
    const uint32_t M = m.E.m;
    const size_t elems = size_t(M) * B;

    float* dst;
    bool owned = false;
    if (out_buf) {
        dst = out_buf;
        owned = false;
    } else {
        void* raw = nullptr;
        posix_memalign(&raw, 64, elems*sizeof(float));
        dst = reinterpret_cast<float*>(raw);
        owned = true;
    }

    if (m.fuse_relu) {
        if (use_mask) ellpack_matmul_fused<true,  true>(m.E, src, B, m.bias_ptr, dst);
        else          ellpack_matmul_fused<false, true>(m.E, src, B, m.bias_ptr, dst);
    } else {
        if (use_mask) ellpack_matmul_fused<true,  false>(m.E, src, B, m.bias_ptr, dst);
        else          ellpack_matmul_fused<false, false>(m.E, src, B, m.bias_ptr, dst);
    }

    return { dst, M, owned };
}
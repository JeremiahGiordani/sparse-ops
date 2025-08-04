#include "sparse_onnx.hpp"
#include "iostream"

RunResult SparseOnnxModel::applyConv(
    const ConvAttr &c,
    const float    *src,  // FORTRAN layout: (Cin*H_in*W_in) × B
    uint32_t        B)
const {

    size_t rows_out  = size_t(c.kernel_dims[0]) * c.H_out * c.W_out;
    size_t total_elems = rows_out * B;
    void* raw_out = nullptr;
    if (posix_memalign(&raw_out, 64, total_elems * sizeof(float)) != 0) {
        throw std::bad_alloc();
    }
    float* dst_all = static_cast<float*>(raw_out);

    // 2) Local im2col buffer
    size_t patch_size  = size_t(c.kernel_dims[1]) * c.kernel_dims[2] * c.kernel_dims[3];
    size_t num_patches = size_t(c.H_out) * c.W_out;
    std::vector<float> col_buf(patch_size * num_patches);

    // 3) For each batch:
    for (uint32_t b = 0; b < B; ++b) {
        const float* src_b = src    + size_t(b) * (patch_size); 
        float*       dst_b = dst_all + size_t(b) * rows_out;

        // scatter with precomputed indices:
        for (size_t i = 0; i < col_buf.size(); ++i) {
            col_buf[i] = src_b[ c.patch_indices[i] ];
        }

        // one sparse‐MatMul: [Cout×patch_size] × [patch_size×num_patches]
        ellpack_matmul(c.E, col_buf.data(), num_patches, c.bias_ptr, dst_b);

    }

    return { dst_all, uint32_t(rows_out), true };
}

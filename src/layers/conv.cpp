#include "sparse_onnx.hpp"


RunResult SparseOnnxModel::applyConv(
    const ConvAttr &c,
    const float    *src,
    uint32_t        rows_in,
    uint32_t        C,
    float*            out_buf
) const {
    // Unpack dims & hyper-params
    int Cout = c.kernel_dims[0];
    int Cin  = c.kernel_dims[1];
    int kH   = c.kernel_dims[2];
    int kW   = c.kernel_dims[3];
    int padH0 = c.pads[0], padW0 = c.pads[1],
        padH1 = c.pads[2], padW1 = c.pads[3];
    int sH = c.strides[0], sW = c.strides[1];
    int dH = c.dilations[0], dW = c.dilations[1];

    // Infer input H×W from rows_in = Cin * H_in * W_in
    int H_in = rows_in / Cin;
    int W_in = H_in;  // assume square for simplicity

    // Compute output spatial dims
    int H_out = (H_in + padH0 + padH1 - dH*(kH-1) - 1) / sH + 1;
    int W_out = (W_in + padW0 + padW1 - dW*(kW-1) - 1) / sW + 1;

    // Sizes
    uint32_t patch_size   = Cin * kH * kW;            // N cols of weight
    uint32_t num_patches  = H_out * W_out;            // output “columns” per sample
    uint32_t rows_out     = Cout * num_patches;       // total rows in output feature map
    size_t   total_elems  = size_t(rows_out) * C;     // for all batch

    // Allocate output buffer
    void* raw_out = nullptr;
    if (posix_memalign(&raw_out, 64, total_elems * sizeof(float)) != 0) {
        throw std::bad_alloc();
    }
    float* dst_all = reinterpret_cast<float*>(raw_out);

    // Temporary im2col buffer for one sample
    std::vector<float> col_buf;
    col_buf.resize(size_t(patch_size) * num_patches);

    // For each example in the batch
    for (uint32_t b = 0; b < C; ++b) {
        const float* src_b = src + size_t(b) * rows_in;
        float*       dst_b = dst_all + size_t(b) * rows_out;

        // Build im2col: each patch is a column
        size_t p = 0;
        for (int hi = 0; hi < H_out; ++hi) {
            int base_h = hi * sH - padH0;
            for (int wi = 0; wi < W_out; ++wi) {
                int base_w = wi * sW - padW0;
                // For each channel and kernel position
                for (int ic = 0; ic < Cin; ++ic) {
                    size_t channel_offset = size_t(ic) * H_in * W_in;
                    for (int r = 0; r < kH; ++r) {
                        int ih = base_h + r * dH;
                        for (int s = 0; s < kW; ++s) {
                            int iw = base_w + s * dW;
                            float val = 0.0f;
                            if ((unsigned)ih < (unsigned)H_in &&
                                (unsigned)iw < (unsigned)W_in) {
                                val = src_b[channel_offset + size_t(ih)*W_in + iw];
                            }
                            col_buf[p++] = val;
                        }
                    }
                }
            }
        }

        // Sparse matmul: [Cout×patch_size] × [patch_size×num_patches] → [Cout×num_patches]
        ellpack_matmul(
            c.E,
            col_buf.data(),
            num_patches,
            c.bias_ptr,
            dst_b
        );
    }

    return { dst_all, rows_out };
}
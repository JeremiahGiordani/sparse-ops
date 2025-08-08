#include "sparse_onnx.hpp"
#include "iostream"

RunResult SparseOnnxModel::applyConv(
    const ConvAttr &c,
    const float    *src,  // FORTRAN layout: (Cin*H_in*W_in) × B
    uint32_t        B)
const {

    size_t rows_out   = size_t(c.kernel_dims[0]) * c.H_out * c.W_out;
    size_t total_elems= rows_out * B;
    void*  raw_out    = nullptr;
    posix_memalign(&raw_out, 64, total_elems * sizeof(float));
    float* dst_all    = static_cast<float*>(raw_out);
    std::fill_n(dst_all, total_elems, 0.0f);

    // im2col dims
    size_t patch_size  = size_t(c.kernel_dims[1])
                       * c.kernel_dims[2]
                       * c.kernel_dims[3];      // K
    size_t in_features = size_t(c.kernel_dims[1]) 
                       * c.H_in 
                       * c.W_in;
    size_t num_patches = size_t(c.H_out) * c.W_out; // N

    std::vector<float> col_buf(patch_size * num_patches);

    // Fortran strides for src[B, Cin, H_in, W_in]
    size_t stride_b = 1;
    size_t stride_c = size_t(B) * stride_b;
    size_t stride_h = size_t(c.kernel_dims[1]) * stride_c; // Cin * B
    size_t stride_w = size_t(c.H_in) * stride_h;           // H_in * Cin * B

    for (uint32_t b = 0; b < B; ++b) {
        float* dst_b = dst_all + size_t(b) * rows_out;

        // --- BUILD col_buf IN ROW-MAJOR ORDER (K rows × N cols) ---
        for (size_t p = 0; p < patch_size; ++p) {
            for (size_t n = 0; n < num_patches; ++n) {
                // pick up the offset for (row p, col n):
                size_t idx_fortran = n * patch_size + p;
                size_t off = c.patch_indices[idx_fortran];

                float val = 0.0f;
                if (off != in_features) {
                    // decode C-order off -> (ic, ih, iw)
                    int ic = int(off / (c.H_in * c.W_in));
                    int rem= int(off % (c.H_in * c.W_in));
                    int ih = rem / c.W_in;
                    int iw = rem % c.W_in;
                    // Fortran‐style linear index into src:
                    size_t fin = size_t(b)*stride_b
                               + size_t(ic)*stride_c
                               + size_t(ih)*stride_h
                               + size_t(iw)*stride_w;
                    val = src[fin];
                }

                // place into col_buf row-major:
                col_buf[p * num_patches + n] = val;
            }
        }

        // --- SPARSE GEMM (now with row-major col_buf) ---
        ellpack_matmul(
          c.E,
          col_buf.data(),
          static_cast<uint32_t>(num_patches),
          c.bias_ptr,
          dst_b
        );
    }

    return { dst_all, uint32_t(rows_out), true };
}


#include <vector>
#include <cstring>

// Performs a single 2D convolution via im2col + GEMM (padding=1, stride=1, no bias).
// weight: pointer to weights of shape (Cout, Cin, kH, kW)
// input:  pointer to input tensor of shape (B, Cin, H, W)
// output: pointer to output tensor of shape (B, Cout, H, W)
// B: batch size, Cin: input channels, H,W: height & width of input
// Cout: output channels, kH,kW: kernel height & width
#include <vector>
#include <cstring>
#include "ellpack_encoder.hpp"
#include "ellpack_matmul.hpp"

// Performs a single 2D convolution via im2col + ELLPACK sparse GEMM (padding=1, stride=1, no bias).
// weight: pointer to weights of shape (Cout, Cin, kH, kW)
// input:  pointer to input tensor of shape (B, Cin, H, W)
// output: pointer to output tensor of shape (B, Cout, H, W)
// B: batch size, Cin: input channels, H,W: height & width of input
// Cout: output channels, kH,kW: kernel height & width
#include <vector>
#include <cstring>
#include "ellpack_encoder.hpp"
#include "ellpack_matmul.hpp"

// Performs a single 2D convolution via im2col + ELLPACK sparse GEMM (padding=1, stride=1, no bias).
// weight: pointer to weights of shape (Cout, Cin, kH, kW) in standard C-order
// input:  pointer to input tensor of shape (Cin*H*W, B) in Fortran-style (column-major)
// output: pointer to output tensor of shape (B, Cout, H, W) in C-order
// B: batch size, Cin: input channels, H,W: height & width of input
// Cout: output channels, kH,kW: kernel height & width
void single_conv_layer(const float* weight,
                       const float* input,
                       float*       output,
                       int          Cin,
                       int          H,
                       int          W,
                       int          Cout,
                       int          kH,
                       int          kW) {
    // Derived dims
    const int pad_h = kH / 2;
    const int pad_w = kW / 2;
    const int H_out = H;
    const int W_out = W;
    const int N = H_out * W_out;
    const int K = Cin * kH * kW;
    const size_t M = static_cast<size_t>(Cout);

    // Zero-out output
    std::fill(output, output + (size_t) M * N, 0.0f);

    // 1) Build dense weight matrix of shape [M x K]
    std::vector<float> weight_mat(M * K);
    std::memcpy(weight_mat.data(), weight, M * K * sizeof(float));

    // 2) Convert weight to ELLPACK
    Ellpack E = convert_to_ellpack(weight_mat.data(), static_cast<uint32_t>(M), static_cast<uint32_t>(K));

    // Allocate buffers
    std::vector<float> col_buf(K * N);
    std::vector<float> out_mat(M * N);

    // For each image in batch
    // for (int b = 0; b < B; ++b) {
        // Build im2col buffer: size K x N
        for (int p = 0; p < K; ++p) {
            int c = p / (kH * kW);
            int rem = p % (kH * kW);
            int kh = rem / kW;
            int kw = rem % kW;
            size_t base_in = (static_cast<size_t> (Cin) + c) * H * W;
            for (int n = 0; n < N; ++n) {
                int y = n / W_out;
                int x = n % W_out;
                int in_y = y + kh - pad_h;
                int in_x = x + kw - pad_w;
                float val = 0.0f;
                if (in_y >= 0 && in_y < H && in_x >= 0 && in_x < W) {
                    val = input[base_in + in_y * W + in_x];
                }
                col_buf[static_cast<size_t>(p) * N + n] = val;
            }
        }

        // 3) Sparse GEMM via ELLPACK: [M x K] * [K x N] -> [M x N]
        ellpack_matmul(E, col_buf.data(), static_cast<uint32_t>(N), nullptr, output);

        // std::memcpy(output, out_mat.data(), M * N * sizeof(float));

        // 4) Scatter out_mat into output tensor
        // for (size_t o = 0; o < M; ++o) {
        //     for (int n = 0; n < N; ++n) {
        //         int y = n / W_out;
        //         int x = n % W_out;
        //         size_t out_idx = ((static_cast<size_t>(b) * M + o) * H_out + y) * W_out + x;
        //         output[out_idx] = out_mat[o * N + n];
        //     }
        // }
    // }
}


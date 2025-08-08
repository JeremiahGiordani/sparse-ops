// ─────────────────────────────────────────────────────────────────────────────
//  bindings.cpp  – pybind11 glue for sparseops_backend
// ─────────────────────────────────────────────────────────────────────────────
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "ellpack_encoder.hpp"   // for ELLPACK encoding
#include "ellpack_matvec.hpp"    // for ellpack_matvec
#include "ellpack_matmul.hpp"    // for ellpack_matmul
#include "sparse_onnx.hpp"       // for SparseOnnxModel

namespace py = pybind11;

void conv2d_implicit_im2col_fmajor(
    const ConvPlan& P,
    const float*    X,
    uint32_t        B,
    uint32_t        H, uint32_t W,
    uint32_t        Hout, uint32_t Wout,
    const float*    bias,   // nullptr if no bias
    float*          Y
);

EllpackW encode_ellpack_from_weight(const float* W,
                                           uint32_t Cout, uint32_t Cin,
                                           uint32_t kH, uint32_t kW);


std::vector<KMap> build_kmap(uint32_t Cin, uint32_t kH, uint32_t kW,
                                    uint32_t pad_h, uint32_t pad_w,
                                    uint32_t dil_h=1, uint32_t dil_w=1);

// ─────────────────────────────────────────────────────────────────────────────
// Encode to ELLPACK format
static Ellpack convert_to_ellpack_py(
    py::array_t<float, py::array::c_style | py::array::forcecast> W)
{
    auto buf = W.request();
    const float* data = static_cast<float*>(buf.ptr);
    uint32_t m = buf.shape[0];
    uint32_t n = buf.shape[1];  // m rows, n columns
    return convert_to_ellpack(data, m, n);
}

// ─────────────────────────────────────────────────────────────────────────────
// Decode from ELLPACK format
static py::array_t<float> decode_from_ellpack_py(const Ellpack& E)
{
    // Create output buffer of size m*n
    py::array_t<float> W_out({E.m, E.n});
    // Fill it with zeros
    std::fill(
        W_out.mutable_data(),
        W_out.mutable_data() + static_cast<size_t>(E.m) * E.n,
        0.0f
    );

    auto buf = W_out.request();
    if (buf.size != static_cast<size_t>(E.m) * E.n) {
        throw std::runtime_error("Output buffer size mismatch");
    }
    decode_from_ellpack(E, static_cast<float*>(buf.ptr));
    return W_out;
}

static ConvPlan setup_conv_from_numpy(const py::array& weights,
                                      int stride=1, int padding=0)
{
    if (weights.ndim() != 4)
        throw std::runtime_error("weights must be 4D (Cout, Cin, kH, kW)");
    auto w = weights.cast<py::array_t<float, py::array::c_style | py::array::forcecast>>();
    auto bw = w.request();
    const uint32_t Cout = static_cast<uint32_t>(bw.shape[0]);
    const uint32_t Cin  = static_cast<uint32_t>(bw.shape[1]);
    const uint32_t kH   = static_cast<uint32_t>(bw.shape[2]);
    const uint32_t kW   = static_cast<uint32_t>(bw.shape[3]);

    const float* Wptr = static_cast<const float*>(bw.ptr);

    ConvPlan P;
    P.Cout = Cout; P.Cin = Cin; P.kH = kH; P.kW = kW;
    P.stride_h = P.stride_w = static_cast<uint32_t>(stride);
    P.pad_h    = P.pad_w    = static_cast<uint32_t>(padding);

    P.kmap = build_kmap(Cin, kH, kW, P.pad_h, P.pad_w, /*dil_h=*/1, /*dil_w=*/1);
    P.W    = encode_ellpack_from_weight(Wptr, Cout, Cin, kH, kW);
    return P;
}

static py::array_t<float, py::array::f_style>
conv2d_ellpack_run(const ConvPlan& P,
                   py::array_t<float, py::array::f_style | py::array::forcecast> X_f,
                   py::object bias_opt = py::none())
{
    auto bx = X_f.request();
    if (bx.ndim != 4) throw std::runtime_error("X must be 4D (B,Cin,H,W) Fortran-ordered");
    const ssize_t item = sizeof(float);

    const uint32_t B   = static_cast<uint32_t>(bx.shape[0]);
    const uint32_t Cin = static_cast<uint32_t>(bx.shape[1]);
    const uint32_t H   = static_cast<uint32_t>(bx.shape[2]);
    const uint32_t W   = static_cast<uint32_t>(bx.shape[3]);

    // strict Fortran contiguity checks for (B,Cin,H,W):
    if (bx.strides[0] != item ||
        bx.strides[1] != static_cast<ssize_t>(B) * item ||
        bx.strides[2] != static_cast<ssize_t>(B) * static_cast<ssize_t>(Cin) * item ||
        bx.strides[3] != static_cast<ssize_t>(B) * static_cast<ssize_t>(Cin) * static_cast<ssize_t>(H) * item) {
        throw std::runtime_error("X must be Fortran contiguous with layout (B,Cin,H,W)");
    }
    if (Cin != P.Cin)
        throw std::runtime_error("X Cin does not match plan.Cin");

    const uint32_t Hout = (H + 2*P.pad_h - P.kH) / P.stride_h + 1;
    const uint32_t Wout = (W + 2*P.pad_w - P.kW) / P.stride_w + 1;

    const float* bias_ptr = nullptr;
    std::vector<float> bias_buf;
    if (!bias_opt.is_none()) {
        auto b = bias_opt.cast<py::array_t<float, py::array::c_style | py::array::forcecast>>();
        auto bb = b.request();
        if (bb.ndim != 1 || static_cast<uint32_t>(bb.shape[0]) != P.Cout)
            throw std::runtime_error("bias must be 1D length Cout");
        bias_ptr = static_cast<const float*>(bb.ptr);
    }

    // Allocate Fortran (B,Cout,Hout,Wout): strides = [1, B, B*Cout, B*Cout*Hout] * item
    std::array<ssize_t,4> y_shape   { B, static_cast<ssize_t>(P.Cout), Hout, Wout };
    std::array<ssize_t,4> y_strides { item,
                                      static_cast<ssize_t>(B) * item,
                                      static_cast<ssize_t>(B) * static_cast<ssize_t>(P.Cout) * item,
                                      static_cast<ssize_t>(B) * static_cast<ssize_t>(P.Cout) * static_cast<ssize_t>(Hout) * item };

    py::array_t<float, py::array::f_style> Y(y_shape, y_strides);
    auto by = Y.request();

    conv2d_implicit_im2col_fmajor(P,
                                  static_cast<const float*>(bx.ptr),
                                  B, H, W, Hout, Wout,
                                  bias_ptr,
                                  static_cast<float*>(by.ptr));
    return Y;
}

// ─────────────────────────────────────────────────────────────────────────────
PYBIND11_MODULE(sparseops_backend, m)
{
    m.doc() = "Sparse CPU kernels (ELLPACK)";

    // — ELLPACK handle —
    py::class_<Ellpack>(m, "Ellpack")
        .def_readonly("m", &Ellpack::m)
        .def_readonly("n", &Ellpack::n)
        .def_readonly("r", &Ellpack::r)
        .def_property_readonly("Wd", [](const Ellpack &E) {
            std::array<ssize_t,2> shape   = { (ssize_t)E.m, (ssize_t)E.r };
            std::array<ssize_t,2> strides = {
                sizeof(float) * E.r,
                sizeof(float)
            };
            return py::array_t<float>(shape, strides, E.Wd.ptr);
        })
        .def_property_readonly("idx", [](const Ellpack &E) {
            std::array<ssize_t,2> shape   = { (ssize_t)E.m, (ssize_t)E.r };
            std::array<ssize_t,2> strides = {
                sizeof(uint32_t) * E.r,
                sizeof(uint32_t)
            };
            return py::array_t<uint32_t>(shape, strides, E.idx.data());
        })
        .def_property_readonly("Xt", [](const Ellpack &E) {
            std::array<ssize_t,2> shape   = { (ssize_t)E.m, (ssize_t)E.r };
            std::array<ssize_t,2> strides = {
                sizeof(float) * E.r,
                sizeof(float)
            };
            return py::array_t<float>(shape, strides, E.Xt.ptr);
        });

    py::class_<ConvPlan>(m, "ConvPlan")
        .def_property_readonly("Cin",  [](const ConvPlan& p){ return p.Cin; })
        .def_property_readonly("Cout", [](const ConvPlan& p){ return p.Cout; })
        .def_property_readonly("kH",   [](const ConvPlan& p){ return p.kH; })
        .def_property_readonly("kW",   [](const ConvPlan& p){ return p.kW; })
        .def_property_readonly("stride_h", [](const ConvPlan& p){ return p.stride_h; })
        .def_property_readonly("stride_w", [](const ConvPlan& p){ return p.stride_w; })
        .def_property_readonly("pad_h",    [](const ConvPlan& p){ return p.pad_h; })
        .def_property_readonly("pad_w",    [](const ConvPlan& p){ return p.pad_w; });

    // — API surface —
    m.def("convert_to_ellpack", &convert_to_ellpack_py,
          "Convert dense NumPy matrix → ELLPACK handle");
    m.def("decode_from_ellpack", &decode_from_ellpack_py,
          "Convert ELLPACK handle → dense NumPy matrix");

    m.def("setup_conv",
          &setup_conv_from_numpy,
          py::arg("weights"),
          py::arg("stride") = 1,
          py::arg("padding") = 0,
          R"doc(
        Build a ConvPlan from weights shaped (Cout, Cin, kH, kW).
        Stride/padding are integers (no dilation, groups=1).
        )doc");


    m.def("ellpack_matvec",
        [](const Ellpack &E,
           py::array_t<float> x_arr,
           py::array_t<float> bias_arr) {
            auto buf_x    = x_arr.request();
            auto buf_bias = bias_arr.request();
            py::array_t<float> y_arr({(ssize_t)E.m});
            auto buf_y    = y_arr.request();

            ellpack_matvec(
                E,
                static_cast<float*>(buf_x.ptr),
                static_cast<float*>(buf_bias.ptr),
                static_cast<float*>(buf_y.ptr)
            );
            return y_arr;
        },
        "Multithreaded bilinear diagonal matvec (ELLPACK format)");

    m.def("ellpack_matmul",
        [](const Ellpack &E,
           py::array_t<float, py::array::c_style | py::array::forcecast> X_arr,
           py::array_t<float, py::array::c_style | py::array::forcecast> bias_arr) {
            auto buf_X    = X_arr.request();
            auto buf_bias = bias_arr.request();
            if (buf_X.ndim != 2) {
                throw std::runtime_error("X must be 2D (n × C)");
            }
            if ((uint32_t)buf_X.shape[0] != E.n) {
                throw std::runtime_error("Input row count must equal E.n");
            }
            if ((uint32_t)buf_bias.size != E.m) {
                throw std::runtime_error("Bias length must equal E.m");
            }

            uint32_t C = static_cast<uint32_t>(buf_X.shape[1]);
            std::array<ssize_t,2> shape = { (ssize_t)E.m, (ssize_t)C };
            py::array_t<float> Y_arr(shape);
            auto buf_Y = Y_arr.request();

            ellpack_matmul(
                E,
                static_cast<const float*>(buf_X.ptr),
                C,
                static_cast<const float*>(buf_bias.ptr),
                static_cast<float*>(buf_Y.ptr)
            );
            return Y_arr;
        },
        R"pbdoc(
            Multithreaded bilinear-diagonal mat-mul (ELLPACK format)

            Computes Y = E × X + bias, where:
              • E is an ELLPACK (m×n) matrix
              • X is an n×C float32 array
              • bias is length-m float32 vector (added to each column)
              • Returns Y as an m×C float32 NumPy array
        )pbdoc");

    // m.def("conv",
    //     [](py::array_t<float> weight, py::array_t<float, py::array::f_style | py::array::forcecast> input) {
    //         // 1) buffer checks
    //         auto bw = weight.request(), bx = input.request();
    //         if (bw.ndim != 4) throw std::runtime_error("weight must be shape (Cout,Cin,kH,kW)");
    //         if (bx.ndim != 4) throw std::runtime_error("input must be shape (B,Cin,H,W)");

    //         // 2) unpack dims
    //         int Cout = bw.shape[0];
    //         int Cin  = bw.shape[1];
    //         int kH   = bw.shape[2];
    //         int kW   = bw.shape[3];

    //         int B = bx.shape[0];
    //         int H = bx.shape[2];
    //         int W = bx.shape[3];

    //         // 3) allocate output (B, Cout, H, W)
    //         std::vector<ssize_t> shape { B, Cout, H, W };
    //         // strides (in bytes): b=1, c=B, h=B*Cout, w=B*Cout*H
    //         ssize_t itemsize = sizeof(float);
    //         std::vector<ssize_t> strides(4);
    //         strides[0] = itemsize;
    //         strides[1] = strides[0] * shape[0];
    //         strides[2] = strides[1] * shape[1];
    //         strides[3] = strides[2] * shape[2];
    //         py::array_t<float, py::array::f_style> out(
    //             shape,
    //             strides
    //         );
    //         auto bo = out.request();

    //         // 4) call your impl
    //         single_conv_layer(
    //             static_cast<float*>(bw.ptr),
    //             static_cast<float*>(bx.ptr),
    //             static_cast<float*>(bo.ptr),
    //             B, Cin, H, W, Cout, kH, kW
    //         );

    //         return out;
    //     },
    //     py::arg("weight"),
    //     py::arg("input"),
    //     "Run a single 2D convolution (no bias, padding=1) on a float tensor"
    // );

    m.def("conv2d_ellpack",
          &conv2d_ellpack_run,
          py::arg("plan"),
          py::arg("X_f"),
          py::arg("bias") = py::none(),
          R"doc(
            Run sparse conv with Fortran (B,Cin,H,W) input; returns Fortran (B,Cout,Hout,Wout).
            )doc");

    // — Sparse ONNX model —
    py::class_<SparseOnnxModel>(m, "SparseOnnxModel")
        .def(py::init<const std::string&>(),
             "Load an ONNX model and pre-encode weights to ELLPACK")
        .def("run",
            [](const SparseOnnxModel &model,
               py::array_t<float, py::array::f_style | py::array::forcecast> X) {
                auto buf = X.request();
                uint32_t n = static_cast<uint32_t>(buf.shape[1]);
                uint32_t C = static_cast<uint32_t>(buf.shape[0]);
                const float* input_ptr = static_cast<const float*>(buf.ptr);

                std::vector<size_t> dims = model.output_shape();
                std::vector<ssize_t> shape(dims.begin(), dims.end());

                // 3) Compute Fortran strides (in bytes)
                std::vector<ssize_t> strides(shape.size());
                ssize_t itemsize = sizeof(float);
                strides[0] = itemsize;
                for (size_t i = 1; i < shape.size(); ++i) {
                    strides[i] = strides[i-1] * shape[i-1];
                }

                // 4) Allocate an f_style array with those strides
                py::array_t<float, py::array::f_style> Y(
                    /* shape  = */ shape,
                    /* strides= */ strides
                );
                auto by = Y.request();

                // 5) Run the model into the Fortran buffer
                model.run(input_ptr,
                            C,
                            static_cast<float*>(by.ptr));

                return Y;
            },
            "Run inference on input X (shape n×C), returns output (m×C)");
}
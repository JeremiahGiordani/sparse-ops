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

void single_conv_layer(const float* weight,
                       const float* input,
                       float*       output,
                    //    int          B,
                       int          Cin,
                       int          H,
                       int          W,
                       int          Cout,
                       int          kH,
                       int          kW);

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

    // — API surface —
    m.def("convert_to_ellpack", &convert_to_ellpack_py,
          "Convert dense NumPy matrix → ELLPACK handle");
    m.def("decode_from_ellpack", &decode_from_ellpack_py,
          "Convert ELLPACK handle → dense NumPy matrix");

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

    m.def("conv",
        [](py::array_t<float> weight, py::array_t<float, py::array::f_style | py::array::forcecast> input) {
            // 1) buffer checks
            auto bw = weight.request(), bx = input.request();
            if (bw.ndim != 4) throw std::runtime_error("weight must be shape (Cout,Cin,kH,kW)");
            if (bx.ndim != 3) throw std::runtime_error("input must be shape (B,Cin,H,W)");

            // 2) unpack dims
            int Cout = bw.shape[0];
            int Cin  = bw.shape[1];
            int kH   = bw.shape[2];
            int kW   = bw.shape[3];

            int H = bx.shape[1];
            int W = bx.shape[2];

            // 3) allocate output (B, Cout, H, W)
            std::vector<ssize_t> shape { Cout, H, W };
            // strides (in bytes): b=1, c=B, h=B*Cout, w=B*Cout*H
            ssize_t itemsize = sizeof(float);
            std::vector<ssize_t> strides(3);
            strides[0] = itemsize;
            strides[1] = strides[0] * shape[0];
            strides[2] = strides[1] * shape[1];
            // strides[3] = strides[2] * shape[2];
            py::array_t<float, py::array::f_style> out(
                shape,
                strides
            );
            auto bo = out.request();

            // 4) call your impl
            single_conv_layer(
                static_cast<float*>(bw.ptr),
                static_cast<float*>(bx.ptr),
                static_cast<float*>(bo.ptr),
                 Cin, H, W, Cout, kH, kW
            );

            return out;
        },
        py::arg("weight"),
        py::arg("input"),
        "Run a single 2D convolution (no bias, padding=1) on a float tensor"
    );

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
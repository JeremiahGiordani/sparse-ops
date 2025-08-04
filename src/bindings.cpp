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
        .def_property_readonly("nnz", [](const Ellpack &E) {
            std::array<ssize_t, 1> shape   = { (ssize_t)E.m };
            std::array<ssize_t, 1> strides = { sizeof(uint32_t) };
            return py::array_t<uint32_t>(shape, strides, E.nnz.data());
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

    m.def("ellpack_matmul_fortran",
        [](const Ellpack &E,
           py::array_t<float, py::array::f_style | py::array::forcecast> X_arr,
           py::array_t<float, py::array::c_style | py::array::forcecast> bias_arr) {
            auto buf_X    = X_arr.request();
            auto buf_bias = bias_arr.request();
            if (buf_X.ndim != 2) {
                throw std::runtime_error("X must be 2D (n × C)");
            }
            if ((uint32_t)buf_X.shape[1] != E.n) {
                throw std::runtime_error("Input row count must equal E.n");
            }
            if ((uint32_t)buf_bias.size != E.m) {
                throw std::runtime_error("Bias length must equal E.m");
            }

            uint32_t C = static_cast<uint32_t>(buf_X.shape[0]);
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

    m.def("ellpack_matmul_outer",
        [](const Ellpack &E,
           py::array_t<float, py::array::c_style | py::array::forcecast> X_arr,
           py::array_t<float, py::array::c_style | py::array::forcecast> bias_arr) {
            auto buf_X    = X_arr.request();
            auto buf_bias = bias_arr.request();
            if (buf_X.ndim != 2) {
                throw std::runtime_error("X must be 2D (n × C)");
            }
            if ((uint32_t)buf_X.shape[1] != E.n) {
                throw std::runtime_error("Input row count must equal E.m");
            }
            if ((uint32_t)buf_bias.size != E.m) {
                throw std::runtime_error("Bias length must equal E.n");
            }

            uint32_t C = static_cast<uint32_t>(buf_X.shape[0]);
            std::array<ssize_t,2> shape = { (ssize_t)E.m, (ssize_t)C };
            py::array_t<float> Y_arr(shape);
            auto buf_Y = Y_arr.request();

            ellpack_matmul_outer(
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

    m.def("ellpack_matmul_tiled",
        [](const Ellpack &E,
           py::array_t<float, py::array::c_style | py::array::forcecast> X_arr,
           py::array_t<float, py::array::c_style | py::array::forcecast> bias_arr) {
            auto buf_X    = X_arr.request();
            auto buf_bias = bias_arr.request();
            if (buf_X.ndim != 2) {
                throw std::runtime_error("X must be 2D (n × C)");
            }
            // if ((uint32_t)buf_X.shape[1] != E.m) {
            //     throw std::runtime_error("Input row count must equal E.m");
            // }
            // if ((uint32_t)buf_bias.size != E.n) {
            //     throw std::runtime_error("Bias length must equal E.n");
            // }

            uint32_t C = static_cast<uint32_t>(buf_X.shape[0]);
            std::array<ssize_t,2> shape = { (ssize_t)E.n, (ssize_t)C };
            py::array_t<float> Y_arr(shape);
            auto buf_Y = Y_arr.request();

            ellpack_matmul_tiled(
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




    // — Sparse ONNX model —
    py::class_<SparseOnnxModel>(m, "SparseOnnxModel")
        .def(py::init<const std::string&>(),
             "Load an ONNX model and pre-encode weights to ELLPACK")
        .def("run",
            [](const SparseOnnxModel &model,
               py::array_t<float, py::array::c_style | py::array::forcecast> X) {
                auto buf = X.request();
                if (buf.ndim != 2) {
                    throw std::runtime_error("Input must be a 2D array");
                }
                uint32_t n = static_cast<uint32_t>(buf.shape[0]);
                uint32_t C = static_cast<uint32_t>(buf.shape[1]);
                const float* input_ptr = static_cast<const float*>(buf.ptr);

                uint32_t M = model.output_rows();
                std::array<ssize_t,2> out_shape = { (ssize_t)M, (ssize_t)C };
                py::array_t<float> Y(out_shape);
                auto bufY = Y.request();
                model.run(input_ptr,
                          C,
                          static_cast<float*>(bufY.ptr));
                return Y;
            },
            "Run inference on input X (shape n×C), returns output (m×C)")
        .def("output_rows",
             &SparseOnnxModel::output_rows,
             "Number of rows in the final output (m)");
}

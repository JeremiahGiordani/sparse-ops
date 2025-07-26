// ─────────────────────────────────────────────────────────────────────────────
//  bindings.cpp  – pybind11 glue for sparseops_backend
// ─────────────────────────────────────────────────────────────────────────────
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "quasi_dense_encoder.hpp"  // for quasi-dense encoding
#include "bilinear_diagonal_matvec.hpp"  // for quasi_dense_matvec_mt
#include "bilinear_diagonal_matmul.hpp"  // for quasi_dense_matmul_mt

namespace py = pybind11;

// ─────────────────────────────────────────────────────────────────────────────
// Encode to quasi-dense format
static QuasiDense convert_to_quasi_dense_py(
    py::array_t<float, py::array::c_style | py::array::forcecast> W)
{
    auto buf = W.request();
    const float* data = static_cast<float*>(buf.ptr);
    uint32_t m = buf.shape[0];
    uint32_t n = buf.shape[1];  // m rows, n columns
    return convert_to_quasi_dense(data, m, n);
}

// ─────────────────────────────────────────────────────────────────────────────
// Decode from quasi-dense format
// Function takes in only quasi-dense handle and constructs output buffer.
static py::array_t<float> decode_from_quasi_dense_py(
    const QuasiDense& Q)
{
    // Create output buffer of size m*n
    py::array_t<float> W_out({Q.m, Q.n});
    // Fill it with zeros
    std::fill(W_out.mutable_data(), W_out.mutable_data() + size_t(Q.m) * Q.n, 0.0f);

    auto buf = W_out.request();
    if (buf.size != size_t(Q.m) * Q.n) {
        throw std::runtime_error("Output buffer size mismatch");
    }
    decode_from_quasi_dense(Q, static_cast<float*>(buf.ptr));
    return W_out;
}

// ─────────────────────────────────────────────────────────────────────────────
PYBIND11_MODULE(sparseops_backend, m)
{
    m.doc() = "Sparse CPU kernels (BCOO-16)";

    // — QuasiDense handle —
    py::class_<QuasiDense>(m, "QuasiDense")
        .def_readonly("m", &QuasiDense::m)
        .def_readonly("n", &QuasiDense::n)
        .def_readonly("r", &QuasiDense::r)
        // Wd is m×r floats, contiguous in row-major
        .def_property_readonly("Wd", [](const QuasiDense &Q) {
            // shape: {rows=m, cols=r}
            std::array<ssize_t,2> shape   = { (ssize_t)Q.m, (ssize_t)Q.r };
            // strides: bytes to skip to next row, then next column
            std::array<ssize_t,2> strides = {
                sizeof(float) * Q.r,
                sizeof(float)
            };
            return py::array_t<float>(
                shape, strides,
                Q.Wd.ptr    // pointer to first element
            );
        })
        // idx is the same shape, uint32_t
        .def_property_readonly("idx", [](const QuasiDense &Q) {
            std::array<ssize_t,2> shape   = { (ssize_t)Q.m, (ssize_t)Q.r };
            std::array<ssize_t,2> strides = {
                sizeof(uint32_t) * Q.r,
                sizeof(uint32_t)
            };
            return py::array_t<uint32_t>(
                shape, strides,
                Q.idx.data()
            );
        })
        .def_property_readonly("Xt", [](const QuasiDense &Q) {
            std::array<ssize_t,2> shape   = { (ssize_t)Q.m, (ssize_t)Q.r };
            std::array<ssize_t,2> strides = {
                sizeof(float) * Q.r,
                sizeof(float)
            };
            return py::array_t<float>(
                shape, strides,
                Q.Xt.ptr 
            );
        });

    // — API surface —
    m.def("convert_to_quasi_dense", &convert_to_quasi_dense_py,
        "Convert dense NumPy matrix → QuasiDense handle");
    m.def("decode_from_quasi_dense", &decode_from_quasi_dense_py,
        "Convert QuasiDense handle → dense NumPy matrix");

    // 1) Standard matvec from raw x
    // Stores result in y, which must be preallocated.
    m.def("bilinear_diagonal_matvec",
        [](const QuasiDense &Q,
        py::array_t<float> x_arr,
        py::array_t<float> bias_arr) {
            auto buf_x     = x_arr.request();
            auto buf_bias  = bias_arr.request();
            py::array_t<float> y_arr({(ssize_t)Q.m});
            auto buf_y     = y_arr.request();

            quasi_dense_matvec(
                Q,
                static_cast<float*>(buf_x.ptr),
                static_cast<float*>(buf_bias.ptr),
                static_cast<float*>(buf_y.ptr)
            );
            return y_arr;
        },
        "Multithreaded bilinear diagonal matvec (quasi‑dense)");

    m.def("bilinear_diagonal_matmul",
    [](const QuasiDense &Q,
       py::array_t<float, py::array::c_style | py::array::forcecast> X_arr,
       py::array_t<float, py::array::c_style | py::array::forcecast> bias_arr) {
        // Request array info
        auto buf_X    = X_arr.request();
        auto buf_bias = bias_arr.request();
        if (buf_X.ndim != 2) {
            throw std::runtime_error("X must be 2D (n × C)");
        }
        if ((uint32_t)buf_X.shape[0] != Q.n) {
            throw std::runtime_error("Input matrix row count must equal Q.n");
        }
        if ((uint32_t)buf_bias.size != Q.m) {
            throw std::runtime_error("Bias length must equal Q.m");
        }

        // Number of columns C
        uint32_t C = (uint32_t)buf_X.shape[1];

        // Allocate output Y (shape m × C)
        std::array<ssize_t,2> shape = { (ssize_t)Q.m, (ssize_t)C };
        py::array_t<float> Y_arr(shape);
        auto buf_Y = Y_arr.request();

        // Call the C++ kernel
        quasi_dense_matmul(
            Q,
            static_cast<const float*>(buf_X.ptr),
            C,
            static_cast<const float*>(buf_bias.ptr),
            static_cast<float*>(buf_Y.ptr)
        );

        return Y_arr;
    },
    R"pbdoc(
        Multithreaded bilinear-diagonal mat-mul (quasi-dense)

        Computes Y = Q × X + bias, where:
          • Q is a QuasiDense (m×n) matrix
          • X is an n×C float32 array
          • bias is length-m float32 vector (added to each column)
          • Returns Y as an m×C float32 NumPy array
          • threads = number of OpenMP threads (0 = auto)
    )pbdoc");
}

// ─────────────────────────────────────────────────────────────────────────────
//  bindings.cpp  – pybind11 glue for sparseops_backend
// ─────────────────────────────────────────────────────────────────────────────
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "ellpack_encoder.hpp"  // for ELLPACK encoding
#include "ellpack_matvec.hpp"  // for ellpack_matvec_mt
#include "ellpack_matmul.hpp"  // for ellpack_matmul_mt

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
// Function takes in only Ellpack handle and constructs output buffer.
static py::array_t<float> decode_from_ellpack_py(
    const Ellpack& E)
{
    // Create output buffer of size m*n
    py::array_t<float> W_out({E.m, E.n});
    // Fill it with zeros
    std::fill(W_out.mutable_data(), W_out.mutable_data() + size_t(E.m) * E.n, 0.0f);

    auto buf = W_out.request();
    if (buf.size != size_t(E.m) * E.n) {
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
        // Wd is m×r floats, contiguous in row-major
        .def_property_readonly("Wd", [](const Ellpack &E) {
            // shape: {rows=m, cols=r}
            std::array<ssize_t,2> shape   = { (ssize_t)E.m, (ssize_t)E.r };
            // strides: bytes to skip to next row, then next column
            std::array<ssize_t,2> strides = {
                sizeof(float) * E.r,
                sizeof(float)
            };
            return py::array_t<float>(
                shape, strides,
                E.Wd.ptr    // pointer to first element
            );
        })
        // idx is the same shape, uint32_t
        .def_property_readonly("idx", [](const Ellpack &E) {
            std::array<ssize_t,2> shape   = { (ssize_t)E.m, (ssize_t)E.r };
            std::array<ssize_t,2> strides = {
                sizeof(uint32_t) * E.r,
                sizeof(uint32_t)
            };
            return py::array_t<uint32_t>(
                shape, strides,
                E.idx.data()
            );
        })
        .def_property_readonly("Xt", [](const Ellpack &E) {
            std::array<ssize_t,2> shape   = { (ssize_t)E.m, (ssize_t)E.r };
            std::array<ssize_t,2> strides = {
                sizeof(float) * E.r,
                sizeof(float)
            };
            return py::array_t<float>(
                shape, strides,
                E.Xt.ptr
            );
        });

    // — API surface —
    m.def("convert_to_ellpack", &convert_to_ellpack_py,
        "Convert dense NumPy matrix → ELLPACK handle");
    m.def("decode_from_ellpack", &decode_from_ellpack_py,
        "Convert ELLPACK handle → dense NumPy matrix");

    // 1) Standard matvec from raw x
    // Stores result in y, which must be preallocated.
    m.def("ellpack_matvec",
        [](const Ellpack &E,
        py::array_t<float> x_arr,
        py::array_t<float> bias_arr) {
            auto buf_x     = x_arr.request();
            auto buf_bias  = bias_arr.request();
            py::array_t<float> y_arr({(ssize_t)E.m});
            auto buf_y     = y_arr.request();

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
        // Request array info
        auto buf_X    = X_arr.request();
        auto buf_bias = bias_arr.request();
        if (buf_X.ndim != 2) {
            throw std::runtime_error("X must be 2D (n × C), got " + std::to_string(buf_X.ndim) + "D");
        }
        if ((uint32_t)buf_X.shape[0] != E.n) {
            throw std::runtime_error("Input matrix row count must equal E.n");
        }
        if ((uint32_t)buf_bias.size != E.m) {
            throw std::runtime_error("Bias length must equal E.m");
        }

        // Number of columns C
        uint32_t C = (uint32_t)buf_X.shape[1];

        // Allocate output Y (shape m × C)
        std::array<ssize_t,2> shape = { (ssize_t)E.m, (ssize_t)C };
        py::array_t<float> Y_arr(shape);
        auto buf_Y = Y_arr.request();

        // Call the C++ kernel
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
          • threads = number of OpenMP threads (0 = auto)
    )pbdoc");
}

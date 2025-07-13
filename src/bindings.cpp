#include "dense_matvec.hpp"
#include <pybind11/pybind11.h>
#include "bcoo16_encoder.hpp"
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

namespace py = pybind11;

py::array_t<float> run_matvec(
    py::array_t<float, py::array::c_style | py::array::forcecast> A,
    py::array_t<float, py::array::c_style | py::array::forcecast> x,
    py::array_t<float, py::array::c_style | py::array::forcecast> b) {
    auto bufA = A.request(), bufx = x.request(), bufb = b.request();
    size_t M = bufA.shape[0], K = bufA.shape[1];
    auto result = py::array_t<float>(M);
    auto bufy = result.request();
    dense_matvec(static_cast<float*>(bufA.ptr), static_cast<float*>(bufx.ptr),
                 static_cast<float*>(bufb.ptr), static_cast<float*>(bufy.ptr), M, K);
    return result;
}

BCOO16 encode_to_bcoo16_py(py::array_t<float> dense_matrix) {
    auto buf = dense_matrix.request();
    std::vector<std::vector<float>> matrix(buf.shape[0], std::vector<float>(buf.shape[1]));
    for (size_t i = 0; i < buf.shape[0]; ++i) {
        for (size_t j = 0; j < buf.shape[1]; ++j) {
            matrix[i][j] = *reinterpret_cast<float*>(reinterpret_cast<char*>(buf.ptr) + i * buf.strides[0] + j * buf.strides[1]);
        }
    }
    BCOO16 bcoo16 = encode_to_bcoo16(matrix);
    return bcoo16;
}

py::array_t<float> decode_from_bcoo16_py(const BCOO16& bcoo16) {
    std::vector<std::vector<float>> matrix = decode_from_bcoo16(bcoo16);

    if (matrix.empty() || matrix[0].empty()) {
        // Return an empty array
        return py::array_t<float>(std::vector<size_t>{0, 0});
    }

    size_t rows = matrix.size();
    size_t cols = matrix[0].size();

    py::array_t<float> result({rows, cols});
    auto result_buf = result.request();
    float* ptr = static_cast<float*>(result_buf.ptr);

    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            ptr[i * cols + j] = matrix[i][j];
        }
    }

    return result;
}

#include "sparse_matvec.hpp"

PYBIND11_MODULE(sparseops_backend, m) {
    m.def("sparse_matvec_avx512", [](const BCOO16& A, py::array_t<float> x, py::array_t<float> b, py::array_t<float> y) {
        auto buf_x = x.request(), buf_b = b.request(), buf_y = y.request();
        sparse_matvec_avx512(A,
                            static_cast<float*>(buf_x.ptr),
                            static_cast<float*>(buf_b.ptr),
                            static_cast<float*>(buf_y.ptr),
                            buf_y.shape[0]);
    }, "Sparse matrix-vector multiplication using AVX-512");
    py::class_<BCOO16>(m, "BCOO16")
        .def(py::init<>())
        .def_readwrite("row_id", &BCOO16::row_id)
        .def_readwrite("first_col", &BCOO16::first_col)
        .def_readwrite("values", &BCOO16::values)
        .def_readwrite("bitmask", &BCOO16::bitmask)
        .def_readwrite("original_num_rows", &BCOO16::original_num_rows)
        .def_readwrite("original_num_cols", &BCOO16::original_num_cols);


    m.def("run_matvec", &run_matvec, "Dense matrix-vector multiplication with bias");
    m.def("encode_to_bcoo16", &encode_to_bcoo16_py, "Encode dense matrix to BCOO-16 format");
    m.def("decode_from_bcoo16", &decode_from_bcoo16_py, "Decode BCOO-16 format to dense matrix");
}

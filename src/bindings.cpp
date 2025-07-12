#include "dense_matvec.hpp"
#include <pybind11/pybind11.h>
#include "bcoo16_encoder.hpp"
#include <pybind11/numpy.h>

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

py::array_t<float> decode_from_bcoo16_py(BCOO16 bcoo16) {
    std::vector<std::vector<float>> matrix = decode_from_bcoo16(bcoo16);
    py::array_t<float> result({matrix.size(), matrix[0].size()});
    auto result_buf = result.request();
    for (size_t i = 0; i < matrix.size(); ++i) {
        for (size_t j = 0; j < matrix[i].size(); ++j) {
            *reinterpret_cast<float*>(reinterpret_cast<char*>(result_buf.ptr) + i * result_buf.strides[0] + j * result_buf.strides[1]) = matrix[i][j];
        }
    }
    return result;
}

PYBIND11_MODULE(sparseops_backend, m) {
    py::class_<BCOO16>(m, "BCOO16")
        .def(py::init<>())
        .def_readwrite("row_id", &BCOO16::row_id)
        .def_readwrite("first_col", &BCOO16::first_col)
        .def_readwrite("values", &BCOO16::values)
        .def_readwrite("bitmask", &BCOO16::bitmask);

    m.def("run_matvec", &run_matvec, "Dense matrix-vector multiplication with bias");
    m.def("encode_to_bcoo16", &encode_to_bcoo16_py, "Encode dense matrix to BCOO-16 format");
    m.def("decode_from_bcoo16", &decode_from_bcoo16_py, "Decode BCOO-16 format to dense matrix");
}

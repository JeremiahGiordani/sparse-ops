#include "dense_matvec.hpp"
#include <pybind11/pybind11.h>
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

PYBIND11_MODULE(sparseops_backend, m) {
    m.def("run_matvec", &run_matvec, "Dense matrix-vector multiplication with bias");
}

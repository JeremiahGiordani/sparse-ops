#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "sparseops.hpp"

namespace py = pybind11;

PYBIND11_MODULE(sparseops_backend, m) {
    m.doc() = "Ultra-fast sparse matrix multiplication backend";
    
    // PreparedA class
    py::class_<sparseops::PreparedA>(m, "PreparedA")
        .def("rows", &sparseops::PreparedA::rows)
        .def("cols", &sparseops::PreparedA::cols)
        .def("sparsity", &sparseops::PreparedA::sparsity);
    
    // Core functions
    m.def("prepare_csr", [](py::array_t<int64_t> indptr,
                            py::array_t<int32_t> indices,
                            py::array_t<float> data,
                            size_t M, size_t N,
                            int block) {
        return sparseops::prepare_csr(
            static_cast<const int64_t*>(indptr.data()),
            static_cast<const int32_t*>(indices.data()),
            static_cast<const float*>(data.data()),
            M, N, block
        );
    }, "Prepare CSR matrix for optimized multiplication",
       py::arg("indptr"), py::arg("indices"), py::arg("data"),
       py::arg("M"), py::arg("N"), py::arg("block") = 16);
    
    m.def("sgemm", [](const sparseops::PreparedA& A,
                      py::array_t<float> B,
                      py::array_t<float> C,
                      bool accumulate,
                      int repeats) {
        auto B_buf = B.request();
        auto C_buf = C.request();
        
        if (B_buf.ndim != 2 || C_buf.ndim != 2) {
            throw std::runtime_error("B and C must be 2D arrays");
        }
        
        size_t N = B_buf.shape[1];
        
        sparseops::sgemm(A,
                         static_cast<const float*>(B_buf.ptr),
                         static_cast<float*>(C_buf.ptr),
                         N, accumulate, repeats);
    }, "Sparse matrix-dense matrix multiplication",
       py::arg("A"), py::arg("B"), py::arg("C"),
       py::arg("accumulate") = false, py::arg("repeats") = 1);
}

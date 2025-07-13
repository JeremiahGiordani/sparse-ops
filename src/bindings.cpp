// ─────────────────────────────────────────────────────────────────────────────
//  bindings.cpp  – pybind11 glue for sparseops_backend
// ─────────────────────────────────────────────────────────────────────────────
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "dense_matvec.hpp"
#include "bcoo16_encoder.hpp"
#include "sparse_matvec_mt.hpp"  // for multithreaded version
#include "sparse_dispatch.hpp"  


namespace py = pybind11;

// ─────────────────────────────────────────────────────────────────────────────
// Dense FP32 matvec (reference)
static py::array_t<float>
run_matvec_py(py::array_t<float, py::array::c_style | py::array::forcecast> A,
              py::array_t<float, py::array::c_style | py::array::forcecast> x,
              py::array_t<float, py::array::c_style | py::array::forcecast> b)
{
    auto bufA = A.request(), bufx = x.request(), bufb = b.request();
    size_t M = bufA.shape[0], K = bufA.shape[1];

    py::array_t<float> y(M);
    auto bufy = y.request();

    dense_matvec(static_cast<float*>(bufA.ptr),
                 static_cast<float*>(bufx.ptr),
                 static_cast<float*>(bufb.ptr),
                 static_cast<float*>(bufy.ptr),
                 M, K);
    return y;
}

// ─────────────────────────────────────────────────────────────────────────────
// Encode NumPy dense → BCOO16  (opaque handle)
static BCOO16 encode_to_bcoo16_py(py::array_t<float,
                               py::array::c_style | py::array::forcecast> dense)
{
    auto buf = dense.request();
    const size_t rows = buf.shape[0];
    const size_t cols = buf.shape[1];

    // Copy into std::vector<std::vector<float>>
    std::vector<std::vector<float>> mat(rows, std::vector<float>(cols));
    const float* src = static_cast<float*>(buf.ptr);

    for (size_t r = 0; r < rows; ++r)
        std::copy(src + r * cols, src + (r + 1) * cols, mat[r].begin());

    return encode_to_bcoo16(mat);
}

// ─────────────────────────────────────────────────────────────────────────────
// Decode BCOO16 → NumPy dense  (for testing)
static py::array_t<float> decode_from_bcoo16_py(const BCOO16& bcoo)
{
    auto dense = decode_from_bcoo16(bcoo);
    size_t rows = dense.size();
    size_t cols = rows ? dense[0].size() : 0;

    py::array_t<float> out({rows, cols});
    auto buf = out.request();
    float* dst = static_cast<float*>(buf.ptr);

    for (size_t r = 0; r < rows; ++r)
        std::copy(dense[r].begin(), dense[r].end(), dst + r * cols);

    return out;
}

// ─────────────────────────────────────────────────────────────────────────────
// Sparse AVX-512 matvec  – returns a fresh NumPy array
// static py::array_t<float>
// sparse_matvec_avx512_py(const BCOO16& A,
//                         py::array_t<float, py::array::c_style | py::array::forcecast> x,
//                         py::array_t<float, py::array::c_style | py::array::forcecast> b)
// {
// #if !defined(__AVX512F__)
//     throw std::runtime_error("CPU lacks AVX-512F support!");
// #endif
//     auto bufx = x.request(), bufb = b.request();
//     if (bufx.ndim != 1 || bufb.ndim != 1)
//         throw std::runtime_error("x and bias must be 1-D arrays");

//     size_t M = A.original_num_rows;
//     if (bufx.shape[0] != static_cast<ssize_t>(A.original_num_cols))
//         throw std::runtime_error("x length != matrix cols");
//     if (bufb.shape[0] != static_cast<ssize_t>(M))
//         throw std::runtime_error("bias length != matrix rows");

//     py::array_t<float> y(M);
//     auto bufy = y.request();

//     sparse_matvec_avx512(A,
//                          static_cast<float*>(bufx.ptr),
//                          static_cast<float*>(bufb.ptr),
//                          static_cast<float*>(bufy.ptr),
//                          M);
//     return y;
// }

static py::array_t<float>
sparse_matvec_avx512_py(const BCOO16& A,
                        py::array_t<float> x,
                        py::array_t<float> b)
{
    auto bufx = x.request(), bufb = b.request();
    size_t M = A.original_num_rows;

    py::array_t<float> y(M);
    auto bufy = y.request();
    std::memcpy(bufy.ptr, bufb.ptr, M*sizeof(float));

    auto fn = get_spmv_kernel(A);
    fn(A.blocks.data(), A.blocks.size(),
       static_cast<float*>(bufx.ptr),
       static_cast<float*>(bufy.ptr));
    return y;
}

// ─────────────────────────────────────────────────────────────────────────────
PYBIND11_MODULE(sparseops_backend, m)
{
    m.doc() = "Sparse CPU kernels (BCOO-16)";

    // — opaque BCOO16 handle (blocks vector not exposed to Python) —
    py::class_<BCOO16>(m, "BCOO16")
        .def(py::init<>())
        .def_readwrite("original_num_rows", &BCOO16::original_num_rows)
        .def_readwrite("original_num_cols", &BCOO16::original_num_cols);

    // — API surface —
    m.def("encode_to_bcoo16",  &encode_to_bcoo16_py,
          "Convert dense NumPy matrix → BCOO-16 handle");
    m.def("decode_from_bcoo16", &decode_from_bcoo16_py,
          "Convert BCOO-16 handle → dense NumPy matrix (testing only)");

    m.def("sparse_matvec_avx512", &sparse_matvec_avx512_py,
          "y = A @ x + b  (AVX-512, BCOO-16 matrix)");

    m.def("run_matvec", &run_matvec_py,
          "Reference dense matvec (Naive C++ kernel)");
    m.def("sparse_matvec_avx512_mt",
        [](const BCOO16& A,
            py::array_t<float> x,
            py::array_t<float> b,
            int threads) {
            auto bufx = x.request(), bufb = b.request();
            py::array_t<float> y(A.original_num_rows);
            auto bufy = y.request();

            sparse_matvec_avx512_mt(
                A,
                static_cast<float*>(bufx.ptr),
                static_cast<float*>(bufb.ptr),
                static_cast<float*>(bufy.ptr),
                threads);
            return y;
        },
        py::arg("A"), py::arg("x"), py::arg("b"),
        py::arg("threads") = 0,
        "Multithreaded sparse matvec (AVX-512)");

}

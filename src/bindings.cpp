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
#include "dense_block_kernel.hpp"  // for dense_block_kernel
#include "quasi_dense_encoder.hpp"  // for quasi-dense encoding
#include "bilinear_diagonal_matvec.hpp"  // for quasi_dense_matvec_mt


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
// Transform input vector x into XtDense format
static XtDense transform_input_py(
    const QuasiDense& Q,
    py::array_t<float, py::array::c_style | py::array::forcecast> x)
{
    auto buf = x.request();
    if (buf.size != Q.n) {
        throw std::runtime_error("Input vector size mismatch");
    }
    return transform_input(Q, static_cast<float*>(buf.ptr));
}


// ─────────────────────────────────────────────────────────────────────────────
PYBIND11_MODULE(sparseops_backend, m)
{
    m.doc() = "Sparse CPU kernels (BCOO-16)";

    // - ─ BCOO16Block handle ────────────────────────────────────────────────
    py::class_<BCOO16Block>(m, "BCOO16Block")
        .def(py::init<>())
        .def_readwrite("row_id", &BCOO16Block::row_id)
        .def_readwrite("first_col", &BCOO16Block::first_col)
        .def_readwrite("bitmask", &BCOO16Block::bitmask)
        .def_readwrite("val_off", &BCOO16Block::val_off);

    // — BCOO16 handle —
    py::class_<BCOO16>(m, "BCOO16")
        .def(py::init<>())
        .def_readwrite("original_num_rows", &BCOO16::original_num_rows)
        .def_readwrite("original_num_cols", &BCOO16::original_num_cols)
        .def_property_readonly("blocks",
            [](const BCOO16& b) {
                py::list out;
                for (const auto& blk : b.blocks) {
                    /* (row_id, first_col, bitmask, val_off) */
                    out.append(py::make_tuple(
                        blk.row_id,
                        blk.first_col,
                        blk.bitmask,
                        blk.val_off));
                }
                return out;        // Python sees a list[tuple[4]]
            })
        .def_property_readonly("values",
            [](const BCOO16& b) {
                return py::array_t<float>(
                    b.values.size(), b.values.data());
            });

    // — QuasiDense handle —
    py::class_<QuasiDense>(m, "QuasiDense")
        .def(py::init<>())
        .def_readwrite("m", &QuasiDense::m)
        .def_readwrite("n", &QuasiDense::n)
        .def_readwrite("r", &QuasiDense::r)
        .def_property_readonly("Wd",
            [](const QuasiDense& Q) {
                return py::array_t<float>(
                    Q.Wd.size(), Q.Wd.data());
            })
        .def_property_readonly("idx",
            [](const QuasiDense& Q) {
                return py::array_t<uint32_t>(
                    Q.idx.size(), Q.idx.data());
            });

    // — XtDense handle —
    py::class_<XtDense>(m, "XtDense")
        .def(py::init<>())
        .def_readwrite("m", &XtDense::m)
        .def_readwrite("r", &XtDense::r)
        .def_property_readonly("Xt",
            [](const XtDense& X) {
                return py::array_t<float>(
                    X.Xt.size(), X.Xt.data());
            });

    // — API surface —
    m.def("encode_to_bcoo16",  &encode_to_bcoo16_py,
          "Convert dense NumPy matrix → BCOO-16 handle");
    m.def("decode_from_bcoo16", &decode_from_bcoo16_py,
          "Convert BCOO-16 handle → dense NumPy matrix (testing only)");
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
    m.def("dense_block_kernel",
      [](py::array_t<float, py::array::c_style|py::array::forcecast> A,
         py::array_t<float, py::array::c_style|py::array::forcecast> x,
         py::array_t<float, py::array::c_style|py::array::forcecast> b)
      {
          auto bufA = A.request(), bufx = x.request(), bufb = b.request();
          size_t M = bufA.shape[0], K = bufA.shape[1];
          py::array_t<float> y(M);
          auto bufy = y.request();

          dense_block_kernel(static_cast<float*>(bufA.ptr),
                             static_cast<float*>(bufx.ptr),
                             static_cast<float*>(bufb.ptr),
                             static_cast<float*>(bufy.ptr),
                             M, K);
          return y;
      },
      "Baseline dense y = A·x + b (blocked AVX-512)");
      m.def("convert_to_quasi_dense", &convert_to_quasi_dense_py,
            "Convert dense NumPy matrix → QuasiDense handle");
      m.def("decode_from_quasi_dense", &decode_from_quasi_dense_py,
            "Convert QuasiDense handle → dense NumPy matrix");
      m.def("transform_input", &transform_input_py,
            "Transform input vector x into XtDense format");
      m.def("bilinear_diagonal_matvec_mt",
            [](const QuasiDense& Q,
               const XtDense& X,
               py::array_t<float> bias,
               int threads) {
                auto buf_bias = bias.request();
                py::array_t<float> y(Q.m);
                auto buf_y = y.request();

                quasi_dense_matvec_mt(
                    Q,
                    X,
                    static_cast<float*>(buf_bias.ptr),
                    static_cast<float*>(buf_y.ptr),
                    threads);
                return y;
            },
            "Multithreaded bilinear diagonal matvec (quasi-dense)");
}

// include/quasi_dense_encoder.hpp
#pragma once

#include <vector>
#include <cstdint>

/// Packed “quasi‑dense” representation.
/// Rows of the original m×n matrix are collapsed into m rows × r columns,
/// where r = max non‑zeros per row.
/// - Wd[i*r + j]: jth non‑zero value of row i (or zero padding).
/// - idx[i*r + j]: original column index of that value.
struct QuasiDense {
    uint32_t m;          ///< number of rows
    uint32_t n;          ///< original number of columns
    uint32_t r;          ///< max non‑zeros per row
    std::vector<float>  Wd;  ///< packed non‑zero values (size m*r)
    std::vector<uint32_t> idx; ///< original column indices (size m*r)
};

/// Transformed input: for each row i, a contiguous block of r values:
/// Xt[i*r + j] = x[ idx[i*r + j] ].
struct XtDense {
    uint32_t m;
    uint32_t r;
    std::vector<float> Xt; ///< gathered x values (size m*r)
};

/// Encode a dense m×n matrix into quasi‑dense form.
QuasiDense convert_to_quasi_dense(const float* W, uint32_t m, uint32_t n);

/// Decode quasi‑dense back into a full dense m×n matrix.
/// W_out must point to an array of size m*n.
void decode_from_quasi_dense(const QuasiDense& Q, float* W_out);

/// Transform input vector x (length n) into XtDense (m×r).
XtDense transform_input(const QuasiDense& Q, const float* x);
#pragma once
#include <vector>
#include <cstdint>
#include "aligned_buffer.hpp"

struct QuasiDense {
    uint32_t m, n, r;
    AlignedBuffer Wd;               // aligned m*r floats
    std::vector<uint32_t> idx;      // m*r indices

    QuasiDense(uint32_t _m, uint32_t _n, uint32_t _r)
      : m(_m), n(_n), r(_r),
        Wd(size_t(_m) * _r),
        idx(size_t(_m) * _r)
    {}
};

struct XtDense {
    uint32_t m, r;
    AlignedBuffer Xt;               // aligned m*r floats

    XtDense(uint32_t _m, uint32_t _r)
      : m(_m), r(_r),
        Xt(size_t(_m) * _r)
    {}
};

/// Encode a dense m×n matrix into quasi‑dense form.
QuasiDense convert_to_quasi_dense(const float* W, uint32_t m, uint32_t n);

/// Decode quasi‑dense back into a full dense m×n matrix.
/// W_out must point to an array of size m*n.
void decode_from_quasi_dense(const QuasiDense& Q, float* W_out);

/// Transform input vector x (length n) into XtDense (m×r).
XtDense transform_input(const QuasiDense& Q, const float* x);
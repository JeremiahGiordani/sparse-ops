#pragma once
#include <vector>
#include <cstdint>
#include "aligned_buffer.hpp"

struct Ellpack {
    uint32_t m;  ///< rows
    uint32_t n;  ///< original columns
    uint32_t r;  ///< max nonzeros per row

    AlignedBuffer           Wd;      ///< m*r floats
    std::vector<uint32_t>   idx;     ///< m*r indices
    AlignedBuffer           Xt;      ///< m*r floats (gather buffer)
    std::vector<uint32_t>   nnz;     ///< m counts

    Ellpack(uint32_t _m, uint32_t _n, uint32_t _r)
      : m(_m), n(_n), r(_r),
        Wd(size_t(_m)*_r),
        idx(size_t(_m)*_r),
        Xt(size_t(_m)*_r),
        nnz(_m)
    {}
};


/// Encode a dense m×n matrix into ELLPACK form.
Ellpack convert_to_ellpack(const float* W, uint32_t m, uint32_t n);

/// Decode ELLPACK back into a full dense m×n matrix.
/// W_out must point to an array of size m*n.
void decode_from_ellpack(const Ellpack& E, float* W_out);
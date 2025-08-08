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

struct SortedEllpack {
    uint32_t m;          // rows
    uint32_t n;          // cols
    uint32_t KB;         // block width in columns (e.g., 16 or 32)
    uint32_t NB;         // number of column blocks = ceil(n / KB)

    // CSR-like over (row, block):
    // rowblk_ptr has size m*(NB+1); rowblk_ptr[i*(NB+1)+b]..[+b+1] indexes into krel/Wd
    std::vector<uint32_t> rowblk_ptr;

    // flattened payload for all (row, block) segments, sorted by block then by row
    AlignedBuffer         Wd;       // float[ total_nnz ]
    std::vector<uint16_t> krel;     // uint16_t[ total_nnz ], col % KB (KB <= 65535)
    // optional: you can keep ELL-like nnz per (row,block) if you prefer, but ptrs suffice

    SortedEllpack(uint32_t _m, uint32_t _n, uint32_t _KB,
                  size_t total_nnz, bool align64=true)
        : m(_m), n(_n), KB(_KB),
          NB( (_n + _KB - 1) / _KB ),
          rowblk_ptr(size_t(_m) * (size_t(NB) + 1u), 0),
          Wd(total_nnz),
          krel(total_nnz, 0)
    {}
};



/// Encode a dense m×n matrix into ELLPACK form.
Ellpack convert_to_ellpack(const float* W, uint32_t m, uint32_t n);

SortedEllpack convert_to_sorted_ellpack(const float* W, uint32_t m, uint32_t n, uint32_t KB);

/// Decode ELLPACK back into a full dense m×n matrix.
/// W_out must point to an array of size m*n.
void decode_from_ellpack(const Ellpack& E, float* W_out);
#pragma once
#include <vector>
#include <cstdint>
#include "aligned_buffer.hpp"

struct CBCOOBlock {
    // for this column block: all entries of all k_rel packed together
    std::vector<uint32_t> koffs;   // length KB+1
    std::vector<uint32_t> rows;    // length nnz_block
    AlignedBuffer         val;     // length nnz_block (float), 64B aligned    

    CBCOOBlock() = default;
    CBCOOBlock(CBCOOBlock&&) noexcept = default;
    CBCOOBlock& operator=(CBCOOBlock&&) noexcept = default;
    CBCOOBlock(const CBCOOBlock&) = delete;
    CBCOOBlock& operator=(const CBCOOBlock&) = delete;
    
};

struct CBCOO {
    uint32_t m, n, KB, NB;        // NB = ceil(n/KB)
    std::vector<CBCOOBlock> blocks; // size NB

    CBCOO() = default; // not strictly required, but helps some toolchains
    CBCOO(CBCOO&&) noexcept = default;
    CBCOO& operator=(CBCOO&&) noexcept = default;
    CBCOO(const CBCOO&) = delete;
    CBCOO& operator=(const CBCOO&) = delete;
    ~CBCOO() = default;
};

CBCOO convert_to_cbcoo(const float* W, uint32_t m, uint32_t n, uint32_t KB);
#pragma once

#include <vector>
#include <memory>
#include <cstdint>
#include <cstring>

namespace sparseops {
namespace detail {

// Blocked CSR representation for 16×k blocks
struct BlockedCSR {
    std::vector<float> values;           // Non-zero values
    std::vector<int32_t> col_indices;    // Column indices (simplified for now)
    std::vector<uint8_t> col_deltas;     // Delta-encoded column indices
    std::vector<uint32_t> block_indptr;  // Pointer to start of each row
    std::vector<uint16_t> block_nnz;     // Number of non-zeros per row
    
    size_t M, N;                         // Matrix dimensions
    int block_size;                      // Block size (typically 16)
    size_t total_nnz;                    // Total non-zeros
    
    BlockedCSR(size_t M, size_t N, int block_size) 
        : M(M), N(N), block_size(block_size), total_nnz(0) {}
};

// Packed matrix representation optimized for micro-kernels
class PackedMatrix {
public:
    PackedMatrix(const int64_t* indptr, const int32_t* indices, 
                 const float* data, size_t M, size_t N, int block_size);
    
    size_t rows() const { return bcsr_->M; }
    size_t cols() const { return bcsr_->N; }
    float sparsity() const { return 1.0f - static_cast<float>(bcsr_->total_nnz) / (bcsr_->M * bcsr_->N); }
    
    const BlockedCSR* get_bcsr() const { return bcsr_.get(); }
    
private:
    std::unique_ptr<BlockedCSR> bcsr_;
    
    void convert_to_blocked_csr(const int64_t* indptr, const int32_t* indices, 
                                const float* data, size_t M, size_t N, int block_size);
};

} // namespace detail
} // namespace sparseops

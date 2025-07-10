#include "packed_matrix.hpp"
#include <algorithm>
#include <cmath>

namespace sparseops {
namespace detail {

PackedMatrix::PackedMatrix(const int64_t* indptr, const int32_t* indices, 
                           const float* data, size_t M, size_t N, int block_size) {
    convert_to_blocked_csr(indptr, indices, data, M, N, block_size);
}

void PackedMatrix::convert_to_blocked_csr(const int64_t* indptr, const int32_t* indices, 
                                          const float* data, size_t M, size_t N, int block_size) {
    bcsr_ = std::make_unique<BlockedCSR>(M, N, block_size);
    
    // Store all values and indices directly (simplified approach)
    for (size_t row = 0; row < M; ++row) {
        for (int64_t idx = indptr[row]; idx < indptr[row + 1]; ++idx) {
            if (data[idx] != 0.0f) {
                bcsr_->values.push_back(data[idx]);
                bcsr_->col_indices.push_back(indices[idx]);
                bcsr_->total_nnz++;
            }
        }
    }
    
    // Create row pointers
    bcsr_->block_indptr.resize(M + 1);
    size_t val_idx = 0;
    for (size_t row = 0; row < M; ++row) {
        bcsr_->block_indptr[row] = val_idx;
        for (int64_t idx = indptr[row]; idx < indptr[row + 1]; ++idx) {
            if (data[idx] != 0.0f) {
                val_idx++;
            }
        }
    }
    bcsr_->block_indptr[M] = val_idx;
    
    // Store row nnz counts
    bcsr_->block_nnz.resize(M);
    for (size_t row = 0; row < M; ++row) {
        bcsr_->block_nnz[row] = bcsr_->block_indptr[row + 1] - bcsr_->block_indptr[row];
    }
}

} // namespace detail
} // namespace sparseops

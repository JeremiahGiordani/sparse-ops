#include "sparseops.hpp"
#include "packed_matrix.hpp"
#include "kernel_cache.hpp"
#include <thread>
#include <vector>
#include <algorithm>
#include <omp.h>

namespace sparseops {

// Global kernel cache instance
static detail::KernelCache g_kernel_cache;

PreparedA::PreparedA(std::shared_ptr<detail::PackedMatrix> handle) 
    : handle_(std::move(handle)) {}

PreparedA::~PreparedA() = default;

size_t PreparedA::rows() const {
    return handle_->rows();
}

size_t PreparedA::cols() const {
    return handle_->cols();
}

float PreparedA::sparsity() const {
    return handle_->sparsity();
}

const detail::PackedMatrix* PreparedA::get_packed_matrix() const {
    return handle_.get();
}

PreparedA prepare_csr(const int64_t* indptr,
                      const int32_t* indices,
                      const float*   data,
                      size_t M, size_t N,
                      int block) {
    auto packed = std::make_shared<detail::PackedMatrix>(indptr, indices, data, M, N, block);
    return PreparedA(packed);
}

void sgemm(const PreparedA& A,
           const float* B,  // column-major N×K
           float* C,        // column-major M×N
           size_t N,
           bool accumulate,
           int   repeats) {
    
    const auto* bcsr = A.get_packed_matrix()->get_bcsr();
    
    // For multiple repeats, we can optimize by reusing computations
    for (int repeat = 0; repeat < repeats; ++repeat) {
        if (!accumulate || repeat == 0) {
            // Zero out C on first iteration or if not accumulating
            std::fill(C, C + A.rows() * N, 0.0f);
        }
        
        // Process each row directly (parallelized)
        #pragma omp parallel for schedule(dynamic, 64)
        for (size_t row = 0; row < bcsr->M; ++row) {
            uint32_t val_start = bcsr->block_indptr[row];
            uint32_t val_end = bcsr->block_indptr[row + 1];
            uint16_t nnz_count = bcsr->block_nnz[row];
            
            if (nnz_count == 0) continue;
            
            // Create a simple kernel key
            detail::KernelKey key;
            key.block_size = bcsr->block_size;
            key.dtype_id = 0;  // f32
            key.sparsity_mask.resize(nnz_count, true);
            
            auto kernel = g_kernel_cache.get_or_create_kernel(key);
            
            // Execute kernel for this row
            float* C_row = C + row * N;
            
            kernel(bcsr->values.data() + val_start,
                   bcsr->col_indices.data() + val_start,
                   B,
                   C_row,
                   nnz_count,
                   N);
        }
    }
}

} // namespace sparseops

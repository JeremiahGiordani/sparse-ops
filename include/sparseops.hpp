#pragma once

#include <cstddef>
#include <memory>
#include <cstdint>

namespace sparseops {

namespace detail {
    class PackedMatrix;
}

// Immutable after construction
class PreparedA {
public:
    PreparedA(std::shared_ptr<detail::PackedMatrix> handle);
    ~PreparedA();
    
    size_t rows() const;
    size_t cols() const;
    float sparsity() const;
    
    // Internal access for sgemm
    const detail::PackedMatrix* get_packed_matrix() const;
    
private:
    std::shared_ptr<detail::PackedMatrix> handle_;
};

// Core API functions
PreparedA prepare_csr(const int64_t* indptr,
                      const int32_t* indices,
                      const float*   data,
                      size_t M, size_t N,
                      int block = 16);

void sgemm(const PreparedA& A,
           const float* B,  // column-major N×K
           float* C,        // column-major M×N
           size_t N,
           bool accumulate = false,
           int   repeats   = 1);

} // namespace sparseops

#include "kernel_cache.hpp"
#include "avx512_kernels.hpp"
#include <cstring>
#include <algorithm>
#include <immintrin.h>

namespace sparseops {
namespace detail {

KernelCache::KernelCache() {}

KernelCache::~KernelCache() {}

MicroKernel KernelCache::get_or_create_kernel(const KernelKey& key) {
    auto it = cache_.find(key);
    if (it != cache_.end()) {
        return it->second;
    }
    
    MicroKernel kernel = generate_kernel(key);
    cache_[key] = kernel;
    return kernel;
}

MicroKernel KernelCache::generate_kernel(const KernelKey& key) {
    // Choose kernel based on sparsity pattern
    int nnz_count = 0;
    for (bool b : key.sparsity_mask) {
        if (b) nnz_count++;
    }
    
    if (nnz_count < 8) {
        // Very sparse - use scatter-gather
        return [](const float* A_vals, const int32_t* col_indices, 
                  const float* B_panel, float* C_row, 
                  int nnz, int N) {
            AVX512Kernels::spmv_sparse_f32(A_vals, col_indices, B_panel, C_row, nnz, N);
        };
    } else if (nnz_count > 64) {
        // Dense-ish - use chunk processing
        return [](const float* A_vals, const int32_t* col_indices, 
                  const float* B_panel, float* C_row, 
                  int nnz, int N) {
            AVX512Kernels::spmv_dense_f32(A_vals, col_indices, B_panel, C_row, nnz, N);
        };
    } else {
        // Medium sparsity - use standard AVX-512
        return [](const float* A_vals, const int32_t* col_indices, 
                  const float* B_panel, float* C_row, 
                  int nnz, int N) {
            AVX512Kernels::spmv_avx512_f32(A_vals, col_indices, B_panel, C_row, nnz, N);
        };
    }
}

MicroKernel KernelCache::generate_generic_kernel(int block_size) {
    return [](const float* A_vals, const int32_t* col_indices, 
              const float* B_panel, float* C_row, 
              int nnz, int N) {
        
        // Optimized kernel with prefetching and better memory patterns
        for (int i = 0; i < nnz; ++i) {
            float a_val = A_vals[i];
            int32_t col = col_indices[i];
            
            // Prefetch next iteration data
            if (i + 1 < nnz) {
                __builtin_prefetch(&B_panel[col_indices[i + 1] * N], 0, 1);
                __builtin_prefetch(&A_vals[i + 1], 0, 1);
            }
            
            const float* B_col = B_panel + col * N;
            
            // Optimized for different N sizes
            if (N >= 16) {
                // Process 16 elements at a time using two AVX256 vectors
                int n = 0;
                for (; n <= N - 16; n += 16) {
                    __m256 a_vec = _mm256_set1_ps(a_val);
                    
                    // First 8 elements
                    __m256 b_vec1 = _mm256_loadu_ps(B_col + n);
                    __m256 c_vec1 = _mm256_loadu_ps(C_row + n);
                    __m256 result1 = _mm256_fmadd_ps(a_vec, b_vec1, c_vec1);
                    _mm256_storeu_ps(C_row + n, result1);
                    
                    // Second 8 elements
                    __m256 b_vec2 = _mm256_loadu_ps(B_col + n + 8);
                    __m256 c_vec2 = _mm256_loadu_ps(C_row + n + 8);
                    __m256 result2 = _mm256_fmadd_ps(a_vec, b_vec2, c_vec2);
                    _mm256_storeu_ps(C_row + n + 8, result2);
                }
                
                // Handle remaining elements with 8-element chunks
                for (; n <= N - 8; n += 8) {
                    __m256 a_vec = _mm256_set1_ps(a_val);
                    __m256 b_vec = _mm256_loadu_ps(B_col + n);
                    __m256 c_vec = _mm256_loadu_ps(C_row + n);
                    __m256 result = _mm256_fmadd_ps(a_vec, b_vec, c_vec);
                    _mm256_storeu_ps(C_row + n, result);
                }
                
                // Handle final scalar elements
                for (; n < N; ++n) {
                    C_row[n] += a_val * B_col[n];
                }
            } else if (N >= 8) {
                // Process 8 elements at a time using AVX
                int n = 0;
                for (; n <= N - 8; n += 8) {
                    __m256 a_vec = _mm256_set1_ps(a_val);
                    __m256 b_vec = _mm256_loadu_ps(B_col + n);
                    __m256 c_vec = _mm256_loadu_ps(C_row + n);
                    __m256 result = _mm256_fmadd_ps(a_vec, b_vec, c_vec);
                    _mm256_storeu_ps(C_row + n, result);
                }
                
                // Handle remaining elements
                for (; n < N; ++n) {
                    C_row[n] += a_val * B_col[n];
                }
            } else {
                // Simple scalar loop for small N
                for (int n = 0; n < N; ++n) {
                    C_row[n] += a_val * B_col[n];
                }
            }
        }
    };
}

} // namespace detail
} // namespace sparseops

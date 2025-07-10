#pragma once

#include <immintrin.h>
#include <cstdint>
#include <cstring>

namespace sparseops {
namespace detail {

// AVX-512 optimized kernels
class AVX512Kernels {
public:
    // Highly optimized sparse matrix-vector multiplication using AVX-512
    static void spmv_avx512_f32(const float* A_vals, const int32_t* col_indices, 
                                 const float* B_panel, float* C_row, 
                                 int nnz, int N) {
        
        // Process 16 elements at a time using AVX-512
        const int simd_width = 16;
        int i = 0;
        
        // Clear output
        std::memset(C_row, 0, N * sizeof(float));
        
        // Main SIMD loop - process 16 outputs at a time
        for (; i + simd_width <= N; i += simd_width) {
            __m512 acc = _mm512_setzero_ps();
            
            // Inner loop over non-zeros
            for (int nz = 0; nz < nnz; ++nz) {
                float a_val = A_vals[nz];
                int32_t col = col_indices[nz];
                
                // Broadcast A value
                __m512 a_vec = _mm512_set1_ps(a_val);
                
                // Load B values (stride pattern)
                __m512 b_vec = _mm512_loadu_ps(&B_panel[col * N + i]);
                
                // FMA: acc += a_vec * b_vec
                acc = _mm512_fmadd_ps(a_vec, b_vec, acc);
            }
            
            // Store result
            _mm512_storeu_ps(&C_row[i], acc);
        }
        
        // Handle remaining elements
        for (; i < N; ++i) {
            float sum = 0.0f;
            for (int nz = 0; nz < nnz; ++nz) {
                sum += A_vals[nz] * B_panel[col_indices[nz] * N + i];
            }
            C_row[i] = sum;
        }
    }
    
    // Specialized kernel for very sparse rows (< 8 non-zeros)
    static void spmv_sparse_f32(const float* A_vals, const int32_t* col_indices, 
                                 const float* B_panel, float* C_row, 
                                 int nnz, int N) {
        
        std::memset(C_row, 0, N * sizeof(float));
        
        // For very sparse rows, use scatter-gather approach
        for (int nz = 0; nz < nnz; ++nz) {
            float a_val = A_vals[nz];
            int32_t col = col_indices[nz];
            const float* b_row = &B_panel[col * N];
            
            // SIMD addition
            int i = 0;
            const int simd_width = 16;
            __m512 a_vec = _mm512_set1_ps(a_val);
            
            for (; i + simd_width <= N; i += simd_width) {
                __m512 b_vec = _mm512_loadu_ps(&b_row[i]);
                __m512 c_vec = _mm512_loadu_ps(&C_row[i]);
                __m512 result = _mm512_fmadd_ps(a_vec, b_vec, c_vec);
                _mm512_storeu_ps(&C_row[i], result);
            }
            
            // Handle remainder
            for (; i < N; ++i) {
                C_row[i] += a_val * b_row[i];
            }
        }
    }
    
    // Specialized kernel for dense-ish rows (> 64 non-zeros)
    static void spmv_dense_f32(const float* A_vals, const int32_t* col_indices, 
                                const float* B_panel, float* C_row, 
                                int nnz, int N) {
        
        std::memset(C_row, 0, N * sizeof(float));
        
        const int simd_width = 16;
        
        // Process in chunks for better cache usage
        for (int i = 0; i < N; i += simd_width) {
            int end_i = (i + simd_width <= N) ? i + simd_width : N;
            __m512 acc = _mm512_setzero_ps();
            
            for (int nz = 0; nz < nnz; ++nz) {
                float a_val = A_vals[nz];
                int32_t col = col_indices[nz];
                __m512 a_vec = _mm512_set1_ps(a_val);
                
                if (i + simd_width <= N) {
                    __m512 b_vec = _mm512_loadu_ps(&B_panel[col * N + i]);
                    acc = _mm512_fmadd_ps(a_vec, b_vec, acc);
                } else {
                    // Handle partial load
                    for (int j = i; j < end_i; ++j) {
                        C_row[j] += a_val * B_panel[col * N + j];
                    }
                }
            }
            
            if (i + simd_width <= N) {
                _mm512_storeu_ps(&C_row[i], acc);
            }
        }
    }
};

} // namespace detail
} // namespace sparseops

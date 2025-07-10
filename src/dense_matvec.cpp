#include "dense_matvec.hpp"
#include <cstddef>
#include <cstring>
#include <omp.h>
#include <immintrin.h>
#include <cstdlib>
#include <cpuid.h>
#include <algorithm>

// Try several block sizes and pick the best for this run
constexpr size_t MC_CANDIDATES[] = {32, 64, 128};
constexpr size_t KC_CANDIDATES[] = {32, 64, 128, 256};

inline float* aligned_alloc_float(size_t n) {
    void* ptr = nullptr;
    if (posix_memalign(&ptr, 64, n * sizeof(float)) != 0) return nullptr;
    return reinterpret_cast<float*>(ptr);
}

static bool cpu_supports_avx512() {
    unsigned int eax, ebx, ecx, edx;
    if (!__get_cpuid_count(7, 0, &eax, &ebx, &ecx, &edx)) return false;
    return (ebx & (1 << 16));
}

// Prefetch helper
inline void prefetch_L1(const void* ptr) {
    _mm_prefetch(reinterpret_cast<const char*>(ptr), _MM_HINT_T0);
}

// Micro-kernel for a block (AVX-512)
void matvec_block_avx512(const float* A, const float* x, float* y, size_t M, size_t K) {
    for (size_t i = 0; i < M; ++i) {
        float acc = 0.0f;
        size_t k = 0;
        __m512 sum_vec = _mm512_setzero_ps();
        for (; k + 15 < K; k += 16) {
            __m512 avec = _mm512_load_ps(A + i * K + k);
            __m512 xvec = _mm512_load_ps(x + k);
            sum_vec = _mm512_add_ps(sum_vec, _mm512_mul_ps(avec, xvec));
        }
        float temp[16];
        _mm512_store_ps(temp, sum_vec);
        for (int j = 0; j < 16; ++j) acc += temp[j];
        for (; k < K; ++k) acc += A[i * K + k] * x[k];
        y[i] = acc;
    }
}

// Micro-kernel for a block (AVX2)
void matvec_block_avx2(const float* A, const float* x, float* y, size_t M, size_t K) {
    for (size_t i = 0; i < M; ++i) {
        float acc = 0.0f;
        size_t k = 0;
        __m256 sum_vec = _mm256_setzero_ps();
        for (; k + 7 < K; k += 8) {
            __m256 avec = _mm256_load_ps(A + i * K + k);
            __m256 xvec = _mm256_load_ps(x + k);
            sum_vec = _mm256_add_ps(sum_vec, _mm256_mul_ps(avec, xvec));
        }
        float temp[8];
        _mm256_store_ps(temp, sum_vec);
        for (int j = 0; j < 8; ++j) acc += temp[j];
        for (; k < K; ++k) acc += A[i * K + k] * x[k];
        y[i] = acc;
    }
}

// Main matvec with autotuned blocking, bias, and prefetching
void dense_matvec(const float* A, const float* x, const float* b, float* y, size_t M, size_t K) {
    static bool use_avx512 = cpu_supports_avx512();
    static size_t best_MC = 64, best_KC = 64;
    static bool tuned = false;
    if (!tuned) {
        // Autotune MC/KC for this shape
        double best_time = 1e9;
        for (size_t MC : MC_CANDIDATES) {
            for (size_t KC : KC_CANDIDATES) {
                float* ytest = aligned_alloc_float(M);
                double t0 = omp_get_wtime();
                #pragma omp parallel for schedule(static)
                for (size_t i0 = 0; i0 < M; i0 += MC) {
                    size_t i_max = std::min(i0 + MC, M);
                    float* Ablock = aligned_alloc_float((i_max - i0) * K);
                    std::memcpy(Ablock, A + i0 * K, (i_max - i0) * K * sizeof(float));
                    float* yblock = ytest + i0;
                    if (use_avx512) matvec_block_avx512(Ablock, x, yblock, i_max - i0, K);
                    else matvec_block_avx2(Ablock, x, yblock, i_max - i0, K);
                    free(Ablock);
                }
                double t1 = omp_get_wtime();
                if (t1 - t0 < best_time) {
                    best_time = t1 - t0;
                    best_MC = MC;
                    best_KC = KC;
                }
                free(ytest);
            }
        }
        tuned = true;
    }
    #pragma omp parallel for schedule(static)
    for (size_t i0 = 0; i0 < M; i0 += best_MC) {
        size_t i_max = std::min(i0 + best_MC, M);
        float* Ablock = aligned_alloc_float((i_max - i0) * K);
        std::memcpy(Ablock, A + i0 * K, (i_max - i0) * K * sizeof(float));
        float* yblock = y + i0;
        // Prefetch next block
        if (i0 + best_MC < M) prefetch_L1(A + (i0 + best_MC) * K);
        if (use_avx512) matvec_block_avx512(Ablock, x, yblock, i_max - i0, K);
        else matvec_block_avx2(Ablock, x, yblock, i_max - i0, K);
        for (size_t i = 0; i < i_max - i0; ++i) yblock[i] += b ? b[i0 + i] : 0.0f;
        free(Ablock);
    }
}

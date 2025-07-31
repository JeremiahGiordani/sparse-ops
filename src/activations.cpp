// src/activations.cpp
#include "activations.hpp"

#include <immintrin.h>   // AVX2 / AVX-512 intrinsics
#include <cmath>         // std::exp, std::tanh
#include <cstddef>       // std::size_t

#if defined(__GNUC__)
static inline bool supports_avx512() {
    return __builtin_cpu_supports("avx512f");
}
static inline bool supports_avx2() {
    return __builtin_cpu_supports("avx2");
}
#else
static inline bool supports_avx512() { return false; }
static inline bool supports_avx2()   { return false; }
#endif

void relu_inplace(float* data, std::size_t len) {
    if (supports_avx512()) {
        const std::size_t stride = 16;
        std::size_t i = 0, n = len - (len % stride);
        __m512 zero = _mm512_setzero_ps();
        for (; i < n; i += stride) {
            __m512 v = _mm512_loadu_ps(data + i);
            v = _mm512_max_ps(v, zero);
            _mm512_storeu_ps(data + i, v);
        }
        for (; i < len; ++i) {
            data[i] = data[i] < 0.0f ? 0.0f : data[i];
        }

    } else if (supports_avx2()) {
        const std::size_t stride = 8;
        std::size_t i = 0, n = len - (len % stride);
        __m256 zero = _mm256_setzero_ps();
        for (; i < n; i += stride) {
            __m256 v = _mm256_loadu_ps(data + i);
            v = _mm256_max_ps(v, zero);
            _mm256_storeu_ps(data + i, v);
        }
        for (; i < len; ++i) {
            data[i] = data[i] < 0.0f ? 0.0f : data[i];
        }

    } else {
        for (std::size_t i = 0; i < len; ++i) {
            data[i] = data[i] < 0.0f ? 0.0f : data[i];
        }
    }
}

void sigmoid_inplace(float* data, std::size_t len) {
    // For now, we use a scalar loop. Can replace with SIMD/poly
    for (std::size_t i = 0; i < len; ++i) {
        // sigmoid(x) = 1 / (1 + exp(-x))
        float v = data[i];
        data[i] = 1.0f / (1.0f + std::exp(-v));
    }
}

void tanh_inplace(float* data, std::size_t len) {
    // Scalar fallback. Can vectorize with polynomial approx in future.
    for (std::size_t i = 0; i < len; ++i) {
        data[i] = std::tanh(data[i]);
    }
}

#include "sparse_onnx.hpp"

static inline float* alloc_aligned(size_t elems) {
    void* p=nullptr; size_t bytes = elems*sizeof(float);
    if (posix_memalign(&p, 64, (bytes+63)&~size_t(63)) != 0 || !p) throw std::bad_alloc();
    return static_cast<float*>(p);
}

RunResult SparseOnnxModel::applyAdd(
    const AddAttr   &/*a*/,
    const float     *in_A,
    const float     *in_B,
    uint32_t         features,   // rows
    uint32_t         B,          // contiguous span per row
    float*           out_buf
) const {
    const size_t tot = size_t(features) * B;

    // Fast path: allow in-place if caller provided out_buf that aliases an input.
    float* dst = out_buf ? out_buf : alloc_aligned(tot);
    const bool owned = (out_buf == nullptr);

    const bool avx512 = supports_avx512();
#if defined(__AVX2__)
    const bool avx2 = !avx512; // prefer 512 if available
#else
    const bool avx2 = false;
#endif

    // Parallelize over feature rows to keep each thread’s working set hot.
    #pragma omp parallel for schedule(static)
    for (uint32_t r = 0; r < features; ++r) {
        const float* A = in_A + size_t(r) * B;
        const float* Bp= in_B + size_t(r) * B;
        float*       D = dst  + size_t(r) * B;

        uint32_t c = 0;

        if (avx512) {
            constexpr uint32_t V = 16;
            for (; c + V <= B; c += V) {
                __m512 av = _mm512_loadu_ps(A + c);
                __m512 bv = _mm512_loadu_ps(Bp + c);
                __m512 dv = _mm512_add_ps(av, bv);
                _mm512_storeu_ps(D + c, dv);
            }
            if (c < B) {
                const uint32_t tail = B - c;
                const __mmask16 mask = (__mmask16(1) << tail) - 1;
                __m512 av = _mm512_maskz_loadu_ps(mask, A + c);
                __m512 bv = _mm512_maskz_loadu_ps(mask, Bp + c);
                __m512 dv = _mm512_add_ps(av, bv);
                _mm512_mask_storeu_ps(D + c, mask, dv);
            }
        }
        else if (avx2) {
#if defined(__AVX2__)
            constexpr uint32_t V = 8;
            for (; c + V <= B; c += V) {
                __m256 av = _mm256_loadu_ps(A + c);
                __m256 bv = _mm256_loadu_ps(Bp + c);
                __m256 dv = _mm256_add_ps(av, bv);
                _mm256_storeu_ps(D + c, dv);
            }
            // tail (<=7) scalar
            for (; c < B; ++c) D[c] = A[c] + Bp[c];
#endif
        }
        else {
            // scalar fallback
            for (; c < B; ++c) D[c] = A[c] + Bp[c];
        }
    }

    return { dst, features, owned };
}
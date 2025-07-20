// include/aligned_buffer.hpp
#pragma once

#include <cstdlib>
#include <stdexcept>

// Wrap a raw float* that must be 64‑byte aligned.
struct AlignedBuffer {
    float* ptr;
    size_t size;  // number of floats

    AlignedBuffer(size_t n): ptr(nullptr), size(n) {
        // posix_memalign requires the allocated size be a multiple of alignment,
        // so round up n*sizeof(float) to the next 64 byte boundary.
        size_t bytes = n * sizeof(float);
        size_t rem   = bytes % 64;
        if (rem) bytes += (64 - rem);

        void* p = nullptr;
        if (posix_memalign(&p, 64, bytes) != 0 || p == nullptr) {
            throw std::bad_alloc();
        }
        ptr = static_cast<float*>(p);
    }

    ~AlignedBuffer() {
        std::free(ptr);
    }

    // disable copy
    AlignedBuffer(const AlignedBuffer&) = delete;
    AlignedBuffer& operator=(const AlignedBuffer&) = delete;

    // enable move
    AlignedBuffer(AlignedBuffer&& o) noexcept
      : ptr(o.ptr), size(o.size) { o.ptr = nullptr; }
    AlignedBuffer& operator=(AlignedBuffer&& o) noexcept {
      std::swap(ptr, o.ptr);
      std::swap(size,o.size);
      return *this;
    }
};

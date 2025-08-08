// include/aligned_buffer.hpp
#pragma once

#include <cstdlib>
#include <stdexcept>

// Wrap a raw float* that must be 64‑byte aligned.
struct AlignedBuffer {
    float* ptr{nullptr};
    size_t size{0};

    AlignedBuffer() = default;
    explicit AlignedBuffer(size_t n) : ptr(nullptr), size(n) {
        size_t bytes = n * sizeof(float);
        size_t rem = bytes % 64;
        if (rem) bytes += (64 - rem);
        void* p = nullptr;
        if (posix_memalign(&p, 64, bytes) != 0 || p == nullptr)
            throw std::bad_alloc();
        ptr = static_cast<float*>(p);
    }
    ~AlignedBuffer() { std::free(ptr); }

    AlignedBuffer(const AlignedBuffer&) = delete;
    AlignedBuffer& operator=(const AlignedBuffer&) = delete;
    AlignedBuffer(AlignedBuffer&& o) noexcept : ptr(o.ptr), size(o.size) { o.ptr = nullptr; }
    AlignedBuffer& operator=(AlignedBuffer&& o) noexcept {
        std::swap(ptr, o.ptr); std::swap(size, o.size); return *this;
    }
};

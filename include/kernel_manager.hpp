#pragma once
#include <cstddef>
#include <functional>
#include <unordered_map>
#include <tuple>

// Custom hash for tuple
namespace std {
    template <typename T>
    inline void hash_combine(std::size_t& seed, const T& v) {
        seed ^= std::hash<T>{}(v) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
    }
    template <typename... TT>
    struct hash<std::tuple<TT...>> {
        size_t operator()(const std::tuple<TT...>& tt) const {
            size_t seed = 0;
            std::apply([&seed](const auto&... args) { (hash_combine(seed, args), ...); }, tt);
            return seed;
        }
    };
}

// Kernel function pointer type
typedef void (*matvec_kernel_fn)(const float* A, const float* x, float* y, size_t M, size_t K);

class KernelManager {
public:
    matvec_kernel_fn get_or_create(size_t M, size_t K, bool avx512);
private:
    std::unordered_map<std::tuple<size_t, size_t, bool>, matvec_kernel_fn> cache_;
};

#pragma once

#include <functional>
#include <unordered_map>
#include <vector>
#include <string>
#include <memory>

namespace sparseops {
namespace detail {

// Function pointer type for micro-kernels
using MicroKernel = std::function<void(const float* A_vals, const int32_t* col_indices, 
                                       const float* B_panel, float* C_row, 
                                       int nnz, int N)>;

// Kernel cache key
struct KernelKey {
    std::vector<bool> sparsity_mask;  // Sparsity pattern within block
    int block_size;
    int dtype_id;  // 0=f32, 1=f16, etc.
    
    bool operator==(const KernelKey& other) const {
        return sparsity_mask == other.sparsity_mask && 
               block_size == other.block_size && 
               dtype_id == other.dtype_id;
    }
};

struct KernelKeyHash {
    std::size_t operator()(const KernelKey& k) const {
        std::size_t seed = 0;
        for (bool b : k.sparsity_mask) {
            seed ^= std::hash<bool>{}(b) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        }
        seed ^= std::hash<int>{}(k.block_size) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        seed ^= std::hash<int>{}(k.dtype_id) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        return seed;
    }
};

// Kernel cache and generator
class KernelCache {
public:
    KernelCache();
    ~KernelCache();
    
    MicroKernel get_or_create_kernel(const KernelKey& key);
    
private:
    std::unordered_map<KernelKey, MicroKernel, KernelKeyHash> cache_;
    
    MicroKernel generate_kernel(const KernelKey& key);
    MicroKernel generate_generic_kernel(int block_size);
};

} // namespace detail
} // namespace sparseops

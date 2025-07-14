#include "sparse_dispatch.hpp"
#include <vector>
#include <algorithm>
#include <sstream>
#include <iomanip>
#include "codegen/spmv_template.hpp"  // for generate_spmv_cpp

static std::string hot_mask_key(const BCOO16& A)
{
    std::unordered_map<uint16_t,size_t> hist;
    for (auto& blk : A.blocks) ++hist[blk.bitmask];

    std::vector<std::pair<uint16_t,size_t>> vec(hist.begin(), hist.end());
    std::sort(vec.begin(), vec.end(),
              [](auto& a, auto& b){ return a.second > b.second; });

    constexpr int MAX_HOT = 32;
    std::vector<uint16_t> hot;
    for (auto& p : vec) {
        if (hot.size()==MAX_HOT) break;
        if (__builtin_popcount(p.first) <= 4 && p.first != 0)
            hot.push_back(p.first);
    }
    std::sort(hot.begin(), hot.end());

    std::ostringstream oss;
    oss << std::hex << std::uppercase;
    for (auto m : hot) oss << std::setw(4) << std::setfill('0') << m;
    return oss.str();
}

KernelFn get_spmv_kernel(const BCOO16& A)
{
    bool dense_test = std::getenv("DENSE_TEST") != nullptr;   // new

    std::string key = (dense_test ? "dense|" : "normal|") +
                      std::to_string(A.original_num_rows) + "|" +
                      std::to_string(A.blocks.size());

    std::string cpp =
        dense_test ? generate_spmv_dense_cpp(A, "spmv_kernel")
                   : generate_spmv_cpp      (A, "spmv_kernel", true);

    return get_or_build_kernel(key, cpp, "spmv_kernel");
}

#include "bcoo16_encoder.hpp"
#include <vector>

std::vector<size_t> make_super_ptr(const BCOO16& A, size_t band_bytes = 32*1024)
{
    const size_t band_words = band_bytes / sizeof(float);     // 8 K floats
    std::vector<size_t> sptr{0};
    uint64_t cur_band = A.blocks.empty() ?
                        0 : (A.blocks[0].first_col / band_words);
    for (size_t i = 0; i < A.blocks.size(); ++i) {
        uint64_t b = A.blocks[i].first_col / band_words;
        if (b != cur_band) { sptr.push_back(i); cur_band = b; }
    }
    sptr.push_back(A.blocks.size());
    return sptr;
}

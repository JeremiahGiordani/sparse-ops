#include "sparse_dispatch.hpp"          // BCOO16Block
#include <sstream>
#include <unordered_map>
#include <vector>
#include <algorithm>
#include <iomanip>
#include <bitset>

static std::string hex4(uint16_t m) {
    std::ostringstream oss;
    oss << "0x" << std::uppercase << std::hex << std::setw(4)
        << std::setfill('0') << m;
    return oss.str();
}

std::string generate_spmv_cpp(const BCOO16& A,
                              const std::string& func_name,
                              bool avx512)
{
    // ──────────────────────────────────────────────────────────────
    // 1. Histogram masks
    std::unordered_map<uint16_t,size_t> hist;
    for (const auto& blk : A.blocks) ++hist[blk.bitmask];

    // ──────────────────────────────────────────────────────────────
    // 2. Choose up to 32 hottest masks with popcnt ≤ 4
    std::vector<std::pair<uint16_t,size_t>> vec(hist.begin(), hist.end());
    std::sort(vec.begin(), vec.end(),
              [](auto& a, auto& b){ return a.second > b.second; });

    constexpr int MAX_HOT = 32;
    std::vector<uint16_t> hot;
    for (auto& p : vec) {
        if (hot.size() == MAX_HOT) break;
        if (__builtin_popcount(p.first) <= 4 && p.first != 0)
            hot.push_back(p.first);
    }
    std::sort(hot.begin(), hot.end());               // deterministic order

    // ──────────────────────────────────────────────────────────────
    // 3. Begin C++ source
    std::ostringstream os;
    os << "#include <immintrin.h>\n"
          "#include \"bcoo16_encoder.hpp\"\n\n";

    // ──────────────────────────────────────────────────────────────
    // 4. Emit helpers for each hot mask
    for (uint16_t m : hot) {
        std::string fname = "dot_" + hex4(m);
        os << "static inline float " << fname
           << "(const BCOO16Block& blk, const float* x) {\n"
           << "    float acc = 0.0f;\n";
        for (int lane = 0; lane < 16; ++lane)
            if (m & (1u << lane))
                os << "    acc += blk.values[" << lane
                   << "] * x[blk.first_col + " << lane << "];\n";
        os << "    return acc;\n}\n\n";
    }

    // ──────────────────────────────────────────────────────────────
    // 5. Generic AVX-512 path
    os << "static inline float dot_generic(const BCOO16Block& blk, const float* x) {\n"
          "    __mmask16 m = blk.bitmask;\n"
          "    if(!m) return 0.0f;\n"
          "    const __m512i idxOff = _mm512_set_epi32("
             "15,14,13,12,11,10,9,8,7,6,5,4,3,2,1,0);\n"
          "    __m512 vals = _mm512_maskz_loadu_ps(m, blk.values);\n"
          "    __m512i idx = _mm512_add_epi32(idxOff, _mm512_set1_epi32(blk.first_col));\n"
          "    __m512 xv   = _mm512_mask_i32gather_ps(_mm512_setzero_ps(), m, idx, x, 4);\n"
          "    return _mm512_reduce_add_ps(_mm512_mul_ps(vals, xv));\n"
          "}\n\n";

    // ──────────────────────────────────────────────────────────────
    // 6. Main kernel with small switch
    os << "extern \"C\" void " << func_name
       << "(const BCOO16Block* __restrict blocks, size_t nBlocks,"
          " const float* __restrict x, float* __restrict y) {\n"
          "  for(size_t bi=0; bi<nBlocks; ++bi){\n"
          "    const auto& blk = blocks[bi];\n"
          "    float acc;\n"
          "    switch(blk.bitmask){\n";
    for (size_t i=0;i<hot.size();++i){
        os << "      case " << hex4(hot[i]) << ": acc = dot_" << hex4(hot[i])
           << "(blk,x); break;\n";
    }
    os << "      default: acc = dot_generic(blk,x); break;\n"
          "    }\n"
          "    y[blk.row_id] += acc;\n"
          "  }\n}\n";

    return os.str();
}

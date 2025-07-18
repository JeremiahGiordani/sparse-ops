/* spmv_template.cpp  – bandwidth‑aware kernel generator */
#include "sparse_dispatch.hpp"
#include <sstream>
#include <unordered_map>
#include <vector>
#include <iomanip>

/* helper: write 0xABCD with 4 hex digits */
static std::string hex4(uint16_t m){
    std::ostringstream o; o<<"0x"<<std::uppercase<<std::hex
                           <<std::setw(4)<<std::setfill('0')<<m; return o.str();
}

/*-------------------------------------------------------------------*/
std::string generate_spmv_cpp(const BCOO16& A,
                              const std::string& fn_name,
                              bool /*avx512*/)
{
    /* ─────────  generate C++ source  ───────── */
    std::ostringstream os;
    os << "#include <immintrin.h>\n"
          "#include \"bcoo16_encoder.hpp\"\n\n";

    /* optional runtime mask histogram */
    os << "#ifdef PROFILE_MASKS\n"
          "extern \"C\" uint64_t hotHits[65536];\n"
          "uint64_t hotHits[65536] = {0};\n"
          "#endif\n\n";

    /* popcnt‑specific helpers ------------------------------------- */

    /* scalar for exactly one lane */
    os << "static inline float dot1(const BCOO16Block& b,const float* x){\n"
          "  int lane = _tzcnt_u32(b.bitmask);\n"
          "  return b.values[0] * x[b.first_col + lane]; }\n\n";

    /* scalar for exactly two lanes */
    os << "static inline float dot2(const BCOO16Block& b,const float* x){\n"
          "  uint32_t m = b.bitmask;\n"
          "  int i = _tzcnt_u32(m);\n"
          "  int j = _tzcnt_u32(m & ~(1u << i));\n"
          "  return b.values[0]*x[b.first_col+i] + b.values[1]*x[b.first_col+j]; }\n\n";

    /* masked gather for popcnt 3‑15 (also used as fallback) */
    os << "static inline float dot_gather(const BCOO16Block& b,const float* x){\n"
          "  __mmask16 m = b.bitmask;\n"
          "  const __m512i off=_mm512_set_epi32("
             "15,14,13,12,11,10,9,8,7,6,5,4,3,2,1,0);\n"
          "  __m512 v=_mm512_maskz_loadu_ps(m,b.values);\n"
          "  __m512 g=_mm512_mask_i32gather_ps(_mm512_setzero_ps(), m,\n"
          "          _mm512_add_epi32(off,_mm512_set1_epi32(b.first_col)), x, 4);\n"
          "  return _mm512_reduce_add_ps(_mm512_mul_ps(v,g)); }\n\n";

    /* full 64‑byte tile for dense block */
    os << "static inline float dot_full(const BCOO16Block& b,const float* x){\n"
          "  __m512 v = _mm512_loadu_ps(b.values);\n"
          "  __m512 xv= _mm512_loadu_ps(x + b.first_col);\n"
          "  return _mm512_reduce_add_ps(_mm512_mul_ps(v,xv)); }\n\n";

    /* main kernel -------------------------------------------------- */
    os << "extern \"C\" void " << fn_name
       << "(const BCOO16Block* blks,size_t n,const float* x,float* y){\n"
          "  for(size_t i=0;i<n;++i){\n"
          "    const auto& b = blks[i];\n"
          "    float acc;\n"
          "    int pc = __builtin_popcount(b.bitmask);\n"
          "    switch(pc){\n"
          "      case 0:  acc = 0.0f; break;\n"
          "      case 1:  acc = dot1(b,x); break;\n"
          "      case 2:  acc = dot2(b,x); break;\n"
          "      case 3:\n"
          "      case 4:  acc = dot_gather(b,x); break;\n"
          "      default: acc = (b.bitmask==0xFFFF) ? dot_full(b,x)\n"
          "                                        : dot_gather(b,x); break;\n"
          "    }\n"
          "#ifdef PROFILE_MASKS\n"
          "    __sync_fetch_and_add(&hotHits[b.bitmask],1);\n"
          "#endif\n"
          "    y[b.row_id] += acc;\n"
          "  }\n"
          "}\n";

    return os.str();
}

/*───────────────────────────────────────────────────────────────
    spmv_template.cpp
    Generates a specialised  AVX‑512 / AVX‑2  micro‑kernel:

        y[row] += Σ  v_i · x[col_i]

    Data layout (packed values):
        struct BCOO16Block {
            uint32_t row_id;
            uint16_t first_col;   // 16‑lane tile base (col & ~15)
            uint16_t bitmask;     // which lanes are live
            uint32_t val_off;     // offset into contiguous values[]
        };
───────────────────────────────────────────────────────────────*/
#include "sparse_dispatch.hpp"
#include <sstream>
#include <iomanip>

/* pretty 0xABCD with leading zeros */
static std::string hex4(uint16_t v){
    std::ostringstream o; o << "0x" << std::uppercase << std::hex
                            << std::setw(4) << std::setfill('0') << v;
    return o.str();
}

/*--------------------------------------------------------------*/
std::string generate_spmv_cpp(const BCOO16& /*A*/,
                              const std::string& fn_name,
                              bool /*avx512*/)
{
    std::ostringstream os;

    /* ---------- headers ---------- */
    os << "#include <immintrin.h>\n"
          "#include \"bcoo16_encoder.hpp\"\n\n";

    /* optional mask histogram (for diagnostics) */
    os << "#ifdef PROFILE_MASKS\n"
          "extern \"C\" uint64_t hotHits[65536];\n"
          "uint64_t hotHits[65536] = {0};\n"
          "#endif\n\n";

    /* ---------- helper kernels ---------- */

    /* popcnt == 1 : scalar */
    os << "static inline float dot1(const BCOO16Block& b,"
          "const float* vals,const float* x){\n"
          "  int lane = _tzcnt_u32(b.bitmask);\n"
          "  return vals[b.val_off] * x[b.first_col + lane]; }\n\n";

    /* popcnt == 2 : scalar */
    os << "static inline float dot2(const BCOO16Block& b,"
          "const float* vals,const float* x){\n"
          "  uint32_t m = b.bitmask;\n"
          "  int i = _tzcnt_u32(m);\n"
          "  int j = _tzcnt_u32(m & ~(1u<<i));\n"
          "  const float* v = vals + b.val_off;\n"
          "  return v[0]*x[b.first_col+i] + v[1]*x[b.first_col+j]; }\n\n";

    /* generic scalar gather (popcnt 3‑15) */
    os << "static inline float dot_gather(const BCOO16Block& b,"
          "const float* vals,const float* x){\n"
          "  const float* v = vals + b.val_off;\n"
          "  uint16_t m = b.bitmask;  int idx = 0;  float acc = 0.0f;\n"
          "  while (m) {\n"
          "    int lane = _tzcnt_u32(m);\n"
          "    acc += v[idx++] * x[b.first_col + lane];\n"
          "    m &= m - 1;   /* clear lowest bit */\n"
          "  }\n"
          "  return acc; }\n\n";

    /* fully dense (0xFFFF) */
    os << "static inline float dot_full(const BCOO16Block& b,"
          "const float* vals,const float* x){\n"
          "  const float* v = vals + b.val_off;\n"
          "  __m512 vv = _mm512_loadu_ps(v);\n"
          "  __m512 xv = _mm512_loadu_ps(x + b.first_col);\n"
          "  return _mm512_reduce_add_ps(_mm512_mul_ps(vv,xv)); }\n\n";

    /* ---------- main kernel ---------- */
    os << "extern \"C\" void " << fn_name
       << "(const BCOO16Block* blks,size_t nBlocks,\n"
          "            const float* values,\n"
          "            const float* x,float* y)\n{\n"
          "  for(size_t i=0;i<nBlocks;++i){\n"
          "    const auto& b = blks[i];\n"
          "    float acc;\n"
          "    int pc = __builtin_popcount(b.bitmask);\n"
          "    switch(pc){\n"
          "      case 0:  acc = 0.0f; break;\n"
          "      case 1:  acc = dot1(b,values,x); break;\n"
          "      case 2:  acc = dot2(b,values,x); break;\n"
          "      case 3:\n"
          "      case 4:  acc = dot_gather(b,values,x); break;\n"
          "      default: acc = (b.bitmask==0xFFFF) ? dot_full(b,values,x)\n"
          "                                         : dot_gather(b,values,x); break;\n"
          "    }\n"
          "#ifdef PROFILE_MASKS\n"
          "    __sync_fetch_and_add(&hotHits[b.bitmask],1);\n"
          "#endif\n"
          "    y[b.row_id] += acc;\n"
          "  }\n"
          "}\n";

    return os.str();
}

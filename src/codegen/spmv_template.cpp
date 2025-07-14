#include "sparse_dispatch.hpp"
#include <sstream>
#include <unordered_map>
#include <vector>
#include <algorithm>
#include <iomanip>

static std::string hex4(uint16_t m){
    std::ostringstream os;
    os<< "0x"<< std::uppercase<< std::hex<< std::setw(4)<< std::setfill('0')<< m;
    return os.str();
}

/* contiguous-run helpers */
static bool is_one_run(uint16_t m){
    if(m==0||m==0xFFFF) return false;
    int l = __builtin_ctz(m);
    int r = 15-(__builtin_clz(m)-16);
    uint16_t span=((1u<<(r-l+1))-1u)<<l;
    return span==m;
}
static bool is_two_run(uint16_t m, int& p, int& q){
    if(is_one_run(m)||m==0||m==0xFFFF) return false;
    /* find first run */
    p = __builtin_ctz(m);
    int p_end = p;
    while(p_end<16 && (m & (1u<<p_end))) ++p_end;
    /* clear first run, check remainder is one run */
    uint16_t rest = m & ~(((1u<<(p_end-p))-1u)<<p);
    if(rest==0) return false;
    if(!is_one_run(rest)) return false;
    q = __builtin_ctz(rest);
    return true;
}

std::string generate_spmv_cpp(const BCOO16& A,
                              const std::string& fn_name,
                              bool avx512)
{
    /*──────── 1. histogram & classify masks ────────*/
    std::unordered_map<uint16_t,size_t> hist;
    for(auto& b:A.blocks) ++hist[b.bitmask];

    std::vector<std::pair<uint16_t,size_t>> vec(hist.begin(),hist.end());
    std::sort(vec.begin(),vec.end(),[](auto&a,auto&b){return a.second>b.second;});

    constexpr int MAX_HOT=32, MAX_2RUN=64;
    std::vector<uint16_t> hot, run2;
    for(auto& p:vec){
        uint16_t m=p.first;
        if(__builtin_popcount(m)<=4 && hot.size()<MAX_HOT && m)  hot.push_back(m);
        else{
            int p_i,q_i;
            if(is_two_run(m,p_i,q_i) && run2.size()<MAX_2RUN) run2.push_back(m);
        }
    }
    std::sort(hot.begin(),hot.end());
    std::sort(run2.begin(),run2.end());

    /*──────── 2. emit C++ source ───────────────────*/
    std::ostringstream os;
    os<<"#include <immintrin.h>\n#include <cstring>\n#include \"bcoo16_encoder.hpp\"\n\n";

    os << "#ifndef _mm256_reduce_add_ps\n"
      "static inline float _mm256_reduce_add_ps(__m256 v){\n"
      "  __m128 lo = _mm256_castps256_ps128(v);\n"
      "  __m128 hi = _mm256_extractf128_ps(v,1);\n"
      "  lo = _mm_add_ps(lo,hi);\n"
      "  __m128 sh = _mm_movehdup_ps(lo);\n"
      "  lo = _mm_add_ps(lo,sh);\n"
      "  sh = _mm_movehl_ps(sh,lo);\n"
      "  lo = _mm_add_ss(lo,sh);\n"
      "  return _mm_cvtss_f32(lo);\n"
      "}\n"
      "#endif\n\n";

    /* 2.a helpers for ≤4-lane hot masks (xTile) */
    for(uint16_t m:hot){
        std::string f="dot_"+hex4(m);
        os<<"static inline float "<<f<<"(const BCOO16Block& b,const float* t,int off){float a=0;";
        for(int l=0;l<16;++l) if(m&(1u<<l)) os<<" a+=b.values["<<l<<"]*t[off+"<<l<<"];";
        os<<" return a;}\n";
    }

    /* 2.b helpers for 2-run masks (direct load) */
    for(uint16_t m:run2){
        int p,q; is_two_run(m,p,q);
        std::string f="dot2_"+hex4(m);
        os<<"static inline float "<<f<<"(const BCOO16Block& b,const float* x){\n"
             "  __m256 v0=_mm256_loadu_ps(b.values+"<<p<<");\n"
             "  __m256 x0=_mm256_loadu_ps(x+b.first_col+"<<p<<");\n"
             "  __m256 v1=_mm256_loadu_ps(b.values+"<<q<<");\n"
             "  __m256 x1=_mm256_loadu_ps(x+b.first_col+"<<q<<");\n"
             "  return _mm256_reduce_add_ps(_mm256_mul_ps(v0,x0))+\n"
             "         _mm256_reduce_add_ps(_mm256_mul_ps(v1,x1)); }\n";
    }

    /* 2.c full-mask helper & generic gather */
    os<<"static inline float dot_full(const BCOO16Block& b,const float* x){\n"
         "  __m512 v=_mm512_loadu_ps(b.values);\n"
         "  __m512 xv=_mm512_loadu_ps(x+b.first_col);\n"
         "  return _mm512_reduce_add_ps(_mm512_mul_ps(v,xv)); }\n\n";
    os<<"static inline float dot_generic(const BCOO16Block& b,const float* x){\n"
         "  __mmask16 m=b.bitmask; if(!m) return 0.0f;\n"
         "  const __m512i off=_mm512_set_epi32(15,14,13,12,11,10,9,8,7,6,5,4,3,2,1,0);\n"
         "  __m512 v=_mm512_maskz_loadu_ps(m,b.values);\n"
         "  __m512i idx=_mm512_add_epi32(off,_mm512_set1_epi32(b.first_col));\n"
         "  __m512 g=_mm512_mask_i32gather_ps(_mm512_setzero_ps(),m,idx,x,4);\n"
         "  return _mm512_reduce_add_ps(_mm512_mul_ps(v,g)); }\n\n";

      os<<"static inline bool is_one_run16(uint16_t m){\n"
         "  if (m==0 || m==0xFFFF) return false;\n"
         "  int l = _tzcnt_u32(m);                                  // first 1-bit\n"
         "  int r = 15 - (_lzcnt_u32((uint32_t)m)-16);              // last 1-bit\n"
         "  uint16_t span = ((1u << (r-l+1)) - 1u) << l;\n"
         "  return span == m;\n"
         "}\n";

      os<<"static inline bool is_two_run16(uint16_t m){\n"
         "  if (is_one_run16(m) || m==0 || m==0xFFFF) return false;\n"
         "  int p = _tzcnt_u32(m);                                  // start of 1st run\n"
         "  int e = p; while (e < 16 && (m & (1u<<e))) ++e;         // end of 1st run\n"
         "  uint16_t rest = m & ~(((1u<<(e-p))-1u) << p);           // remove 1st run\n"
         "  return is_one_run16(rest);                              // rest is 1 run?\n"
         "}\n";
      os<<"static inline float dot_contig(const BCOO16Block& b,const float* x){\n"
         "  __m512 xv = _mm512_loadu_ps(x + b.first_col);           // contiguous slice\n"
         "  __mmask16 m = b.bitmask;\n"
         "  __m512 v  = _mm512_maskz_loadu_ps(m, b.values);         // zero dead lanes\n"
         "  return _mm512_reduce_add_ps(_mm512_mul_ps(v, xv));\n"
         "}\n";


    /* 2.d main kernel with prefetch+unroll and tile reuse gate */
    os<<"extern \"C\" void "<<fn_name
      <<"(const BCOO16Block* blks,size_t n,const float* x,float* y){\n"
        "  alignas(64) float t[16]; int tileBase=-64;\n"
        "  for(size_t i=0;i<n;++i){\n"
        "    if(i+8<n) _mm_prefetch(&blks[i+8],_MM_HINT_T0);\n"
        "    const auto& b=blks[i]; int need=b.first_col & ~15;\n"
        "    bool have=(need==tileBase);\n"
        "    bool reuse=(i+1<n && ((blks[i+1].first_col & ~15)==need));\n"
        "    float acc;\n"
        "    if(have||reuse){ if(!have){std::memcpy(t,x+need,64); tileBase=need;} switch(b.bitmask){\n"
        "      case 0xFFFF: acc=dot_full(b,x); break;\n";
    for(uint16_t m:hot)
        os<<"      case "<<hex4(m)<<": acc=dot_"<<hex4(m)
          <<"(b,t,b.first_col-tileBase); break;\n";
    for(uint16_t m:run2)
        os<<"      case "<<hex4(m)<<": acc=dot2_"<<hex4(m)
          <<"(b,x); break;\n";
    os<<"      default: acc=dot_generic(b,x); break; }\n"
         "    }else{\n"
         "      /* one-off block */\n"
         "      acc = "<<
         "(is_one_run16(b.bitmask)? dot_full(b,x) :\n"
         "       (is_two_run16(b.bitmask)? dot_contig(b,x) : dot_generic(b,x)));\n"
         "    }\n"
         "    y[b.row_id]+=acc;\n"
         "  }\n}\n";

    return os.str();
}

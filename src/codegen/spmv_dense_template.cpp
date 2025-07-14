/*  spmv_dense_template.cpp       (row-blocked version, B = 4)
 *
 *  Treat every BCOO16 block as dense (16 lanes) but process a *panel* of up to
 *  four consecutive blocks that share the same 64-byte slice of x.  Each tile
 *  of x is therefore loaded once and reused 2-4×, slashing L1 miss penalty.
 */
#include "sparse_dispatch.hpp"
#include <sstream>

std::string generate_spmv_dense_cpp(const BCOO16& A,
                                    const std::string& func_name)
{
    std::ostringstream os;
    os<<"#include <immintrin.h>\n#include \"bcoo16_encoder.hpp\"\n\n";
    os << "extern \"C\" void " << func_name
    << "(const BCOO16Block* blks, size_t n, const float* x, float* y){\n"
        "  for(size_t bi = 0; bi < n; ){                      // iterate blocks\n"
        "    size_t run = 0;\n"
        "    const int base = blks[bi].first_col & ~15;\n"
        "    __m512 xv = _mm512_loadu_ps(x + base);\n"
        "    __m512 acc0=_mm512_setzero_ps(), acc1=acc0, acc2=acc0, acc3=acc0;\n"
        "\n"
        "    /* fuse up to 4 consecutive rows that share this tile */\n"
        "    while(run < 4 && bi+run < n &&\n"
        "          (blks[bi+run].first_col & ~15) == base){\n"
        "      __m512 v = _mm512_loadu_ps(blks[bi+run].values);\n"
        "      switch(run){\n"
        "        case 0: acc0 = _mm512_fmadd_ps(v, xv, acc0); break;\n"
        "        case 1: acc1 = _mm512_fmadd_ps(v, xv, acc1); break;\n"
        "        case 2: acc2 = _mm512_fmadd_ps(v, xv, acc2); break;\n"
        "        case 3: acc3 = _mm512_fmadd_ps(v, xv, acc3); break;\n"
        "      }\n"
        "      ++run;\n"
        "    }\n"
        "    /* horizontal reduction once per row */\n"
        "    if(run>0) y[blks[bi+0].row_id] += _mm512_reduce_add_ps(acc0);\n"
        "    if(run>1) y[blks[bi+1].row_id] += _mm512_reduce_add_ps(acc1);\n"
        "    if(run>2) y[blks[bi+2].row_id] += _mm512_reduce_add_ps(acc2);\n"
        "    if(run>3) y[blks[bi+3].row_id] += _mm512_reduce_add_ps(acc3);\n"
        "    bi += run;\n"
        "  }\n"
        "}\n";
    return os.str();
}

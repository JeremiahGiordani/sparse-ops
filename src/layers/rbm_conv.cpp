// src/rbm_conv.cpp
#include "rbm_conv.hpp"
#include "ellpack_encoder.hpp"
#include <algorithm>
#include <unordered_map>

RBMPlan build_rbm_from_ellpack(const Ellpack& E, const float* bias_or_null, uint32_t Ct_max) {
    RBMPlan P;
    P.M = E.m;
    P.N = E.n;
    P.Ct_max = Ct_max;
    P.blocks.reserve( (E.m + Ct_max - 1) / Ct_max );

    for (uint32_t M0 = 0; M0 < E.m; M0 += Ct_max) {
        RBMBlock B;
        B.M0 = M0;
        B.Ct = std::min(Ct_max, E.m - M0);

        // 1) Merge-scan all rows’ column indices -> union list
        // Collect (k_rel, row_id, weight) triples, then group by k_rel.
        struct Trip { uint32_t k; uint16_t r; float w; };
        std::vector<Trip> all;
        all.reserve(size_t(B.Ct) * E.r);
        for (uint16_t r = 0; r < B.Ct; ++r) {
            const uint32_t i   = M0 + r;
            const uint32_t cnt = E.nnz[i];
            const size_t   base= size_t(i) * E.r;
            for (uint32_t j = 0; j < cnt; ++j) {
                all.push_back(Trip{ E.idx[base + j], r, E.Wd.ptr[base + j] });
            }
        }
        if (all.empty()) {
            // degenerate: no nnz across block
            B.krel.clear(); B.colptr = {0};
            P.blocks.push_back(std::move(B));
            continue;
        }
        std::sort(all.begin(), all.end(),
                  [](const Trip& a, const Trip& b){ return a.k < b.k || (a.k==b.k && a.r < b.r); });

        // 2) Build krel / colptr / pairs
        B.krel.reserve(all.size()); // upper bound
        B.colptr.clear();
        B.colptr.push_back(0);
        B.pairs.reserve(all.size());

        uint32_t cur_k = all[0].k;
        size_t   start = 0;
        for (size_t t = 0; t <= all.size(); ++t) {
            if (t == all.size() || all[t].k != cur_k) {
                // flush [start..t)
                B.krel.push_back(cur_k);
                for (size_t u = start; u < t; ++u) {
                    B.pairs.push_back(RBMColPair{ all[u].r, all[u].w });
                }
                B.colptr.push_back( (uint32_t)B.pairs.size() );
                if (t == all.size()) break;
                // next run
                cur_k = all[t].k;
                start = t;
            }
        }

        // 3) Bias slice
        if (bias_or_null) {
            B.bias.resize(B.Ct);
            for (uint16_t r = 0; r < B.Ct; ++r) B.bias[r] = bias_or_null[M0 + r];
        }

        P.blocks.push_back(std::move(B));
    }
    return P;
}

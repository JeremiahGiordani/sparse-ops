#include "sparse_onnx.hpp"

RunResult SparseOnnxModel::applyFlatten(
    const FlattenAttr &f,
    const float       *src,
    uint32_t           features,  // already == C*H*W for 4D inputs; unchanged
    uint32_t           B,
    float*             /*out_buf*/
) const {
    // Zero-copy: just reinterpret the buffer as (B, features) in Fortran.
    // No allocation, and we do NOT take ownership.
    return { const_cast<float*>(src), features, /*owned=*/false };
}
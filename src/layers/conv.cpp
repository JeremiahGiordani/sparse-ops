#include "sparse_onnx.hpp"


RunResult SparseOnnxModel::applyConv(
    const ConvAttr   &/*c*/,
    const float      *src,
    uint32_t          C,
    float*            out_buf
) const {
    // Stub: just pass src through
    // Assuming rows_map provides correct rows, we’ll just copy
    // but we need the rows; for now, assume rows_map[src_name]
    // has been captured externally; so we throw:
    throw std::runtime_error("applyConv stub: not yet implemented");
}
#include "sparse_linear.hpp"
#include <cnpy.h>
#include <stdexcept>
#include <algorithm>

SparseLinear::SparseLinear(const std::string& npz_path, int in_dim, int out_dim)
    : input_dim_(in_dim), output_dim_(out_dim) {
    
    auto data = cnpy::npz_load(npz_path);
    std::cerr << "Loading from: " << npz_path << std::endl;
    values_ = data["values"].as_vec<float>();
    indices_ = data["indices"].as_vec<int>();
    indptr_ = data["indptr"].as_vec<int>();
    bias_ = data["bias"].as_vec<float>();

    std::cerr << "expected output_dim: " << output_dim_ << ", bias.size(): " << bias_.size() << ", indptr.size(): " << indptr_.size() << std::endl;

    if (bias_.size() != output_dim_ || indptr_.size() != output_dim_ + 1) {
        throw std::runtime_error("Shape mismatch in sparse layer: expected bias of size " + std::to_string(output_dim_) +
                         ", got " + std::to_string(bias_.size()) +
                         "; indptr size = " + std::to_string(indptr_.size()) +
                         ", expected " + std::to_string(output_dim_ + 1));
    }
}

std::vector<float> SparseLinear::forward(const std::vector<float>& input) const {
    std::cerr << "Expected input_dim: " << input_dim_ << ", got: " << input.size() << std::endl;
    if (input.size() != input_dim_) {
        throw std::runtime_error("Input size mismatch in SparseLinear");
    }
    std::vector<float> output(output_dim_, 0.0f);
    for (int row = 0; row < output_dim_; ++row) {
        float sum = bias_[row];
        for (int idx = indptr_[row]; idx < indptr_[row + 1]; ++idx) {
            int col = indices_[idx];
            sum += values_[idx] * input[col];
        }
        output[row] = sum;
    }
    return output;
}

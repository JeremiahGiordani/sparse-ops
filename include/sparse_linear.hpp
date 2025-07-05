#include "model.hpp"
#include <string>
#include <vector>

class SparseLinear : public Layer {
public:
    SparseLinear(const std::string& npz_path, int in_dim, int out_dim);
    std::vector<float> forward(const std::vector<float>& input) const override;

private:
    int input_dim_;
    int output_dim_;
    std::vector<float> values_;
    std::vector<int> indices_;
    std::vector<int> indptr_;
    std::vector<float> bias_;
};

class ReLU : public Layer {
public:
    std::vector<float> forward(const std::vector<float>& input) const override {
        std::vector<float> output = input;
        for (auto& val : output) {
            val = std::max(0.0f, val);
        }
        return output;
    }
};
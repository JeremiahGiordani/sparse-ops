#pragma once

#include <vector>

class Layer {
public:
    virtual std::vector<float> forward(const std::vector<float>& input) const = 0;
    virtual ~Layer() = default;
};

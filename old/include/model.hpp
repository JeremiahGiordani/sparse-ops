#pragma once

#include <string>
#include <vector>
#include <memory>

class Layer {
public:
    virtual std::vector<float> forward(const std::vector<float>& input) const = 0;
    virtual ~Layer() = default;
};

class Model {
public:
    void load(const std::string& json_path);
    std::vector<float> forward(const std::vector<float>& input) const;
private:
    std::vector<std::unique_ptr<Layer>> layers_;
};
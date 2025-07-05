#include "model.hpp"
#include "sparse_linear.hpp"
#include <fstream>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

void Model::load(const std::string& json_path) {
    std::ifstream file(json_path);
    if (!file.is_open()) throw std::runtime_error("Could not open model config");

    json config;
    file >> config;

    for (const auto& layer : config["layers"]) {
        std::string type = layer["type"];
        if (type == "SparseLinear") {
            int in_dim = layer["input_dim"];
            int out_dim = layer["output_dim"];
            std::string path = layer["path"];
            layers_.emplace_back(std::make_unique<SparseLinear>("data/" + path, in_dim, out_dim));
        } else if (type == "ReLU") {
            layers_.emplace_back(std::make_unique<ReLU>());
        } else {
            throw std::runtime_error("Unsupported layer type: " + type);
        }
    }
}

std::vector<float> Model::forward(const std::vector<float>& input) const {
    std::vector<float> current = input;
    for (const auto& layer : layers_) {
        current = layer->forward(current);
    }
    return current;
}
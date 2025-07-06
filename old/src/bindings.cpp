// bindings.cpp

#include "model.hpp"
#include <cstring>

extern "C" {
    void run_inference(const char* model_json_path,
                       const float* input,
                       float* output,
                       int input_size) {
        // Load model
        Model model;
        model.load(model_json_path);

        // Copy input into std::vector
        std::vector<float> input_vec(input, input + input_size);

        // Run inference
        std::vector<float> result = model.forward(input_vec);

        // Copy output to provided pointer
        std::memcpy(output, result.data(), result.size() * sizeof(float));
    }
}
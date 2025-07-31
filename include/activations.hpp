// include/activations.hpp
#pragma once

#include <cstddef>

/// \file activations.hpp
/// SIMD-optimized in-place activation function declarations.

/// Applies the ReLU activation (max(0, x)) to each element of `data[0..len)`.
void relu_inplace(float* data, std::size_t len);

/// Applies the Sigmoid activation (1 / (1 + exp(-x))) to each element of `data[0..len)`.
void sigmoid_inplace(float* data, std::size_t len);

/// Applies the Tanh activation to each element of `data[0..len)`.
void tanh_inplace(float* data, std::size_t len);

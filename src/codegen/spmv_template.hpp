#pragma once
#include <string>
#include "bcoo16_encoder.hpp"

std::string generate_spmv_cpp(const BCOO16& A,
                              const std::string& func_name,
                              bool avx512);

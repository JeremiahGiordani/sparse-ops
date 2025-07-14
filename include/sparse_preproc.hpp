#pragma once

#include "bcoo16_encoder.hpp"
#include <cstddef>

std::vector<size_t> make_super_ptr(const BCOO16&, size_t band_bytes = 32*1024);

// utils.hpp
#pragma once
#include <immintrin.h>

inline bool supports_avx512() {
#if defined(__GNUC__)
  return __builtin_cpu_supports("avx512f");
#else
  return false;
#endif
}

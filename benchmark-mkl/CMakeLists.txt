cmake_minimum_required(VERSION 3.14)
project(SparseOpsBenchmarks LANGUAGES CXX)

# Use C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Always build in Release with high‑opt flags
set(CMAKE_BUILD_TYPE       Release  CACHE STRING "" FORCE)
set(CMAKE_CXX_FLAGS_RELEASE
    "${CMAKE_CXX_FLAGS_RELEASE} -O3 -march=native -funroll-loops"
)

# Find threading and MKL dependencies
find_package(OpenMP REQUIRED)
find_package(MKL REQUIRED)

# Subdirectories
add_subdirectory(common)
add_subdirectory(matvec)
add_subdirectory(matmul)

# benchmark/CMakeLists.txt

cmake_minimum_required(VERSION 3.14)
project(SparseOpsMatVecBenchmark LANGUAGES CXX)

# Use C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Ensure we get optimized code in Release builds
set(CMAKE_BUILD_TYPE       Release  CACHE STRING "" FORCE)
set(CMAKE_CXX_FLAGS_RELEASE
    "${CMAKE_CXX_FLAGS_RELEASE} -O3 -march=native -funroll-loops"
)

# Find OpenMP for threading
find_package(OpenMP REQUIRED)

# Find Intel MKL for sparse BLAS
find_package(MKL REQUIRED)

# Sources for the benchmark executable
set(BENCH_SOURCES
    main.cpp
    data_gen.cpp
    bench_matvec.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/quasi_dense_encoder.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/bilinear_diagonal_matvec.cpp
)

# Define the executable
add_executable(sparse_matvec_bench ${BENCH_SOURCES})

# Include directories for your headers
target_include_directories(sparse_matvec_bench PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
    ${CMAKE_CURRENT_SOURCE_DIR}/../src
)

# Link against OpenMP and MKL
target_link_libraries(sparse_matvec_bench PRIVATE
    OpenMP::OpenMP_CXX
    MKL::MKL
)

# Enable AVX2 and AVX-512F so our intrinsics will compile
if (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_options(sparse_matvec_bench PRIVATE
    -march=native
  )
endif()

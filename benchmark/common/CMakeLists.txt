# Build a small static library with data_gen & encoder
add_library(benchmark_common STATIC
    data_gen.cpp
    ${CMAKE_SOURCE_DIR}/../src/ellpack_encoder.cpp
    ${CMAKE_SOURCE_DIR}/../src/ellpack_matmul.cpp
    ${CMAKE_SOURCE_DIR}/../src/ellpack_matvec.cpp
)

# Expose this directory for headers like data_gen.hpp, ellpack_encoder.hpp
target_include_directories(benchmark_common PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/../include
)


# Pull in MKL’s include dirs / defines so data_gen.cpp can use MKL_INT
target_link_libraries(benchmark_common PUBLIC
    OpenMP::OpenMP_CXX
    MKL::MKL
)

target_compile_options(benchmark_common
    PRIVATE
    -O3
    -march=native
    -funroll-loops
)
# benchmark/matmul/CMakeLists.txt

# mat‑mul benchmark executable
add_executable(sparse_matmul_bench
    main_matmul.cpp
    bench_matmul.cpp
)

# Headers from common/
target_include_directories(sparse_matmul_bench PRIVATE
    ${CMAKE_SOURCE_DIR}/common
    ${CMAKE_SOURCE_DIR}/../include
)

# Link against shared code, OpenMP, and MKL
target_link_libraries(sparse_matmul_bench PRIVATE
    benchmark_common
    OpenMP::OpenMP_CXX
    MKL::MKL
)

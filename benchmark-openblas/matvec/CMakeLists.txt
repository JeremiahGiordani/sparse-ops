# benchmark/matvec/CMakeLists.txt

# mat‑vec benchmark executable
add_executable(sparse_matvec_bench
    main_matvec.cpp
    bench_matvec.cpp
)

# Headers from common/
target_include_directories(sparse_matvec_bench PRIVATE
    ${CMAKE_SOURCE_DIR}/common
    ${CMAKE_SOURCE_DIR}/../include
)

# Link against shared code, OpenMP, and OpenBLAS
target_link_libraries(sparse_matvec_bench PRIVATE
    benchmark_common
    OpenMP::OpenMP_CXX
)

find_library(OPENBLAS_LIB openblas REQUIRED)
target_link_libraries(benchmark_common PRIVATE ${OPENBLAS_LIB})

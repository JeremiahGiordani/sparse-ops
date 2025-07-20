cmake_minimum_required(VERSION 3.15)
project(sparseops LANGUAGES CXX)

# ──────────────────────────────────────────────────────────────
#  Compiler & ISA flags
# ──────────────────────────────────────────────────────────────
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(CheckCXXCompilerFlag)
check_cxx_compiler_flag("-mavx512f" HAVE_AVX512F)
check_cxx_compiler_flag("-mavx2"    HAVE_AVX2)

set(SIMD_FLAGS "")
list(APPEND SIMD_FLAGS "-march=native")

# ──────────────────────────────────────────────────────────────
#  Optional OpenMP
# ──────────────────────────────────────────────────────────────
find_package(OpenMP QUIET)

# ──────────────────────────────────────────────────────────────
#  pybind11
# ──────────────────────────────────────────────────────────────
find_package(pybind11 CONFIG QUIET)
if (NOT pybind11_FOUND)
    # Fallback: query Python for include flags
    execute_process(
        COMMAND python3 -m pybind11 --includes
        OUTPUT_VARIABLE PYBIND11_CFLAGS
        OUTPUT_STRIP_TRAILING_WHITESPACE)
    separate_arguments(PYBIND11_CFLAGS)
endif()

# ──────────────────────────────────────────────────────────────
#  pybind11 (header-only fallback)
# ──────────────────────────────────────────────────────────────
# Hard-coded include path to your venv:
include_directories(
    /scratch/network/jg0037/env/lib64/python3.9/site-packages/pybind11/include)

# Also grab Python include / link flags
find_package(Python3 COMPONENTS Development REQUIRED)

# ──────────────────────────────────────────────────────────────
#  Static library: code generator + JIT cache
# ──────────────────────────────────────────────────────────────
add_library(sparse_codegen STATIC
    src/codegen/jit_cache.cpp
    src/codegen/spmv_template.cpp
    src/codegen/spmv_dense_template.cpp)

set_target_properties(sparse_codegen PROPERTIES POSITION_INDEPENDENT_CODE ON)


target_include_directories(sparse_codegen
    PUBLIC ${CMAKE_SOURCE_DIR}/include)

target_link_libraries(sparse_codegen
    PRIVATE pthread dl)

# ──────────────────────────────────────────────────────────────
#  Python extension module
# ──────────────────────────────────────────────────────────────
add_library(sparseops_backend MODULE
    src/bindings.cpp
    src/dense_matvec.cpp
    src/bcoo16_encoder.cpp
    src/sparse_matvec_mt.cpp
    src/sparse_dispatch.cpp
    src/sparse_preproc.cpp
    src/dense_block_kernel.cpp
    src/quasi_dense_encoder.cpp
    src/bilinear_diagonal_matvec.cpp)
target_include_directories(sparseops_backend PRIVATE ${PYBIND11_INCLUDES})
# Naming: produce "sparseops_backend.so" directly
set_target_properties(sparseops_backend PROPERTIES
    PREFIX ""
    SUFFIX ".so")

target_include_directories(sparseops_backend
    PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${pybind11_INCLUDE_DIRS}      # empty if find_package failed
        ${PYBIND11_CFLAGS})           # fallback include flags

target_link_libraries(sparseops_backend
    PRIVATE
        sparse_codegen
        Python3::Python
        $<$<BOOL:${OpenMP_CXX_FOUND}>:OpenMP::OpenMP_CXX>)


target_compile_options(sparseops_backend
    PRIVATE
        -O3 -falign-functions=64 -DNDEBUG
        ${SIMD_FLAGS}
        $<$<BOOL:${OpenMP_CXX_FOUND}>:-fopenmp>
        -DPYBIND11_DETAILED_ERROR_MESSAGES)

# ──────────────────────────────────────────────────────────────
#  Install rules (optional)
# ──────────────────────────────────────────────────────────────
install(TARGETS sparseops_backend LIBRARY DESTINATION lib)
install(DIRECTORY include/ DESTINATION include)

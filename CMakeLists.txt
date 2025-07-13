cmake_minimum_required(VERSION 3.10)
project(ultrasparse_runtime)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Detect AVX-512 and AVX2 support
include(CheckCXXCompilerFlag)
check_cxx_compiler_flag("-mavx512f" COMPILER_SUPPORTS_AVX512F)
check_cxx_compiler_flag("-mavx2" COMPILER_SUPPORTS_AVX2)

if(COMPILER_SUPPORTS_AVX512F)
    set(AVX512_FLAGS "-mavx512f")
endif()

if(COMPILER_SUPPORTS_AVX2)
    set(AVX2_FLAGS "-mavx2")
endif()

# Output shared library
add_library(ultrasparse SHARED
    src/bindings.cpp
    src/dense_matvec.cpp
    # src/sparse_matvec.cpp  # (optional: keep for legacy, but not needed for dense)
)

# Build Python extension module
add_library(sparseops_backend MODULE src/bindings.cpp src/dense_matvec.cpp src/bcoo16_encoder.cpp src/sparse_matvec.cpp)
set_target_properties(sparseops_backend PROPERTIES
    PREFIX ""
    SUFFIX ".so"
)

# Include our C++ headers and pybind11
include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(/scratch/network/jg0037/env/lib64/python3.9/site-packages/pybind11/include)
include_directories(/usr/include/python3.9)
# Get pybind11 include dirs from Python
execute_process(
    COMMAND python3 -m pybind11 --includes
    OUTPUT_VARIABLE PYBIND11_INCLUDES
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
separate_arguments(PYBIND11_INCLUDES)
target_include_directories(ultrasparse PRIVATE ${PYBIND11_INCLUDES})
target_include_directories(sparseops_backend PRIVATE ${PYBIND11_INCLUDES})

# Link OpenMP if available
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    message(STATUS "Found OpenMP: enabling")
    target_link_libraries(ultrasparse PRIVATE OpenMP::OpenMP_CXX)
    target_link_libraries(sparseops_backend PRIVATE OpenMP::OpenMP_CXX)
else()
    message(STATUS "OpenMP not found: manually setting for macOS")
    include_directories(SYSTEM "/opt/homebrew/opt/libomp/include")
    link_directories("/opt/homebrew/opt/libomp/lib")
    target_link_libraries(ultrasparse PRIVATE omp)
    target_link_libraries(sparseops_backend PRIVATE omp)
endif()

target_compile_options(ultrasparse PRIVATE -O3 ${AVX512_FLAGS} ${AVX2_FLAGS} -mfma -fopenmp -falign-functions=64 -DNDEBUG)
# target_compile_options(sparseops_backend PRIVATE -O3 ${AVX512_FLAGS} ${AVX2_FLAGS} -mfma -fopenmp -falign-functions=64 -DNDEBUG -DPYBIND11_DETAILED_ERROR_MESSAGES)

target_compile_options(sparseops_backend PRIVATE -O3 ${AVX512_FLAGS} ${AVX2_FLAGS} -mfma -fopenmp -falign-functions=64 -DNDEBUG -DPYBIND11_DETAILED_ERROR_MESSAGES)


# Install rules (optional)
install(TARGETS ultrasparse DESTINATION lib)
install(DIRECTORY include/ DESTINATION include)

cmake_minimum_required(VERSION 3.10)
project(ultrasparse_runtime)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Output shared library
add_library(ultrasparse SHARED
    src/bindings.cpp
    src/dense_matvec.cpp
    # src/sparse_matvec.cpp  # (optional: keep for legacy, but not needed for dense)
)

# Build Python extension module
add_library(sparseops_backend MODULE src/bindings.cpp src/dense_matvec.cpp src/kernel_manager.cpp)
set_target_properties(sparseops_backend PROPERTIES
    PREFIX ""
    SUFFIX ".so"
)

# Include our C++ headers and pybind11
include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(/scratch/network/jg0037/env/lib64/python3.9/site-packages/pybind11/include)
include_directories(/usr/include/python3.9)
include_directories(${CMAKE_SOURCE_DIR}/third_party/xbyak)
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

target_compile_options(ultrasparse PRIVATE -O3 -mavx512f -mavx512vl -mavx2 -mfma -fopenmp -falign-functions=64 -DNDEBUG)
target_compile_options(sparseops_backend PRIVATE -O3 -mavx512f -mavx512vl -mavx2 -mfma -fopenmp -falign-functions=64 -DNDEBUG)

# Install rules (optional)
install(TARGETS ultrasparse DESTINATION lib)
install(DIRECTORY include/ DESTINATION include)

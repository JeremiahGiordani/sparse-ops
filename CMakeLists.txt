cmake_minimum_required(VERSION 3.10)
project(ultrasparse_runtime)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Add aggressive optimization flags
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -march=native -mtune=native -ffast-math -funroll-loops")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O3 -g -DNDEBUG -march=native -mtune=native -ffast-math -funroll-loops")

# Enable release mode by default
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Add optimization flags
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -mtune=native -ffast-math")
set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g")

# Set build type to Release if not specified
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Find required packages
find_package(pybind11 QUIET)

# Output shared library (original)
add_library(ultrasparse SHARED
    src/bindings.cpp
    src/sparse_matvec.cpp
)

# New sparseops library
add_library(sparseops_core STATIC
    src/sparseops.cpp
    src/packed_matrix.cpp
    src/kernel_cache.cpp
)

# Set position independent code for static library
set_property(TARGET sparseops_core PROPERTY POSITION_INDEPENDENT_CODE ON)

# Include our C++ headers
target_include_directories(ultrasparse PRIVATE include)
target_include_directories(sparseops_core PRIVATE include)

# Pybind11 module (if available)
if(pybind11_FOUND)
    pybind11_add_module(sparseops_backend src/pybind_interface.cpp)
    target_include_directories(sparseops_backend PRIVATE include)
    target_link_libraries(sparseops_backend PRIVATE sparseops_core)
    
    # Compiler-specific definitions
    target_compile_definitions(sparseops_backend PRIVATE VERSION_INFO=${EXAMPLE_VERSION_INFO})
else()
    message(WARNING "pybind11 not found. Python interface will not be built.")
endif()

# Link OpenMP if available
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    message(STATUS "Found OpenMP: enabling")
    target_link_libraries(ultrasparse PRIVATE OpenMP::OpenMP_CXX)
    target_link_libraries(sparseops_core PRIVATE OpenMP::OpenMP_CXX)
    if(TARGET sparseops_backend)
        target_link_libraries(sparseops_backend PRIVATE OpenMP::OpenMP_CXX)
    endif()
else()
    message(STATUS "OpenMP not found: manually setting for macOS")
    include_directories(SYSTEM "/opt/homebrew/opt/libomp/include")
    link_directories("/opt/homebrew/opt/libomp/lib")
    target_link_libraries(ultrasparse PRIVATE omp)
    target_link_libraries(sparseops_core PRIVATE omp)
    if(TARGET sparseops_backend)
        target_link_libraries(sparseops_backend PRIVATE omp)
    endif()
endif()

# Install rules (optional)
install(TARGETS ultrasparse DESTINATION lib)
install(DIRECTORY include/ DESTINATION include)

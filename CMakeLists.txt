cmake_minimum_required(VERSION 3.10)
project(ultrasparse_runtime)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Output shared library
add_library(ultrasparse SHARED
    src/bindings.cpp
    src/sparse_matvec.cpp
)

# Include our C++ headers
target_include_directories(ultrasparse PRIVATE
    include
)

# Link OpenMP if available
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    message(STATUS "Found OpenMP: enabling")
    target_link_libraries(ultrasparse PRIVATE OpenMP::OpenMP_CXX)
else()
    message(STATUS "OpenMP not found: manually setting for macOS")
    include_directories(SYSTEM "/opt/homebrew/opt/libomp/include")
    link_directories("/opt/homebrew/opt/libomp/lib")
    target_link_libraries(ultrasparse PRIVATE omp)
endif()

# Install rules (optional)
install(TARGETS ultrasparse DESTINATION lib)
install(DIRECTORY include/ DESTINATION include)

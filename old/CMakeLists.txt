cmake_minimum_required(VERSION 3.10)
project(ultrasparse_runtime)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Output shared library
add_library(ultrasparse SHARED
    src/bindings.cpp
    src/model.cpp
    src/sparse_linear.cpp
)

# Include our project headers
target_include_directories(ultrasparse PRIVATE
    include
    third_party/cnpy
)

# Add cnpy subdirectory
add_subdirectory(third_party/cnpy)

# Link cnpy and ZLIB (cnpy depends on zlib)
find_package(ZLIB REQUIRED)
target_link_libraries(ultrasparse PRIVATE cnpy ZLIB::ZLIB)

# Find and link nlohmann_json (installed via Homebrew or system-wide)
find_package(nlohmann_json REQUIRED)
target_link_libraries(ultrasparse PRIVATE nlohmann_json::nlohmann_json)

# Optionally install
install(TARGETS ultrasparse DESTINATION lib)
install(DIRECTORY include/ DESTINATION include)

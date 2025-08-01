cmake_minimum_required(VERSION 3.15)
project(sparseops LANGUAGES CXX)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# ──────────────────────────────────────────────────────────────
#  Compiler & ISA flags
# ──────────────────────────────────────────────────────────────
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(SIMD_FLAGS "-march=native")

include(CheckCXXCompilerFlag)
check_cxx_compiler_flag("-mavx512f" HAVE_AVX512F)
check_cxx_compiler_flag("-mavx2"    HAVE_AVX2)

# ──────────────────────────────────────────────────────────────
#  Optional OpenMP
# ──────────────────────────────────────────────────────────────
find_package(OpenMP QUIET)

# ──────────────────────────────────────────────────────────────
#  Python3 dev (headers & libs)
# ──────────────────────────────────────────────────────────────
find_package(Python3 COMPONENTS Development REQUIRED)
find_package(Python3 COMPONENTS Interpreter REQUIRED)

# ──────────────────────────────────────────────────────────────
#  pybind11 (header-only)
# ──────────────────────────────────────────────────────────────
find_package(pybind11 CONFIG QUIET)
if (NOT pybind11_FOUND)
  execute_process(
    COMMAND python3 -m pybind11 --includes
    OUTPUT_VARIABLE PYBIND11_CFLAGS
    OUTPUT_STRIP_TRAILING_WHITESPACE
  )
  separate_arguments(PYBIND11_CFLAGS)
endif()

set(PY_VER "${Python3_VERSION_MAJOR}.${Python3_VERSION_MINOR}")

set(_cand1 "${CMAKE_SOURCE_DIR}/env/lib/python${PY_VER}/site-packages/pybind11/include")
set(_cand2 "${CMAKE_SOURCE_DIR}/env/lib64/python${PY_VER}/site-packages/pybind11/include")

if (EXISTS "${_cand1}")
  set(PYBIND11_INCLUDE_DIRS "${_cand1}")
elseif (EXISTS "${_cand2}")
  set(PYBIND11_INCLUDE_DIRS "${_cand2}")
else()
  message(FATAL_ERROR
    "Could not find pybind11 in your venv!\n"
    "Tried:\n"
    "  ${_cand1}\n"
    "  ${_cand2}\n"
    "Make sure you have run: \n"
    "  source env/bin/activate && pip install pybind11"
  )
endif()


# ──────────────────────────────────────────────────────────────
#  ONNX protos (vendored under third_party/onnx)
# ──────────────────────────────────────────────────────────────
set(ONNX_PROTO_ROOT ${CMAKE_SOURCE_DIR}/third_party/onnx)
set(GENERATED_PROTO "${ONNX_PROTO_ROOT}/generated/onnx.pb.cc")

if (EXISTS ${GENERATED_PROTO})
  set(PROTO_SRCS ${ONNX_PROTO_ROOT}/generated/onnx.pb.cc)
  set(PROTO_HDRS ${ONNX_PROTO_ROOT}/generated/onnx.pb.h)
  set(LINK_PROTOBUF FALSE)
else()
  message(STATUS "Generated Protobuf files not found. Falling back to build-time generation.")
  find_package(Protobuf REQUIRED)
  file(GLOB ONNX_PROTOS
    ${ONNX_PROTO_ROOT}/onnx/onnx.proto
    # ${ONNX_PROTO_ROOT}/onnx/onnx-ml.proto  # remove this if not used!
  )
  protobuf_generate_cpp(PROTO_SRCS PROTO_HDRS ${ONNX_PROTOS})
  set(LINK_PROTOBUF TRUE)
endif()

add_library(onnx_protos STATIC ${PROTO_SRCS} ${PROTO_HDRS})

if (LINK_PROTOBUF)
  target_link_libraries(onnx_protos PUBLIC ${Protobuf_LIBRARIES})
endif()

# ──────────────────────────────────────────────────────────────
#  Your Python extension
# ──────────────────────────────────────────────────────────────
add_library(sparseops_backend MODULE
  src/bindings.cpp
  src/ellpack_encoder.cpp
  src/ellpack_matvec.cpp
  src/ellpack_matmul.cpp
  src/activations.cpp
  src/sparse_onnx.cpp
)
set_target_properties(sparseops_backend PROPERTIES
  PREFIX ""
  SUFFIX ".so"
)

# pull in all the include paths
target_include_directories(sparseops_backend PRIVATE
  ${CMAKE_SOURCE_DIR}/include
  ${Protobuf_INCLUDE_DIRS}
  ${CMAKE_CURRENT_BINARY_DIR}      # for generated onnx/*.pb.h
  ${PYBIND11_INCLUDE_DIRS}
)

# link against Python, Protobuf, OpenMP and our onnx_protos
target_link_libraries(sparseops_backend PRIVATE
  Python3::Python
  $<$<BOOL:${OpenMP_FOUND}>:OpenMP::OpenMP_CXX>
  onnx_protos
)

target_compile_options(sparseops_backend PRIVATE
  -O3
  -falign-functions=64
  -DNDEBUG
  ${SIMD_FLAGS}
  $<$<BOOL:${OpenMP_FOUND}>:-fopenmp>
  -DPYBIND11_DETAILED_ERROR_MESSAGES
)

# ──────────────────────────────────────────────────────────────
#  (Optional) install rules
# ──────────────────────────────────────────────────────────────
install(TARGETS sparseops_backend LIBRARY DESTINATION lib)
install(DIRECTORY include/ DESTINATION include)

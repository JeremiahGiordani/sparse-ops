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
#  Protobuf (for ONNX protos)
# ──────────────────────────────────────────────────────────────
find_package(Protobuf REQUIRED)

# ──────────────────────────────────────────────────────────────
#  ONNX protos (no sudo, vendored under third_party/onnx)
# ──────────────────────────────────────────────────────────────
set(ONNX_PROTO_ROOT ${CMAKE_SOURCE_DIR}/third_party/onnx)
file(GLOB ONNX_PROTOS
  ${ONNX_PROTO_ROOT}/onnx/onnx.proto
  ${ONNX_PROTO_ROOT}/onnx/onnx-ml.proto
)

# This macro comes from FindProtobuf.cmake
protobuf_generate_cpp(
  PROTO_SRCS  PROTO_HDRS
  ${ONNX_PROTOS}
)

# bundle generated code into a tiny static lib
add_library(onnx_protos STATIC
  ${PROTO_SRCS}
  ${PROTO_HDRS}        # so IDEs know about the headers
)
target_link_libraries(onnx_protos PUBLIC ${Protobuf_LIBRARIES})
target_include_directories(onnx_protos PUBLIC
  ${CMAKE_CURRENT_BINARY_DIR}  # where onnx/*.pb.h now lives
)

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
  src/model_run.cpp
  src/layers/activation.cpp
  src/layers/add.cpp
  src/layers/conv.cpp
  src/layers/flatten.cpp
  src/layers/matmul.cpp
  src/layers/pool.cpp
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
  ${Protobuf_LIBRARIES}
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

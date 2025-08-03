# SparseOps Backend: ELLPACK Sparse Matrix Kernels

A high-performance CPU backend for sparse **matrix–matrix** and **matrix–vector** operations, plus **Sparse ONNX model inference** using the ELLPACK format.

---

## Features

* **High-Performance Sparse Matrix–Matrix Multiply** (`ellpack_matmul`)

  * Ultra-tight SIMD loops with register blocking and tiling to `simd_width`
  * Pulls data from contiguous aligned buffers to minimize cache misses
  * Avoids unnecessary FLOPs by iterating only over non-zero entries
  * Runtime detection of AVX‑512 / AVX2, fallback to scalar loops
  * OpenMP parallelism over independent rows
  * 64‑byte alignment for maximal memory throughput
* **Efficient Sparse Matrix–Vector Multiply** (`ellpack_matvec`)

  * Single-pass gather-then-dot approach
  * FMA-accelerated fused multiply-adds
  * Optional bias initialization via `memcpy` or zero-fill
* **Sparse ONNX Model Inference** (`SparseOnnxModel`)

  * Load and preprocess an ONNX graph at load time
  * Extract and ELLPACK-encode all weight matrices
  * Support for MatMul, Relu, Sigmoid, Tanh layers
  * Zero-allocation inference with fixed batch size
* **Python API & Bindings** via `pybind11`
* **Unit tests** for correctness and **benchmark scripts** for performance exploration

---

## Repository Structure

```text
.
├── include/                # Public headers
│   ├── aligned_buffer.hpp  # 64-byte aligned float buffer wrapper
│   ├── ellpack_encoder.hpp # ELLPACK format encoder/decoder
│   ├── ellpack_matmul.hpp  # ELLPACK sparse MatMul API
│   ├── ellpack_matvec.hpp  # ELLPACK sparse MatVec API
│   └── sparse_onnx.hpp     # Sparse ONNX model interface
│
├── src/                    # Core C++ implementations
│   ├── aligned_buffer.cpp
│   ├── ellpack_encoder.cpp
│   ├── ellpack_matmul.cpp
│   ├── ellpack_matvec.cpp
│   ├── sparse_onnx.cpp     # ONNX parsing & inference
│   └── bindings.cpp        # pybind11 glue
│
├── python/                 # Python convenience wrappers
│   └── cpp_backend.py      # Thin wrapper: encode, matvec, matmul, SparseOnnxModel
│
├── tests/                  # End-to-end benchmarks & demos (Python)
│   ├── test_matvec.py
│   └── test_matmul.py
│
├── tests/unit/             # Unit tests (pytest)
│   ├── test_ellpack_encoder.py
│   ├── test_matvec.py
│   └── test_matmul.py
│
├── benchmark-mkl/          # Standalone C++ benchmarks vs. Intel MKL
├── benchmark-openblas/     # Standalone C++ benchmarks vs. OpenBLAS
│
├── CMakeLists.txt          # Root CMake build for Python library
├── requirements.txt        # Python dependencies (NumPy, pybind11)
└── README.md               # This document
```

---

## Ellpack Format Overview

The ELLPACK format packs each row’s non-zero entries into a fixed-width buffer:

```cpp
struct Ellpack {
    uint32_t m, n, r;       // # rows, # cols, max non-zeros per row
    AlignedBuffer Wd;       // [m × r] packed non-zero values
    std::vector<uint32_t> idx;   // [m × r] column indices
    AlignedBuffer Xt;       // scratch for gathers or decode
    std::vector<uint32_t> nnz, rev_off, rev_pos; // helpers
};
```

* **Encoding**: count non-zeros → allocate `Ellpack(m,n,r)` → pack values & indices into contiguous buffers → build reverse offsets for optional decoding.
* **Decoding**: scatter back into a dense buffer (used in tests) by iterating only over `nnz[i]` entries per row.

---

## Core API

### Sparse Matrix–Matrix Multiply (`ellpack_matmul`)

```cpp
void ellpack_matmul(
    const Ellpack &E,
    const float* X,    // shape = [E.n × C]
    uint32_t C,        // # columns in X
    const float* bias, // length = E.m (or nullptr)
    float* Y           // out: [E.m × C]
);
```

**Algorithm Highlights:**

1. **Row Blocking:** parallelize over rows with OpenMP, each thread handles one or more rows independently.
2. **Column Tiling:** split `C` into blocks of `simd_width` (e.g., 16 for AVX‑512) so each vector register holds a full chunk.
3. **Initialization:** load bias (or zero) into output register block.
4. **Accumulate:** for each non-zero `(w, j)` in the row, broadcast `w`, gather `X[j, cb…cb+simd_width)`, perform FMA.
5. **Store:** write back the register block once per tile.

> **Why it’s efficient:**
>
> * **Contiguous memory:** both `Wd` and `idx` are tightly packed, improving cache line utilization.
> * **Avoid FLOPs:** skip zero entries completely, doing only one multiply-add per non-zero.
> * **Register reuse:** each output tile stays in registers across all non-zero updates.
> * **Hardware prefetchers** can coalesce indirect accesses in `X` thanks to tiled gathers.

### Sparse Matrix–Vector Multiply (`ellpack_matvec`)

```cpp
void ellpack_matvec(
    const Ellpack &E,
    const float* x,     // length = E.n
    const float* bias,  // length = E.m (or nullptr)
    float* y            // length = E.m
);
```

* **Gather:** load `x` entries based on `idx` into scratch buffer `Xt` in one pass.
* **Dot:** FMA-based horizontal reduction over `Wd` and `Xt`.
* **Bias:** optional via `memcpy` or zero-fill.

---

## Sparse ONNX Model Inference

```cpp
SparseOnnxModel model("model.onnx");
model.resize_buffers(batch_size);
model.run(input_data, batch_size, output_data);
```

* Parses initializers and graph topology in `sparse_onnx.cpp`.
* Encodes each weight matrix to ELLPACK.
* Supports fixed batch inference (MatMul, activation).

---

## Python API & Bindings

```python
from sparseops_backend import convert_to_ellpack, ellpack_matvec, ellpack_matmul, SparseOnnxModel
from python.cpp_backend import encode, matvec, matmul

E = encode(W)                 # dense → Ellpack
Y = matmul(E, X, bias)        # sparse mat-mul
y = matvec(E, x, bias)        # sparse mat-vec
model = SparseOnnxModel("m.onnx")
Y_onx = model.run(X)
```

---

## Usage Examples

### Vector Multiply

```python
import numpy as np
from python.cpp_backend import encode, matvec

W = np.random.randn(512, 256).astype(np.float32)
W[W < 0.8] = 0.0                          # sparsify
E = encode(W)                            # build Ellpack
x = np.random.randn(256).astype(np.float32)
bias = np.random.randn(512).astype(np.float32)

y = matvec(E, x, bias)                   # shape = (512,)
# y  ==  W @ x + bias
```

### Matrix Multiply

```python
import numpy as np
from python.cpp_backend import encode, matmul

W = np.random.randn(1024, 512).astype(np.float32)
W[W < 0.9] = 0.0
E = encode(W)

X = np.random.randn(512, 10).astype(np.float32)
bias = np.random.randn(1024).astype(np.float32)

Y = matmul(E, X, bias)                   # shape = (1024, 10)
# Y  ==  W @ X  +  bias[:,None]
```

The scripts under `tests/` (e.g. `test_matvec.py`, `test_matmul.py`) show how to integrate this into a benchmark loop, comparing against PyTorch, NumPy, and SciPy.

### ONNX Model Inference

```python
import numpy as np
from python.cpp_backend import OnnxModel

model = OnnxModel("model.onnx")
output = model.run(x_infer)
```

---

## Tests & Benchmarks

### Python Tests & Benchmarks

* **Unit tests:** `pytest tests/unit`
* **Python benchmarks:**

  ```bash
  python tests/test_matvec.py
  python tests/test_matmul.py
  ```

### C++ Benchmarks

Build and compare ELLPACK vs. MKL and OpenBLAS:

```bash
# Intel MKL benchmark
cd benchmark-mkl
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j

# OpenBLAS benchmark
cd benchmark-openblas
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

**Generated executables:**

* `build/matvec/sparse_matvec_bench`
* `build/matmul/sparse_matmul_bench`

**Usage flags:**

| Flag            | Description                               |
| --------------- | ----------------------------------------- |
| `--M`           | Number of rows of the sparse matrix       |
| `--N`           | Number of cols of the sparse matrix       |
| `--C`           | Number of columns in dense input (MatMul) |
| `--sparsity`    | Fraction of entries set to zero           |
| `--runs`        | Number of timed repetitions (median)      |
| `--mkl-threads` | MKL thread‑pool size                      |
| `--omp-threads` | OpenMP team size for ELLPACK kernels      |
| `--seed`        | RNG seed for reproducible matrices        |
| `--irregular`   | Force last row to be fully dense (0/1)    |

*Example:*

```bash
./build/matvec/sparse_matvec_bench --M 2000 --N 2000 --sparsity 0.9 --runs 50 --mkl-threads 1 --omp-threads 4 --seed 42
```

---

## Building the Python Library

### Prerequisites

* C++17 compiler with OpenMP & AVX2/AVX512 support
* Python 3, NumPy
* `pybind11` (via `requirements.txt`)

### Build & Install

```bash
git clone ...
cd sparse-ops
python -m venv env
source env/bin/activate
pip install -r requirements.txt
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
# yields lib(sparseops_backend).so, importable by Python
```

---

With this setup, you have a flexible, high-performance sparse CPU backend for NumPy/PyTorch workflows and sparse ONNX model inference. Enjoy!

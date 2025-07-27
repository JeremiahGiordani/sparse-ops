# SparseOps Backend: Quasi‑Dense Sparse Matrix Kernels

This repository implements a custom CPU backend for very efficient sparse **matrix–vector** and **matrix–matrix** multiplications.  It achieves this by:

1. **Encoding** a sparse matrix into a “quasi‑dense” format that packs each row’s non‑zero entries into a fixed‑width buffer.  
2. Exposing two highly‑optimized kernels—one for **mat‑vec** and one for **mat‑mul**—that operate directly on this format, leveraging  
   - runtime detection of AVX‑512 / AVX2,  
   - OpenMP for parallelism,  
   - 64‑byte alignment for maximal throughput.  
3. Wrapping everything in a **pybind11**‑based Python API, so you can do
   ```python
   Q    = encode(dense_matrix)          # build the QuasiDense handle
   y    = matvec(Q, x, bias)            # sparse matrix–vector multiply
   Y    = matmul(Q, X, bias)            # sparse matrix–matrix multiply
    ````

4. Providing both **unit tests** (via pytest) for correctness and **benchmark scripts** for performance exploration.

---

## 📂 Repository Structure

```
.
├── include/
│   ├── aligned_buffer.hpp            — 64‑byte‑aligned float buffer wrapper
│   ├── quasi_dense_encoder.hpp       — QuasiDense struct + encode/decode APIs
│   ├── bilinear_diagonal_matvec.hpp  — declaration of quasi_dense_matvec
│   └── bilinear_diagonal_matmul.hpp  — declaration of quasi_dense_matmul
│
├── src/
│   ├── aligned_buffer.cpp            — (none; header‑only)
│   ├── quasi_dense_encoder.cpp       — implementation of convert/decode
│   ├── bilinear_diagonal_matvec.cpp  — vector kernel
│   ├── bilinear_diagonal_matmul.cpp  — matrix kernel
│   └── bindings.cpp                  — pybind11 glue
│
├── python/
│   └── cpp_backend.py                — thin Python wrapper over the C++ module
│
├── tests/
│   ├── test_matvec.py                — end‑to‑end mat‑vec benchmark/demo
│   └── test_matmul.py                — end‑to‑end mat‑mul benchmark/demo
│
├── tests/unit/
│   ├── test_quasi_encoder.py         — correctness of encode/decode
│   ├── test_matvec.py                — correctness of mat‑vec kernel
│   └── test_matmul.py                — correctness of mat‑mul kernel
│
├── CMakeLists.txt                    — build instructions
├── rebuild.sh                        — helper script
└── README.md                         — this document
```

---

## 🔧 Quasi‑Dense Format

### `include/quasi_dense_encoder.hpp` → `src/quasi_dense_encoder.cpp`

```cpp
struct QuasiDense {
    uint32_t m;    // # rows
    uint32_t n;    // # cols (original)
    uint32_t r;    // max non‑zeros in any row

    AlignedBuffer     Wd;      // [m × r] packed non‑zero values
    std::vector<uint32_t> idx; // [m × r] column indices
    AlignedBuffer     Xt;      // [m × r] scratch buffer for gathers
    std::vector<uint32_t> nnz;     // actual nnz per row
    std::vector<uint32_t> rev_off; // CSR‑style row offsets for decode
    std::vector<uint32_t> rev_pos; // flattened positions for decode
};
```

1. **Encoding** (`convert_to_quasi_dense`):

   * **Scan** each row of the original dense matrix `W` to **count** its non‑zeros → `rowCounts[i]`.
   * **Find** `r = max(rowCounts)`.
   * **Allocate** a `QuasiDense Q(m,n,r)`.
   * **Zero‑initialize** `Q.Wd` then **pack** every non‑zero `v` at `(i,j)` into `Q.Wd.ptr[i*r + pos]` and `Q.idx[i*r + pos] = j`.
   * **Build** reverse offsets `rev_off` and flattened positions `rev_pos` so you can **scatter** back if desired.
2. **Decoding** (`decode_from_quasi_dense`):

   * Zero out an `m×n` output buffer.
   * For each row `i`, for `j` in `[0..nnz[i])` scatter `Q.Wd.ptr[i*r + j]` back to its original column index.

---

## 🚀 Sparse Matrix–Vector Multiply

### API

```cpp
void quasi_dense_matvec(
    const QuasiDense &Q,
    const float*      x,     // length = Q.n
    const float*      bias,  // length = Q.m (or nullptr to zero‐init)
    float*            y      // length = Q.m
);
```

### Key points (see `src/bilinear_diagonal_matvec.cpp`)

1. **Output Init**

   * If `bias` is non‐null, `memcpy(y, bias, …)`, else `fill(y, 0)`.
2. **Threading**

   * Read `OMP_NUM_THREADS`; fall back to `omp_get_max_threads()`.
   * `#pragma omp parallel for` over the `m` rows.
3. **SIMD**

   * At runtime call `supports_avx512()` → if available, use \_mm512 intrinsics; otherwise fall back to AVX2/\_mm256 or plain loops.
   * **Gather**: load the `r` elements of `x` into the aligned scratch buffer `Q.Xt.ptr + i*r`.
   * **Dot**: FMA‐accelerated fused multiply‐adds over `Q.Wd` and the gathered `Q.Xt` row.
   * **Horizontal reduction** to collapse the SIMD register to a scalar, then `y[i] += acc`.

---

## 🚀 Sparse Matrix–Matrix Multiply

### API

```cpp
void quasi_dense_matmul(
    const QuasiDense &Q,
    const float*      X,     // shape = [Q.n × C]
    uint32_t          C,
    const float*      bias,  // length = Q.m
    float*            Y      // out buffer shape = [Q.m × C]
);
```

### High‑Level Algorithm (in `src/bilinear_diagonal_matmul.cpp`)

```text
for each row i in 0..m-1:
    yrow = Y + i        // pointer to row i storage
    yrow[*] ← bias[i]   // broadcast bias to all C columns

    // For each packed non‑zero in row i:
    for t in 0..r-1:  
        w = Q.Wd.ptr[i*r + t]       // value
        col = Q.idx[i*r + t]        // original column
        xrow = X + col*C            // pointer to that column in X

        // vectorized:  Y_row[0..C) += w * X[col,0..C)
        // tail‑handle for C % VLEN
```

* Parallelized over `i` with OpenMP.
* Uses AVX2 intrinsics (`_mm256_fmadd_ps`) for the inner loop.

---

## 🐍 Python Bindings

All C++ types/functions are exposed by **pybind11** in `src/bindings.cpp` as the module **`sparseops_backend`**:

* **Classes & Handles**

  ```python
  Q = sparseops_backend.convert_to_quasi_dense(np_matrix)
  ```
* **Methods**

  ```python
  Y = sparseops_backend.decode_from_quasi_dense(Q)
  y = sparseops_backend.bilinear_diagonal_matvec(Q, x, bias)
  Y = sparseops_backend.bilinear_diagonal_matmul(Q, X, bias)
  ```
* **Properties** on `QuasiDense`:

  * `.m, .n, .r` (dimensions)
  * `.Wd` → NumPy view of the packed values (shape `[m, r]`)
  * `.idx` → NumPy view of the packed column‐indices (shape `[m, r]`)
  * `.Xt` → the scratch buffer (usually you don’t need this directly)
* A thin helper wrapper in `python/cpp_backend.py` re‑exports these as `encode`, `matvec`, and `matmul` for convenience.

---

## 🔍 Usage Examples

### Vector Multiply

```python
import numpy as np
from python.cpp_backend import encode, matvec

W = np.random.randn(512, 256).astype(np.float32)
W[W < 0.8] = 0.0                          # sparsify
Q = encode(W)                            # build QuasiDense
x = np.random.randn(256).astype(np.float32)
bias = np.random.randn(512).astype(np.float32)

y = matvec(Q, x, bias)                   # shape = (512,)
# y  ==  W @ x + bias
```

### Matrix Multiply

```python
import numpy as np
from python.cpp_backend import encode, matmul

W = np.random.randn(1024, 512).astype(np.float32)
W[W < 0.9] = 0.0
Q = encode(W)

X = np.random.randn(512, 10).astype(np.float32)
bias = np.random.randn(1024).astype(np.float32)

Y = matmul(Q, X, bias)                   # shape = (1024, 10)
# Y  ==  W @ X  +  bias[:,None]
```

The scripts under `tests/` (e.g. `test_matvec.py`, `test_matmul.py`) show how to integrate this into a benchmark loop, comparing against PyTorch, NumPy, and SciPy.

---

## ✅ Correctness & Benchmarks

* **Unit tests** in `tests/unit/` validate:

  * Round‑trip **encode ↔ decode**
  * **matvec** correctness against a dense reference
  * **matmul** correctness across various sizes & sparsities
* **Benchmark scripts** in `tests/` measure the throughput over many runs, illustrating the performance gains on modern CPUs when using AVX and multi‑threading.

---

## ⚙️ Building

1. Install prerequisites:

   * C++17 compiler with OpenMP & AVX2/AVX512 support
   * [pybind11](https://github.com/pybind/pybind11)
   * Python 3, NumPy

2. From the repo root:

   ```bash
   mkdir build
   cmake --build build -j
   # this produces `lib(sparseops_backend).so` that Python can import
   ```

3. (Optional) Run unit tests:

   ```bash
   pytest tests/unit
   ```

---

With this setup, you get a flexible, high‑performance sparse CPU backend that can slot into your NumPy/PyTorch workflows with minimal boilerplate. Enjoy!

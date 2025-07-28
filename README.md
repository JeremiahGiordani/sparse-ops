# SparseOps Backend: ELLPACK Sparse Matrix Kernels

This repository implements a custom CPU backend for very efficient sparse **matrix–vector** and **matrix–matrix** multiplications.  It achieves this by:

1. **Encoding** a sparse matrix into ELLPACK format that packs each row’s non‑zero entries into a fixed‑width buffer.  
2. Exposing two highly‑optimized kernels—one for **mat‑vec** and one for **mat‑mul**—that operate directly on this format, leveraging  
   - runtime detection of AVX‑512 / AVX2,  
   - OpenMP for parallelism,  
   - 64‑byte alignment for maximal throughput.  
3. Wrapping everything in a **pybind11**‑based Python API, so you can do
   ```python
   E    = encode(dense_matrix)          # build the Ellpack handle
   y    = matvec(E, x, bias)            # sparse matrix–vector multiply
   Y    = matmul(E, X, bias)            # sparse matrix–matrix multiply
    ````

4. Providing both **unit tests** (via pytest) for correctness and **benchmark scripts** for performance exploration.

---

## Repository Structure

```
.
├── include/
│   ├── aligned_buffer.hpp            — 64‑byte‑aligned float buffer wrapper
│   ├── ellpack_encoder.hpp           — Ellpack struct + encode/decode APIs
│   ├── ellpack_matvec.hpp            — declaration of ellpack_matvec
│   └── ellpack_matmul.hpp            — declaration of ellpack_matmul
│
├── src/
│   ├── aligned_buffer.cpp            — (none; header‑only)
│   ├── ellpack_encoder.cpp           — implementation of convert/decode
│   ├── ellpack_matvec.cpp            — vector kernel
│   ├── ellpack_matmul.cpp            — matrix kernel
│   └── bindings.cpp                  — pybind11 bindings
│
├── python/
│   └── cpp_backend.py                — thin Python wrapper over the C++ module
│
├── tests/
│   ├── test_matvec.py                — end‑to‑end mat‑vec benchmark/demo
│   └── test_matmul.py                — end‑to‑end mat‑mul benchmark/demo
│
├── tests/unit/
│   ├── test_ellpack_encoder.py       — correctness of encode/decode
│   ├── test_matvec.py                — correctness of mat‑vec kernel
│   └── test_matmul.py                — correctness of mat‑mul kernel
│
├── CMakeLists.txt                    — build instructions
├── rebuild.sh                        — helper script
└── README.md                         — this document
```

---

## Ellpack Format

### `include/ellpack_encoder.hpp` → `src/ellpack_encoder.cpp`

```cpp
struct Ellpack {
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

1. **Encoding** (`convert_to_ellpack`):

   * **Scan** each row of the original dense matrix `W` to **count** its non‑zeros → `rowCounts[i]`.
   * **Find** `r = max(rowCounts)`.
   * **Allocate** a `Ellpack E(m,n,r)`.
   * **Zero‑initialize** `E.Wd` then **pack** every non‑zero `v` at `(i,j)` into `E.Wd.ptr[i*r + pos]` and `E.idx[i*r + pos] = j`.
   * **Build** reverse offsets `rev_off` and flattened positions `rev_pos` so you can **scatter** back if desired.
2. **Decoding** (`decode_from_ellpack`):

   * Zero out an `m×n` output buffer.
   * For each row `i`, for `j` in `[0..nnz[i])` scatter `E.Wd.ptr[i*r + j]` back to its original column index.

---

## Sparse Matrix–Vector Multiply

### API

```cpp
void ellpack_matvec(
    const Ellpack &E,
    const float*      x,     // length = E.n
    const float*      bias,  // length = E.m (or nullptr to zero‐init)
    float*            y      // length = E.m
);
```

### Key points (see `src/ellpack_matvec.cpp`)

1. **Output Init**

   * If `bias` is non‐null, `memcpy(y, bias, …)`, else `fill(y, 0)`.
2. **Threading**

   * Read `OMP_NUM_THREADS`; fall back to `omp_get_max_threads()`.
   * `#pragma omp parallel for` over the `m` rows.
3. **SIMD**

   * At runtime call `supports_avx512()` → if available, use \_mm512 intrinsics; otherwise fall back to AVX2/\_mm256 or plain loops.
   * **Gather**: load the `r` elements of `x` into the aligned scratch buffer `E.Xt.ptr + i*r`.
   * **Dot**: FMA‐accelerated fused multiply‐adds over `E.Wd` and the gathered `E.Xt` row.
   * **Horizontal reduction** to collapse the SIMD register to a scalar, then `y[i] += acc`.

---

## Sparse Matrix–Matrix Multiply

### API

```cpp
void ellpack_matmul(
    const Ellpack &E,
    const float*      X,     // shape = [E.n × C]
    uint32_t          C,
    const float*      bias,  // length = E.m
    float*            Y      // out buffer shape = [E.m × C]
);
```

### High‑Level Algorithm (in `src/ellpack_matmul.cpp`)

1. **For each row** `i = 0…m-1`

   * Point to the output slice `yrow = Y[i,*]`
   * **Initialize** `yrow[c] = bias[i]` (or zero)

2. **Tile the columns** into chunks of `simd_width` (e.g. 16 for AVX‑512):

   ```text
   for cb in 0, simd_width, 2·simd_width, … < C:
       // handle columns [cb … cb+simd_width)
   ```

3. **Within each block**
   a. **Load** the current `yrow[cb…]` into a vector register
   b. **Accumulate** every non‑zero `(wj, colj)` in that row:

   ```
   for each packed non‑zero in row i:
       yv += wj * X[colj, cb…]
   ```

   i.e. broadcast the weight, gather that slice of `X[colj,*]`, and do a fused‑multiply‑add
   c. **Store** the updated block back to `yrow[cb…]`

4. **Repeat** until all columns are covered.

> When `C == 1`, this collapses to the same gather‑then‑dot approach used in **mat‑vec**, giving you a single‑pass, ultra‑tight SIMD loop.

---

**Why this makes sense on CPUs:** by blocking the output, we load and store each chunk of `yrow` exactly once and keep it in registers across all non‑zero updates, vastly reducing memory traffic; meanwhile tiling to `simd_width` guarantees full‐width vector FMAs and lets hardware prefetchers coalesce the otherwise indirect gathers from `X`, and since each row is independent this naturally parallelizes over threads with minimal synchronization.

---

## Python Bindings

All C++ types/functions are exposed by **pybind11** in `src/bindings.cpp` as the module **`sparseops_backend`**:

* **Classes & Handles**

  ```python
  E = sparseops_backend.convert_to_ellpack(np_matrix)
  ```
* **Methods**

  ```python
  Y = sparseops_backend.decode_from_ellpack(E)
  y = sparseops_backend.ellpack_matvec(E, x, bias)
  Y = sparseops_backend.ellpack_matmul(E, X, bias)
  ```
* **Properties** on `Ellpack`:

  * `.m, .n, .r` (dimensions)
  * `.Wd` → NumPy view of the packed values (shape `[m, r]`)
  * `.idx` → NumPy view of the packed column‐indices (shape `[m, r]`)
  * `.Xt` → the scratch buffer (usually you don’t need this directly)
* A thin helper wrapper in `python/cpp_backend.py` re‑exports these as `encode`, `matvec`, and `matmul` for convenience.

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

---

## Correctness & Python Benchmarks

* **Unit tests** in `tests/unit/` validate:

  * Round‑trip **encode ↔ decode**
  * **matvec** correctness against a dense reference
  * **matmul** correctness across various sizes & sparsities
* **Benchmark scripts** in `tests/` measure the throughput over many runs, illustrating the performance gains on modern CPUs when using AVX and multi‑threading.

---


## C++ Benchmarks

In addition to the Python tests, we provide two standalone C++ benchmarking executables—one for **sparse mat‑vec** and one for **sparse mat‑mul**—that let you exercise MKL vs. ELLPACK vs. BLAS on your local hardware.

### Directory layout

```text
benchmark/
├── common/                   # shared data‐gen & encoder library
│   └── …
├── matvec/                   # ELLPACK mat‑vec benchmark
│   ├── main_matvec.cpp       # parses --M/--N/--sparsity/--runs/--mkl-threads/--omp-threads
│   └── bench_matvec.cpp      # timing hooks for MKL CSR, MKL dense sgemv, and ellpack_matvec
├── matmul/                   # ELLPACK mat‑mul benchmark
│   ├── main_matmul.cpp       # parses --M/--N/--C/--sparsity/--runs/--mkl-threads/--omp-threads
│   └── bench_matmul.cpp      # timing hooks for MKL SpMM, BLAS GEMM, and ellpack_matmul
└── CMakeLists.txt            # top‐level, adds common/, matvec/, matmul/
```

### Building

```bash
cd benchmark
mkdir build
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

This will produce:

* `build/matvec/sparse_matvec_bench`
* `build/matmul/sparse_matmul_bench`

### Usage

Each executable supports a common set of flags:

| Flag            | Description                              |
| --------------- | ---------------------------------------- |
| `--M`           | # rows of the sparse matrix              |
| `--N`           | # cols of the sparse matrix              |
| `--sparsity`    | fraction of entries set to zero          |
| `--runs`        | # timed repetitions (median reported)    |
| `--mkl-threads` | MKL thread‐pool size                     |
| `--omp-threads` | OpenMP team size for the ELLPACK kernel  |
| `--seed`        | RNG seed for reproducible matrices       |
| `--irregular`   | if `1`, force last row to be fully dense |

The **matmul** binary also takes:

| Flag  | Description              |
| ----- | ------------------------ |
| `--C` | # columns in dense input |

#### Mat‑vec example

```bash
./build/matvec/sparse_matvec_bench \
  --M 2000 --N 2000 --sparsity 0.9 \
  --runs 50 --mkl-threads 1 --omp-threads 1 --seed 42
```

Prints:

```
=== Sparse MatVec Benchmark ===
Matrix dims:     2000×2000
Sparsity:        0.9
Repetitions:     10
MKL Threads:     1
OpenMP threads:  1
RNG seed:        42
Irregular last row: no

MKL sparse matvec:   547.862 µs
MKL dense matvec:    672.266 µs
**Ellpack matvec:      417.514 µs**
```

#### Mat‑mul example

```bash
./build/matmul/sparse_matmul_bench \
  --M 2000 --N 2000 --C 120 --sparsity 0.8 \
  --runs 10 --mkl-threads 8 --omp-threads 8 --seed 44
```

Prints:

```
=== Sparse MatMul Benchmark ===
Matrix dims:     2000×2000
Dense cols C:    120
Sparsity:        0.8
Repetitions:     10
MKL threads:     8
OpenMP threads:  8
RNG seed:        44
Irregular last row: 0

MKL sparse×dense  : 13738.5 µs
BLAS dense GEMM   : 8256.7 µs
**Ellpack matmul    : 1856.68 µs**
```

This C++ benchmark gives you a direct, low‑overhead way to measure raw kernel performance on identical data and hardware—ideal for comparing ELLPACK against MKL’s sparse and dense kernels.

---


## Building

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

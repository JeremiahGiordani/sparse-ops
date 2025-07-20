## SparseOps

This repository implements a high‑performance custom backend for matrix‑vector multiplication, focusing on both dense and sparse workloads. It leverages:

* **BCOO‑16 sparse format**: A block‑compressed sparse representation that groups nonzero entries in 16‑wide tiles, capturing both location (via a bitmask) and packed values.
* **PyBind11 bindings**: Exposes C++ routines to Python with minimal overhead.
* **JIT‑compiled micro‑kernels**: Generates specialized AVX‑2/AVX‑512 code on the fly for each unique sparse structure, caching compiled kernels to disk.
* **Multi‑threading with OpenMP**: Partitions work by sparse blocks (not rows), keeping threads busy and reducing synchronization overhead.

---

## Repository Structure

```text
.
├── CMakeLists.txt            # Build configuration for C++ library and Python extension
├── rebuild.sh                # Helper script to rebuild the extension
├── include/                  # Public headers
│   ├── bcoo16_encoder.hpp    # BCOO‑16 data structures & encode/decode API
│   ├── dense_block_kernel.hpp
│   ├── dense_matvec.hpp
│   ├── jit_cache.hpp         # JIT cache API
│   ├── sparse_dispatch.hpp   # Kernel dispatch API
│   ├── sparse_matvec_mt.hpp  # Multi‑threaded sparse matvec API
│   └── sparse_preproc.hpp
├── src/                      # C++ implementation
│   ├── bcoo16_encoder.cpp    # encode_to_bcoo16 / decode_from_bcoo16
│   ├── dense_block_kernel.cpp
│   ├── dense_matvec.cpp
│   ├── sparse_dispatch.cpp   # get_spmv_kernel (dispatch)
│   ├── sparse_matvec_mt.cpp  # sparse_matvec_avx512_mt (or AVX2)
│   ├── sparse_preproc.cpp
│   └── codegen/              # JIT codegen & cache
│       ├── jit_cache.cpp     # File‑based cache + dlopen
│       ├── spmv_template.cpp # source generator for micro‑kernels
│       └── spmv_template.hpp
├── python/                   # Python interface & utilities
│   ├── cpp_backend.py        # pybind11 wrappers: run_matvec, encode/decode, run_sparse_matvec
│   └── utils.py              # Helpers (e.g. to_csr)
└── tests/                    # Examples, unit tests, and benchmarks
    ├── full_matmul.py        # Demonstrates Python API + benchmarks across frameworks
    ├── test_matmul.py        # Verifies correctness of encode/decode & matvec
    └── …                      # Additional profiling & edge‑case tests
```

---

## Python Interface

Located in `python/cpp_backend.py`, this module provides:

* `run_matvec(weight, bias, x) → y`
  Dense matrix‐vector + bias using a blocked AVX kernel (`dense_block_kernel`).

* `encode_to_bcoo16(dense: np.ndarray) → BCOO16`
  Converts a NumPy 2D array to the opaque `BCOO16` handle.

* `decode_from_bcoo16(bcoo: BCOO16) → np.ndarray`
  Reconstructs a dense array from a `BCOO16` handle (for testing).

* `run_sparse_matvec(bcoo, bias, x, threads) → y`
  Multithreaded sparse matvec on BCOO‑16 via JIT’ed AVX‑512/AVX2 micro‑kernel.

Helper `to_csr` in `python/utils.py` converts PyTorch layers to CSR format for comparisons.

---

## BCOO‑16 Sparse Format

Defined in `include/bcoo16_encoder.hpp` / `src/bcoo16_encoder.cpp`:

```cpp
struct BCOO16Block {
    uint32_t row_id;    // row index
    uint16_t first_col; // base column (multiple of 16)
    uint16_t bitmask;   // 16‑bit mask: which lanes (columns) are nonzero
    uint32_t val_off;   // offset into contiguous values[]
};

struct BCOO16 {
    uint32_t original_num_rows, original_num_cols;
    std::vector<BCOO16Block> blocks;
    std::vector<float>       values;  // packed nonzero values in block order
};
```

* **Encoding (`encode_to_bcoo16`)** loops each row in strides of 16 columns, computes a 16‑bit mask of nonzero lanes, skips empty tiles, appends a `BCOO16Block` header (including `val_off = values.size()`), then pushes live values in lane order.
* **Decoding (`decode_from_bcoo16`)** allocates a dense matrix, iterates blocks, and scatters values back to their original positions.

---

## Dense MatVec

Two implementations:

1. **Reference naive** in `dense_matvec.cpp`, exposed as `run_matvec_py`.
2. **Blocked AVX** in `dense_block_kernel.cpp` for improved throughput, also exposed.

---

## Sparse MatVec (Multi‐Threaded)

`src/sparse_matvec_mt.cpp` implements `sparse_matvec_avx512_mt`:

1. **Thread count** determined by OpenMP (`threads > 0 ? threads : omp_get_max_threads()`).
2. **Fast path** if single thread: optionally copy bias into `y`, then invoke the JIT micro‑kernel on all blocks, writing directly into `y`.
3. **Multi‐threaded**:

   * Allocate private per‑thread output buffers `ypriv[T][M]`.
   * Partition the **blocks** array evenly among threads (block‑centric, not row‑centric) to balance work when rows have uneven nonzero counts.
   * Each thread runs:

     ```cpp
     kernel(
       A.blocks.data() + start_block,
       num_blocks_for_thread,
       A.values.data(),
       x,
       ypriv[thread_id]
     );
     ```
   * **Reduction**: sum `ypriv[t]` into the global `y` vector.

The `kernel` function pointer is retrieved via `get_spmv_kernel(A)`.

---

## Kernel Dispatch & JIT Compilation

**`get_spmv_kernel`** (`src/sparse_dispatch.cpp`):

1. **Cache key**:
   `"{rows}|{cols}|{#blocks}|{#values}|{isa_flag}"`
   Ensures a unique kernel per matrix shape and sparsity pattern.

2. **Code generation**:
   Calls `generate_spmv_cpp(A, "spmv_kernel", avx512_supported)` to emit C++ source as a string.

3. **Compile / Load**:
   `get_or_build_kernel(key, source, "spmv_kernel")` looks for `~/.cache/sparseops/<key>.so`; if missing, writes `<key>.cpp`, compiles with:

   ```
   ${CXX:-g++} -std=c++17 -shared -fPIC ${CXXFLAGS:-"-O3 -march=native"}
     [-DPROFILE_MASKS] -I<repo>/include
     <key>.cpp -o <key>.build.so
   mv <key>.build.so <key>.so
   ```

   and finally `dlopen` + `dlsym` to retrieve the function pointer.

---

## Micro‐Kernel Generation (`spmv_template.cpp`)

`generate_spmv_cpp` emits a specialized loop that:

1. **Includes** `<immintrin.h>` and the `BCOO16Block` definition.
2. **Optional profiling** (if `PROFILE_MASKS` is set) to record a histogram of mask frequencies.
3. **Helper routines** for different sparsity patterns:

   * `dot1` for a single nonzero (scalar multiply-add).
   * `dot2` for exactly two nonzeros (pairwise vector ops).
   * `dot_full` when all 16 lanes are live (bulk AVX‑512 dot product).
   * `dot_gather` for intermediate cases (uses gathers + vectorized FMA).
4. **Main body**:
   A tight `for`‑loop over blocks:

   ```cpp
   for (size_t i = 0; i < num_blocks; ++i) {
     const auto &b = blocks[i];
     const float* values = base_values + b.val_off;
     float acc;
     switch (popcount(b.bitmask)) {
       case 1: acc = dot1(b,values,x); break;
       case 2: acc = dot2(b,values,x); break;
       case 3: case 4: acc = dot_gather(b,values,x); break;
       default:
         acc = (b.bitmask==0xFFFF) ? dot_full(b,values,x)
                                  : dot_gather(b,values,x);
     }
     #ifdef PROFILE_MASKS
       hotHits[b.bitmask]++;
     #endif
     y[b.row_id] += acc;
   }
   ```

   By specializing based on the mask & popcount, the compiler can emit branchless AVX‑512 code for common cases, maximizing throughput.

---

## Tests & Benchmarks

* **`tests/full_matmul.py`** showcases usage of:

  * `run_matvec` (dense)
  * `run_sparse_matvec` (sparse)
  * Comparisons against PyTorch, NumPy, SciPy, TensorFlow, OpenVINO, and Torch‑Sparse.
  * Block‑sparsity patterns and average runtime measurements.

* **Unit tests** (`tests/test_matmul.py`, `tests/test_bcoo16.py`, etc.) verify correctness of encoding/decoding and numerical equivalence to reference implementations.

---

## Key Takeaways

* **Block‑compression (BCOO‑16)** trades a little storage for highly regular SIMD‑friendly tiles.
* **Per‑matrix JIT micro‑kernels** ensure that sparsity structure is exploited with minimal overhead.
* **Work partitioning by blocks** (vs rows) achieves better load balancing in multi‑threading.
* **Modular design** cleanly separates Python binding, sparse format encoding, dispatch logic, JIT cache, and kernel generation.

This combination delivers both **performance** (through AVX‑512 specialization and threading) and **flexibility** (automatic kernel generation for arbitrary sparsity patterns) under a simple Python API.

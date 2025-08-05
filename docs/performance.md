# Performance Benchmarks

## Kernel Benchmark

This section presents the runtime performance of the custom kernel compared to **Intel MKL Sparse GEMM** and **OpenBLAS Dense GEMM** under various matrix configurations.

Benchmarks were run on:
- **CPU**: Intel Xeon Gold 6338
- **Compiler**: GCC 12.2.0 with -O3 -march=native -funroll-loops
- **MKL Version**: 2025.2
- **OpenBLAS Version**: 0.3.21+ds4
- **Custom Kernel**: Built with AVX-512

---

### Hyperparameter Definitions

- `M`: Number of rows in the first matrix
- `N`: Number of columns in the first matrix (and rows in the second matrix)
- `C`: Number of columns in the second matrix
- `SPARSITY`: Fraction of zeros in the input matrix (0 = dense, 0.9 = 90% zeros)

## Results

### Single threaded

The table below shows runtimes (in miliseconds) for various matrix multiplication configurations using three different backends. 

> ✅ Denotes the best performance for each test configuration.


| M    | N    | C    | Sparsity | - | Ellpack Matmul | MKL Sparse | OpenBLAS Dense |
|------|------|------|----------|-|----------------|------------|----------------|
| 512  | 512  | 64   | 0.90     | | **0.122** ✅        | 0.234      | 0.406          |
| 512  | 512  | 64   | 0.80     | | **0.245**  ✅        | 0.343      | 0.402          |
| 512  | 512  | 64   | 0.70     | | 0.442          | 0.538      | **0.406**      ✅     |
| 512  | 512  | 64   | 0.50     | | 0.761          | 1.12       | **0.401**     ✅      |
|                |
| 1024 | 1024 | 64   | 0.90     | | **0.539**    ✅       | 0.697      | 1.61           |
| 1024 | 1024 | 64   | 0.80     | | **1.27**    ✅       | 1.41       | 1.61           |
| 1024 | 1024 | 64   | 0.70     | | 1.91           | 2.14       |**1.62**      ✅      |
| 1024 | 1024 | 64   | 0.50     | | 3.02           | 3.82       | **1.62**     ✅       |
|                   |
| 2048 | 2048 | 64   | 0.90     | | **2.45**    ✅        | 3.22       | 6.72           |
| 2048 | 2048 | 64   | 0.80     | | **4.97**    ✅        | 6.12       | 6.75           |
| 2048 | 2048 | 64   | 0.70     | | 7.49           | 9.73       | **6.70**      ✅      |
| 2048 | 2048 | 64   | 0.50     | | 12.8           | 16.1       | **6.75**     ✅       |

### Multi threaded (8 Threads)



| M    | N    | C    | Sparsity | - | Ellpack Matmul | MKL Sparse | OpenBLAS Dense |
|------|------|------|----------|-|----------------|------------|----------------|
| 512  | 512  | 64   | 0.90     | | **0.0229** ✅       | 0.0455     | 0.329          |
| 512  | 512  | 64   | 0.80     | | **0.0336** ✅         | 0.0732     | 0.332          |
| 512  | 512  | 64   | 0.70     | | **0.0611** ✅         | 0.0848     | 0.339          |
| 512  | 512  | 64   | 0.50     | | **0.0899** ✅         | 0.126      | 0.341          |
|                 |
| 1024 | 1024 | 64   | 0.90     | | **0.0580** ✅         | 0.136      | 1.09           |
| 1024 | 1024 | 64   | 0.80     | | **0.162** ✅          | 0.200      | 1.13           |
| 1024 | 1024 | 64   | 0.70     | | **0.247** ✅          | 0.452      | 1.10           |
| 1024 | 1024 | 64   | 0.50     | | **0.341** ✅          | 0.520      | 1.07           |
|               |
| 2048 | 2048 | 64   | 0.90     | | **0.251** ✅          | 0.502      | 3.70           |
| 2048 | 2048 | 64   | 0.80     | | **0.637** ✅          | 0.799      | 3.82           |
| 2048 | 2048 | 64   | 0.70     | | **0.972** ✅          | 1.19       | 3.56           |
| 2048 | 2048 | 64   | 0.50     | | **1.623** ✅          | 2.15       | 3.80           
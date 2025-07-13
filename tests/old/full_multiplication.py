#!/usr/bin/env python3
import os
import time
import ctypes
import numpy as np
import scipy.sparse as sp
import torch
from scipy.linalg.blas import sgemv
import pytaco as pt
from pytaco import compressed, dense
from python.cpp_backend import run_matvec
from python.utils import to_csr

from tests.old.mkl_ops import mkl_dense_run

from sparse_dot_mkl import dot_product_mkl, mkl_set_num_threads

# ---------------------------------------------
# CONFIG
# ---------------------------------------------
os.environ["OMP_NUM_THREADS"] = "8"
os.environ["MKL_NUM_THREADS"] = "8"
os.environ["OPENBLAS_NUM_THREADS"] = "8"
torch.set_num_threads(8)

INPUT_DIM = 2096
OUTPUT_DIM = 2096
SPARSITY = 0.96
N_RUNS = 100
SEED = 42

# ---------------------------------------------
# DATA CREATION
# ---------------------------------------------
np.random.seed(SEED)
torch.manual_seed(SEED)

def apply_nm_sparsity(weight, N=2, M=4):
    weight = weight.copy()
    rows, cols = weight.shape
    assert cols % M == 0, "Number of columns must be divisible by M"

    for row in range(rows):
        for block_start in range(0, cols, M):
            block = weight[row, block_start:block_start+M]
            # Keep N largest in magnitude
            top_indices = np.argsort(np.abs(block))[-N:]
            mask = np.zeros(M, dtype=bool)
            mask[top_indices] = True
            block *= mask
            weight[row, block_start:block_start+M] = block
    return weight

def apply_block_sparsity(weight, block_rows=4, block_cols=4, keep_prob=0.001):
    weight = weight.copy()
    for i in range(0, weight.shape[0], block_rows):
        for j in range(0, weight.shape[1], block_cols):
            if np.random.rand() > keep_prob:
                weight[i:i+block_rows, j:j+block_cols] = 0
    return weight

weight = np.random.randn(OUTPUT_DIM, INPUT_DIM).astype(np.float32)
weight *= (np.random.rand(*weight.shape) > SPARSITY)
# weight = apply_nm_sparsity(weight, N=2, M=4)
# weight = apply_block_sparsity(weight)
A_f = np.asfortranarray(weight)
bias = np.random.randn(OUTPUT_DIM).astype(np.float32)
input_vec = np.random.randn(INPUT_DIM).astype(np.float32)

weight_t = torch.tensor(weight)
input_t = torch.tensor(input_vec)
bias_t = torch.tensor(bias)

weight_np = weight_t.numpy()
bias_np   = bias_t.numpy()
input_np  = input_t.numpy()

csr_mat = sp.csr_matrix(weight)
_, _, _ = to_csr(torch.tensor(weight))

# ---------------------------------------------
# TIMING UTILITY
# ---------------------------------------------
def timed_avg(fn, n=N_RUNS):
    total = 0.0
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        t1 = time.perf_counter()
        total += t1 - t0
    return total / n

# ---------------------------------------------
# BACKEND RUNNERS
# ---------------------------------------------
def torch_run():
    return torch.matmul(weight_t, input_t) + bias_t

def numpy_run():
    return weight @ input_vec + bias

def scipy_run():
    return csr_mat.dot(input_vec) + bias

def custom_run():
    return run_matvec(weight_np, bias_np, input_np)

def taco_setup():
    csr = pt.format([dense, compressed])
    dv = pt.format([dense])

    A = pt.tensor(weight.shape, csr)
    rows, cols = csr_mat.nonzero()
    vals = csr_mat.data
    for r, c, v in zip(rows, cols, vals):
        A.insert([r, c], float(v))

    x = pt.tensor([INPUT_DIM], dv)
    for i, val in enumerate(input_vec):
        x.insert([i], float(val))

    z = pt.tensor([OUTPUT_DIM], dv)
    for i, val in enumerate(bias):
        z.insert([i], float(val))

    y = pt.tensor([OUTPUT_DIM], dv)
    i, j = pt.get_index_vars(2)
    y[i] = A[i, j] * x[j] + z[i]
    return y

y_taco = taco_setup()
def taco_run():
    y_taco.evaluate()
    return y_taco

def blas_run():
    return sgemv(1.0, A_f, input_vec, beta=1.0, y=bias)

# ---------------------------------------------
# MKL Sparse via ctypes
# ---------------------------------------------

def test_mkl_dot(input_vec, csr_mat, n_runs=1000):
    """
    input_vec: np.ndarray of shape [input_dim]
    csr_mat: scipy.sparse.csr_matrix of shape [output_dim, input_dim]
    Returns: output vector, avg runtime
    """
    mkl_set_num_threads(8)  # Match with OMP_NUM_THREADS if needed

    # Warm-up
    dot_product_mkl(csr_mat, input_vec, cast=True, dense=True)

    # Benchmark
    start = time.perf_counter()
    for _ in range(n_runs):
        out = dot_product_mkl(csr_mat, input_vec, cast=True, dense=True)
    end = time.perf_counter()

    avg_time_ms = (end - start) * 1000 / n_runs
    return out, avg_time_ms

# ---------------------------------------------
# CORRECTNESS CHECK
# ---------------------------------------------
print("=== Verifying correctness ===")
out_torch = torch_run().numpy()
print("Torch vs NumPy       :", np.allclose(out_torch, numpy_run(), atol=1e-4))
print("Torch vs SciPy       :", np.allclose(out_torch, scipy_run(), atol=1e-4))
print("Torch vs Custom      :", np.allclose(out_torch, custom_run(), atol=1e-4))
print("Torch vs TACO        :", np.allclose(out_torch, taco_run().to_array(), atol=1e-4))
print("Torch vs BLAS        :", np.allclose(out_torch, blas_run(), atol=1e-4))
out_torch_without_bias = out_torch - bias_t.numpy()
y = test_mkl_dot(input_vec, csr_mat, n_runs=1)
print("Torch vs MKL Sparse  :", np.allclose(out_torch_without_bias, y[0], atol=1e-4))
y = mkl_dense_run(weight_np, input_np, bias_np, n_runs=1)
print("Torch vs MKL Dense   :", np.allclose(out_torch, y[0], atol=1e-4))

# ---------------------------------------------
# BENCHMARK RESULTS
# ---------------------------------------------
print(f"\n=== Average runtime over {N_RUNS:,} runs ===")
print(f"[PyTorch]      {timed_avg(torch_run)*1000:.3f} ms")
print(f"[NumPy]        {timed_avg(numpy_run)*1000:.3f} ms")
print(f"[SciPy Sparse] {timed_avg(scipy_run)*1000:.3f} ms")
print(f"[Custom]       {timed_avg(custom_run)*1000:.3f} ms")
print(f"[TACO]         {timed_avg(taco_run)*1000:.3f} ms")
print(f"[BLAS SGEMV]   {timed_avg(blas_run)*1000:.3f} ms")

_, mkl_spmv = test_mkl_dot(input_vec, csr_mat, n_runs=1000)
print(f"[MKL Sparse]   {mkl_spmv:.3f} ms")
_, mkl_dense = mkl_dense_run(weight_np, input_np, bias_np, n_runs=1000)
print(f"[MKL Dense]    {mkl_dense:.3f} ms")
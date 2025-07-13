#!/usr/bin/env python3
# test_matmul_avg.py

import os
import time
import numpy as np
import scipy.sparse as sp
import torch
from scipy.linalg.blas import sgemv

import pytaco as pt
from pytaco import compressed, dense

from python.cpp_backend import run_matvec
from python.utils import to_csr


# os.environ["OMP_NUM_THREADS"] = "8"
# os.environ["MKL_NUM_THREADS"] = "8"
# os.environ["OPENBLAS_NUM_THREADS"] = "8"
# os.environ["NUMEXPR_NUM_THREADS"] = "8"

# ----------------------------------------------------------------------
# Experiment parameters
# ----------------------------------------------------------------------
# INPUT_DIM  = 512
# OUTPUT_DIM = 256
INPUT_DIM = 1096
OUTPUT_DIM = 1096
SPARSITY   = 0.2
N_RUNS     = 1_000
SEED       = 42

# ----------------------------------------------------------------------
# Data generation
# ----------------------------------------------------------------------
np.random.seed(SEED)
torch.manual_seed(SEED)

weight = np.random.randn(OUTPUT_DIM, INPUT_DIM).astype(np.float32)
mask   = np.random.rand(*weight.shape) > SPARSITY
weight *= mask

bias       = np.random.randn(OUTPUT_DIM).astype(np.float32)
input_vec  = np.random.randn(INPUT_DIM).astype(np.float32)

weight_t   = torch.tensor(weight)
bias_t     = torch.tensor(bias)
input_t    = torch.tensor(input_vec)

weight_np = weight_t.numpy()
bias_np   = bias_t.numpy()
input_np  = input_t.numpy()

csr_mat = sp.csr_matrix(weight)
_, _, _ = to_csr(torch.tensor(weight))

assert np.isfinite(input_vec).all()
assert np.isfinite(bias_np).all()
assert np.isfinite(csr_mat.data).all()

# ----------------------------------------------------------------------
# Utility: average wall-time over n runs
# ----------------------------------------------------------------------
def timed_avg(fn, n=N_RUNS):
    total = 0.0
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        t1 = time.perf_counter()
        total += (t1 - t0)
    return total / n

# ----------------------------------------------------------------------
# Define callables we want to benchmark
# ----------------------------------------------------------------------
def torch_run():
    return torch.matmul(weight_t, input_t) + bias_t

def numpy_run():
    return weight @ input_vec + bias

def scipy_run():
    return csr_mat.dot(input_vec) + bias

def custom_run():
    return run_matvec(weight_np, bias_np, input_np)

def blas_run():
    # Ensure A is Fortran-contiguous (sgemv assumes column-major layout)
    A_f = np.asfortranarray(weight_np)  # shape (M, N)
    
    # sgemv: y = alpha * A @ x + beta * y
    # Inputs:
    #   alpha = 1.0
    #   beta  = 1.0
    #   A     = A_f (2D float32)
    #   x     = input_vec (1D float32)
    #   y     = bias (1D float32)
    #
    # Returns: output vector (float32)
    return sgemv(1.0, A_f, input_vec, beta=1.0, y=bias)

# === TACO benchmark ===
def taco_setup():

    # Define formats: CSR for sparse matrix, dense vectors
    csr = pt.format([dense, compressed])
    dv  = pt.format([dense])

    # Create sparse matrix tensor and insert CSR data
    A = pt.tensor(weight_np.shape, csr)
    rows, cols = csr_mat.nonzero()
    vals = csr_mat.data
    for r, c, v in zip(rows, cols, vals):
        A.insert([r, c], v)

    # Dense input vector x and bias vector z
    x = pt.from_array(input_vec)

    z = pt.from_array(bias_np)

    # Output vector y
    y = pt.tensor([OUTPUT_DIM], dv)

    # Declare index variables and define SpMV computation: y[i] = A[i,j]*x[j] + z[i]
    i, j = pt.get_index_vars(2)
    y[i] = A[i, j] * x[j] + z[i]

    return A, x, z, y

A_taco, x_taco, z_taco, y_taco = taco_setup()

def taco_run():
    y_taco.evaluate()
    return y_taco.to_array()

# ----------------------------------------------------------------------
# Correctness check (one-shot)
# ----------------------------------------------------------------------
print("=== Verifying correctness ===")
out_torch = torch_run().numpy()
print("Torch vs NumPy :", np.allclose(out_torch, numpy_run(), atol=1e-4))
print("Torch vs SciPy :", np.allclose(out_torch, scipy_run(), atol=1e-4))
print("Torch vs Custom:", np.allclose(out_torch, custom_run(), atol=1e-4))
print("Torch vs TACO  :", np.allclose(out_torch, taco_run(), atol=1e-4))
print("Torch vs BLAS  :", np.allclose(out_torch, blas_run(), atol=1e-4))
print()

# ----------------------------------------------------------------------
# Benchmark
# ----------------------------------------------------------------------
print(f"=== Average runtime over {N_RUNS:,} runs ===")
print(f"[PyTorch]      {timed_avg(torch_run)*1000:.3f} ms")
print(f"[NumPy]        {timed_avg(numpy_run)*1000:.3f} ms")
print(f"[SciPy Sparse] {timed_avg(scipy_run)*1000:.3f} ms")
print(f"[Custom]       {timed_avg(custom_run)*1000:.3f} ms")
print(f"[TACO]         {timed_avg(taco_run)*1000:.3f} ms")
print(f"[BLAS]         {timed_avg(blas_run)*1000:.3f} ms")
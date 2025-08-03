#!/usr/bin/env python3
# test_matmul_avg.py
#
# Measure average runtime of several matrix-vector multiplications
# over N_RUNS iterations.
import os
import time
import numpy as np
import scipy.sparse as sp
import torch

from python.cpp_backend import (
    encode,
    matmul,
)

# ----------------------------------------------------------------------
# Experiment parameters
# ----------------------------------------------------------------------
# Resnet toy example dimensions:
INPUT_DIM  = 147
OUTPUT_DIM = 64
C = 12544

# Benchmark dimensions:
INPUT_DIM  = 1000
OUTPUT_DIM = 1000
C = 64
SPARSITY   = 0.90
N_RUNS     = 100     # <-- number of repetitions
SEED       = 42
# ----------------------------------------------------------------------

# ----------------------------------------------------------------------
# Data generation (identical to original script)
# ----------------------------------------------------------------------
np.random.seed(SEED)
torch.manual_seed(SEED)

weight = np.random.randn(OUTPUT_DIM, INPUT_DIM).astype(np.float32)
mask   = np.random.rand(*weight.shape) > SPARSITY
weight *= mask

bias       = np.random.randn(OUTPUT_DIM).astype(np.float32)
input_mat  = np.random.randn(INPUT_DIM, C).astype(np.float32)

weight_t   = torch.tensor(weight)
bias_t     = torch.tensor(np.expand_dims(bias, axis=1))
input_t    = torch.tensor(input_mat)

weight_np = weight_t.detach().cpu().numpy().astype(np.float32)
bias_np   = bias_t.detach().cpu().numpy().astype(np.float32)
input_np  = input_t.detach().cpu().numpy().astype(np.float32)

csr_mat    = sp.csr_matrix(weight)            # build once

ellpack = encode(weight_np)


# num_threads = int(os.environ.get("OMP_NUM_THREADS", "1"))
# torch.set_num_threads(num_threads)
num_threads = None


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
    return total / n   # seconds

# ----------------------------------------------------------------------
# Define callables we want to benchmark
# ----------------------------------------------------------------------
def torch_run():
    return torch.matmul(weight_t, input_t) + bias_t

def numpy_run():
    return weight @ input_mat + bias_np

def scipy_run():
    return csr_mat.dot(input_mat) + bias_np

def bilinear_diagonal_run():
    return matmul(ellpack, input_mat, bias_np)

# ----------------------------------------------------------------------
# Correctness check (one-shot)
# ----------------------------------------------------------------------
print("=== Verifying correctness ===")
out_torch  = torch_run().numpy()
print("Torch vs NumPy :", np.allclose(out_torch, numpy_run(), atol=1e-4))
print("Torch vs SciPy :", np.allclose(out_torch, scipy_run(), atol=1e-4))
print("Torch vs Bilinear Diagonal:", np.allclose(out_torch, bilinear_diagonal_run(), atol=1e-4))
print()

# ----------------------------------------------------------------------
# Benchmark
# ----------------------------------------------------------------------
print(f"=== Average runtime over {N_RUNS:,} runs ===")
print(f"=== Sparsity: {SPARSITY:.2f}, Input dim: {INPUT_DIM:,}, Output dim: {OUTPUT_DIM:,}, Columns: {C} ===")
print(f"=== Threads: {num_threads} ===")
print(f"[PyTorch]      {timed_avg(torch_run)*1000:.3f} ms")
print(f"[NumPy]        {timed_avg(numpy_run)*1000:.3f} ms")
print(f"[Bilinear Diagonal] {timed_avg(bilinear_diagonal_run)*1000:.3f} ms")

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
    matvec,
)

# ----------------------------------------------------------------------
# Experiment parameters
# ----------------------------------------------------------------------
INPUT_DIM  = 2000
OUTPUT_DIM = 2000
SPARSITY   = 0.9
N_RUNS     = 100     # <-- number of repetitions
SEED       = 42
# ----------------------------------------------------------------------

# ----------------------------------------------------------------------
# Data generation (identical to original script)
# ----------------------------------------------------------------------
np.random.seed(SEED)
torch.manual_seed(SEED)

weight = np.random.randn(OUTPUT_DIM, INPUT_DIM).astype(np.float32)
# print(f"Minimum weight value: {np.min(weight)}, maximum weight value: {np.max(weight)}")
mask   = np.random.rand(*weight.shape) > SPARSITY
weight *= mask
# weight[weight < SPARSITY] = 0.0
# Ensure all values are non-negative
# weight[weight < 0] *= -1.0

bias       = np.random.randn(OUTPUT_DIM).astype(np.float32)
input_vec  = np.random.randn(INPUT_DIM).astype(np.float32)

weight_t   = torch.tensor(weight)
bias_t     = torch.tensor(bias)
input_t    = torch.tensor(input_vec)

weight_np = weight_t.detach().cpu().numpy().astype(np.float32)
bias_np   = bias_t.detach().cpu().numpy().astype(np.float32)
input_np  = input_t.detach().cpu().numpy().astype(np.float32)

csr_mat    = sp.csr_matrix(weight)            # build once

ellpack = encode(weight_np)


num_threads = int(os.environ.get("OMP_NUM_THREADS", "1"))
torch.set_num_threads(num_threads)
# num_threads = None  # Disable threading for this test

print(f"MAX NNZ in row: {ellpack.r}")
# Compute the minimum number of non-zero entries (NNZ) in any row, using weight_np
print(f"MIN NNZ in row: {np.min(np.sum(weight_np != 0, axis=1))}")


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
    return weight @ input_vec + bias

def scipy_run():
    return csr_mat.dot(input_vec) + bias

def bilinear_diagonal_run():
    return matvec(ellpack, input_np, bias_np)
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
print(f"=== Sparsity: {SPARSITY:.2f}, Input dim: {INPUT_DIM:,}, Output dim: {OUTPUT_DIM:,} ===")
print(f"=== Threads: {num_threads} ===")
print(f"[PyTorch]      {timed_avg(torch_run)*1000:.3f} ms")
print(f"[NumPy]        {timed_avg(numpy_run)*1000:.3f} ms")
print(f"[SciPy Sparse] {timed_avg(scipy_run)*1000:.3f} ms")
print(f"[Bilinear Diagonal] {timed_avg(bilinear_diagonal_run)*1000:.3f} ms")

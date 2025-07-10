#!/usr/bin/env python3
# test_matmul_avg.py
#
# Measure average runtime of several matrix-vector multiplications
# over N_RUNS iterations.

import time
import numpy as np
import scipy.sparse as sp
import torch

from python.cpp_backend import run_sparse_matvec
from python.utils import to_csr

# ----------------------------------------------------------------------
# Experiment parameters
# ----------------------------------------------------------------------
INPUT_DIM  = 512
OUTPUT_DIM = 256
SPARSITY   = 0.9
N_RUNS     = 1_000      # <-- number of repetitions
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
input_vec  = np.random.randn(INPUT_DIM).astype(np.float32)

weight_t   = torch.tensor(weight)
bias_t     = torch.tensor(bias)
input_t    = torch.tensor(input_vec)

csr_mat    = sp.csr_matrix(weight)            # build once
_, _, _    = to_csr(torch.tensor(weight))     # pre-compute CSR pieces if needed

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

def custom_run():
    return run_sparse_matvec(weight_t, bias_t, input_t)

# ----------------------------------------------------------------------
# Correctness check (one-shot)
# ----------------------------------------------------------------------
print("=== Verifying correctness ===")
out_torch  = torch_run().numpy()
print("Torch vs NumPy :", np.allclose(out_torch, numpy_run(), atol=1e-4))
print("Torch vs SciPy :", np.allclose(out_torch, scipy_run(), atol=1e-4))
print("Torch vs Custom:", np.allclose(out_torch, custom_run().numpy(), atol=1e-4))
print()

# ----------------------------------------------------------------------
# Benchmark
# ----------------------------------------------------------------------
print(f"=== Average runtime over {N_RUNS:,} runs ===")
print(f"[PyTorch]      {timed_avg(torch_run)*1000:.3f} ms")
print(f"[NumPy]        {timed_avg(numpy_run)*1000:.3f} ms")
print(f"[SciPy Sparse] {timed_avg(scipy_run)*1000:.3f} ms")
print(f"[Custom]       {timed_avg(custom_run)*1000:.3f} ms")

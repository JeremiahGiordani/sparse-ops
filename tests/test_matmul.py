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
    run_matvec,
    decode_bcoo16,
    convert_to_bcoo16,
    run_sparse_matvec,
    run_bilinear_diagonal_matvec,
    encode_to_quasi_dense,
    transform_input
)
from python.utils import to_csr

# ----------------------------------------------------------------------
# Experiment parameters
# ----------------------------------------------------------------------
INPUT_DIM  = 2000
OUTPUT_DIM = 2000
SPARSITY   = 0.95
N_RUNS     = 100      # <-- number of repetitions
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
_, _, _    = to_csr(torch.tensor(weight))     # pre-compute CSR pieces if needed

bcoo_16 = convert_to_bcoo16(weight)

quasi_dense = encode_to_quasi_dense(weight_np)
transformed_input = transform_input(quasi_dense, input_np)

num_threads = int(os.environ.get("OMP_NUM_THREADS", "1"))


# Test converting back and forth between BCOO-16 and dense
def encode_decode_and_check(mat, name):
    bcoo = convert_to_bcoo16(mat)
    decoded = decode_bcoo16(bcoo)

    print(f"Decoded shape: {decoded.shape}, original shape: {mat.shape}")
    
    if not np.allclose(decoded, mat, atol=1e-4):
        raise AssertionError(f"Decoded matrix does not match original for test: {name}")
    
    print(f"✅ Test passed: {name}")

# Run tests
encode_decode_and_check(weight_np, "Weight matrix")
encode_decode_and_check(bias_np.reshape(-1, 1), "Bias vector")
encode_decode_and_check(input_np.reshape(-1, 1), "Input vector")

# Check correctness of sparse matrix-vector multiplication
def sparse_matvec_correctness_check():
    bcoo = convert_to_bcoo16(weight_np)
    output = run_sparse_matvec(bcoo, bias_np, input_np, threads=num_threads)
    
    expected_output = weight @ input_vec + bias
    print(f"Expected output shape: {expected_output.shape}, actual output shape: {output.shape}")
    print(f"Expected output: {expected_output}")
    print(f"Actual output: {output}")
    if not np.allclose(output, expected_output, atol=1e-4):
        raise AssertionError("Sparse matrix-vector multiplication did not match expected output.")
    
    print("✅ Sparse matrix-vector multiplication correctness check passed.")

# sparse_matvec_correctness_check()
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
    return run_matvec(weight_np, bias_np, input_np)

def custom_sparse_run():
    return run_sparse_matvec(bcoo_16, bias_np, input_np, threads=num_threads)

def bilinear_diagonal_run():
    return run_bilinear_diagonal_matvec(quasi_dense, transformed_input, bias_np, threads=num_threads)

# ----------------------------------------------------------------------
# Correctness check (one-shot)
# ----------------------------------------------------------------------
print("=== Verifying correctness ===")
out_torch  = torch_run().numpy()
out_expected = weight @ input_vec + bias
out_custom_sparse = custom_sparse_run()
# print("Expected vs Custom Sparse:", np.allclose(out_torch, custom_sparse_run(), atol=1e-4))
# print("number of correct elements:", np.sum(np.isclose(out_expected, out_custom_sparse, atol=1e-4)))
# print("Location of incorrect elements:", np.where(~np.isclose(out_expected, out_custom_sparse, atol=1e-4)))
# assert False
print("Torch vs NumPy :", np.allclose(out_torch, numpy_run(), atol=1e-4))
print("Torch vs SciPy :", np.allclose(out_torch, scipy_run(), atol=1e-4))
print("Torch vs Custom:", np.allclose(out_torch, custom_run(), atol=1e-4))
print("Torch vs Custom Sparse:", np.allclose(out_torch, custom_sparse_run(), atol=1e-4))
print("Torch vs Bilinear Diagonal:", np.allclose(out_torch, bilinear_diagonal_run(), atol=1e-4))
print()

# ----------------------------------------------------------------------
# Benchmark
# ----------------------------------------------------------------------
print(f"=== Average runtime over {N_RUNS:,} runs ===")
print(f"[PyTorch]      {timed_avg(torch_run)*1000:.3f} ms")
print(f"[NumPy]        {timed_avg(numpy_run)*1000:.3f} ms")
print(f"[SciPy Sparse] {timed_avg(scipy_run)*1000:.3f} ms")
print(f"[Custom]       {timed_avg(custom_run)*1000:.3f} ms")
print(f"[Custom Sparse]{timed_avg(custom_sparse_run)*1000:.3f} ms")
print(f"[Bilinear Diagonal] {timed_avg(bilinear_diagonal_run)*1000:.3f} ms")

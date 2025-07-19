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
import collections

from python.cpp_backend import run_matvec, decode_bcoo16, convert_to_bcoo16, run_sparse_matvec
from python.utils import to_csr

# ----------------------------------------------------------------------
# Experiment parameters
# ----------------------------------------------------------------------
INPUT_DIM  = 2000
OUTPUT_DIM = 2000
SPARSITY   = 0.90
N_RUNS     = 1000
SEED       = 42

np.random.seed(SEED)
torch.manual_seed(SEED)

weight = np.random.randn(OUTPUT_DIM, INPUT_DIM).astype(np.float32)
mask   = np.random.rand(*weight.shape) > SPARSITY
weight *= mask

# print(weight[-1, -32:-16])

bias       = np.random.randn(OUTPUT_DIM).astype(np.float32)
# bias = np.zeros_like(bias)  # Ensure bias is zero for consistency
input_vec  = np.random.randn(INPUT_DIM).astype(np.float32)

bcoo_16 = convert_to_bcoo16(weight)
decoded = decode_bcoo16(bcoo_16)

popcounts = [bin(blk[2]).count('1') for blk in bcoo_16.blocks]
counts = collections.Counter(popcounts)
print("Bitmask popcount histogram:")
for pc, count in sorted(counts.items()):
    print(f"{pc}: {count}")

assert np.all(np.array([b[0] for b in bcoo_16.blocks]) < 2000), "Found out-of-bounds row_id"

# print([b[0] for b in bcoo_16.blocks[:10]])
# print([b[0] for b in bcoo_16.blocks[-10:]])

print("Any NaNs in values?:", np.isnan(bcoo_16.values).any())
print("Any extreme values?:", np.max(np.abs(bcoo_16.values)))

# Print final 10 blocks:
print("Last 10 blocks:", bcoo_16.blocks[-10:])
print("Total length values array:", len(bcoo_16.values))

np.set_printoptions(suppress=True, precision=4)
print("First 20 values:", bcoo_16.values[:20])

x_max_index = INPUT_DIM - 1
for blk in bcoo_16.blocks:
    for k in range(16):
        if blk[2] & (1 << k):
            index = blk[1] + k
            if index > x_max_index:
                print(f"❌ OOB x access: blk.first_col={blk[1]}, lane={k}, access={index}")

# print("First 10 blocks:", bcoo_16.blocks[:10])
# print("First 10 values:", bcoo_16.values[:10])
# print("First 10 decoded rows:", decoded[:10])
assert np.allclose(weight, decoded), "BCOO-16 encoding/decoding mismatch"
assert len(bcoo_16.values) == np.count_nonzero(weight), "BCOO-16 values count mismatch"
# print("Nonzeros:", np.count_nonzero(weight))
# print("BCOO block count:", len(bcoo_16.blocks)) 
num_threads = int(os.environ.get("OMP_NUM_THREADS", "1"))

out_expected = weight @ input_vec + bias
out_custom_sparse = run_sparse_matvec(bcoo_16, bias, input_vec, threads=num_threads)

print("Expected vs Custom Sparse:", np.allclose(out_expected, out_custom_sparse, atol=1e-4))
print("number of correct elements:", np.sum(np.isclose(out_expected, out_custom_sparse, atol=1e-4)))
print("Location of incorrect elements:", np.where(~np.isclose(out_expected, out_custom_sparse, atol=1e-4)))
print(f"out_expected: {out_expected}")
print(f"out_custom_sparse: {out_custom_sparse}")
diffs = np.abs(out_expected - out_custom_sparse)
bad_rows = np.where(diffs > 1e-4)[0]
print("First 10 bad rows:")
for i in bad_rows[:10]:
    print(f"Row {i}: expected={out_expected[i]}, got={out_custom_sparse[i]}, diff={diffs[i]}")
print(bias[:10])

assert np.allclose(out_expected, out_custom_sparse, atol=1e-4), "Mismatch in custom sparse output"
print("✅ Custom sparse matvec correctness test passed.")
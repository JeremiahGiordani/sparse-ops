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

bias       = np.random.randn(OUTPUT_DIM).astype(np.float32)
input_vec  = np.random.randn(INPUT_DIM).astype(np.float32)

bcoo_16 = convert_to_bcoo16(weight)

num_threads = int(os.environ.get("OMP_NUM_THREADS", "1"))

out_expected = weight @ input_vec + bias
out_custom_sparse = run_sparse_matvec(bcoo_16, bias, input_vec, threads=num_threads)


assert np.allclose(out_expected, out_custom_sparse, atol=1e-4), "Mismatch in custom sparse output"
print("✅ Custom sparse matvec correctness test passed.")
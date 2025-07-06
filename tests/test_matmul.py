# test_matmul.py

import torch
import numpy as np
import scipy.sparse
import time
from python.cpp_backend import run_sparse_matvec
from python.utils import to_csr

# Generate a random dense weight matrix with sparsity
input_dim = 5120
output_dim = 2560
sparsity = 0.1

np.random.seed(42)
torch.manual_seed(42)

# Create sparse matrix
weight = np.random.randn(output_dim, input_dim).astype(np.float32)
mask = np.random.rand(*weight.shape) > sparsity
weight *= mask

bias = np.random.randn(output_dim).astype(np.float32)
input_vec = np.random.randn(input_dim).astype(np.float32)

# Convert to PyTorch
weight_torch = torch.tensor(weight)
bias_torch = torch.tensor(bias)
input_torch = torch.tensor(input_vec)

print("=== Verifying correctness and measuring runtime ===")

# --- Torch Dense ---
t1 = time.perf_counter()
output_torch = torch.matmul(weight_torch, input_torch) + bias_torch
t2 = time.perf_counter()
print(f"[PyTorch] Time: {(t2 - t1)*1000:.3f} ms")

# --- NumPy Dense ---
t1 = time.perf_counter()
output_numpy = weight @ input_vec + bias
t2 = time.perf_counter()
print(f"[NumPy] Time: {(t2 - t1)*1000:.3f} ms")

# --- SciPy Sparse ---
csr = scipy.sparse.csr_matrix(weight)
t1 = time.perf_counter()
output_scipy = csr.dot(input_vec) + bias
t2 = time.perf_counter()
print(f"[SciPy Sparse] Time: {(t2 - t1)*1000:.3f} ms")

# --- Custom Kernel ---
values, indices, indptr = to_csr(torch.tensor(weight))
t1 = time.perf_counter()
output_custom = run_sparse_matvec(torch.tensor(weight), torch.tensor(bias), torch.tensor(input_vec))
t2 = time.perf_counter()
print(f"[Custom Kernel] Time: {(t2 - t1)*1000:.3f} ms")

# --- Check correctness ---
print("\nChecking correctness:")
print("Torch vs NumPy:", np.allclose(output_torch.numpy(), output_numpy, atol=1e-4))
print("Torch vs SciPy:", np.allclose(output_torch.numpy(), output_scipy, atol=1e-4))
print("Torch vs Custom:", np.allclose(output_torch.numpy(), output_custom.numpy(), atol=1e-4))

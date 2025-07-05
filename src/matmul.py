# matmul_ctypes_benchmark.py

import numpy as np
import ctypes
from numpy.ctypeslib import ndpointer
import time

# Load shared library
lib = ctypes.CDLL('./libmatmul.so')

# Define function signature
lib.matmul.argtypes = [
    ndpointer(ctypes.c_float, flags='C_CONTIGUOUS'),
    ndpointer(ctypes.c_float, flags='C_CONTIGUOUS'),
    ndpointer(ctypes.c_float, flags='C_CONTIGUOUS'),
    ctypes.c_int, ctypes.c_int, ctypes.c_int
]
lib.matmul.restype = None

# Matrix dimensions
m, k, n = 512, 512, 512

# Initialize input matrices
A = np.random.rand(m, k).astype(np.float32)
B = np.random.rand(k, n).astype(np.float32)
C_cpp = np.zeros((m, n), dtype=np.float32)

# Time the C++ call
start_cpp = time.perf_counter()
lib.matmul(A, B, C_cpp, m, k, n)
end_cpp = time.perf_counter()
cpp_time_ms = (end_cpp - start_cpp) * 1000

# Time the NumPy baseline
start_np = time.perf_counter()
C_np = A @ B
end_np = time.perf_counter()
np_time_ms = (end_np - start_np) * 1000


# Print results
print(f"[C++] matmul({m}x{k}) · ({k}x{n}) took {cpp_time_ms:.3f} ms")
print(f"[NumPy] matmul took {np_time_ms:.3f} ms")

# Validate correctness
if np.allclose(C_cpp, C_np, atol=1e-5):
    print("✅ Results match!")
else:
    print("❌ Results differ!")

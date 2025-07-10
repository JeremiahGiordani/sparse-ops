import ctypes
import numpy as np
import platform
from pathlib import Path

from typing import TYPE_CHECKING
import torch
import sys
import os

# Try to use the new backend first
try:
    sys.path.insert(0, str(Path(__file__).parent.parent / "build"))
    import sparseops_backend
    NEW_BACKEND_AVAILABLE = True
except ImportError:
    NEW_BACKEND_AVAILABLE = False

if not NEW_BACKEND_AVAILABLE:
    # Fallback to old backend
    # Determine platform-specific shared library name
    libname = "libultrasparse.dylib" if platform.system() == "Darwin" else "libultrasparse.so"
    lib_path = Path(__file__).parent.parent / "build" / libname

    # Load the shared library
    lib = ctypes.CDLL(str(lib_path))

    # Define argument and return types
    lib.sparse_matvec.argtypes = [
        np.ctypeslib.ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),  # input
        ctypes.c_int,
        np.ctypeslib.ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),  # values
        np.ctypeslib.ndpointer(ctypes.c_int32, flags="C_CONTIGUOUS"),  # indices
        np.ctypeslib.ndpointer(ctypes.c_int32, flags="C_CONTIGUOUS"),  # indptr
        ctypes.c_int,
        np.ctypeslib.ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),  # bias
        np.ctypeslib.ndpointer(ctypes.c_float, flags="C_CONTIGUOUS")   # output
    ]
    lib.sparse_matvec.restype = None

def run_sparse_matvec(weight: torch.Tensor, bias: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
    import torch
    
    if NEW_BACKEND_AVAILABLE:
        # Use new optimized backend
        import scipy.sparse as sp
        
        # Convert to CSR format
        weight_csr = sp.csr_matrix(weight.detach().cpu().numpy())
        
        # Prepare matrix
        A_prepared = sparseops_backend.prepare_csr(
            weight_csr.indptr, weight_csr.indices, weight_csr.data,
            weight_csr.shape[0], weight_csr.shape[1])
        
        # Prepare input
        input_np = input_tensor.detach().cpu().numpy().astype(np.float32)
        if input_np.ndim == 1:
            input_np = input_np.reshape(-1, 1)
        else:
            input_np = input_np.T  # Transpose for correct layout
        
        # Allocate output
        output_np = np.zeros((weight_csr.shape[0], input_np.shape[1]), dtype=np.float32)
        
        # Perform multiplication
        sparseops_backend.sgemm(A_prepared, input_np, output_np)
        
        # Add bias and return
        result = output_np.flatten() + bias.detach().cpu().numpy()
        return torch.from_numpy(result)
    
    else:
        # Fallback to old backend
        from .utils import to_csr

        values, indices, indptr = to_csr(weight)
        input_np = input_tensor.detach().numpy().astype(np.float32)
        bias_np = bias.detach().numpy().astype(np.float32)
        output_np = np.zeros(bias_np.shape, dtype=np.float32)

        lib.sparse_matvec(
            input_np,
            input_np.shape[0],
            values.astype(np.float32),
            indices.astype(np.int32),
            indptr.astype(np.int32),
            bias_np.shape[0],
            bias_np,
            output_np
        )

        return torch.from_numpy(output_np)

import ctypes
import numpy as np
import platform
from pathlib import Path

from typing import TYPE_CHECKING
import torch

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
    import torch  # Local import to avoid circular dependency
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

import importlib
import numpy as np
import platform
import torch
from pathlib import Path

# Import the new pybind11 backend
# sparseops_backend = importlib.import_module("sparseops_backend")
import sparseops_backend

def run_matvec(weight: np.ndarray, bias: np.ndarray, input_tensor: np.ndarray) -> np.ndarray:
    """
    Dense matrix-vector multiplication with bias using the C++ backend.
    Args:
        weight: (M, K) torch.Tensor
        bias: (M,) torch.Tensor
        input_tensor: (K,) torch.Tensor
    Returns:
        (M,) torch.Tensor
    """
    return sparseops_backend.run_matvec(weight, input_tensor, bias)

def convert_to_bcoo16(weight: np.ndarray):
    """
    Convert a dense matrix to BCOO-16 format.
    Args:
        weight: (M, K) torch.Tensor
    Returns:
        BCOO16 object
    """
    return sparseops_backend.encode_to_bcoo16(weight)

def decode_bcoo16(bcoo):
    """
    Decode a BCOO-16 object back to a dense matrix.
    Args:
        bcoo: BCOO16 object
    Returns:
        (M, K) torch.Tensor
    """
    return sparseops_backend.decode_from_bcoo16(bcoo)

def run_sparse_matvec(bcoo, bias: np.ndarray, input_tensor: np.ndarray, threads: int) -> np.ndarray:
    """
    Sparse matrix-vector multiplication with bias using the C++ backend.
    Args:
        weight: (M, K) torch.Tensor in BCOO format
        bias: (M,) torch.Tensor
        input_tensor: (K,) torch.Tensor
    Returns:
        (M,) torch.Tensor
    """
    return sparseops_backend.sparse_matvec_avx512_mt(bcoo, input_tensor, bias, threads)

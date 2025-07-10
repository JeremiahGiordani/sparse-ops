import importlib
import numpy as np
import platform
import torch
from pathlib import Path

# Import the new pybind11 backend
sparseops_backend = importlib.import_module("sparseops_backend")

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

import importlib
import numpy as np
import platform
import torch
from pathlib import Path

# Import the new pybind11 backend
sparseops_backend = importlib.import_module("sparseops_backend")

def run_matvec(weight: torch.Tensor, bias: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
    """
    Dense matrix-vector multiplication with bias using the C++ backend.
    Args:
        weight: (M, K) torch.Tensor
        bias: (M,) torch.Tensor
        input_tensor: (K,) torch.Tensor
    Returns:
        (M,) torch.Tensor
    """
    A = weight.detach().cpu().numpy().astype(np.float32)
    x = input_tensor.detach().cpu().numpy().astype(np.float32)
    b = bias.detach().cpu().numpy().astype(np.float32)
    y = sparseops_backend.run_matvec(A, x, b)
    return torch.from_numpy(np.array(y))

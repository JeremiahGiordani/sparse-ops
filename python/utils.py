# utils.py

import torch
import numpy as np

def is_pruned(layer: torch.nn.Linear) -> bool:
    """Check if a linear layer has been pruned."""
    return "weight_orig" in layer._parameters and "weight_mask" in layer._buffers

def to_csr(weight: torch.Tensor):
    """Convert a 2D PyTorch tensor to CSR format."""
    weight_np = weight.detach().cpu().numpy()
    values = []
    indices = []
    indptr = [0]

    for row in weight_np:
        for j, val in enumerate(row):
            if val != 0:
                values.append(val)
                indices.append(j)
        indptr.append(len(values))

    return (np.array(values, dtype=np.float32),
            np.array(indices, dtype=np.int32),
            np.array(indptr, dtype=np.int32))
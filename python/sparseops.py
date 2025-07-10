"""
sparseops: Ultra-fast sparse matrix multiplication library
"""

import numpy as np
import scipy.sparse as sp
import warnings
from typing import Optional, Union
from .sparseops_backend import PreparedA, prepare_csr, sgemm as _sgemm

__version__ = "0.1.0"

def prepare(A: Union[sp.csr_matrix, sp.csr_array], block: int = 16) -> PreparedA:
    """
    Prepare a sparse matrix for optimized multiplication.
    
    Parameters:
    -----------
    A : scipy.sparse.csr_matrix or csr_array
        Input sparse matrix in CSR format
    block : int, default=16
        Block size for internal representation
        
    Returns:
    --------
    PreparedA
        Prepared matrix handle for efficient multiplication
    """
    if not sp.isspmatrix_csr(A) and not isinstance(A, sp.csr_array):
        raise ValueError("Input matrix must be in CSR format")
    
    if A.dtype != np.float32:
        warnings.warn("Converting matrix to float32 for optimal performance")
        A = A.astype(np.float32)
    
    # Warn if matrix is not very sparse
    density = A.nnz / (A.shape[0] * A.shape[1])
    if density > 0.3:
        warnings.warn(f"Matrix density is {density:.2%}. Dense multiplication may be faster.")
    
    return prepare_csr(A.indptr, A.indices, A.data, A.shape[0], A.shape[1], block)

def sgemm(A: PreparedA, B: np.ndarray, C: Optional[np.ndarray] = None, 
          accumulate: bool = False, repeats: int = 1) -> np.ndarray:
    """
    Sparse matrix-dense matrix multiplication: C = A @ B
    
    Parameters:
    -----------
    A : PreparedA
        Prepared sparse matrix
    B : np.ndarray
        Dense matrix (K x N)
    C : np.ndarray, optional
        Output matrix (M x N). If None, will be allocated.
    accumulate : bool, default=False
        If True, accumulate results into C instead of overwriting
    repeats : int, default=1
        Number of times to repeat the multiplication (for benchmarking)
        
    Returns:
    --------
    np.ndarray
        Result matrix C
    """
    if B.dtype != np.float32:
        B = B.astype(np.float32)
    
    if B.ndim != 2:
        raise ValueError("B must be a 2D array")
    
    if B.shape[0] != A.cols():
        raise ValueError(f"Dimension mismatch: A.cols()={A.cols()}, B.shape[0]={B.shape[0]}")
    
    if C is None:
        C = np.zeros((A.rows(), B.shape[1]), dtype=np.float32)
    else:
        if C.dtype != np.float32:
            raise ValueError("C must be float32")
        if C.shape != (A.rows(), B.shape[1]):
            raise ValueError(f"C shape mismatch. Expected {(A.rows(), B.shape[1])}, got {C.shape}")
    
    _sgemm(A, B, C, accumulate, repeats)
    return C

# Convenience function that matches the old interface
def run_sparse_matvec(weight: np.ndarray, bias: np.ndarray, input_tensor: np.ndarray) -> np.ndarray:
    """
    Backward compatibility function for sparse matrix-vector multiplication
    """
    import torch
    
    # Convert to scipy sparse matrix
    weight_csr = sp.csr_matrix(weight.detach().cpu().numpy())
    
    # Prepare the matrix
    A_prepared = prepare(weight_csr)
    
    # Reshape input for matrix multiplication
    input_np = input_tensor.detach().cpu().numpy().reshape(-1, 1)
    
    # Perform multiplication
    result = sgemm(A_prepared, input_np.T)
    
    # Add bias and return as torch tensor
    result_with_bias = result.flatten() + bias.detach().cpu().numpy()
    return torch.from_numpy(result_with_bias)

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

def encode_to_quasi_dense(sparse_matrix: np.ndarray) -> sparseops_backend.QuasiDense:
    """
    Convert a sparse matrix to a quasi-dense representation.
    Args:
        sparse_matrix: (M, K) numpy.ndarray
    Returns:
        QuasiDense object
    """
    return sparseops_backend.convert_to_quasi_dense(sparse_matrix)

def transform_input(quasi_dense: sparseops_backend.QuasiDense, input_vector: np.ndarray) -> np.ndarray:
    """
    Transform an input vector using a quasi-dense representation.
    Args:
        quasi_dense: QuasiDense object
        input_vector: (K,) numpy.ndarray
    Returns:
        Transformed input vector as XtDense object
    """
    return sparseops_backend.transform_input(quasi_dense, input_vector)

def run_quasi_dense_matvec_gather(quasi_dense: sparseops_backend.QuasiDense, input_vector: np.ndarray, bias: np.ndarray, threads: int) -> np.ndarray:
    """
    Perform matrix-vector multiplication on-the-fly using quasi-dense representation.
    Args:
        quasi_dense: QuasiDense object
        input_vector: (K,) numpy.ndarray
        bias: (M,) numpy.ndarray
        threads: int, number of threads to use
    Returns:
        (M,) numpy.ndarray
    """
    return sparseops_backend.quasi_dense_matvec_gather(quasi_dense, input_vector, bias, threads)


def run_bilinear_diagonal_matvec(Q, x, bias: np.ndarray, threads: int) -> np.ndarray:
    """
    Bilinear diagonal matrix-vector multiplication using the C++ backend.
    Args:
        Q: QuasiDense object (m, n)
        X: XtDense object OR (n,) np.ndarray
        bias: (m,) torch.Tensor
        threads: int, number of threads to use
    Returns:
        (m,) torch.Tensor
    """
    return sparseops_backend.bilinear_diagonal_matvec_mt(Q, x, bias, threads)


def run_quasi_dense_matvec_hidden(Q: sparseops_backend.QuasiDense, Q_next: sparseops_backend.QuasiDense, x: np.ndarray, bias: np.ndarray, threads: int) -> np.ndarray:
    """
    Perform hidden-layer fused quasi-dense matrix-vector multiplication.
    Args:
        Q: QuasiDense object (m, n)
        Q_next: QuasiDense object for the next layer (n, k)
        x: Input tensor (XtDense or (n,))
        bias: Bias vector (m,)
        threads: int, number of threads to use
    Returns:
        (m,) numpy.ndarray
    """
    return sparseops_backend.quasi_dense_matvec_hidden(Q, Q_next, x, bias, threads)
import numpy as np
import sparseops_backend

def encode(sparse_matrix: np.ndarray) -> sparseops_backend.Ellpack:
    """
    Convert a sparse matrix to an ELLPACK representation.
    Args:
        sparse_matrix: (M, K) numpy.ndarray
    Returns:
        Ellpack object
    """
    return sparseops_backend.convert_to_ellpack(sparse_matrix)


def matvec(E, x, bias: np.ndarray) -> np.ndarray:
    """
    Bilinear diagonal matrix-vector multiplication using the C++ backend.
    Args:
        E: Ellpack object (m, n)
        X: XtDense object OR (n,) np.ndarray
        bias: (m,) torch.Tensor
    Returns:
        (m,) torch.Tensor
    """
    return sparseops_backend.ellpack_matvec(E, x, bias)

def matmul(E: sparseops_backend.Ellpack, X: np.ndarray, bias: np.ndarray) -> np.ndarray:
    """
    Bilinear diagonal matrix multiplication using the C++ backend.
    Args:
        E: Ellpack object (m, n)
        X: (n, C) np.ndarray
        bias: (m,) np.ndarray
    Returns:
        (m,C) np.ndarray
    """
    return sparseops_backend.ellpack_matmul(E, X, bias)

def matmul_outer(E: sparseops_backend.Ellpack, X: np.ndarray, bias: np.ndarray) -> np.ndarray:
    """
    Bilinear diagonal outer matrix multiplication using the C++ backend.
    Args:
        E: Ellpack object (n, m)
        X: (B,n) np.ndarray
        bias: (m,) np.ndarray
    Returns:
        (m,C) np.ndarray
    """
    return sparseops_backend.ellpack_matmul_outer(E, X, bias)


def matmul_outer_dis(E: sparseops_backend.Ellpack, X: np.ndarray, bias: np.ndarray) -> np.ndarray:
    """
    Bilinear diagonal outer matrix multiplication using the C++ backend.
    Args:
        E: Ellpack object (n, m)
        X: (B,n) np.ndarray
        bias: (m,) np.ndarray
    Returns:
        (m,C) np.ndarray
    """
    return sparseops_backend.ellpack_matmul_outer_dis(E, X, bias)
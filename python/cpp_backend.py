import numpy as np
import sparseops_backend

def encode(sparse_matrix: np.ndarray) -> sparseops_backend.QuasiDense:
    """
    Convert a sparse matrix to a quasi-dense representation.
    Args:
        sparse_matrix: (M, K) numpy.ndarray
    Returns:
        QuasiDense object
    """
    return sparseops_backend.convert_to_quasi_dense(sparse_matrix)


def matvec(Q, x, bias: np.ndarray) -> np.ndarray:
    """
    Bilinear diagonal matrix-vector multiplication using the C++ backend.
    Args:
        Q: QuasiDense object (m, n)
        X: XtDense object OR (n,) np.ndarray
        bias: (m,) torch.Tensor
    Returns:
        (m,) torch.Tensor
    """
    return sparseops_backend.bilinear_diagonal_matvec(Q, x, bias)

def matmul(Q: sparseops_backend.QuasiDense, X: np.ndarray, bias: np.ndarray) -> np.ndarray:
    """
    Bilinear diagonal matrix multiplication using the C++ backend.
    Args:
        Q: QuasiDense object (m, n)
        X: XtDense object OR (n,) np.ndarray
        bias: (m,) torch.Tensor
    Returns:
        (m,) torch.Tensor
    """
    return sparseops_backend.bilinear_diagonal_matmul(Q, X, bias)
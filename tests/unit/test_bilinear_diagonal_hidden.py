import pytest
import sparseops_backend
import numpy as np
import os

num_threads = int(os.environ.get("OMP_NUM_THREADS", "1"))

def generate_sparse_matrix(M, K, sparsity=0.3):
    """Generates a random sparse matrix with given dimensions and sparsity."""
    dense = np.random.rand(M, K).astype(np.float32)
    dense[dense < sparsity] = 0  # Apply sparsity
    return dense

def get_quasi_dense_matrix(M, K, sparsity=0.3):
    """Generates a quasi-dense matrix with given dimensions and sparsity."""
    sparse_matrix = generate_sparse_matrix(M, K, sparsity)
    return sparseops_backend.convert_to_quasi_dense(sparse_matrix)

def transform_input(quasi_dense, K):
    """Generates a transformed input vector for testing."""
    input_vector = np.random.rand(K).astype(np.float32)
    sparseops_backend.transform_input(quasi_dense, input_vector)

def test_bilinear_diagonal_hidden_multiplication_shape():
    M, K, N, sparsity = 16, 32, 64, 0.3
    Q = get_quasi_dense_matrix(M, K, sparsity)
    Q_next = get_quasi_dense_matrix(N, M, sparsity)
    bias1 = np.random.rand(K).astype(np.float32)
    bias2 = np.random.rand(N).astype(np.float32)

    sparseops_backend.bilinear_diagonal_matvec_hidden_mt(Q, Q_next, bias1, num_threads)
    y = sparseops_backend.bilinear_diagonal_matvec_mt(Q_next, bias2, num_threads)

    assert y.shape == (N,), "Output shape mismatch"

def test_bilinear_diagonal_hidden_multiplication_correctness():
    M, K, N, Z, sparsity = 16, 32, 64, 128, 0.3
    sparse_matrix_1 = generate_sparse_matrix(M, K, sparsity)
    sparse_matrix_2 = generate_sparse_matrix(N, M, sparsity)
    sparse_matrix_3 = generate_sparse_matrix(Z, N, sparsity)
    quasi_dense_1 = sparseops_backend.convert_to_quasi_dense(sparse_matrix_1)
    quasi_dense_2 = sparseops_backend.convert_to_quasi_dense(sparse_matrix_2)
    quasi_dense_3 = sparseops_backend.convert_to_quasi_dense(sparse_matrix_3)

    bias1 = np.random.rand(M).astype(np.float32)
    bias2 = np.random.rand(N).astype(np.float32)
    bias3 = np.random.rand(Z).astype(np.float32)

    input_vector = np.random.rand(K).astype(np.float32)

    sparseops_backend.bilinear_diagonal_matvec_hidden_mt(quasi_dense_1, quasi_dense_2, input_vector, bias1, num_threads)
    sparseops_backend.bilinear_diagonal_matvec_hidden_mt(quasi_dense_2, quasi_dense_3, bias2, num_threads)
    y = sparseops_backend.bilinear_diagonal_matvec_mt(quasi_dense_3, bias3, num_threads)

    y_1 = sparse_matrix_1 @ input_vector + bias1
    y_2 = sparse_matrix_2 @ y_1 + bias2
    y_expected = sparse_matrix_3 @ y_2 + bias3
    assert y.shape == (Z,), "Output shape mismatch for correctness test"
    assert np.allclose(y, y_expected, atol=1e-5), "Bilinear diagonal multiplication mismatch"

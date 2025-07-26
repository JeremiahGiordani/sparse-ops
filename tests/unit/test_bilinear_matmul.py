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


def test_bilinear_diagonal_matmul_multiplication_shape():
    M, K, N, sparsity = 16, 32, 64, 0.3
    Q = get_quasi_dense_matrix(M, K, sparsity)
    X = generate_sparse_matrix(K, N, sparsity)
    bias = np.random.rand(M,).astype(np.float32)

    # Test the bilinear diagonal multiplication
    print(f"Type of Q: {type(Q)}")
    y = sparseops_backend.bilinear_diagonal_matmul_mt(Q, X, bias, num_threads)
    assert y.shape == (M,N), "Output shape mismatch"


def test_bilinear_diagonal_matmul_multiplication_single_column_input_matrix_correctness():
    M, K, N, sparsity = 16, 4, 1, 0.3
    sparse_matrix = generate_sparse_matrix(M, K, sparsity)
    Q = sparseops_backend.convert_to_quasi_dense(sparse_matrix)
    X = np.random.rand(K, N).astype(np.float32)
    bias = np.zeros(M,).astype(np.float32)

    # Test the bilinear diagonal multiplication
    y = sparseops_backend.bilinear_diagonal_matmul_mt(Q, X, bias, num_threads)
    y_expected = sparse_matrix @ X + np.expand_dims(bias, axis=1)
    assert y.shape == (M,N), "Output shape mismatch for correctness test"
    assert y.shape == y_expected.shape, "Output shape mismatch for correctness test"
    print(f"y: {y},\ny_expected: {y_expected}")
    assert np.allclose(y, y_expected, atol=1e-5), "Bilinear diagonal multiplication mismatch"


def test_bilinear_diagonal_matmul_multiplication_single_column_input_matrix_non_zero_bias_correctness():
    M, K, N, sparsity = 16, 4, 1, 0.3
    sparse_matrix = generate_sparse_matrix(M, K, sparsity)
    Q = sparseops_backend.convert_to_quasi_dense(sparse_matrix)
    X = np.random.rand(K, N).astype(np.float32)
    bias = np.random.rand(M,).astype(np.float32)

    # Test the bilinear diagonal multiplication
    y = sparseops_backend.bilinear_diagonal_matmul_mt(Q, X, bias, num_threads)
    y_expected = sparse_matrix @ X + np.expand_dims(bias, axis=1)
    assert y.shape == (M,N), "Output shape mismatch for correctness test"
    assert y.shape == y_expected.shape, "Output shape mismatch for correctness test"
    print(f"y: {y},\ny_expected: {y_expected}")
    assert np.allclose(y, y_expected, atol=1e-5), "Bilinear diagonal multiplication mismatch"


def test_bilinear_diagonal_matmul_multiplication_two_column_input_matrix_correctness():
    M, K, N, sparsity = 4, 4, 2, 0.3
    sparse_matrix = generate_sparse_matrix(M, K, sparsity)
    Q = sparseops_backend.convert_to_quasi_dense(sparse_matrix)
    X = np.random.rand(K, N).astype(np.float32)
    bias = np.zeros(M,).astype(np.float32)

    # Test the bilinear diagonal multiplication
    y = sparseops_backend.bilinear_diagonal_matmul_mt(Q, X, bias, num_threads)
    y_expected = sparse_matrix @ X + np.expand_dims(bias, axis=1)
    assert y.shape == (M,N), "Output shape mismatch for correctness test"
    assert y.shape == y_expected.shape, "Output shape mismatch for correctness test"
    print(f"y: {y},\ny_expected: {y_expected}")
    assert np.allclose(y, y_expected, atol=1e-5), "Bilinear diagonal multiplication mismatch"


def test_bilinear_diagonal_matmul_multiplication_correctness_zero_bias_small():
    M, K, N, sparsity = 2, 4, 4, 0.3
    sparse_matrix = generate_sparse_matrix(M, K, sparsity)
    quasi_dense = sparseops_backend.convert_to_quasi_dense(sparse_matrix)
    X = np.random.rand(K, N).astype(np.float32)

    bias = np.zeros(M,).astype(np.float32)

    y_expected = sparse_matrix @ X + np.expand_dims(bias, axis=1)
    y = sparseops_backend.bilinear_diagonal_matmul_mt(quasi_dense, X, bias, num_threads)

    assert y.shape == (M,N), "Output shape mismatch for correctness test"
    assert y.shape == y_expected.shape, "Output shape mismatch for correctness test"
    print(f"y: {y},\ny_expected: {y_expected}")
    assert np.allclose(y, y_expected, atol=1e-5), "Bilinear diagonal multiplication mismatch"

def test_bilinear_diagonal_matmul_multiplication_correctness_zero_bias():
    M, K, N, sparsity = 16, 32, 64, 0.3
    sparse_matrix = generate_sparse_matrix(M, K, sparsity)
    quasi_dense = sparseops_backend.convert_to_quasi_dense(sparse_matrix)
    X = np.random.rand(K, N).astype(np.float32)

    bias = np.zeros(M,).astype(np.float32)

    y_expected = sparse_matrix @ X + np.expand_dims(bias, axis=1)
    y = sparseops_backend.bilinear_diagonal_matmul_mt(quasi_dense, X, bias, num_threads)

    assert y.shape == (M,N), "Output shape mismatch for correctness test"
    assert y.shape == y_expected.shape, "Output shape mismatch for correctness test"
    assert np.allclose(y, y_expected, atol=1e-5), "Bilinear diagonal multiplication mismatch"

def test_bilinear_diagonal_matmul_multiplication_correctness():
    M, K, N, sparsity = 16, 32, 64, 0.3
    sparse_matrix = generate_sparse_matrix(M, K, sparsity)
    quasi_dense = sparseops_backend.convert_to_quasi_dense(sparse_matrix)
    X = np.random.rand(K, N).astype(np.float32)

    bias = np.random.rand(M,).astype(np.float32)

    y_expected = sparse_matrix @ X + np.expand_dims(bias, axis=1)
    y = sparseops_backend.bilinear_diagonal_matmul_mt(quasi_dense, X, bias, num_threads)

    assert y.shape == (M,N), "Output shape mismatch for correctness test"
    assert y.shape == y_expected.shape, "Output shape mismatch for correctness test"
    assert np.allclose(y, y_expected, atol=1e-5), "Bilinear diagonal multiplication mismatch"

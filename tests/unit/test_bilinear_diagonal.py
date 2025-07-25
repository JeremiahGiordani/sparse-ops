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

def get_transformed_input_vector(quasi_dense, K):
    """Generates a transformed input vector for testing."""
    input_vector = np.random.rand(K).astype(np.float32)
    return sparseops_backend.transform_input(quasi_dense, input_vector)

def test_bilinear_diagonal_multiplication_shape():
    M, K, sparsity = 16, 32, 0.3
    Q = get_quasi_dense_matrix(M, K, sparsity)
    X = get_transformed_input_vector(Q, K)
    bias = np.random.rand(M).astype(np.float32)

    # Test the bilinear diagonal multiplication
    print(f"Type of Q: {type(Q)}")
    print(f"Type of X: {type(X)}")
    y = sparseops_backend.bilinear_diagonal_matvec_mt(Q, X, bias, num_threads)
    assert y.shape == (M,), "Output shape mismatch"

# Test across a range of dimensions and sparsity levels
@pytest.mark.parametrize("M, K, sparsity", [
    # Regular shapes
    (16, 32, 0.3),
    (64, 128, 0.5),
    (128, 256, 0.7),
    (256, 512, 0.9),
    # Edge cases
    (1, 1, 0.0),  # Single element
    (1, 1000, 0.1),  # Single row wide matrix
    (1000, 1, 0.1),  # Single column tall matrix
    (0, 0, 0.0),  # Empty matrix
    (32, 32, 1.0),  # Fully dense matrix
    (10, 10, 0.0),  # Dense matrix
    (10, 10, 0.5),  # Sparse matrix
    (10, 10, 0.9),  # Very sparse matrix
    (20, 20, 0.2),
    (100, 100, 0.2),  # Moderate sparsity
    (1000, 1000, 0.1),  # Large matrix
    (500, 500, 0.05),  # Large sparse matrix
    (2000, 2000, 0.3),  # Very large matrix
    (3000, 3000, 0.4),  # Very large sparse matrix
])
def test_bilinear_diagonal_multiplication_correctness_parametrized(M, K, sparsity):
    sparse_matrix = generate_sparse_matrix(M, K, sparsity)
    quasi_dense = sparseops_backend.convert_to_quasi_dense(sparse_matrix)
    input_vector = np.random.rand(K).astype(np.float32)
    transformed_input = sparseops_backend.transform_input(quasi_dense, input_vector)
    Q = quasi_dense
    X = transformed_input
    bias = np.random.rand(M).astype(np.float32)

    # Test the bilinear diagonal multiplication
    y = sparseops_backend.bilinear_diagonal_matvec_mt(Q, X, bias, num_threads)

    y_expected = sparse_matrix @ input_vector + bias
    assert y.shape == (M,), "Output shape mismatch for parametrized test"
    assert np.allclose(y, y_expected, atol=1e-5), "Bilinear diagonal multiplication mismatch"
    

# Test a few thread counts
@pytest.mark.parametrize("threads", [1, 2, 4, 8])
def test_bilinear_diagonal_multiplication_threads(threads):
    M, K, sparsity = 64, 128, 0.5
    Q = get_quasi_dense_matrix(M, K, sparsity)
    X = get_transformed_input_vector(Q, K)
    bias = np.random.rand(M).astype(np.float32)

    # Test the bilinear diagonal multiplication with different thread counts
    y = sparseops_backend.bilinear_diagonal_matvec_mt(Q, X, bias, threads)
    assert y.shape == (M,), "Output shape mismatch for thread test"
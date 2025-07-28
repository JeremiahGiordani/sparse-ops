import pytest
import sparseops_backend
import numpy as np

def generate_sparse_matrix(M, K, sparsity=0.3):
    """Generates a random sparse matrix with given dimensions and sparsity."""
    dense = np.random.rand(M, K).astype(np.float32)
    dense[dense < sparsity] = 0  # Apply sparsity
    return dense

# Test across a range of dimensions, number of columns, and sparsity levels
@pytest.mark.parametrize("M, K, C, sparsity", [
    # Regular shapes
    (16, 32, 1, 0.3),
    (16, 32, 10, 0.3),
    (16, 32, 100, 0.3),
    (64, 128, 10, 0.5),
    (128, 256, 20, 0.7),
    (256, 512, 10, 0.9),
    # Edge cases
    (1, 1, 1, 0.0),  # Single element
    (1, 1000, 1, 0.1),  # Single row wide matrix
    (1000, 1, 1, 0.1),  # Single column tall matrix
    (0, 0, 0, 0.0),  # Empty matrix
    (32, 32, 1, 1.0),  # Fully dense matrix
    (10, 10, 1, 0.0),  # Dense matrix
    (10, 10, 10, 0.5),  # Sparse matrix
    (10, 10, 10, 0.9),  # Very sparse matrix
    (20, 20, 10, 0.2),
    (100, 100, 10, 0.2),  # Moderate sparsity
    (1000, 1000, 10, 0.1),  # Large matrix
    (500, 500, 10, 0.05),  # Large sparse matrix
    (2000, 2000, 10, 0.3),  # Very large matrix
    (3000, 3000, 10, 0.4),  # Very large sparse matrix
])
def test_ellpack_matmul_correctness_parametrized(M, K, C, sparsity):
    sparse_matrix = generate_sparse_matrix(M, K, sparsity)
    ellpack = sparseops_backend.convert_to_ellpack(sparse_matrix)
    input_matrix = np.random.rand(K, C).astype(np.float32)
    E = ellpack
    bias = np.random.rand(M,).astype(np.float32)

    # Test the ellpack multiplication
    y = sparseops_backend.ellpack_matmul(E, input_matrix, bias)

    y_expected = sparse_matrix @ input_matrix + np.expand_dims(bias, axis=1)
    assert y.shape == (M,C), "Output shape mismatch for parametrized test"
    assert np.allclose(y, y_expected, atol=1e-5), "Ellpack multiplication mismatch"
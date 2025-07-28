import pytest
import sparseops_backend
import numpy as np

def generate_sparse_matrix(M, K, sparsity=0.3):
    """Generates a random sparse matrix with given dimensions and sparsity."""
    dense = np.random.rand(M, K).astype(np.float32)
    dense[dense < sparsity] = 0  # Apply sparsity
    return dense

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
def test_ellpack_multiplication_correctness_parametrized(M, K, sparsity):
    sparse_matrix = generate_sparse_matrix(M, K, sparsity)
    ellpack = sparseops_backend.convert_to_ellpack(sparse_matrix)
    input_vector = np.random.rand(K).astype(np.float32)
    E = ellpack
    bias = np.random.rand(M).astype(np.float32)

    # Test the bilinear diagonal multiplication
    y = sparseops_backend.ellpack_matvec(E, input_vector, bias)

    y_expected = sparse_matrix @ input_vector + bias
    assert y.shape == (M,), "Output shape mismatch for parametrized test"
    assert np.allclose(y, y_expected, atol=1e-5), "Bilinear diagonal multiplication mismatch"
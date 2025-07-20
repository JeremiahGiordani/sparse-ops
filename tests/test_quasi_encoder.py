# Unit tests for sparseops_backend's quasi-dense encoding functionality
# Tested using pytest framework

import sparseops_backend
import numpy as np

def generate_sparse_matrix(M, K, sparsity=0.3):
    """Generates a random sparse matrix with given dimensions and sparsity."""
    dense = np.random.rand(M, K).astype(np.float32)
    dense[dense < sparsity] = 0  # Apply sparsity
    return dense

def test_quasi_dense_encoding():
    M, K = 64, 64
    sparse_matrix = generate_sparse_matrix(M, K)
    quasi_dense = sparseops_backend.convert_to_quasi_dense(sparse_matrix)
    decoded = sparseops_backend.decode_from_quasi_dense(quasi_dense)

    # Check if the original and decoded matrices are close
    assert np.allclose(sparse_matrix, decoded, atol=1e-5)

def test_quasi_dense_encoding_shape():
    M, K = 128, 128
    sparse_matrix = generate_sparse_matrix(M, K)
    quasi_dense = sparseops_backend.convert_to_quasi_dense(sparse_matrix)

    # Check if the shape of the quasi-dense representation is correct
    assert quasi_dense.m == M
    assert quasi_dense.n == K

def test_quasi_dense_encoder_correctly_sets_max_non_zeros_per_row():
    M, K = 32, 32
    sparse_matrix = generate_sparse_matrix(M, K)
    quasi_dense = sparseops_backend.convert_to_quasi_dense(sparse_matrix)

    # Check if max_non_zeros_per_row is set correctly
    max_non_zeros = np.max(np.count_nonzero(sparse_matrix, axis=1))
    print(max_non_zeros)
    assert quasi_dense.r == max_non_zeros

def test_quasi_dense_encoder_on_empty_matrix():
    M, K = 32, 32
    sparse_matrix = np.zeros((M, K), dtype=np.float32)
    quasi_dense = sparseops_backend.convert_to_quasi_dense(sparse_matrix)

    # Check if the quasi-dense representation is empty
    assert quasi_dense.m == 32
    assert quasi_dense.n == 32
    assert quasi_dense.r == 0 

def test_quasi_dense_encoder_correctly_sets_wd_on_manual_input():
    matrix = np.array([[1, 0, 0, 2],
                       [0, 3, 0, 0],
                       [0, 0, 4, 0],
                       [0, 0, 0, 5]], dtype=np.float32)
    quasi_dense = sparseops_backend.convert_to_quasi_dense(matrix)
    # Check if the quasi-dense representation has correct wd
    print(quasi_dense.Wd)
    assert quasi_dense.Wd.shape == (8,) # 4 rows * 2 columns = 8 elements
    assert quasi_dense.Wd[0] == 1.0
    assert quasi_dense.Wd[1] == 2.0
    assert quasi_dense.Wd[2] == 3.0
    assert quasi_dense.Wd[3] == 0.0 # padding
    assert quasi_dense.Wd[4] == 4.0

def test_quasi_dense_encoder_sets_column_indices_correctly():
    matrix = np.array([[1, 0, 0, 2],
                       [0, 3, 0, 0],
                       [0, 0, 4, 0],
                       [0, 0, 0, 5]], dtype=np.float32)
    quasi_dense = sparseops_backend.convert_to_quasi_dense(matrix)
    # Check if the column indices are set correctly
    assert quasi_dense.idx.shape == (8,)
    assert np.array_equal(quasi_dense.idx[:4], [0, 3, 1, 0]) # Last value is padding, col idx 0
    assert np.array_equal(quasi_dense.idx[4:], [2, 0, 3, 0]) # Second and fourth values are padding

def test_quasi_dense_encoder_handles_non_square_matrices():
    M, K = 64, 32
    sparse_matrix = generate_sparse_matrix(M, K)
    quasi_dense = sparseops_backend.convert_to_quasi_dense(sparse_matrix)

    # Check if the quasi-dense representation has correct dimensions
    assert quasi_dense.m == M
    assert quasi_dense.n == K
    assert quasi_dense.r == np.max(np.count_nonzero(sparse_matrix, axis=1))

def test_quasi_dense_decoder_handles_non_square_matrices():
    M, K = 64, 32
    sparse_matrix = generate_sparse_matrix(M, K)
    quasi_dense = sparseops_backend.convert_to_quasi_dense(sparse_matrix)
    decoded = sparseops_backend.decode_from_quasi_dense(quasi_dense)

    # Check if the decoded matrix has correct dimensions
    assert decoded.shape == (M, K)
    assert np.allclose(sparse_matrix, decoded, atol=1e-5)

# ----------------------------------------------------------------------

def test_transform_input_shape():
    M, K = 64, 32
    sparse_matrix = generate_sparse_matrix(M, K)
    quasi_dense = sparseops_backend.convert_to_quasi_dense(sparse_matrix)

    input_vec = np.random.randn(32).astype(np.float32)
    transformed = sparseops_backend.transform_input(quasi_dense, input_vec)

    print(transformed)

    # Check if the transformed input has correct shape
    assert transformed.m == M
    assert transformed.r == np.max(np.count_nonzero(sparse_matrix, axis=1))

def test_transform_input_shape_with_empty_matrix():
    M, K = 32, 32
    sparse_matrix = np.zeros((M, K), dtype=np.float32)
    quasi_dense = sparseops_backend.convert_to_quasi_dense(sparse_matrix)

    input_vec = np.random.randn(32).astype(np.float32)
    transformed = sparseops_backend.transform_input(quasi_dense, input_vec)

    # Check if the transformed input has correct shape
    assert transformed.m == M
    assert transformed.r == 0  # No non-zero elements in the matrix

def test_transform_input_with_manual_input():
    matrix = np.array([[1, 0, 0, 2],
                       [0, 3, 0, 0],
                       [0, 0, 4, 0],
                       [0, 0, 0, 5]], dtype=np.float32)
    quasi_dense = sparseops_backend.convert_to_quasi_dense(matrix)

    input_vec = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    transformed = sparseops_backend.transform_input(quasi_dense, input_vec)

    # Check if the transformed input has correct values
    assert transformed.m == 4
    assert transformed.r == 2
    assert np.array_equal(transformed.Xt[:4], [1.0, 4.0, 2.0, 1.0])
    assert np.array_equal(transformed.Xt[4:], [3.0, 1.0, 4.0, 1.0])  # Column indices for the non-zero elements

def test_manual_multiplication_with_objects_equals_expected():
    M, K = 7, 7
    sparse_matrix = generate_sparse_matrix(M, K)
    quasi_dense = sparseops_backend.convert_to_quasi_dense(sparse_matrix)

    input_vec = np.random.randn(K).astype(np.float32)
    transformed = sparseops_backend.transform_input(quasi_dense, input_vec)

    # Perform manual multiplication
    result = np.zeros(M, dtype=np.float32)
    for i in range(M):
        curr_idx = i * transformed.r
        for j in range(transformed.r):
            result[i] += transformed.Xt[curr_idx + j] * quasi_dense.Wd[curr_idx + j]

    # Compare with the expected result
    expected = sparse_matrix @ input_vec
    print(f"Result: {result}")
    print(f"Expected: {expected}")
    assert np.allclose(result, expected)

if __name__ == "__main__":
    test_manual_multiplication_with_objects_equals_expected()
    
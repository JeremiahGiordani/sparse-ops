# Unit tests for sparseops_backend's quasi-dense encoding functionality
# Tested using pytest framework

import sparseops_backend
import numpy as np

def generate_sparse_matrix(M, K, sparsity=0.3):
    """Generates a random sparse matrix with given dimensions and sparsity."""
    dense = np.random.rand(M, K).astype(np.float32)
    dense[dense < sparsity] = 0  # Apply sparsity
    return dense

def test_ellpack_encoding():
    M, K = 64, 64
    sparse_matrix = generate_sparse_matrix(M, K)
    ellpack = sparseops_backend.convert_to_ellpack(sparse_matrix)
    decoded = sparseops_backend.decode_from_ellpack(ellpack)

    # Check if the original and decoded matrices are close
    assert np.allclose(sparse_matrix, decoded, atol=1e-5)

def test_ellpack_encoding_shape():
    M, K = 128, 128
    sparse_matrix = generate_sparse_matrix(M, K)
    ellpack = sparseops_backend.convert_to_ellpack(sparse_matrix)

    # Check if the shape of the quasi-dense representation is correct
    assert ellpack.m == M
    assert ellpack.n == K

def test_ellpack_encoder_correctly_sets_max_non_zeros_per_row():
    M, K = 32, 32
    sparse_matrix = generate_sparse_matrix(M, K)
    ellpack = sparseops_backend.convert_to_ellpack(sparse_matrix)

    # Check if max_non_zeros_per_row is set correctly
    max_non_zeros = np.max(np.count_nonzero(sparse_matrix, axis=1))
    print(max_non_zeros)
    assert ellpack.r == max_non_zeros

def test_ellpack_encoder_on_empty_matrix():
    M, K = 32, 32
    sparse_matrix = np.zeros((M, K), dtype=np.float32)
    ellpack = sparseops_backend.convert_to_ellpack(sparse_matrix)

    # Check if the quasi-dense representation is empty
    assert ellpack.m == 32
    assert ellpack.n == 32
    assert ellpack.r == 0 

def test_ellpack_encoder_correctly_sets_wd_on_manual_input():
    matrix = np.array([[1, 0, 0, 2],
                       [0, 3, 0, 0],
                       [0, 0, 4, 0],
                       [0, 0, 0, 5]], dtype=np.float32)
    ellpack = sparseops_backend.convert_to_ellpack(matrix)
    # Check if the quasi-dense representation has correct wd
    print(ellpack.Wd)
    assert ellpack.Wd.shape == (4,2) # 4 rows * 2 columns = 8 elements
    assert ellpack.Wd[0, 0] == 1.0
    assert ellpack.Wd[0, 1] == 2.0
    assert ellpack.Wd[1, 0] == 3.0
    assert ellpack.Wd[1, 1] == 0.0 # padding
    assert ellpack.Wd[2, 0] == 4.0

def test_ellpack_encoder_sets_column_indices_correctly():
    matrix = np.array([[1, 0, 0, 2],
                       [0, 3, 0, 0],
                       [0, 0, 4, 0],
                       [0, 0, 0, 5]], dtype=np.float32)
    ellpack = sparseops_backend.convert_to_ellpack(matrix)
    # Check if the column indices are set correctly
    assert ellpack.idx.shape == (4, 2)
    assert np.array_equal(ellpack.idx[0], [0, 3])
    assert np.array_equal(ellpack.idx[1], [1, 0])
    assert np.array_equal(ellpack.idx[2], [2, 0])
    assert np.array_equal(ellpack.idx[3], [3, 0])

def test_ellpack_encoder_handles_non_square_matrices():
    M, K = 64, 32
    sparse_matrix = generate_sparse_matrix(M, K)
    ellpack = sparseops_backend.convert_to_ellpack(sparse_matrix)

    # Check if the quasi-dense representation has correct dimensions
    assert ellpack.m == M
    assert ellpack.n == K
    assert ellpack.r == np.max(np.count_nonzero(sparse_matrix, axis=1))

def test_ellpack_decoder_handles_non_square_matrices():
    M, K = 64, 32
    sparse_matrix = generate_sparse_matrix(M, K)
    ellpack = sparseops_backend.convert_to_ellpack(sparse_matrix)
    decoded = sparseops_backend.decode_from_ellpack(ellpack)

    # Check if the decoded matrix has correct dimensions
    assert decoded.shape == (M, K)
    assert np.allclose(sparse_matrix, decoded, atol=1e-5)

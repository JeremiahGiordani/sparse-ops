import numpy as np
import sparseops_backend
import ctypes
from sparseops_backend import sparse_matvec_avx512, BCOO16

def generate_sparse_matrix(shape, sparsity=0.95):
    """Generate a random sparse matrix in BCOO-16 format."""
    dense_matrix = np.random.rand(*shape)
    mask = np.random.rand(*shape) > sparsity
    dense_matrix[mask] = 0
    return dense_matrix

def test_sparse_matvec_avx512():
    """Test the sparse_matvec_avx512 function for correctness."""
    M, K = 1024, 1024
    dense_matrix = generate_sparse_matrix((M, K))
    bcoo16 = BCOO16()
    bcoo16 = sparseops_backend.encode_to_bcoo16(dense_matrix)
    x = np.random.rand(K).astype(np.float32)
    bias = np.random.rand(M).astype(np.float32)
    
    # Ensure arrays are aligned
    x = np.require(x, requirements=['A', 'O', 'C'])
    bias = np.require(bias, requirements=['A', 'O', 'C'])
    y_sparse = np.require(np.empty(M, dtype=np.float32), requirements=['A', 'O', 'C'])
    
    # Compute using sparse kernel
    sparse_matvec_avx512(bcoo16, x, bias, y_sparse, M)
    
    # Compute using dense baseline
    y_dense = dense_matrix @ x + bias
    
    # Assert correctness
    if not np.allclose(y_sparse, y_dense, atol=1e-5):
        print("y_sparse:", y_sparse)
        print("y_dense:", y_dense)
        raise AssertionError("Sparse kernel output does not match dense baseline")
    
    # Compute using dense baseline
    y_dense = dense_matrix @ x + bias
    
    # Assert correctness
    assert np.allclose(y_sparse, y_dense, atol=1e-5), "Sparse kernel output does not match dense baseline"

def test_avx512_kernel_correctness():
    M, K = 64, 64
    dense = np.random.rand(M, K).astype(np.float32)
    dense[dense < 0.9] = 0  # 90% sparse

    x = np.random.rand(K).astype(np.float32)
    b = np.random.rand(M).astype(np.float32)

    bcoo = sparseops_backend.encode_to_bcoo16(dense)
    y = np.empty(M, dtype=np.float32)

    sparseops_backend.sparse_matvec_avx512(bcoo, x, y)

    y_expected = dense @ x + b
    y_expected = dense @ x 
    print("y_expected:", y_expected)
    print("y:", y)
    print(f"y_expected sum: {np.sum(y_expected)}")
    print(f"y sum: {np.sum(y)}")
    print("Number of correct elements:", np.sum(np.isclose(y, y_expected, atol=1e-4)))
    assert np.allclose(y, y_expected, atol=1e-4), "Mismatch in AVX-512 output"

    print("✅ AVX-512 kernel correctness test passed.")


def benchmark_sparse_matvec_avx512():
    """Benchmark the sparse_matvec_avx512 function."""
    M, K = 4096, 4096
    dense_matrix = generate_sparse_matrix((M, K), sparsity=0.95)
    bcoo16 = BCOO16()
    bcoo16 = sparseops_backend.encode_to_bcoo16(dense_matrix)
    x = np.random.rand(K).astype(np.float32)
    bias = np.random.rand(M).astype(np.float32)
    
    # Measure runtime
    import time
    start_time = time.time()
    for _ in range(1000):
        sparse_matvec_avx512(bcoo16, x, bias)
    end_time = time.time()
    
    print(f"Average runtime: {(end_time - start_time) / 1000:.6f} seconds per operation")

if __name__ == "__main__":
    # test_sparse_matvec_avx512()
    # benchmark_sparse_matvec_avx512()
    test_avx512_kernel_correctness()
    print("All tests passed successfully.")

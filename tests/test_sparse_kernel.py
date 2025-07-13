import numpy as np
import sparseops_backend
from sparseops_backend import BCOO16, sparse_matvec_avx512

def pad_block(active_vals, bitmask):
    """Return a length-16 list with zeros in inactive lanes."""
    out = [0.0] * 16
    idx = 0
    for lane in range(16):
        if (bitmask >> lane) & 1:
            out[lane] = active_vals[idx]
            idx += 1
    return out

def test_sparse_matvec_avx512():
    # -------- build block 0 --------
    block0_vals = pad_block([1.0, 2.0], 0b0000000000000011)
    # -------- build block 1 --------
    block1_vals = pad_block([3.0, 4.0], 0b0000000000000011)

    bcoo16 = BCOO16()
    bcoo16.row_id      = [0, 1]
    bcoo16.first_col   = [0, 2]
    bcoo16.bitmask     = [0b11, 0b11]
    bcoo16.values      = block0_vals + block1_vals    # 32 floats!
    bcoo16.original_num_cols = 4
    bcoo16.original_num_rows = 2

    x     = np.array([1, 2, 3, 4], dtype=np.float32)
    bias  = np.array([0.0, 0.0],  dtype=np.float32)
    y     = np.zeros(2, dtype=np.float32)

    sparse_matvec_avx512(bcoo16, x, bias, y)

    expected = np.array([5.0, 25.0], dtype=np.float32)
    assert np.allclose(y, expected, atol=1e-6), f"got {y}, want {expected}"
    print("✅ SpMV AVX-512 test passed.")

def test_avx512_single_block():
    bcoo = BCOO16()
    bcoo.row_id = [0]
    bcoo.first_col = [0]
    bcoo.bitmask = [0b0101]  # indices 0 and 2
    block0_vals = pad_block([1.0, 2.0], 0b0000000000000101)
    bcoo.values = block0_vals

    bcoo.original_num_cols = 4
    bcoo.original_num_rows = 1

    x = np.array([10.0, 0.0, 20.0, 0.0], dtype=np.float32)
    bias = np.array([0.0], dtype=np.float32)
    y = np.zeros(1, dtype=np.float32)

    sparse_matvec_avx512(bcoo, x, bias, y)
    expected = 1.0 * 10.0 + 2.0 * 20.0
    assert np.isclose(y[0], expected), f"Expected {expected}, got {y[0]}"
    print("✅ Single block test passed.")


def test_avx512_two_blocks_noncontiguous():
    bcoo = BCOO16()
    bcoo.row_id = [0, 1]
    bcoo.first_col = [0, 16]
    bcoo.bitmask = [0b1001, 0b0011]  # [0, 3], [16, 17]
    bcoo.values = pad_block([1.0, 2.0], 0b0000000000001001) + pad_block([3.0, 4.0], 0b0000000000000011)
    bcoo.original_num_cols = 32
    bcoo.original_num_rows = 2

    x = np.zeros(32, dtype=np.float32)
    x[0] = 1.0
    x[3] = 2.0
    x[16] = 3.0
    x[17] = 4.0
    bias = np.array([0.0, 0.0], dtype=np.float32)
    y = np.zeros(2, dtype=np.float32)

    sparse_matvec_avx512(bcoo, x, bias, y)
    assert np.allclose(y, [1.0 + 2.0 * 2, 3.0 * 3.0 + 4.0 * 4.0])
    print("✅ Two blocks non-contiguous test passed.")

def test_avx512_zero_mask_skipped():
    bcoo = BCOO16()
    bcoo.row_id = [0]
    bcoo.first_col = [0]
    bcoo.bitmask = [0b0000]  # no active elements
    bcoo.values = pad_block([], 0b0000000000000000)  # empty block
    bcoo.original_num_cols = 4
    bcoo.original_num_rows = 1

    x = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    bias = np.array([0.0], dtype=np.float32)
    y = np.array([999.0], dtype=np.float32)

    sparse_matvec_avx512(bcoo, x, bias, y)
    assert np.isclose(y[0], 0.0), f"Expected 0.0, got {y[0]}"
    print("✅ Zero mask skipped test passed.")

def test_avx512_single_block_sparse():
    dense = np.zeros((1, 16), dtype=np.float32)
    dense[0, [0, 2, 5]] = [1.0, 2.0, 3.0]
    x = np.arange(16, dtype=np.float32)
    b = np.array([10.0], dtype=np.float32)

    bcoo = sparseops_backend.encode_to_bcoo16(dense)
    y = np.empty(1, dtype=np.float32)
    sparseops_backend.sparse_matvec_avx512(bcoo, x, b, y)

    expected = dense @ x + b
    assert np.allclose(y, expected, atol=1e-4), f"Expected {expected}, got {y}"
    print("✅ test_avx512_single_block_sparse passed")

def test_avx512_multiple_blocks_one_row():
    dense = np.zeros((1, 32), dtype=np.float32)
    dense[0, [0, 1, 2, 16, 17, 18]] = [1, 2, 3, 4, 5, 6]
    x = np.arange(32, dtype=np.float32)
    b = np.array([0.0], dtype=np.float32)

    bcoo = sparseops_backend.encode_to_bcoo16(dense)
    y = np.empty(1, dtype=np.float32)
    sparseops_backend.sparse_matvec_avx512(bcoo, x, b, y)

    expected = dense @ x
    assert np.allclose(y, expected, atol=1e-4), f"Expected {expected}, got {y}"
    print("✅ test_avx512_multiple_blocks_one_row passed")


def test_avx512_multirow_multiblock():
    dense = np.zeros((4, 32), dtype=np.float32)
    dense[0, [0, 3]] = [1, 2]
    dense[1, [4, 8]] = [3, 4]
    dense[2, [16, 17]] = [5, 6]
    dense[3, [30, 31]] = [7, 8]

    x = np.arange(32, dtype=np.float32)
    b = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)

    bcoo = sparseops_backend.encode_to_bcoo16(dense)
    y = np.empty(4, dtype=np.float32)
    sparseops_backend.sparse_matvec_avx512(bcoo, x, b, y)

    expected = dense @ x + b
    assert np.allclose(y, expected, atol=1e-4), f"Expected {expected}, got {y}"
    print("✅ test_avx512_multirow_multiblock passed")


def test_avx512_dense_block():
    dense = np.arange(16, dtype=np.float32).reshape(1, 16)
    x = np.ones(16, dtype=np.float32)
    b = np.array([0.0], dtype=np.float32)

    bcoo = sparseops_backend.encode_to_bcoo16(dense)
    y = np.empty(1, dtype=np.float32)
    sparseops_backend.sparse_matvec_avx512(bcoo, x, b, y)

    expected = dense @ x
    assert np.allclose(y, expected, atol=1e-4), f"Expected {expected}, got {y}"
    print("✅ test_avx512_dense_block passed")


def test_avx512_diagonal_matrix():
    mat = np.eye(32, dtype=np.float32)
    x = np.ones(32, dtype=np.float32)
    b = np.zeros(32, dtype=np.float32)

    bcoo = sparseops_backend.encode_to_bcoo16(mat)
    y = np.empty(32, dtype=np.float32)
    sparseops_backend.sparse_matvec_avx512(bcoo, x, b, y)

    expected = mat @ x
    assert np.allclose(y, expected, atol=1e-4), f"Expected {expected}, got {y}"
    print("✅ test_avx512_diagonal_matrix passed")

def test_avx512_col_edge_case():
    dense = np.zeros((1, 64), dtype=np.float32)
    dense[0, [48, 49, 50, 63]] = [1.0, 2.0, 3.0, 4.0]

    x = np.arange(64).astype(np.float32)
    b = np.array([5.0], dtype=np.float32)

    bcoo = sparseops_backend.encode_to_bcoo16(dense)
    y = np.empty(1, dtype=np.float32)
    sparseops_backend.sparse_matvec_avx512(bcoo, x, b, y)

    expected = dense @ x + b
    assert np.allclose(y, expected, atol=1e-4), f"Expected {expected}, got {y}"
    print("✅ test_avx512_col_edge_case passed")


def test_avx512_correctness_fixed_seed():
    np.random.seed(42)
    M, K = 64, 64
    dense = np.random.rand(M, K).astype(np.float32)
    dense[dense < 0.9] = 0  # 90% sparse
    # print("dense:", dense[0])


    x = np.random.rand(K).astype(np.float32)
    b = np.zeros(M).astype(np.float32)
    b = np.random.rand(M).astype(np.float32)

    # print("x:", x)
    # print("b:", b)

    bcoo = sparseops_backend.encode_to_bcoo16(dense)

    # print("bcoo:", bcoo)
    # print("bcoo.values:", bcoo.values)
    # print("bcoo.row_id:", bcoo.row_id)
    # print("bcoo.first_col:", bcoo.first_col)
    # print("bcoo.bitmask:", bcoo.bitmask)
    # print("bcoo.original_num_rows:", bcoo.original_num_rows)

    y = np.empty(M, dtype=np.float32)
    sparseops_backend.sparse_matvec_avx512(bcoo, x, b, y)

    y_expected = dense @ x + b
    # print("y_expected:", y_expected)
    # print("y:", y)
    # print(f"y_expected sum: {np.sum(y_expected)}")
    # print(f"y sum: {np.sum(y)}")
    assert np.allclose(y, y_expected, atol=1e-4), "Fixed seed test failed"
    print("✅ test_avx512_correctness_fixed_seed passed")

def test_avx512_correctness_dense_matrix():
    M, K = 64, 64
    dense = np.random.rand(M, K).astype(np.float32)

    x = np.random.rand(K).astype(np.float32)
    b = np.random.rand(M).astype(np.float32)

    bcoo = sparseops_backend.encode_to_bcoo16(dense)
    y = np.empty(M, dtype=np.float32)
    sparseops_backend.sparse_matvec_avx512(bcoo, x, b, y)

    y_expected = dense @ x + b
    assert np.allclose(y, y_expected, atol=1e-4), "Dense matrix test failed"
    print("✅ test_avx512_correctness_dense_matrix passed")


def test_avx512_correctness_half_sparse():
    np.random.seed(0)
    M, K = 64, 64
    dense = np.random.rand(M, K).astype(np.float32)
    dense[dense < 0.5] = 0  # 50% sparse

    x = np.random.rand(K).astype(np.float32)
    b = np.zeros(M).astype(np.float32)
    b = np.random.rand(M).astype(np.float32)

    bcoo = sparseops_backend.encode_to_bcoo16(dense)
    y = np.empty(M, dtype=np.float32)
    sparseops_backend.sparse_matvec_avx512(bcoo, x, b, y)

    y_expected = dense @ x + b
    assert np.allclose(y, y_expected, atol=1e-4), "50% sparse test failed"
    print("✅ test_avx512_correctness_half_sparse passed")



def test_avx512_kernel_correctness():
    M, K = 64, 64
    dense = np.random.rand(M, K).astype(np.float32)
    dense[dense < 0.9] = 0  # 90% sparse

    x = np.random.rand(K).astype(np.float32)
    b = np.random.rand(M).astype(np.float32)

    bcoo = sparseops_backend.encode_to_bcoo16(dense)

    y = np.empty(M, dtype=np.float32)

    sparseops_backend.sparse_matvec_avx512(bcoo, x, b, y)

    y_expected = dense @ x + b
    # print("y_expected:", y_expected)
    # print("y:", y)
    # print(f"y_expected sum: {np.sum(y_expected)}")
    # print(f"y sum: {np.sum(y)}")
    # print("number of correct elements:", np.sum(np.isclose(y, y_expected, atol=1e-4)))
    assert np.allclose(y, y_expected, atol=1e-4), "Mismatch in AVX-512 output"

    print("✅ AVX-512 kernel correctness test passed.")

def test_avx512_kernel_large():
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
    sparse_matvec_avx512(bcoo16, x, bias, y_sparse)
    
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
    print("✅ AVX-512 kernel large test passed.")

def test_avx512_kernel_large_irregular_square():
    """Test the sparse_matvec_avx512 function for correctness."""
    M, K = 1001, 1001
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
    sparse_matvec_avx512(bcoo16, x, bias, y_sparse)
    
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
    print("✅ AVX-512 kernel large irregular square test passed.")


def test_avx512_kernel_large_irregular_non_square():
    """Test the sparse_matvec_avx512 function for correctness."""
    M, K = 1001, 543
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
    sparse_matvec_avx512(bcoo16, x, bias, y_sparse)
    
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
    print("✅ AVX-512 kernel large irregular non-square test passed.")

def generate_sparse_matrix(shape, sparsity=0.95):
    """Generate a random sparse matrix in BCOO-16 format."""
    dense_matrix = np.random.rand(*shape)
    mask = np.random.rand(*shape) > sparsity
    dense_matrix[mask] = 0
    return dense_matrix

def benchmark_sparse_matvec_avx512():
    """Benchmark the sparse_matvec_avx512 function."""
    M, K = 2000, 2000
    dense_matrix = generate_sparse_matrix((M, K), sparsity=0.999999)
    bcoo16 = BCOO16()
    bcoo16 = sparseops_backend.encode_to_bcoo16(dense_matrix)
    x = np.random.rand(K).astype(np.float32)
    bias = np.random.rand(M).astype(np.float32)
    y = np.empty(M, dtype=np.float32)
    
    # Measure runtime
    import time
    start_time = time.time()
    for _ in range(1000):
        sparse_matvec_avx512(bcoo16, x, bias, y)
    end_time = time.time()

    print(f"Average runtime: {(end_time - start_time):.6f} milliseconds per operation")
    print("✅ Benchmarking completed.")


if __name__ == "__main__":
    test_sparse_matvec_avx512()
    test_avx512_single_block()
    test_avx512_two_blocks_noncontiguous()
    test_avx512_zero_mask_skipped()

    test_avx512_col_edge_case()

    test_avx512_single_block_sparse()
    test_avx512_multiple_blocks_one_row()
    test_avx512_multirow_multiblock()
    test_avx512_dense_block()
    test_avx512_diagonal_matrix()

    test_avx512_correctness_fixed_seed()
    test_avx512_correctness_half_sparse()
    test_avx512_correctness_dense_matrix()

    test_avx512_kernel_correctness()
    test_avx512_kernel_large()
    test_avx512_kernel_large_irregular_square()
    test_avx512_kernel_large_irregular_non_square()

    benchmark_sparse_matvec_avx512()
    print("✅ All AVX-512 kernel tests passed.")
import numpy as np
import sparseops_backend
from sparseops_backend import BCOO16, sparse_matvec_avx512_mt
import os

num_threads = int(os.environ.get("OMP_NUM_THREADS", "1"))

def test_avx512_kernel_correctness():
    M, K = 64, 64
    dense = np.random.rand(M, K).astype(np.float32)
    dense[dense < 0.9] = 0  # 90% sparse

    x = np.random.rand(K).astype(np.float32)
    b = np.random.rand(M).astype(np.float32)
    # b = np.zeros(M).astype(np.float32)  # Ensure bias is zero for simplicity

    bcoo = sparseops_backend.encode_to_bcoo16(dense)

    y = np.empty(M, dtype=np.float32)

    y = sparseops_backend.sparse_matvec_avx512_mt(bcoo, x, b, num_threads)

    y_expected = dense @ x + b
    print("y_expected:", y_expected)
    print("y:", y)
    print(f"y_expected sum: {np.sum(y_expected)}")
    print(f"y sum: {np.sum(y)}")
    print("number of correct elements:", np.sum(np.isclose(y, y_expected, atol=1e-4)))
    assert np.allclose(y, y_expected, atol=1e-4), "Mismatch in AVX-512 output"

    print("✅ AVX-512 kernel correctness test passed.")


if __name__ == "__main__":

    test_avx512_kernel_correctness()
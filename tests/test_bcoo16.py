# test_bcoo16.py
# Test suite for BCOO-16 encoder and decoder.

import numpy as np
import sparseops_backend

def test_encode_decode():
    # Create a random sparse matrix
    dense_matrix = np.random.rand(100, 100)
    dense_matrix[dense_matrix < 0.95] = 0  # Make it sparse

    # Encode to BCOO-16
    bcoo16 = sparseops_backend.encode_to_bcoo16(dense_matrix)

    # Decode back to dense
    decoded_matrix = sparseops_backend.decode_from_bcoo16(bcoo16)

    # Validate correctness
    assert np.allclose(dense_matrix, decoded_matrix), "Decoded matrix does not match original"

if __name__ == "__main__":
    test_encode_decode()
    print("All tests passed!")

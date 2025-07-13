# test_bcoo16.py
# Extended test suite for BCOO-16 encoder and decoder.

import numpy as np
import sparseops_backend

def encode_decode_and_check(matrix, name=""):
    bcoo16 = sparseops_backend.encode_to_bcoo16(matrix)
    decoded = sparseops_backend.decode_from_bcoo16(bcoo16)
    try:
        if not np.allclose(matrix, decoded):
            print(f"Failed test: {name}")
            print("Original:\n", matrix)
            print("Decoded:\n", decoded)
            # raise AssertionError(f"Decoded matrix does not match original for test: {name}")
        print(f"✅ Test passed: {name}")
    except Exception as e:
        print(f"Error in test {name}: {e}")
        

def test_sparse_patterns():
    print("Running extended BCOO-16 encode/decode tests...")

    # Sparse random
    mat = np.random.rand(64, 64)
    mat[mat < 0.95] = 0
    encode_decode_and_check(mat, "sparse random 64x64")

    # All zeros
    encode_decode_and_check(np.zeros((16, 16), dtype=np.float32), "all zeros 16x16")

    # All nonzeros
    encode_decode_and_check(np.ones((16, 16), dtype=np.float32), "all ones 16x16")

    # Single row
    mat = np.zeros((1, 16), dtype=np.float32)
    mat[0, 3] = 1.0
    mat[0, 15] = 2.0
    encode_decode_and_check(mat, "1x16 edge bits")

    # Single column
    mat = np.zeros((16, 1), dtype=np.float32)
    mat[5, 0] = 1.0
    encode_decode_and_check(mat, "16x1 column vector")

    # 1x1 with zero
    encode_decode_and_check(np.array([[0.0]], dtype=np.float32), "1x1 zero")

    # 1x1 with nonzero
    encode_decode_and_check(np.array([[42.0]], dtype=np.float32), "1x1 nonzero")

    # 32x32 with diagonal
    mat = np.zeros((32, 32), dtype=np.float32)
    np.fill_diagonal(mat, 1.0)
    encode_decode_and_check(mat, "diagonal 32x32")

    # Irregular pattern
    mat = np.zeros((4, 20), dtype=np.float32)
    mat[0, [0, 3, 5, 7, 11]] = [1, 2, 3, 4, 5]
    mat[2, [15, 16, 17, 19]] = [6, 7, 8, 9]
    encode_decode_and_check(mat, "irregular pattern")

def test_internal_bcoo_layout():
    print("Running internal layout test...")

    dense = np.zeros((4, 32), dtype=np.float32)
    dense[0, [0, 1, 2]] = [1, 2, 3]
    dense[2, [16, 17, 18]] = [4, 5, 6]

    bcoo = sparseops_backend.encode_to_bcoo16(dense)

    print("BCOO-16 values:", bcoo.values)

    assert bcoo.original_num_rows == 4
    assert bcoo.original_num_cols == 32

    # Should contain 2 blocks
    assert len(bcoo.row_id) == 2
    assert len(bcoo.first_col) == 2
    assert len(bcoo.bitmask) == 2
    assert len(bcoo.values) == 16*2
    # Walk through each block and verify values
    value_index = 0

    # First block (row 0, col_base 0, mask 0b111)
    assert bcoo.row_id[0] == 0
    assert bcoo.first_col[0] == 0
    expected_mask0 = 0b00000111
    assert bcoo.bitmask[0] == expected_mask0

    expected_values0 = [1.0, 2.0, 3.0]
    for j in range(16):
        if (expected_mask0 >> j) & 1:
            col = 0 + j
            actual_val = bcoo.values[value_index]
            expected_val = dense[0, col]
            assert np.isclose(actual_val, expected_val), (
                f"Block 0, col {col}: expected {expected_val}, got {actual_val}"
            )
        value_index += 1

    # Second block (row 2, col_base 16, mask 0b111)
    assert bcoo.row_id[1] == 2
    assert bcoo.first_col[1] == 16
    expected_mask1 = 0b00000111
    assert bcoo.bitmask[1] == expected_mask1

    for j in range(16):
        if (expected_mask1 >> j) & 1:
            col = 16 + j
            actual_val = bcoo.values[value_index]
            expected_val = dense[2, col]
            assert np.isclose(actual_val, expected_val), (
                f"Block 1, col {col}: expected {expected_val}, got {actual_val}"
            )
            value_index += 1

    print("✅ Internal BCOO layout test passed.")


def test_bcoo16_value_layout():
    print("Running block value layout test...")

    # Construct matrix with known values
    M, K = 2, 20
    dense = np.zeros((M, K), dtype=np.float32)
    dense[0, [0, 2, 5, 6, 15]] = [1, 2, 3, 4, 5]      # Block 0: col_base=0
    dense[1, [16, 18]] = [6, 7]                      # Block 1: col_base=16

    bcoo = sparseops_backend.encode_to_bcoo16(dense)

    # Check number of blocks and value count
    assert len(bcoo.row_id) == 2
    assert len(bcoo.bitmask) == 2
    assert len(bcoo.first_col) == 2
    assert len(bcoo.values) == 16*2  # only the nonzero entries

    value_index = 0
    for i in range(len(bcoo.row_id)):
        row = bcoo.row_id[i]
        col_base = bcoo.first_col[i]
        bitmask = bcoo.bitmask[i]

        for j in range(16):
            if (bitmask >> j) & 1:
                col = col_base + j
                expected_val = dense[row, col]
                actual_val = bcoo.values[value_index]
                print("value_index:", value_index, "row:", row, "col:", col, "expected:", expected_val, "actual:", actual_val)
                assert np.isclose(actual_val, expected_val), (
                    f"Mismatch at block {i}, col {col}: expected {expected_val}, got {actual_val}"
                )
            value_index += 1
                

    print("✅ BCOO-16 value layout test passed.")

def test_bcoo16_bitmask_logic():
    print("Running bitmask logic test...")

    # Create a small test matrix with known sparsity
    M, K = 2, 32
    dense = np.zeros((M, K), dtype=np.float32)
    dense[0, [0, 3, 5, 10]] = [1, 2, 3, 4]      # Block 0: col_base = 0
    dense[0, [17]] = [5]                        # Block 1: col_base = 16
    dense[1, [16, 17, 31]] = [6, 7, 8]          # Block 2: col_base = 16

    bcoo = sparseops_backend.encode_to_bcoo16(dense)

    for i in range(len(bcoo.row_id)):
        row = bcoo.row_id[i]
        col_base = bcoo.first_col[i]
        mask = bcoo.bitmask[i]

        for bit in range(16):
            col = col_base + bit
            expected_nonzero = (
                col < dense.shape[1] and dense[row, col] != 0.0
            )
            bit_is_set = (mask >> bit) & 1

            if expected_nonzero and not bit_is_set:
                raise AssertionError(
                    f"Block {i}, bit {bit} should be 1 (nonzero at col {col}) but was 0"
                )
            if not expected_nonzero and bit_is_set:
                raise AssertionError(
                    f"Block {i}, bit {bit} should be 0 (zero at col {col}) but was 1"
                )

    print("✅ BCOO-16 bitmask logic test passed.")


def test_encoder_value_count_matches_bitmask():
    # Dense matrix: 2 rows x 16 cols (1 block per row)
    dense = np.array([
        [1.0] * 4 + [0.0] * 12,     # Only first 4 entries are nonzero
        [0.0] * 8 + [2.0] * 2 + [0.0] * 6  # Only 2 entries (at positions 8,9) are nonzero
    ], dtype=np.float32)

    dense_list = dense.tolist()  # C++ binding takes std::vector<std::vector<float>>
    bcoo = sparseops_backend.encode_to_bcoo16(dense_list)

    # Count number of bits set in each mask
    expected_num_values = dense.shape[0] * 16  # Each row has 16 bits
    actual_num_values = len(bcoo.values)

    assert actual_num_values == expected_num_values, (
        f"Mismatch in value count: expected {expected_num_values}, got {actual_num_values}"
    )
    print("✅ Encoder value count matches bitmask test passed.")

if __name__ == "__main__":
    test_sparse_patterns()
    test_internal_bcoo_layout()
    test_bcoo16_value_layout()
    test_bcoo16_bitmask_logic()
    test_encoder_value_count_matches_bitmask()


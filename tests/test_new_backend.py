#!/usr/bin/env python3
# test_new_backend.py

import sys
import os

# Add the build directory to sys.path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'build'))

try:
    import sparseops_backend
    print("✅ Successfully imported sparseops_backend")
    
    # Test basic functionality
    import numpy as np
    import scipy.sparse as sp
    
    print("\n=== Testing basic functionality ===")
    
    # Create a simple test matrix
    M, N = 100, 50
    density = 0.1
    
    # Generate sparse matrix
    A_dense = np.random.randn(M, N).astype(np.float32)
    mask = np.random.rand(M, N) > (1 - density)
    A_dense *= mask
    A_csr = sp.csr_matrix(A_dense)
    
    print(f"Created {M}x{N} matrix with {A_csr.nnz} non-zeros ({A_csr.nnz/(M*N)*100:.1f}% density)")
    
    # Test prepare_csr
    A_prepared = sparseops_backend.prepare_csr(A_csr.indptr, A_csr.indices, A_csr.data, M, N)
    print(f"Prepared matrix: {A_prepared.rows()}x{A_prepared.cols()}, sparsity={A_prepared.sparsity():.2%}")
    
    # Test sgemm
    K = 20
    B = np.random.randn(N, K).astype(np.float32)
    C = np.zeros((M, K), dtype=np.float32)
    
    sparseops_backend.sgemm(A_prepared, B, C)
    print(f"SGEMM completed: output shape {C.shape}")
    
    # Verify correctness
    C_reference = A_dense @ B
    error = np.abs(C - C_reference).max()
    print(f"Maximum error vs reference: {error}")
    
    if error < 1e-4:
        print("✅ Basic functionality test PASSED")
    else:
        print("❌ Basic functionality test FAILED")
        
except ImportError as e:
    print(f"❌ Failed to import sparseops_backend: {e}")
    sys.exit(1)
except Exception as e:
    print(f"❌ Test failed with error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

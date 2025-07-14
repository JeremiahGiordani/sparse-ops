"""
Profile and time sparseops_backend SpMV.

Usage examples
--------------
# single-thread, 2000×2000, 90 % sparse
python tests/profile_spmv.py --m 2000 --k 2000 --sparsity 0.90 --threads 1

# 8 threads, 2000×2000, 99.9 % sparse, 30 repetitions
python tests/profile_spmv.py -m 2000 -k 2000 -p 0.999 -t 8 -r 30
"""
import argparse, os, time, numpy as np
import sparseops_backend as so

def gen_sparse_matrix(shape, sparsity):
    mat = np.random.rand(*shape).astype(np.float32)
    mat[np.random.rand(*shape) < sparsity] = 0.0
    return mat

def ms_per_op(fn, reps=10):
    fn()                                 # warm-up (compiles if needed)
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    return (time.perf_counter() - t0) * 1e3 / reps

def run(m, k, sparsity, threads, reps, use_dense=False):
    dense = gen_sparse_matrix((m, k), sparsity)
    bcoo  = so.encode_to_bcoo16(dense)
    x     = np.random.rand(k).astype(np.float32)
    bias  = np.random.rand(m).astype(np.float32)

    if use_dense:
        # use dense matrix-vector multiplication
        fn = lambda: so.dense_block_kernel(dense, x, bias)
    else:
        fn = lambda: so.sparse_matvec_avx512_mt(bcoo, x, bias, threads)

    ms = ms_per_op(fn, reps)
    print(f"{m}×{k}, sparsity {sparsity*100:.1f} %, "
          f"{threads} thr → {ms:.3f} ms/op over {reps} reps")

    # correctness check (1-shot)
    if m <= 512:     # avoid huge dense @
        y_expected = dense @ x + bias
        y_spmv    = fn()
        assert np.allclose(y_spmv, y_expected, atol=1e-4)
        print("   ✓ correctness verified (≤512 rows)")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--m",        type=int, default=2000)
    ap.add_argument("-k", "--k",        type=int, default=2000)
    ap.add_argument("-p", "--sparsity", type=float, default=0.90,
                    help="fraction of zeros (e.g. 0.90 = 90 % sparse)")
    ap.add_argument("-t", "--threads",  type=int, default=8)
    ap.add_argument("-r", "--reps",     type=int, default=10)
    ap.add_argument("-d", "--dense",    type=bool, default=False)
    args = ap.parse_args()

    # show compile / hit lines from jit_cache
    # os.environ["SPARSEOPS_VERBOSE"] = "1"

    run(args.m, args.k, args.sparsity, args.threads, args.reps, args.dense)

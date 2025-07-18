import os, ctypes, numpy as np, pathlib, sparseops_backend as so

os.environ["PROFILE_MASKS"] = "1"          # pass macro to the JIT compiler

# ---- build a 90 % sparse test matrix ---------------------------------
M = K = 2000
dense = np.random.rand(M, K).astype(np.float32)
dense[dense < 0.90] = 0
bcoo  = so.encode_to_bcoo16(dense)
x     = np.random.rand(K).astype(np.float32)
y     = np.empty(M, dtype=np.float32)

# ---- run ONE SpMV  (this compiles the kernel & increments counters) ---
so.sparse_matvec_avx512_mt(bcoo, x, np.zeros_like(y), 1)

# ---- locate the .so just compiled (newest in cache dir) ---------------
cache = pathlib.Path.home() / ".cache" / "sparseops"
jit_so = max(cache.glob("*.so"), key=lambda p: p.stat().st_mtime)
print("reading counters from", jit_so)

lib   = ctypes.CDLL(str(jit_so), mode=os.RTLD_NOLOAD)  # attach to same copy
Arr   = ctypes.c_uint64 * 65536
hits  = Arr.in_dll(lib, "hotHits")
vals  = np.frombuffer(hits, dtype=np.uint64)

live  = vals.nonzero()[0]
print("unique masks:", len(live))
for m in sorted(live, key=lambda m: -vals[m])[:20]:
    print(hex(m), int(vals[m]))

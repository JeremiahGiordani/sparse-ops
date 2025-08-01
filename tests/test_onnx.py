# test_onnx.py
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import onnxruntime as ort

# import the new C++-backed Sparse ONNX model
from sparseops_backend import SparseOnnxModel

# ────────────────────────────────────────────────────────────────
#  Helper: average runtime over N calls
# ────────────────────────────────────────────────────────────────
def average_runtime(func, n_runs: int = 100):
    total = 0.0
    for _ in range(n_runs):
        t0 = time.perf_counter()
        func()
        t1 = time.perf_counter()
        total += (t1 - t0)
    return total / n_runs

# ────────────────────────────────────────────────────────────────
#  Define a simple 2-layer MLP in PyTorch and prune it
# ────────────────────────────────────────────────────────────────
class M(nn.Module):
    def __init__(self, fc_1_in=8, fc_1_out=16, fc_2_in=16, fc_2_out=4):
        super().__init__()
        self.fc1 = nn.Linear(fc_1_in, fc_1_out)
        self.fc2 = nn.Linear(fc_2_in, fc_2_out)
    def forward(self, x):
        return torch.relu(self.fc2(torch.relu(self.fc1(x))))

# ────────────────────────────────────────────────────────────────
#  Configuration
# ────────────────────────────────────────────────────────────────
FC_1_IN, FC_1_OUT = 1000, 1000
FC_2_IN, FC_2_OUT = FC_1_OUT, 120
INPUT_DIM       = FC_1_IN
SPARSITY        = 0.9
N_RUNS          = 100
SEED            = 42

BATCH_DIM       = 50

# ────────────────────────────────────────────────────────────────
#  Prepare and export ONNX model
# ────────────────────────────────────────────────────────────────
torch.manual_seed(SEED)
m = M(fc_1_in=FC_1_IN,
      fc_1_out=FC_1_OUT,
      fc_2_in=FC_2_IN,
      fc_2_out=FC_2_OUT).eval()

# dummy input for export
dummy = torch.randn(BATCH_DIM, INPUT_DIM, dtype=torch.float32)

# apply random unstructured pruning to each weight matrix
prune.random_unstructured(m.fc1, name="weight", amount=SPARSITY)
prune.random_unstructured(m.fc2, name="weight", amount=SPARSITY)

# export the pruned model
onnx_path = "test_fc.onnx"
torch.onnx.export(m, dummy, onnx_path, opset_version=14)

# Limit threading
num_threads = int(os.environ.get("OMP_NUM_THREADS", "1"))
torch.set_num_threads(num_threads)

os.environ["OMP_NUM_THREADS"]        = str(num_threads)
os.environ["OPENBLAS_NUM_THREADS"]   = str(num_threads)
os.environ["MKL_NUM_THREADS"]        = str(num_threads)
sess_opts = ort.SessionOptions()
sess_opts.intra_op_num_threads  = num_threads
sess_opts.inter_op_num_threads  = num_threads

# ────────────────────────────────────────────────────────────────
#  Set up inference sessions
# ────────────────────────────────────────────────────────────────
# 1) Custom C++ sparse-ELLPACK backend
model = SparseOnnxModel(onnx_path)

# 2) ONNX Runtime session
session = ort.InferenceSession(onnx_path, sess_options=sess_opts)
input_name = session.get_inputs()[0].name


# ────────────────────────────────────────────────────────────────
#  Prepare input vectors
# ────────────────────────────────────────────────────────────────
# 1D host vector
x = np.random.randn(BATCH_DIM, INPUT_DIM).astype(np.float32)

# for ONNX Runtime: shape (1, INPUT_DIM)
x_onnx = x

# for custom model: shape (INPUT_DIM, 1)
x_custom = x.T

# ────────────────────────────────────────────────────────────────
#  Define runner functions
# ────────────────────────────────────────────────────────────────
def torch_run():
    # input shape (BATCH_DIM, INPUT_DIM)
    with torch.no_grad():
        return m(torch.from_numpy(x))

def custom_run():
    # returns shape (FC_2_OUT, 1) → reshape to (BATCH_DIM, FC_2_OUT)
    Y = model.run(x_custom)
    return Y.reshape(BATCH_DIM, FC_2_OUT)

def onnx_run():
    return session.run(None, {input_name: x_onnx})[0]  # returns (BATCH_DIM, FC_2_OUT)

# warm up
for _ in range(10):
    torch_run(); custom_run(); onnx_run()

# ────────────────────────────────────────────────────────────────
#  Correctness check
# ────────────────────────────────────────────────────────────────
y_ref   = torch_run().numpy()          # shape (BATCH_DIM, FC_2_OUT)
y_sp    = custom_run()                 # shape (BATCH_DIM, FC_2_OUT)
y_onnx  = onnx_run()                   # shape (BATCH_DIM, FC_2_OUT)

print("=== Verifying correctness ===")
print("Torch vs ONNX Runtime:  ", np.allclose(y_ref,   y_onnx,  atol=1e-4))
print("Torch vs SparseModel:   ", np.allclose(y_ref,   y_sp,    atol=1e-4))


# ────────────────────────────────────────────────────────────────
#  Benchmarking
# ────────────────────────────────────────────────────────────────
print("\n=== Benchmarking over", N_RUNS, "runs ===")
print(f"Dimensions: FC1 {FC_1_IN}→{FC_1_OUT}, FC2 {FC_2_IN}→{FC_2_OUT}, Sparsity {SPARSITY:.1%}")
print(f"Batch Dim: {BATCH_DIM}")
print(f"[PyTorch]       {average_runtime(torch_run,   N_RUNS)*1000:.3f} ms")
print(f"[Sparse Model]  {average_runtime(custom_run,  N_RUNS)*1000:.3f} ms")
print(f"[ONNX Runtime]  {average_runtime(onnx_run,    N_RUNS)*1000:.3f} ms")

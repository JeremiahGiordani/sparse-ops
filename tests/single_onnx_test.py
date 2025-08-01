# test_onnx.py
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import torch.nn.functional as F

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
    def __init__(self, fc_in=8, fc_out=16, num_layers=10):
        super().__init__()
        layers = []
        for i in range(num_layers):
            # first layer goes fc_in→fc_out, all others fc_out→fc_out
            in_f  = fc_in  if i == 0 else fc_out
            out_f = fc_out
            layers.append(nn.Linear(in_f, out_f))
        # register them properly
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        for layer in self.layers:
            x = F.relu(layer(x))
        return x
# ────────────────────────────────────────────────────────────────
#  Configuration
# ────────────────────────────────────────────────────────────────
FC_IN, FC_OUT = 1000, 1000
INPUT_DIM       = FC_IN
NUM_LAYERS      = 1
SPARSITY        = 0.90
N_RUNS          = 100
SEED            = 42

BATCH_DIM       = 64

# ────────────────────────────────────────────────────────────────
#  Prepare and export ONNX model
# ────────────────────────────────────────────────────────────────
torch.manual_seed(SEED)
m = M(fc_in=FC_IN,
      fc_out=FC_OUT,
      num_layers=NUM_LAYERS).eval()

# dummy input for export
dummy = torch.randn(BATCH_DIM, INPUT_DIM, dtype=torch.float32)

# apply random unstructured pruning to each weight matrix
for layer in m.layers:
    prune.random_unstructured(layer, name="weight", amount=SPARSITY)

# export the pruned model
onnx_path = "test_fc.onnx"
torch.onnx.export(m, dummy, onnx_path, opset_version=14)

# Limit threading
num_threads = int(os.environ.get("OMP_NUM_THREADS", "1"))
torch.set_num_threads(num_threads)

os.environ["OMP_NUM_THREADS"]        = str(num_threads)

# ────────────────────────────────────────────────────────────────
#  Set up inference sessions
# ────────────────────────────────────────────────────────────────
# 1) Custom C++ sparse-ELLPACK backend
model = SparseOnnxModel(onnx_path)


# ────────────────────────────────────────────────────────────────
#  Prepare input vectors
# ────────────────────────────────────────────────────────────────
# 1D host vector
x = np.random.randn(BATCH_DIM, INPUT_DIM).astype(np.float32)


# for custom model: shape (INPUT_DIM, 1)
x_custom = x.T

# ────────────────────────────────────────────────────────────────
#  Define runner functions
# ────────────────────────────────────────────────────────────────

def custom_run():
    # returns shape (FC_2_OUT, 1) → reshape to (BATCH_DIM, FC_2_OUT)
    return model.run(x_custom)

# warm up
for _ in range(10):
    custom_run()

# ────────────────────────────────────────────────────────────────
#  Correctness check
# ────────────────────────────────────────────────────────────────
y_sp    = custom_run()                 # shape (BATCH_DIM, FC_2_OUT)

# ────────────────────────────────────────────────────────────────
#  Benchmarking
# ────────────────────────────────────────────────────────────────
print("\n=== Benchmarking over", N_RUNS, "runs ===")
print(f"Dimensions: FC {FC_IN}→{FC_OUT}, NUM_LAYERS {NUM_LAYERS}, Sparsity {SPARSITY:.1%}")
print(f"Batch Dim: {BATCH_DIM}")
print(f"[Sparse Model]  {average_runtime(custom_run,  N_RUNS)*1000:.3f} ms")


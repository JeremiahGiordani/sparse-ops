import numpy as np
from python.sparse_onnx import OnnxSparseModel
from onnx import numpy_helper
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import onnxruntime as ort
import time
import os

class M(nn.Module):
    def __init__(self, fc_1_in=8, fc_1_out=16, fc_2_in=16, fc_2_out=4): 
        super().__init__()
        self.fc1 = nn.Linear(fc_1_in, fc_1_out)
        self.fc2 = nn.Linear(fc_2_in, fc_2_out)
    def forward(self, x):
        return torch.relu(self.fc2(torch.relu(self.fc1(x))))
    

FC_1_IN  = 1000
FC_1_OUT = 1000
FC_2_IN  = FC_1_OUT
FC_2_OUT = 4
INPUT_DIM  = FC_1_IN
SPARSITY   = 0.5
N_RUNS     = 100   # <-- number of repetitions
SEED       = 42

def average_runtime(func, n_runs=N_RUNS):
    total = 0.0
    for _ in range(n_runs):
        t0 = time.perf_counter()
        func()
        t1 = time.perf_counter()
        total += t1 - t0
    return total / n_runs


torch.manual_seed(SEED)
m = M(fc_1_in=FC_1_IN, fc_1_out=FC_1_OUT, fc_2_in=FC_2_IN, fc_2_out=FC_2_OUT).eval()
dummy = torch.randn(1, INPUT_DIM, dtype=torch.float32)
# Prune the model layers
prune.random_unstructured(m.fc1, name="weight", amount=SPARSITY)
prune.random_unstructured(m.fc2, name="weight", amount=SPARSITY)
# Save the pruned model
torch.onnx.export(m, dummy, "test_fc.onnx", opset_version=14)

num_threads = int(os.environ.get("OMP_NUM_THREADS", "1"))
torch.set_num_threads(num_threads)

model = OnnxSparseModel("test_fc.onnx")

session = ort.InferenceSession("test_fc.onnx")
input_name = session.get_inputs()[0].name


x = np.random.randn(INPUT_DIM).astype(np.float32)
x_t = torch.from_numpy(x)
x_onnx = x.reshape(1, -1)  # ONNX expects batch dimension

def torch_run():
    return m(x_t)

def custom_run():
    return model.run(x)

def onnx_run():
    return session.run(None, {input_name: x_onnx})

for _ in range(500):
    custom_run(); torch_run(); onnx_run()

y_ref = torch_run().detach().numpy()
y_sp  = custom_run()
y_onnx = onnx_run()[0]

print("=== Verifying correctness ===")
print("Torch vs SM :", np.allclose(y_ref, y_sp, atol=1e-4))
print("Torch vs ONNX:", np.allclose(y_ref, y_onnx, atol=1e-4))

print("=== Benchmarking ===")
print(f"=== Average runtime over {N_RUNS} runs ===")
print(f"=== FC1 dims {FC_1_IN}x{FC_1_OUT}, FC2 dims {FC_2_IN}x{FC_2_OUT} ===")
print(f"=== Sparsity {SPARSITY:.2%} ===")
print(f"[PyTorch]      {average_runtime(torch_run)*1000:.3f} ms")
print(f"[Sparse Model] {average_runtime(custom_run)*1000:.3f} ms")
print(f"[ONNX]        {average_runtime(onnx_run)*1000:.3f} ms")
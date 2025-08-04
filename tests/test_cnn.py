import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import onnxruntime as ort

from sparseops_backend import SparseOnnxModel

def average_runtime(func, n_runs: int = 100):
    total = 0.0
    for _ in range(n_runs):
        t0 = time.perf_counter()
        func()
        t1 = time.perf_counter()
        total += (t1 - t0)
    return total / n_runs

# ────────────────────────────────────────────────────────────────
# CNN → FC → ReLU Model
# ────────────────────────────────────────────────────────────────
class CNNFC(nn.Module):
    def __init__(self, in_channels=3, conv_out=16, fc_out=12, img_size=32):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, conv_out, kernel_size=3, padding=1)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(conv_out * img_size * img_size, fc_out)

    def forward(self, x):
        x = self.conv(x)
        x = self.flatten(x)
        x = self.fc(x)
        return torch.relu(x)

# ────────────────────────────────────────────────────────────────
# Config
# ────────────────────────────────────────────────────────────────
IMG_SIZE     = 32
BATCH_DIM    = 1
IN_CHANNELS  = 3
CONV_OUT     = 16
FC_OUT       = 12
SPARSITY     = 0.90
N_RUNS       = 1
SEED         = 42

torch.manual_seed(SEED)
np.random.seed(SEED)
model = CNNFC(in_channels=IN_CHANNELS, conv_out=CONV_OUT, fc_out=FC_OUT, img_size=IMG_SIZE).eval()
dummy = torch.randn(BATCH_DIM, IN_CHANNELS, IMG_SIZE, IMG_SIZE)

with torch.no_grad():
    if model.conv.bias is not None:
        model.conv.bias.zero_()
    if model.fc.bias is not None:
        model.fc.bias.zero_()

# Apply pruning
prune.random_unstructured(model.conv, name="weight", amount=SPARSITY)
prune.random_unstructured(model.fc,   name="weight", amount=SPARSITY)

onnx_path = "test_cnn_fc.onnx"
torch.onnx.export(model, dummy, onnx_path, opset_version=14)

# ────────────────────────────────────────────────────────────────
# ONNX + Sparse Setup
# ────────────────────────────────────────────────────────────────
num_threads = int(os.environ.get("OMP_NUM_THREADS", "1"))
torch.set_num_threads(num_threads)
os.environ["OMP_NUM_THREADS"] = str(num_threads)
os.environ["OPENBLAS_NUM_THREADS"] = str(num_threads)
os.environ["MKL_NUM_THREADS"] = str(num_threads)

sess_opts = ort.SessionOptions()
sess_opts.intra_op_num_threads = num_threads
sess_opts.inter_op_num_threads = num_threads

session = ort.InferenceSession(
    onnx_path, sess_options=sess_opts, providers=["CPUExecutionProvider"]
)
input_name = session.get_inputs()[0].name
custom_model = SparseOnnxModel(onnx_path)



# ────────────────────────────────────────────────────────────────
# Input: NHWC style for ONNX
# ────────────────────────────────────────────────────────────────
x = np.random.randn(BATCH_DIM, IN_CHANNELS, IMG_SIZE, IMG_SIZE).astype(np.float32)
x_onnx = x
x_t = torch.from_numpy(x)


# ────────────────────────────────────────────────────────────────
# Runners
# ────────────────────────────────────────────────────────────────
def torch_run():
    with torch.no_grad():
        return model(x_t).numpy()

def custom_run():
    return custom_model.run(x)

def onnx_run():
    return session.run(None, {input_name: x_onnx})[0]

for _ in range(0):
    torch_run(); custom_run(); onnx_run()

# ────────────────────────────────────────────────────────────────
# Correctness
# ────────────────────────────────────────────────────────────────
y_ref   = torch_run()
y_sp    = custom_run()
y_onnx  = onnx_run()

print("torch reference")
print(y_ref)
print("="*50)
print("Sparse output:")
print(y_sp.T)
print(y_sp.shape)

print("=== Verifying correctness ===")
print("Torch vs ONNX Runtime:  ", np.allclose(y_ref, y_onnx, atol=1e-4))
print("Torch vs SparseModel:   ", np.allclose(y_ref, y_sp.T, atol=1e-4))

# ────────────────────────────────────────────────────────────────
# Benchmark
# ────────────────────────────────────────────────────────────────
print("\n=== Benchmarking over", N_RUNS, "runs ===")
print(f"Conv → FC → ReLU | Input: {IN_CHANNELS}×{IMG_SIZE}×{IMG_SIZE}, Conv out: {CONV_OUT}, FC out: {FC_OUT}")
print(f"Sparsity: {SPARSITY:.1%}, Batch: {BATCH_DIM}")
print(f"[PyTorch]       {average_runtime(torch_run,   N_RUNS)*1000:.3f} ms")
print(f"[Sparse Model]  {average_runtime(custom_run,  N_RUNS)*1000:.3f} ms")
print(f"[ONNX Runtime]  {average_runtime(onnx_run,    N_RUNS)*1000:.3f} ms")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import time

import sparseops_backend

BATCH_DIM   = 8
IN_CHANNELS = 9
CONV_OUT    = 5
IMG_SIZE    = 32
KERNEL_SIZE = 3
PADDING     = 1
STRIDE      = 1
BIAS        = False
SEED = 42

SPARSITY   = 0.90
N_RUNS     = 100

torch.manual_seed(SEED)
np.random.seed(SEED)


def average_runtime(func, n_runs: int = 100):
    total = 0.0
    for _ in range(n_runs):
        t0 = time.perf_counter()
        func()
        t1 = time.perf_counter()
        total += (t1 - t0)
    return total / n_runs

# ---- Torch reference
conv = nn.Conv2d(IN_CHANNELS, CONV_OUT, kernel_size=KERNEL_SIZE, padding=PADDING, stride=STRIDE, bias=BIAS)

prune.random_unstructured(conv, name="weight", amount=SPARSITY)

input = torch.randn(BATCH_DIM, IN_CHANNELS, IMG_SIZE, IMG_SIZE, dtype=torch.float32)
with torch.no_grad():
    output_ref = conv(input)
print("Output ref shape:", tuple(output_ref.shape))

# ---- NumPy views
conv_weight_np = conv.weight.detach().cpu().numpy().astype(np.float32)    # (Cout,Cin,kH,kW)
input_np       = input.detach().cpu().numpy().astype(np.float32)          # (B,Cin,H,W)


non_zero_count = np.count_nonzero(conv_weight_np)
total_elements = conv_weight_np.size
print(f"Sparsity of weight = {1 - non_zero_count/total_elements}")

# Fortran input: batch-fast
input_f = np.asfortranarray(input_np)  # (B,Cin,H,W)_F

plan   = sparseops_backend.setup_conv(conv_weight_np, stride=STRIDE, padding=PADDING)
output = sparseops_backend.conv2d_ellpack(plan, input_f)  # (B,Cout,Hout,Wout)_F

def torch_run():
    with torch.no_grad():
        return conv(input)

def custom_run():
    return sparseops_backend.conv2d_ellpack(plan, input_f)

for _ in range(10):
    torch_run(); custom_run()

print(f"Output ref shape {output_ref.shape}")
print(f"Output shape {output.shape}")
# print("="*50)
# print(output_ref)
# print("="*50)
# print(output)

print("Torch vs Conv:   ", np.allclose(output_ref.detach().cpu(), output, atol=1e-4))

print("\n=== Benchmarking over", N_RUNS, "runs ===")
print(f" Sparsity {SPARSITY:.1%}")
print(f"Batch Dim: {BATCH_DIM}")
print(f"[PyTorch]       {average_runtime(torch_run,   N_RUNS)*1000:.3f} ms")
print(f"[Sparse Model]  {average_runtime(custom_run,  N_RUNS)*1000:.3f} ms")

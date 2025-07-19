#!/usr/bin/env python3
# test_matmul_avg.py
#
# Measure average runtime of several matrix-vector multiplications
# over N_RUNS iterations.

import os
import time
import numpy as np
import scipy.sparse as sp
import torch
import tensorflow as tf  # <- Added

from python.cpp_backend import run_matvec, decode_bcoo16, convert_to_bcoo16, run_sparse_matvec
from python.utils import to_csr

import onnx
import tempfile
import torch.nn as nn
from openvino.runtime import Core

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

num_threads = int(os.environ.get("OMP_NUM_THREADS", "1"))

# ----------------------------------------------------------------------
# Experiment parameters
# ----------------------------------------------------------------------
INPUT_DIM  = 2000
OUTPUT_DIM = 2000
SPARSITY   = 0.9
N_RUNS     = 100
SEED       = 42
# ----------------------------------------------------------------------


class MatMulModel(nn.Module):
    def __init__(self, weight, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(weight))
        self.bias = nn.Parameter(torch.tensor(bias))

    def forward(self, x):
        return torch.matmul(self.weight, x) + self.bias

def export_onnx_model(weight, bias, input_shape):
    model = MatMulModel(weight, bias)
    dummy_input = torch.randn(input_shape)
    onnx_path = tempfile.NamedTemporaryFile(suffix=".onnx", delete=False).name
    torch.onnx.export(model, dummy_input, onnx_path, input_names=["input"], output_names=["output"], opset_version=13)
    return onnx_path

def apply_block_sparsity(weight, block_rows=4, block_cols=4, sparsity=SPARSITY):
    keep_prob = 1 - sparsity  # Convert to probability of keeping a block
    weight = weight.copy()
    for i in range(0, weight.shape[0], block_rows):
        for j in range(0, weight.shape[1], block_cols):
            if np.random.rand() > keep_prob:
                weight[i:i+block_rows, j:j+block_cols] = 0
    return weight


# ----------------------------------------------------------------------
# Data generation
# ----------------------------------------------------------------------
np.random.seed(SEED)
torch.manual_seed(SEED)

weight = np.random.randn(OUTPUT_DIM, INPUT_DIM).astype(np.float32)
# mask   = np.random.rand(*weight.shape) > SPARSITY
# weight *= mask

weight = apply_block_sparsity(weight)

bias       = np.random.randn(OUTPUT_DIM).astype(np.float32)
input_vec  = np.random.randn(INPUT_DIM).astype(np.float32)

# Torch
weight_t   = torch.tensor(weight)
bias_t     = torch.tensor(bias)
input_t    = torch.tensor(input_vec)

# Numpy
weight_np = weight.astype(np.float32)
bias_np   = bias.astype(np.float32)
input_np  = input_vec.astype(np.float32)

# TensorFlow
weight_tf = tf.constant(weight_np)
bias_tf   = tf.constant(bias_np)
input_tf  = tf.constant(input_np)

# SciPy sparse (CSR)
csr_mat    = sp.csr_matrix(weight)
_, _, _    = to_csr(torch.tensor(weight))     # pre-compute CSR if needed

# tvm_func, A_tvm, x_tvm, b_tvm, y_tvm = tvm_setup(weight_np, input_np, bias_np)

coo_indices = torch.nonzero(weight_t).t()
coo_values = weight_t[coo_indices[0], coo_indices[1]]
sparse_tensor = torch.sparse_coo_tensor(coo_indices, coo_values, size=weight_t.shape)

bcoo_16 = convert_to_bcoo16(weight_np)


# ----------------------------------------------------------------------
# Utility
# ----------------------------------------------------------------------
def timed_avg(fn, n=N_RUNS):
    total = 0.0
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        t1 = time.perf_counter()
        total += (t1 - t0)
    return total / n

# ----------------------------------------------------------------------
# Backend Functions
# ----------------------------------------------------------------------
def torch_run():
    return torch.matmul(weight_t, input_t) + bias_t

def numpy_run():
    return weight @ input_vec + bias

def scipy_run():
    return csr_mat.dot(input_vec) + bias

def custom_run():
    return run_matvec(weight_np, bias_np, input_np)

def custom_sparse_run():
    return run_sparse_matvec(bcoo_16, bias_np, input_np, threads=num_threads)

def tensorflow_run():
    return tf.linalg.matvec(weight_tf, input_tf) + bias_tf

def openvino_run(n_runs=N_RUNS):
    onnx_path = export_onnx_model(weight, bias, (INPUT_DIM,))
    core = Core()
    model = core.read_model(onnx_path)
    compiled = core.compile_model(model, "CPU")
    infer_request = compiled.create_infer_request()
    input_tensor = input_vec.astype(np.float32)

    # Warmup
    infer_request.infer({0: input_tensor})

    # Benchmark
    start = time.perf_counter()
    for _ in range(n_runs):
        infer_request.infer({0: input_tensor})
    end = time.perf_counter()

    avg_ms = (end - start) * 1000 / n_runs
    output = infer_request.get_output_tensor().data
    return output, avg_ms

# def tvm_run(n_runs=N_RUNS):
#     start = time.perf_counter()
#     for _ in range(n_runs):
#         tvm_func(A_tvm, x_tvm, b_tvm, y_tvm)
#     end = time.perf_counter()
#     avg_ms = (end - start) * 1000 / n_runs
#     return y_tvm.numpy(), avg_ms

def torch_sparse_run(n_runs=N_RUNS):
    unsqueezed_input = input_t.unsqueeze(1)  # Convert to column vector
    start = time.perf_counter()
    for _ in range(n_runs):
        result = torch.sparse.mm(sparse_tensor, unsqueezed_input)
    end = time.perf_counter()
    avg_ms = (end - start) * 1000 / n_runs
    result = result.squeeze() + bias_t  # Squeeze to remove the extra dimension
    return result, avg_ms

# ----------------------------------------------------------------------
# Correctness
# ----------------------------------------------------------------------
print("=== Verifying correctness ===")
out_torch = torch_run().numpy()
print("Torch vs NumPy :", np.allclose(out_torch, numpy_run(), atol=1e-4))
print("Torch vs SciPy :", np.allclose(out_torch, scipy_run(), atol=1e-4))
print("Torch vs Custom:", np.allclose(out_torch, custom_run(), atol=1e-4))
print("Torch vs Custom Sparse:", np.allclose(out_torch, custom_sparse_run(), atol=1e-4))
print("Torch vs TF    :", np.allclose(out_torch, tensorflow_run().numpy(), atol=1e-4))
print("Torch vs OpenVINO:", np.allclose(out_torch, openvino_run(n_runs=1)[0], atol=1e-4))
# print("Torch vs TVM   :", np.allclose(out_torch, tvm_run(n_runs=1)[0], atol=1e-4))
print("Torch vs Torch Sparse:", np.allclose(out_torch, torch_sparse_run(n_runs=1)[0], atol=1e-4))
print()

# ----------------------------------------------------------------------
# Benchmark
# ----------------------------------------------------------------------
print(f"=== Average runtime over {N_RUNS:,} runs ===")
print(f"[PyTorch]      {timed_avg(torch_run)*1000:.3f} ms")
print(f"[NumPy]        {timed_avg(numpy_run)*1000:.3f} ms")
print(f"[SciPy Sparse] {timed_avg(scipy_run)*1000:.3f} ms")
print(f"[Custom]       {timed_avg(custom_run)*1000:.3f} ms")
print(f"[Custom Sparse]{timed_avg(custom_sparse_run)*1000:.3f} ms")
print(f"[TensorFlow]   {timed_avg(tensorflow_run)*1000:.3f} ms")
print(f"[OpenVINO]      {openvino_run()[1]:.3f} ms")
# print(f"[TVM]          {tvm_run()[1]:.3f} ms")
print(f"[Torch Sparse] {torch_sparse_run()[1]:.3f} ms")

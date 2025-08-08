import os, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import onnxruntime as ort

from sparseops_backend import SparseOnnxModel

def average_runtime(fn, n=50):
    # warmup
    for _ in range(5): fn()
    t0 = time.perf_counter()
    for _ in range(n): fn()
    return (time.perf_counter() - t0) / n

class CNNBothPoolsAdd(nn.Module):
    def __init__(self, in_channels=3, c1=16, c2=16, fc_out=12):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, c1, kernel_size=3, padding=1, bias=False)
        self.pool  = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(c1, c2, kernel_size=3, padding=1, bias=False)
        self.skip  = nn.Conv2d(c1, c2, kernel_size=1, bias=False)   # shape-match for Add
        self.gap   = nn.AdaptiveAvgPool2d((1, 1))                   # → GlobalAveragePool
        self.flatten = nn.Flatten()
        self.fc    = nn.Linear(c2, fc_out, bias=False)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        p = self.pool(x)
        y = F.relu(self.conv2(p))
        s = self.skip(p)
        x = y + s                    # <-- Add (same shape)
        x = self.gap(x)              # GlobalAveragePool
        x = self.flatten(x)
        x = self.fc(x)
        return F.relu(x)

if __name__ == "__main__":
    torch.manual_seed(0); np.random.seed(0)

    # Config
    BATCH, IMG, IN_CH, C1, C2, FC_OUT = 2, 16, 3, 16, 16, 12
    OMP = int(os.environ.get("OMP_NUM_THREADS", "1"))
    torch.set_num_threads(OMP)
    os.environ["OMP_NUM_THREADS"] = str(OMP)
    os.environ["MKL_NUM_THREADS"] = str(OMP)
    os.environ["OPENBLAS_NUM_THREADS"] = str(OMP)

    model = CNNBothPoolsAdd(IN_CH, C1, C2, FC_OUT).eval()
    x_t   = torch.randn(BATCH, IN_CH, IMG, IMG, dtype=torch.float32)

    # Torch reference
    with torch.no_grad():
        y_ref = model(x_t).cpu().numpy()

    # Export ONNX
    onnx_path = "cnn_both_pools_add.onnx"
    torch.onnx.export(model, x_t, onnx_path, opset_version=14, do_constant_folding=True)

    # ONNX Runtime
    sess_opts = ort.SessionOptions()
    sess_opts.intra_op_num_threads = OMP
    sess_opts.inter_op_num_threads = OMP
    sess = ort.InferenceSession(onnx_path, sess_options=sess_opts, providers=["CPUExecutionProvider"])
    x_np = x_t.cpu().numpy().astype(np.float32)
    y_onnx = sess.run(None, {sess.get_inputs()[0].name: x_np})[0]

    # Your backend (expects Fortran B-fast)
    x_f = np.asfortranarray(x_np)
    sp = SparseOnnxModel(onnx_path)
    y_sp = sp.run(x_f)

    print("Shapes:")
    print(" Torch:", y_ref.shape)
    print(" ONNX :", y_onnx.shape)
    print(" Sparse:", y_sp.shape)
    print("Correctness:")
    print(" Torch vs ONNX  :", np.allclose(y_ref, y_onnx, atol=1e-4))
    print(" Torch vs Sparse:", np.allclose(y_ref, y_sp,   atol=1e-4))

    # Bench (optional)
    N = 50
    print("\nBenchmark ({} runs, OMP_NUM_THREADS={}):".format(N, OMP))
    t_torch = average_runtime(lambda: model(x_t).detach().cpu().numpy(), N) * 1000
    t_onnx  = average_runtime(lambda: sess.run(None, {sess.get_inputs()[0].name: x_np})[0], N) * 1000
    t_sp    = average_runtime(lambda: sp.run(x_f), N) * 1000
    print(" Torch         : {:6.3f} ms".format(t_torch))
    print(" ONNX Runtime  : {:6.3f} ms".format(t_onnx))
    print(" Sparse backend: {:6.3f} ms".format(t_sp))

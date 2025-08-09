#!/usr/bin/env python3
import argparse, os, time
import numpy as np
import onnxruntime as ort
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import torchvision.models as tvm

from sparseops_backend import SparseOnnxModel

def set_threads(n):
    torch.set_num_threads(n)
    os.environ["OMP_NUM_THREADS"] = str(n)
    os.environ["MKL_NUM_THREADS"] = str(n)
    os.environ["OPENBLAS_NUM_THREADS"] = str(n)

def avg_time(fn, runs=30, warmup=5):
    for _ in range(warmup):
        fn()
    t0 = time.perf_counter()
    for _ in range(runs):
        fn()
    return (time.perf_counter() - t0) / runs

def apply_unstructured_pruning(model: nn.Module, amount: float):
    if amount <= 0.0:
        return
    # Apply module-wise pruning, then bake masks into weights.
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            prune.random_unstructured(m, name="weight", amount=amount)
            prune.remove(m, "weight")  # make pruning permanent

def measure_sparsity(model: nn.Module):
    total_elems = 0
    zero_elems = 0
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            W = m.weight.detach()
            total_elems += W.numel()
            zero_elems += (W == 0).sum().item()
    return (zero_elems, total_elems, (zero_elems / max(1,total_elems)))

def export_resnet18_to_onnx(path, img_size=224, batch=1, pretrained=False, sparsity=0.0, seed=0):
    torch.manual_seed(seed)
    np.random.seed(seed)

    # torchvision >= 0.13 weights enum handling
    weights = None
    if pretrained:
        try:
            weights = tvm.ResNet18_Weights.DEFAULT
        except Exception:
            weights = None

    model = tvm.resnet18(weights=weights).eval()

    if sparsity > 0:
        apply_unstructured_pruning(model, sparsity)
        zeros, total, frac = measure_sparsity(model)
        print(f"[prune] Applied unstructured pruning: "
              f"{zeros}/{total} zeros ({frac:.1%})")

    dummy = torch.randn(batch, 3, img_size, img_size, dtype=torch.float32)
    torch.onnx.export(
        model, dummy, path, opset_version=14, do_constant_folding=True,
        input_names=["input"], output_names=["output"],
        dynamic_axes=None  # fixed-batch graph
    )
    return model  # return Torch model so we can compute a reference

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", type=str, default="resnet18.onnx",
                    help="Path to ONNX file (created if --export)")
    ap.add_argument("--export", action="store_true",
                    help="Export ResNet18 from torchvision to --onnx first")
    ap.add_argument("--pretrained", action="store_true",
                    help="Use pretrained ImageNet weights for export")
    ap.add_argument("--sparsity", type=float, default=0.0,
                    help="Fraction (0..1) of weights to prune in Conv/FC during export")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--img", type=int, default=224)
    ap.add_argument("--runs", type=int, default=30)
    ap.add_argument("--threads", type=int, default=int(os.environ.get("OMP_NUM_THREADS", "1")))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    set_threads(args.threads)
    torch_model = None

    if args.export:
        print(f"[export] writing {args.onnx} (pretrained={args.pretrained}, sparsity={args.sparsity})…")
        torch_model = export_resnet18_to_onnx(
            args.onnx, img_size=args.img, batch=args.batch,
            pretrained=args.pretrained, sparsity=args.sparsity, seed=args.seed
        )

    # Inputs
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    x = torch.randn(args.batch, 3, args.img, args.img, dtype=torch.float32)
    x_np = x.cpu().numpy().astype(np.float32)
    x_f = np.asfortranarray(x_np)  # your backend expects B-fast Fortran

    # Torch reference only if we exported (so weights match)
    if torch_model is None:
        try:
            torch_model = tvm.resnet18(weights=None).eval()
            print("[warn] Torch reference is randomly initialized; it will not match ONNX unless you exported here.")
        except Exception:
            torch_model = None

    y_torch = None
    if torch_model is not None and args.export:
        with torch.no_grad():
            y_torch = torch_model(x).cpu().numpy()

    # ONNX Runtime
    sess_opts = ort.SessionOptions()
    sess_opts.intra_op_num_threads = args.threads
    sess_opts.inter_op_num_threads = args.threads
    sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    ort_sess = ort.InferenceSession(args.onnx, sess_options=sess_opts, providers=["CPUExecutionProvider"])

    # Your backend
    sp = SparseOnnxModel(args.onnx)

    # Correctness
    y_onnx = ort_sess.run(None, {ort_sess.get_inputs()[0].name: x_np})[0]
    y_sparse = sp.run(x_f)

    print("\nShapes:")
    print(" torch :", None if y_torch is None else y_torch.shape)
    print(" onnx  :", y_onnx.shape)
    print(" sparse:", y_sparse.shape)

    if y_torch is not None and args.export:
        print("\nCorrectness vs Torch:")
        print(" sparse:", np.allclose(y_torch, y_sparse, atol=1e-4))
        print(" onnx  :", np.allclose(y_torch, y_onnx,  atol=1e-4))
    else:
        print("\nCorrectness (ONNX vs Sparse):", np.allclose(y_onnx, y_sparse, atol=1e-4))

    # Benchmarks
    def run_torch():
        with torch.no_grad():
            torch_model(x).cpu().numpy()

    def run_onnx():
        ort_sess.run(None, {ort_sess.get_inputs()[0].name: x_np})[0]

    def run_sparse():
        sp.run(x_f)

    print(f"\nBenchmark (runs={args.runs}, threads={args.threads}, batch={args.batch}, img={args.img}, sparsity={args.sparsity})")
    if torch_model is not None and args.export:
        t_torch = avg_time(run_torch, runs=args.runs) * 1000
        print(f" Torch         : {t_torch:7.3f} ms")
    t_onnx  = avg_time(run_onnx,  runs=args.runs) * 1000
    t_sparse= avg_time(run_sparse, runs=args.runs) * 1000
    print(f" ONNX Runtime  : {t_onnx:7.3f} ms")
    print(f" Sparse backend: {t_sparse:7.3f} ms")

if __name__ == "__main__":
    main()

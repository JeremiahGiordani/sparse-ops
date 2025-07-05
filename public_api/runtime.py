# runtime.py

import os
import tempfile
import numpy as np
import ctypes
from pathlib import Path
from numpy.ctypeslib import ndpointer
import torch
from .export import export_sparse_model

import ctypes
import platform
from pathlib import Path

libname = "libultrasparse.dylib" if platform.system() == "Darwin" else "libultrasparse.so"
lib_path = Path(__file__).parent.parent / "build" / libname

class SparseRuntime:
    def __init__(self, pt_model_path: str):
        # Create a temp dir to hold the exported files
        self.outdir = Path("data")
        print(f"Outdir {self.outdir}")

        # Generate model.json, .npz weights, and input placeholder
        dummy_input = torch.randn(1, 8)  # assume input_dim = 8 for now
        export_sparse_model(pt_model_path, dummy_input[0], self.outdir)

        # Load the shared C++ library
        self.lib = ctypes.CDLL(str(lib_path))

        # Define function signature
        self.lib.run_inference.argtypes = [
            ctypes.c_char_p,  # model.json
            ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),  # input
            ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),  # output
            ctypes.c_int     # input size
        ]
        self.lib.run_inference.restype = None

        # Load model.json to get input/output size
        import json
        with open(self.outdir / "model.json") as f:
            self.model_spec = json.load(f)
        self.input_dim = self.model_spec["layers"][0]["input_dim"]
        self.output_dim = self.model_spec["layers"][-1]["output_dim"]

    def run(self, input_vec: np.ndarray) -> np.ndarray:
        print(f"Input shape {input_vec.shape}")
        assert input_vec.shape == (self.input_dim,), f"Expected input shape ({self.input_dim},)"
        input_np = input_vec.astype(np.float32).copy()
        output_np = np.zeros((self.output_dim,), dtype=np.float32)

        self.lib.run_inference(
            str(self.outdir / "model.json").encode("utf-8"),
            input_np,
            output_np,
            self.input_dim
        )

        return output_np

    def __del__(self):
        pass

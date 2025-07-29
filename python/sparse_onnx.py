import onnx
import numpy as np
from python.cpp_backend import encode, matvec, matmul
from onnx import numpy_helper

class OnnxSparseModel:
    def __init__(self, onnx_path: str):
        # 1. Load and check graph
        model = onnx.load(onnx_path)
        graph = model.graph

        # 2. Map initializer name → numpy array
        self._initializers = {
            init.name:  # shape: Tuple[int,...]
              numpy_helper.to_array(init).astype(np.float32)
            for init in graph.initializer
        }

        # 3. Build execution plan: a list of (op, E, bias, attrs)
        self.layers = []
        for node in graph.node:
            if node.op_type in ('Gemm','MatMul'):
                W = self._initializers[node.input[1]]    # weight
                b = None
                if node.op_type=='Gemm' and len(node.input)>2:
                    b = self._initializers[node.input[2]]
                E = encode(W)                             # pack
                self.layers.append(('matmul', E, b, {}))

            elif node.op_type in ('Relu','Sigmoid','Tanh'):
                self.layers.append((node.op_type.lower(), None, None, {}))

            else:
                raise NotImplementedError(f"Unsupported op {node.op_type}")

    def run(self, x: np.ndarray) -> np.ndarray:
        """Single‑batch inference: x shape must match first layer’s input dim."""
        v = x.astype(np.float32)
        for op, E, b, attrs in self.layers:
            if op == 'matmul':
                # if v is 1D, use matvec; else matmul
                if v.ndim == 1:
                    v = matvec(E, v, b)
                else:
                    # v: [n, C] → output [m, C]
                    v = matmul(E, v, b)
            elif op == 'relu':
                v = np.maximum(v, 0, out=v)
            elif op == 'sigmoid':
                v = 1/(1 + np.exp(-v))
            elif op == 'tanh':
                v = np.tanh(v)
        return v

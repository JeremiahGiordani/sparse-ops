import torch
from .cpp_backend import run_sparse_matvec
from .utils import is_pruned


class SparseRuntime:
    def __init__(self, pt_model_path: str):
        self.model = torch.load(pt_model_path, map_location="cpu")
        self.model.eval()

    def run(self, input_tensor: torch.Tensor) -> torch.Tensor:
        x = input_tensor.detach().cpu()

        for name, layer in self.model.named_modules():
            if name == "":  # Skip top-level module
                continue

            print(f"Name {name}")
            print(f"Is pruned {isinstance(layer, torch.nn.Linear) and is_pruned(layer)}")

            if isinstance(layer, torch.nn.Linear) and is_pruned(layer):
                W = layer.weight.detach().cpu()
                B = layer.bias.detach().cpu()
                print("RUNNING SPARSE MATVEC")
                x = run_sparse_matvec(W, B, x)
            elif isinstance(layer, torch.nn.ReLU):
                x = torch.relu(x)
            elif isinstance(layer, torch.nn.Linear):
                x = layer(x)
            else:
                raise NotImplementedError(f"Unsupported layer type: {type(layer)}")

        return x

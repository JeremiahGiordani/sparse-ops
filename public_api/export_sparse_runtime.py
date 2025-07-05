# export_sparse_runtime.py

import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import numpy as np
import os
import json

# Convert a dense weight tensor to CSR format
def to_csr(tensor):
    tensor = tensor.detach().cpu().numpy()
    indptr = [0]
    indices = []
    values = []
    for row in tensor:
        for j, val in enumerate(row):
            if val != 0:
                indices.append(j)
                values.append(val)
        indptr.append(len(indices))
    return np.array(values, dtype=np.float32), np.array(indices, dtype=np.int32), np.array(indptr, dtype=np.int32)

# Load a pruned PyTorch model and export sparse layers
def export_sparse_model(pt_path, input_vec, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    print("CALLING EXPORT")

    state_dict = torch.load(pt_path, map_location='cpu')
    layers = []

    for i, (name, weight) in enumerate(state_dict.items()):
        print(f"NAME {name}")
        if 'weight' in name:
            base_name = name.rsplit('.', 1)[0]  # remove ".weight"
            bias_name = base_name + ".bias"
            bias = state_dict.get(bias_name, torch.zeros(weight.size(0)))

            values, indices, indptr = to_csr(weight)
            layer_path = f"layer{i}.npz"
            np.savez(os.path.join(output_dir, layer_path), values=values, indices=indices, indptr=indptr, bias=bias.detach().cpu().numpy().astype(np.float32))

            in_dim = weight.size(1)
            out_dim = weight.size(0)
            layers.append({
                "type": "SparseLinear",
                "input_dim": in_dim,
                "output_dim": out_dim,
                "path": layer_path
            })
        elif 'relu' in name.lower():
            layers.append({"type": "ReLU"})

    with open(os.path.join(output_dir, "model.json"), 'w') as f:
        json.dump({"layers": layers}, f, indent=2)

    # Save the input vector for testing
    input_vec = input_vec.detach().cpu().numpy().astype(np.float32)
    np.save(os.path.join(output_dir, "input.npy"), input_vec)

    print("✅ Export complete: model.json, weights, and input saved.")

# Example usage for development:
if __name__ == '__main__':
    class SimpleMLP(nn.Module):
        def __init__(self, input_dim, hidden_dim, output_dim):
            super().__init__()
            self.fc1 = nn.Linear(input_dim, hidden_dim)
            self.relu = nn.ReLU()
            self.fc2 = nn.Linear(hidden_dim, output_dim)

        def forward(self, x):
            return self.fc2(self.relu(self.fc1(x)))

    model = SimpleMLP(8, 16, 4)
    prune.l1_unstructured(model.fc1, name="weight", amount=0.9)
    prune.l1_unstructured(model.fc2, name="weight", amount=0.9)
    torch.save(model.state_dict(), "example_model.pt")

    example_input = torch.randn(8)
    export_sparse_model("example_model.pt", example_input, "../data")

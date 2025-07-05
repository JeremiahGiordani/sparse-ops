# export.py

import torch
import numpy as np
import os
import json

# Convert a dense tensor to CSR (Compressed Sparse Row) format
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
    return (
        np.array(values, dtype=np.float32),
        np.array(indices, dtype=np.int32),
        np.array(indptr, dtype=np.int32),
    )

# Export model weights to JSON + .npz files for runtime

def export_sparse_model(pt_path, input_vec, output_dir="../data"):
    os.makedirs(output_dir, exist_ok=True)

    model = torch.load(pt_path, map_location="cpu")
    model.eval()
    layers = []
    layer_idx = 0

    for name, layer in model.named_modules():
        print(f"NAME {name}")
        # Skip top-level model itself
        if name == "":
            continue

        # Only handle leaf modules
        if isinstance(layer, torch.nn.Linear):
            W = layer.weight
            B = layer.bias
            values, indices, indptr = to_csr(W)
            layer_path = f"layer{layer_idx}.npz"
            np.savez(output_dir / layer_path, values=values, indices=indices, indptr=indptr, bias=B.detach().numpy())

            layers.append({
                "type": "SparseLinear",
                "input_dim": W.shape[1],
                "output_dim": W.shape[0],
                "path": layer_path
            })
            layer_idx += 1

        elif isinstance(layer, torch.nn.ReLU):
            layers.append({"type": "ReLU"})

    if layers[-1]["type"] == "ReLU":
        layers.pop()  # remove final ReLU if not needed

    with open(os.path.join(output_dir, "model.json"), "w") as f:
        json.dump({"layers": layers}, f, indent=2)

    np.save(os.path.join(output_dir, "input.npy"), input_vec.detach().cpu().numpy().astype(np.float32))

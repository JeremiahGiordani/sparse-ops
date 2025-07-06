# test_runtime.py

import numpy as np
import torch
import time
import torch.nn.utils.prune as prune

from public_api.runtime import SparseRuntime

# Create a dummy MLP model and export its pruned state_dict
class TestMLP(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(8, 16)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(16, 4)

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))

model = TestMLP()
prune.l1_unstructured(model.fc1, name="weight", amount=0.9)
prune.l1_unstructured(model.fc2, name="weight", amount=0.9)
torch.save(model, "test_model.pt")

# Generate a random input
input_tensor = torch.randn(8)

# Use ultrasparse runtime to execute the model
start_sparse = time.perf_counter()
runtime = SparseRuntime("test_model.pt")
output_vec = runtime.run(input_tensor.detach().numpy())
end_sparse = time.perf_counter()

# Compare to PyTorch output
model.eval()
with torch.no_grad():
    start_torch = time.perf_counter()
    output_ref = model(input_tensor).detach().numpy()
    end_torch = time.perf_counter()

print("Output from SparseRuntime:", output_vec)
print("Reference from PyTorch:", output_ref)
print(f"⏱ SparseRuntime time: {(end_sparse - start_sparse) * 1000:.3f} ms")
print(f"⏱ PyTorch time: {(end_torch - start_torch) * 1000:.3f} ms")

if np.allclose(output_vec, output_ref, atol=1e-3):
    print("✅ Test passed: outputs match")
else:
    print("❌ Test failed: outputs do not match")
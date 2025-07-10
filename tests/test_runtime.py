# test_runtime.py

import torch
import numpy as np
import time
from python.runtime import SparseRuntime
import warnings

warnings.filterwarnings(
    "ignore",
    message="You are using `torch.load` with `weights_only=False`.*",
    category=FutureWarning
)

import torch.nn.utils.prune as prune

class TestMLP(torch.nn.Module):
    def __init__(self, multiplier=1):
        super().__init__()
        self.fc1 = torch.nn.Linear(8*multiplier, 16*multiplier)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(16*multiplier, 4*multiplier)

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))

# Define a simple model with one pruned linear layer and ReLU
def create_test_model(multiplier=1, amount=0.8):

    model = TestMLP(multiplier)
    prune.l1_unstructured(model.fc1, name="weight", amount=amount)
    prune.l1_unstructured(model.fc2, name="weight", amount=amount)
    return model

# Create and save model
multiplier = 1
amount=0.9
model = create_test_model(multiplier, amount)
torch.save(model, "test_model.pt")

# Generate input
input_tensor = torch.randn(8*multiplier)

# Dry run
model = torch.load("test_model.pt", weights_only=False)
model.eval()
with torch.no_grad():
    model(input_tensor).detach().numpy()

# Run using SparseRuntime
runtime = SparseRuntime("test_model.pt")
t1 = time.perf_counter()
output = runtime.run(input_tensor)
t2 = time.perf_counter()
# print("Output from SparseRuntime:", output.detach().numpy())
print(f"⏱ SparseRuntime time: {(t2 - t1) * 1000:.3f} ms")

# Compare with PyTorch baseline
model = torch.load("test_model.pt", weights_only=False)
model.eval()
with torch.no_grad():
    t3 = time.perf_counter()
    reference = model(input_tensor).detach().numpy()
    t4 = time.perf_counter()




# print("Reference from PyTorch:", reference)
print(f"⏱ PyTorch time: {(t4 - t3) * 1000:.3f} ms")

# Validate correctness
if np.allclose(output.detach().numpy(), reference, atol=1e-3):
    print("✅ Test passed: outputs match")
else:
    print("❌ Test failed: outputs do not match")
import numpy as np
from python.sparse_onnx import OnnxSparseModel
from onnx import numpy_helper
import torch
import torch.nn as nn

class M(nn.Module):
    def __init__(self): 
        super().__init__()
        self.fc1 = nn.Linear(8, 16)
        self.fc2 = nn.Linear(16, 4)
    def forward(self, x):
        return torch.relu(self.fc2(torch.relu(self.fc1(x))))
    

# 
m = torch.load("test_fc.pt", weights_only=False)
# m = M().eval()
# dummy = torch.randn(1,8)
# torch.onnx.export(m, dummy, "test_fc.onnx", opset_version=14)

model = OnnxSparseModel("test_fc.onnx")
x = np.random.randn(8).astype(np.float32)
y_ref = m(torch.from_numpy(x)).detach().numpy()
y_sp  = model.run(x)
print("Reference output:", y_ref)
print("Sparse ONNX output:", y_sp)
print("Torch vs SM :", np.allclose(y_ref, y_sp, atol=1e-4))

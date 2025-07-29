# create_model.py
import torch
import torch.nn as nn

class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(8, 16)
        self.fc2 = nn.Linear(16, 4)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return torch.relu(self.fc2(x))

if __name__ == "__main__":
    # 1) Instantiate and export to ONNX
    model = M().eval()
    dummy_in = torch.randn(1, 8, dtype=torch.float32)
    torch.onnx.export(
        model,
        dummy_in,
        "test_fc.onnx",
        input_names=["input"],
        output_names=["output"],
        opset_version=14,
    )
    print("✅ Wrote test_fc.onnx")

    # 2) Save the full PyTorch model for reference
    torch.save(model, "test_fc.pt")
    print("✅ Wrote test_fc.pt")

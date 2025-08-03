# create_resnet_model.py
import torch
import torchvision.models as models

if __name__ == "__main__":
    # 1) Instantiate and export ResNet-18 to ONNX
    model = models.resnet18(pretrained=False).eval()
    dummy_in = torch.randn(1, 3, 224, 224, dtype=torch.float32)
    torch.onnx.export(
        model,
        dummy_in,
        "resnet18.onnx",
        input_names=["input"],
        output_names=["output"],
        opset_version=14,
    )
    print("✅ Wrote resnet18.onnx")

    # 2) Save the full PyTorch model for reference
    torch.save(model, "resnet18.pt")
    print("✅ Wrote resnet18.pt")

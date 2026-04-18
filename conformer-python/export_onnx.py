import torch
import sys
sys.path.insert(0, "src")
from conformer_python.conformer import Conformer

model = Conformer(
    dim=144,
    depth=16,
    heads=4,
    conv_kernel_size=32,
)

model.eval()

x = torch.randn(1, 50, 144)

torch.onnx.export(
    model,
    x,
    "../conformer-rs/model.onnx",
    input_names=["input"],
    output_names=["output"],
)

print("Exported to ../conformer-rs/model.onnx")

with torch.no_grad():
    out = model(x)
    print(f"PyTorch output shape: {out.shape}")
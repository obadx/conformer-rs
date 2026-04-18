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

output_path = "../conformer-rs/model.onnx"

torch.onnx.export(
    model,
    x,
    output_path,
    input_names=["input"],
    output_names=["output"],
    export_params=True,
)

print(f"Exported to {output_path}")

from onnxsim import simplify
import onnx

m = onnx.load(output_path)
m_simp, check = simplify(m)
onnx.save(m_simp, output_path)
print(f"Simplified (check passed: {check})")

with torch.no_grad():
    out = model(x)
    print(f"PyTorch output shape: {out.shape}")
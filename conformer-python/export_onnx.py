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

print("Exporting to ONNX...")
torch.onnx.export(
    model,
    x,
    output_path,
    input_names=["input"],
    output_names=["output"],
    export_params=True,
)

print(f"Exported to {output_path}")

print("Simplifying ONNX (embedding weights)...")
from onnxsim import simplify
import onnx

m = onnx.load(output_path)
m_simp, check = simplify(m)
onnx.save(m_simp, output_path)
print(f"Simplified (check passed: {check})")

print("Converting to NNEF for fast mobile loading...")
import subprocess
subprocess.run([
    "tract", "model.onnx", "dump", "--nnef-dir", "model.nnef.d"
], check=True)
print("NNEF created: model.nnef.d/")

with torch.no_grad():
    out = model(x)
    print(f"PyTorch output shape: {out.shape}")
    print("\nDone! Files created:")
    print("  - model.onnx  (50MB, single file, slow load ~30s)")
    print("  - model.nnef.d/  (50MB, 25 files, fast load ~0.3s)")
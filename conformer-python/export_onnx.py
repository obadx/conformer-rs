import torch
import numpy as np
import os

MODEL_PATH = "/../quran_world/muaalm_quran/conformer-rs/conformer-python/tiny_muaalem.pt"
ONNX_PATH = "/../quran_world/muaalm_quran/conformer-rs/conformer-python/tiny_muaalem.onnx"

print("Loading PyTorch model...")
model = torch.jit.load(MODEL_PATH)
model.eval()

# Infer input shapes from the first graph node
code = str(model.graph)
print("\n=== Model Graph (first 500 chars) ===")
print(code[:500])

# Try to find input dimensions from graph
import re
matches = re.findall(r'Float\([^)]*\)', code)
print("\nFound tensor signatures:", matches[:5])

# Create dummy inputs based on expected conformer inputs
# tiny_muaalem typically uses: [batch, seq_len, 80] for audio features
# Let's try common shapes
batch_size = 1
seq_len = 100
n_mels = 80

dummy_input = torch.randn(batch_size, seq_len, n_mels)

print(f"\nExporting to ONNX with input shape: {dummy_input.shape}")
try:
    torch.onnx.export(
        model,
        dummy_input,
        ONNX_PATH,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch_size", 1: "seq_len"},
            "output": {0: "batch_size", 1: "seq_len"},
        },
        opset_version=17,
    )
    print(f"ONNX model saved to: {ONNX_PATH}")
    print(f"Size: {os.path.getsize(ONNX_PATH) / 1024 / 1024:.2f} MB")
except Exception as e:
    print(f"Export failed: {e}")
    # Try with different input shape or multiple inputs
    print("\nTrying with 2D input [batch, seq_len * n_mels]...")
    dummy_input_2d = torch.randn(batch_size, seq_len * n_mels)
    try:
        torch.onnx.export(
            model,
            dummy_input_2d,
            ONNX_PATH,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={
                "input": {0: "batch_size"},
                "output": {0: "batch_size"},
            },
            opset_version=17,
        )
        print(f"ONNX model saved to: {ONNX_PATH}")
        print(f"Size: {os.path.getsize(ONNX_PATH) / 1024 / 1024:.2f} MB")
    except Exception as e2:
        print(f"2D export also failed: {e2}")

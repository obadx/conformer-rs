import onnx
import numpy as np
import onnxruntime

# Load ONNX model
model = onnx.load("../conformer-rs/model.onnx")
onnx.checker.check_model(model)
print("ONNX model loaded and verified successfully")
# Run inference
sess = onnxruntime.InferenceSession("../conformer-rs/model.onnx")
input_name = sess.get_inputs()[0].name
output_name = sess.get_outputs()[0].name
x = np.random.randn(1, 50, 144).astype(np.float32)
result = sess.run([output_name], {input_name: x})
print(f"Input shape: {x.shape}")
print(f"Output shape: {result[0].shape}")

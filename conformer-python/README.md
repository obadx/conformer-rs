# Muaalem ONNX Conversion & Benchmark (Python)

## Setup
```bash
# Using uv (not python)
uv add onnx onnxruntime onnxsim torch transformers
# or just use the existing pyproject.toml
uv sync
```

## Export Models
```bash
uv run python convert_muaalem_to_onnx.py
```

## Run Benchmark (Linux)
```bash
uv run python run_muaalem_onnx.py
```

### Results (Linux x86_64, Python/onnxruntime)

| Model    | Load (ms) | Inference (ms) | Notes |
|----------|-----------|----------------|-------|
| float32  | 43.9     | 8.73±0.37     | ✅ Works |
| float16  | 57.3     | 10.12±0.89   | ✅ Works |
| int8     | 38.6     | 25.89±0.17   | ✅ Works |

## Model Files

Generated files are saved to `conformer-rs/models/`:

| File | Size | Description |
|------|------|-------------|
| `tiny_muaalem_float32.onnx` | 22 MB | Full precision |
| `tiny_muaalem_float16.onnx` | 11 MB | Half precision |
| `tiny_muaalem_int8.onnx` | 8.1 MB | Quantized (int8 weights) |

## Notes

- **float16**: Works on x86_64, requires float16 input
- **int8**: Uses hybrid quantization (MatMul/Conv → int8, LayerNorm → fp32)
- Model outputs 11 levels: phonemes + 10 sifa attributes
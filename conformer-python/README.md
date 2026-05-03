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

| Model   | Load (ms) | Inference (ms) | Notes   |
| ------- | --------- | -------------- | ------- |
| float32 | 43.9      | 8.73±0.37      | ✅ Works |
| float16 | 57.3      | 10.12±0.89     | ✅ Works |
| int8    | 38.6      | 25.89±0.17     | ✅ Works |

## Model Files

Generated files are saved to `conformer-rs/models/`:

| File                        | Size   | Description              |
| --------------------------- | ------ | ------------------------ |
| `tiny_muaalem_float32.onnx` | 22 MB  | Full precision           |
| `tiny_muaalem_float16.onnx` | 11 MB  | Half precision           |
| `tiny_muaalem_int8.onnx`    | 8.1 MB | Quantized (int8 weights) |

## Notes

- **float16**: Works on x86_64, requires float16 input
- **int8**: Uses hybrid quantization (MatMul/Conv → int8, LayerNorm → fp32)
- Model outputs 11 levels: phonemes + 10 sifa attributes

## LiteRT Suppport

### The Full Cycle

#### Why `.tflite`?

PyTorch models can't run directly on Android. `.tflite` is LiteRT's native format — it's a compact, optimized flatbuffer that the LiteRT runtime can execute efficiently on CPU, GPU, and NPU on Android devices.

#### The Pipeline

```
HuggingFace (PyTorch weights) 
        ↓
[convert_tiny_to_tflite.py]
        ↓
tiny_muaalem_float32.tflite  (21MB)
tiny_muaalem_int8.tflite     (6.9MB)
tiny_muaalem_int4.tflite     (4.7MB)
        ↓
[benchmark_litert.py]          ← test on your laptop (CPU)
[adb push + benchmark APK]     ← test on real Android device (CPU/NNAPI)
        ↓
[compile_for_npu.py]           ← upload to QAI Hub, compile for Qualcomm NPU
        ↓
tiny_muaalem_int8_qnn.tflite   ← NPU-optimized model
        ↓
[profile_npu.py]               ← run on real Qualcomm device remotely
        ↓
[download_profile_results.py]  ← download JSON results
        ↓
[parse_profile_results.py]     ← human-readable summary
```

## Learned Lessons

### Why we should use quantize to tflite from float32 not used prequantized version

`litert_torch.convert()` uses `torch.export` internally, which traces the PyTorch model graph.

When you try to convert an **already quantized** PyTorch model (via PT2E), the quantized ops (`uniform_dequantize`, int8 tensors) are not supported by `litert_torch`'s MLIR lowering pipeline.

So the correct order is:

1. Convert float32 PyTorch → float32 `.tflite` (via `litert_torch`)
2. Quantize the `.tflite` → int8/int4 `.tflite` (via `ai_edge_quantizer`)

The quantizer works on the `.tflite` flatbuffer directly, which is why it works cleanly.

## Android Device Testing (CPU/NNAPI)

These are the steps that is need to profile the model on Local Android Device with ADB command line.

### Step 1: Push TFLite Model to Device

```bash
# Push float32 model
adb push tiny_muaalem_float32.tflite /data/local/tmp/

# Or quantized versions
adb push tiny_muaalem_int8.tflite /data/local/tmp/
```

### Step 2: Download LiteRT Benchmark Tool

Get the `benchmark_model` binary from TensorFlow Lite releases (works with LiteRT `.tflite` files):

```bash
wget https://storage.googleapis.com/tensorflow-nightly-public/prod/tensorflow/release/lite/tools/nightly/latest/android_aarch64_benchmark_model
chmod +x android_aarch64_benchmark_model
adb push android_aarch64_benchmark_model /data/local/tmp/
```

### Step 3: Run CPU Benchmark

```bash
adb shell /data/local/tmp/android_aarch64_benchmark_model \
  --graph=/data/local/tmp/tiny_muaalem_float32.tflite \
  --num_runs=100 \
  --warmup_runs=5
```

### Step 4: Run with NNAPI (Deprecated but functional)

```bash
adb shell /data/local/tmp/android_aarch64_benchmark_model \
  --graph=/data/local/tmp/tiny_muaalem_float32.tflite \
  --num_runs=100 \
  --warmup_runs=5 \
  --use_nnapi=true
```

For the NEW NPU we should use QAI hub to compile the model their then redownload the model.

## NPU Compilation via QAI Hub

Qualcomm AI Hub (QAI Hub) compiles models for specific Snapdragon NPUs. This offloads heavy operators (MatMul, Conv) from CPU to the dedicated NPU for 10-100x speedup.

### Why QAI Hub?

- **Device-specific optimization**: Generates `.tflite` with Qualcomm's QNN delegate
- **Real hardware profiling**: Runs on actual devices (S24, S25, etc.)
- **No local setup needed**: Everything happens in Qualcomm's cloud

### Prerequisites

```bash
pip install qai-hub
export QAI_HUB_API_TOKEN="your_token_here"
```

### Step 1: Compile for NPU

Converts ONNX → NPU-optimized `.tflite`:

```bash
uv run python compile_for_npu.py \
  --model ../models/tiny_muaalem_int8.onnx \
  --device "Samsung Galaxy S24 (Family)" \
  --os 14 \
  --output tiny_muaalem_int8_qnn.tflite
```

**Output**: ~24 MB `.tflite` ready for the S24's Snapdragon 8 Gen 3 NPU.

### Step 2: Profile on Real Device

Runs inference 100+ times on an actual S24 and measures latency, memory, NPU utilization:

```bash
uv run python profile_npu.py --model tiny_muaalem_int8_qnn.tflite
```

**Output**: Job ID + results saved to `qai_hub_results/`

### Step 3: Download & Parse Results

```bash
# If connection dropped, re-download by job ID
uv run python download_profile_results.py --job-id jxxxxxxxxx

# Parse to human-readable summary
uv run python parse_profile_results.py \
  --results qai_hub_results/tiny_muaalem_int8_qnn.tflite_jxxxxxxxxx_results.json \
  --device "Samsung Galaxy S24 (Snapdragon 8 Gen 3)"
```

## Results Summary

| Platform | Device | Backend | Model | Inference Time | Memory |
|:---|:---|:---|:---|---:|---:|
| Linux x86_64 | Laptop (i7 + 1660Ti) | LiteRT CPU | float32 | 16.1 ms | — |
| Linux x86_64 | Laptop (i7 + 1660Ti) | LiteRT CPU | int8 | 16.5 ms | — |
| Android | OnePlus Pad (Snapdragon 8 Gen 1) | LiteRT CPU | float32 | 11.8 ms | 57 MB |
| **Android** | **Samsung S24 (Snapdragon 8 Gen 3)** | **QNN NPU** | **int8_qnn** | **8.5 ms** | **168 MB** |

### Key Findings

1. **NPU vs CPU**: Snapdragon 8 Gen 3 NPU achieves ~28% speedup over Gen 1 CPU (8.5ms vs 11.8ms) for this model
2. **100% NPU utilization**: All 5,649 operators mapped to NPU (0 CPU/GPU fallback)
3. **First load penalty**: 6.5 seconds cold start, 195ms warm start — important for real-time apps
4. **Memory overhead**: NPU uses more memory (168MB) than CPU (57MB) — QNN delegate workspace
5. **Quantization is mixed**: int8 can be slower on some x86 CPUs due to dequantization overhead

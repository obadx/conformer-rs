# Muaalem ONNX Benchmark (Rust/ort-tract)

## Build & Run (Linux)

```bash
cd conformer-rs
cargo run --release -- --models ../models --iterations 50
```

### Results (Linux x86_64)

| Model    | Load (ms) | Inference (ms) | Notes |
|----------|-----------|----------------|-------|
| float32  | 216.9    | 10.29±0.18    | ✅ Works |
| float16  | -        | -              | ❌ LayerNorm not supported |
| int8     | -        | -              | ❌ Quantization not supported |

## Build for Android

```bash
cd conformer-rs
cargo ndk -t arm64-v8a build --release
```

Output: `target/aarch64-linux-android/release/conformer-rs` (19 MB)

## Run on Android

```bash
# Push binary and models
adb push target/aarch64-linux-android/release/conformer-rs /data/local/
adb push models/tiny_muaalem_float32.onnx /data/local/

# Run benchmark
adb shell /data/local/conformer-rs --models /data/local --iterations 50
```

### Results (Android - Redmi 9, MediaTek Helio G80)

| Model    | Load (ms)  | Inference (ms)      | Notes |
|----------|------------|---------------------|-------|
| float32  | 2013.0    | 90.17±88.46        | ✅ Works |
| float16  | -         | -                   | ❌ LayerNorm not supported |
| int8     | -         | -                   | ❌ Quantization not supported |

## CLI Options

```bash
conformer-rs --models <path>     # Model directory (default: models)
conformer-rs -n <N>              # Iterations (default: 10)
conformer-rs -w <N>               # Warmup iterations (default: 2)
conformer-rs --shape B,T,F       # Input shape (default: 1,49,160)
conformer-rs --help              # Show help
```

### Example

```bash
# Linux
./target/release/conformer-rs --models ./models --iterations 10 --warmup 2

# Android
adb shell /data/local/conformer-rs --models /sdcard/models -n 50
```

## Model Support Matrix

| Precision | Python/onnxruntime | Rust/ort-tract (Linux) | Android |
|------------|-------------------|--------------------------|--------|
| float32   | ✅ 8.73 ms      | ✅ 10.29 ms            | ✅ 90 ms |
| float16   | ✅ 10.12 ms     | ❌ LayerNorm           | ❌ LayerNorm |
| int8      | ✅ 25.89 ms     | ❌ Quantization       | ❌ Quantization |

## Architecture

- **Backend**: ort-tract (alternative backend based on tract)
- **Why not ONNX Runtime**: Requires building ONNX Runtime from source for Android (Bionic vs glibc)
- **Limitations**: tract doesn't support LayerNorm and quantization operators

## Troubleshooting

### "Failed to parse model: Translating node LayerNorm"
- LayerNorm is not supported by tract backend
- Use float32 model only

### "Failed to parse model: Failed analyse for node ConvHir"
- Quantization (QDQ ops) not supported by tract backend
- Use float32 model only
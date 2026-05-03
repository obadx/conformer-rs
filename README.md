<img src="./conformer-conv-module.png" width="600px"></img>

## Conformer

Implementation of the convolutional module from the [Conformer](https://arxiv.org/abs/2005.08100) paper, for improving the local inductive bias in Transformers.

## Usage

The Conformer convolutional module, the main novelty of the paper

```python
import torch
from conformer_python import ConformerConvModule

layer = ConformerConvModule(
    dim = 512,
    causal = False,             # auto-regressive or not - 1d conv will be made causal with padding if so
    expansion_factor = 2,       # what multiple of the dimension to expand for the depthwise convolution
    kernel_size = 31,           # kernel size, 17 - 31 was said to be optimal
    dropout = 0.                # dropout at the very end
)

x = torch.randn(1, 1024, 512)
x = layer(x) + x
```

1 Conformer Block

```python
import torch
from conformer import ConformerBlock

block = ConformerBlock(
    dim = 512,
    dim_head = 64,
    heads = 8,
    ff_mult = 4,
    conv_expansion_factor = 2,
    conv_kernel_size = 31,
    attn_dropout = 0.,
    ff_dropout = 0.,
    conv_dropout = 0.
)

x = torch.randn(1, 1024, 512)
block(x) # (1, 1024, 512)
```

Conformer - just multiple `ConformerBlock` from above

```python
import torch
from conformer import Conformer

conformer = Conformer(
    dim = 512,
    depth = 12,          # 12 blocks
    dim_head = 64,
    heads = 8,
    ff_mult = 4,
    conv_expansion_factor = 2,
    conv_kernel_size = 31,
    attn_dropout = 0.,
    ff_dropout = 0.,
    conv_dropout = 0.
)

x = torch.randn(1, 1024, 512)
conformer(x) # (1, 1024, 512)
```

## Model Configuration

This project includes a pre-configured Conformer model for ASR using ConformerBlock:

- **Depth**: 16 encoder layers
- **Dim**: 144
- **Heads**: 4
- **Conv Kernel Size**: 32
- **Input**: (batch, 50, 144) - 50 frames representing 2 seconds of speech
- **Output**: (batch, 50, 144)

---

# Muaalem: Tiny Arabic ASR Model

Muaalem uses ConformerBlock as the building block for its encoder. This section documents exporting to ONNX and benchmarking.

## Model Architecture

- **Base Model**: Uses ConformerBlock as building block (16 layers)
- **Parameters**: hidden_size=144, heads=4, conv_kernel=32
- **Source**: `obadx/muaalem-model-v3_2` (HuggingFace)
- **Input Shape**: (batch, 49, 160) - audio features (1 second)
- **Output**: 11 levels (phonemes + 10 sifa attributes):
  - phonemes (43 classes)
  - hams_or_jahr, shidda_or_rakhawa, tafkheem_or_taqeeq, itbaq, safeer, qalqla, tikraar, tafashie, istitala, ghonna

## Step 1: Export ONNX Models

```bash
cd conformer-python
uv sync
uv run python convert_muaalem_to_onnx.py
```

This creates 3 ONNX model files in `conformer-rs/models/`:

| File | Size | Precision |
|------|------|-----------|
| `tiny_muaalem_float32.onnx` | 22 MB | Full float32 |
| `tiny_muaalem_float16.onnx` | 11 MB | Half precision |
| `tiny_muaalem_int8.onnx` | 8.1 MB | Quantized int8 |

## Step 2: Benchmark (Python/Linux)

```bash
cd conformer-python
uv run python run_muaalem_onnx.py
```

### Results (Linux x86_64, Python/onnxruntime)

| Model    | Load (ms) | Inference (ms) | Notes |
|----------|-----------|----------------|-------|
| float32  | 43.9     | 8.73±0.37     | ✅ Works |
| float16  | 57.3     | 10.12±0.89   | ✅ Works |
| int8     | 38.6     | 25.89±0.17   | ✅ Works |

## Step 3: Benchmark (Rust/Linux)

```bash
cd conformer-rs
cargo run --release -- --models ../models --iterations 50
```

### Results (Linux x86_64, Rust/ort-tract)

| Model    | Load (ms) | Inference (ms) | Notes |
|----------|-----------|----------------|-------|
| float32  | 216.9    | 10.29±0.18    | ✅ Works |
| float16  | -        | -              | ❌ LayerNorm not supported |
| int8     | -        | -              | ❌ Quantization not supported |

## Step 4: Android Build & Run

### Build for Android ARM64

```bash
cd conformer-rs
cargo ndk -t arm64-v8a build --release
```

Output: `target/aarch64-linux-android/release/conformer-rs` (19 MB)

### Run on Android Device

```bash
# Push binary and model
adb push target/aarch64-linux-android/release/conformer-rs /data/local/
adb push models/tiny_muaalem_float32.onnx /data/local/

# Run benchmark
adb shell /data/local/conformer-rs --models /data/local --iterations 50
```

### Results (Android - Redmi 9, MediaTek Helio G80)

| Model    | Window Length (seconds) |  Load (ms)  | Inference (ms)      | Notes |
|----------|-------------------------|-------------|---------------------|-------|
| float32  | 1                       | 3090.6    | 79.51±18.91        | ✅ Works |
| float32  | 5.02                    | 4087.3    | 693.95±175.45      | ✅ Works |
| float16  | | -         | -                   | ❌ LayerNorm not supported |
| int8     | | -         | -                   | ❌ Quantization not supported |

### Results for tract (single core CPU)

```bash
~/muaalem $ ./conformer-rs --models ./models/ --iterations 1000
==================================================
Muaalem ONNX Benchmark (Rust/ort-tract)
==================================================
Models dir: ./models/
Input shape: (1, 49, 160)
Iterations: 1000 (warmup: 2)
Found 3 model(s)

========================================
Precision: tiny_muaalem_float16
========================================
  Failed to load model: Failed to parse model: Translating node #3 "node_layer_norm" LayerNorm ToTypedTranslator

========================================
Precision: tiny_muaalem_float32
========================================
  Load time: 3090.6ms
  Input: input_features Some([1, 49, 160])
  Input dtype: Tensor { ty: Float32, shape: [1, 49, 160], dimension_symbols: SymbolicDimensions(["", "", ""]) }
  Outputs: ["ghonna", "hams_or_jahr", "istitala", "itbaq", "phonemes", "qalqla", "safeer", "shidda_or_rakhawa", "tafashie", "tafkheem_or_taqeeq", "tikraar"]
  Inference (ms): avg=79.51±18.91, min=71.40, max=452.33
  Output count: 11

========================================
Precision: tiny_muaalem_int8
========================================
  Failed to load model: Failed to parse model: Failed analyse for node #458 "node_Conv_1490_quant" ConvHir

==================================================
Summary
==================================================
Model                Load (ms)    Inference (ms)  Std (ms)  
--------------------------------------------------
tiny_muaalem_float32       3090.6           79.51      18.91
```

### Results for ruten crate

```bash
 $ ./conformer-rten --models ./models/ --iterations 1000
==================================================
Muaalem ONNX Benchmark (Rust/rten)
==================================================
Models dir: ./models/
Input shape: (1, 49, 160)
Iterations: 1000 (warmup: 2)
Found 3 model(s)

========================================
Precision: tiny_muaalem_float16
========================================
  Failed to load model: in node "model.wav2vec2_bert.feature_projection.layer_norm.weight": graph error: initializer has unsupported data type FLOAT16

========================================
Precision: tiny_muaalem_float32
========================================
  Load time: 185.0ms
  Input: input_features Some([1, 49, 160])
  Outputs: 11 outputs
    - ghonna
    - hams_or_jahr
    - istitala
    ... and 8 more
  Inference (ms): avg=179.49±95.21, min=73.79, max=492.09
  Output count: 11

========================================
Precision: tiny_muaalem_int8
========================================
  Load time: 111.5ms
  Input: input_features Some([1, 49, 160])
  Outputs: 11 outputs
    - ghonna
    - hams_or_jahr
    - istitala
    ... and 8 more
  Inference (ms): avg=106.42±51.06, min=56.86, max=311.79
  Output count: 11

==================================================
Summary
==================================================
Model                Load (ms)    Inference (ms)  Std (ms)  
--------------------------------------------------
tiny_muaalem_float32        185.0          179.49      95.21
tiny_muaalem_int8           111.5          106.42      51.06

```bash

## CLI Options (Rust)

```bash
conformer-rs --models <path>     # Model directory (default: models)
conformer-rs -n <N>              # Iterations (default: 10)
conformer-rs -w <N>             # Warmup iterations (default: 2)
conformer-rs --shape B,T,F      # Input shape (default: 1,49,160)
conformer-rs --help             # Show help
```

## Model Support Matrix

| Precision | Python/onnxruntime | Rust/ort-tract (Linux) | Android |
|------------|-------------------|--------------------------|--------|
| float32   | ✅ 8.73 ms      | ✅ 10.29 ms            | ✅ 90 ms |
| float16   | ✅ 10.12 ms     | ❌ LayerNorm           | ❌ LayerNorm |
| int8      | ✅ 25.89 ms     | ❌ Quantization       | ❌ Quantization |

## Troubleshooting

### Rust/Tract Errors

- **"Failed to parse model: Translating node LayerNorm"** - LayerNorm not supported by tract backend
- **"Failed to parse model: Failed analyse for node ConvHir"** - Quantization (QDQ) not supported

Solution: Use float32 model only for Rust/Android benchmarks.

### Why Not ONNX Runtime for Android?

The `ort` crate includes ONNX Runtime source code that uses glibc functions which don't exist on Android (Bionic libc). Building from source takes ~1 hour. Using tract backend avoids this but loses float16/int8 support.

---

## Todo

- [ ] Implement Streaming conformer with KV caching and 5.02 seconds as input (250 frames)
- [ ] Implement Streaming conformer with KV caching with sliding widnow attention and 5.02 seconds as input (250 frames)
- [ ] Implement Streaming conformer LSTM Architecture [here](https://github.com/obadx/quran-muaalem/issues/10) and 5.02 seconds as input (250 frames)


## Citations

```bibtex
@misc{gulati2020conformer,
    title   = {Conformer: Convolution-augmented Transformer for Speech Recognition},
    author  = {Anmol Gulati and James Qin and Chung-Cheng Chiu and Niki Parmar and Yu Zhang and Jiahui Yu and Wei Han and Shibo Wang and Zhengdong Zhang and Yonghui Wu and Ruoming Pang},
    year    = {2020},
    eprint  = {2005.08100},
    archivePrefix = {arXiv},
    primaryClass = {eess.AS}
}
```

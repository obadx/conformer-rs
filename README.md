<img src="./conformer-conv-module.png" width="600px"></img>

## Conformer

Implementation of the convolutional module from the <a href="https://arxiv.org/abs/2005.08100">Conformer</a> paper, for improving the local inductive bias in Transformers.

## Install

```bash
$ pip install conformer
```

## Usage

The Conformer convolutional module, the main novelty of the paper

```python
import torch
from conformer import ConformerConvModule

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

This project includes a pre-configured Conformer model for ASR:

- **Depth**: 16 encoder layers
- **Dim**: 144
- **Heads**: 4
- **Conv Kernel Size**: 32
- **Input**: (batch, 50, 144) - 50 frames representing 2 seconds of speech
- **Output**: (batch, 50, 144)

## ONNX Export (Python)

Export Conformer model to ONNX for Rust inference:

```bash
cd conformer-python
source .venv/bin/activate
python export_onnx.py
```

This creates `conformer-rs/model.onnx` (~50MB with embedded weights).

## NNEF Conversion (Recommended for Mobile)

Convert ONNX to NNEF format for ~100x faster model loading on mobile:

```bash
cd conformer-rs
tract model.onnx dump --nnef-dir model.nnef.d
```

This creates `model.nnef.d/` directory (~50MB, 25 files).

**Why NNEF?**
- ONNX loading: ~30 seconds (parses & optimizes every run)
- NNEF loading: ~0.3 seconds (pre-optimized)

## Rust Inference

Run locally:

```bash
cd conformer-rs

# Copy model to release directory (for NNEF)
cp -r model.nnef.d target/release/model.nnef

# Run
cargo run --release
```

Output:
```
Using model: "/path/to/model.nnef"
Output shape: [1, 50, 144]

real    0m0.2xxs
```

## Android Build

Cross-compile for Android (arm64):

```bash
cd conformer-rs
cargo ndk -t arm64-v8a build --release
```

**Output files:**
- Binary: `target/aarch64-linux-android/release/conformer-rs` (~16MB)
- Model: `target/aarch64-linux-android/release/model.nnef/` (~50MB, 25 files)

## Run on Android Phone

### Files Needed

```
/data/local/tmp/
├── conformer-rs    # Binary (~16MB)
└── model.nnef/     # NNEF model directory (~50MB, 25 files)
```

### Copy to Phone

```bash
adb push conformer-rs/target/aarch64-linux-android/release/conformer-rs /data/local/tmp/
adb push conformer-rs/target/aarch64-linux-android/release/model.nnef/ /data/local/tmp/
adb shell chmod +x /data/local/tmp/conformer-rs
```

### Run

```bash
adb shell /data/local/tmp/conformer-rs
```

Expected output:
```
Using model: "/data/local/tmp/model.nnef"
Output shape: [1, 50, 144]

real    0m0.2xxs
```

## Todo

- [ ] switch to a better relative positional encoding. shaw's is dated
- [ ] flash attention with a better RPE

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
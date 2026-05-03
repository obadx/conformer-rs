#!/usr/bin/env python3
"""
Convert tiny Muaalem model to LiteRT (.tflite) format.
Produces float32, int8, and int4 quantized versions.

Usage:
    python convert_tiny_to_tflite.py [--output-dir OUTPUT_DIR]
"""

import argparse
import os
import torch
import litert_torch
from transformers import AutoFeatureExtractor
from ai_edge_quantizer import quantizer, recipe
from conformer_python.muaalem_offline import (
    Wav2Vec2BertForMultilevelCTC,
    Wav2Vec2BertForMultilevelCTCConfig,
    vocab,
)

MODEL_ID = "obadx/muaalem-model-v3_2"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def load_tiny_model():
    processor = AutoFeatureExtractor.from_pretrained(MODEL_ID)
    config = Wav2Vec2BertForMultilevelCTCConfig.from_pretrained(
        MODEL_ID,
        pad_token_id=vocab.PAD_TOKEN_IDX,
        attention_dropout=0.0,
        hidden_dropout=0.0,
        feat_proj_dropout=0.0,
        mask_time_prob=0.0,
        layerdrop=0.0,
        ctc_loss_reduction="mean",
        add_adapter=False,
        num_hidden_layers=16,
        hidden_size=144,
        output_hidden_size=144,
        intermediate_size=256,
        num_attention_heads=4,
        adapter_stride=1,
    )
    model = Wav2Vec2BertForMultilevelCTC.from_pretrained(
        MODEL_ID,
        config=config,
        ignore_mismatched_sizes=True,
    )
    model.eval()
    return model, processor


class TinyMuaalemWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.level_names = list(model.level_to_lm_head.keys())

    def forward(self, input_features):
        out = self.model(input_features, return_dict=True)
        return tuple(out["logits"][name] for name in self.level_names)


def main():
    parser = argparse.ArgumentParser(description="Convert tiny Muaalem to TFLite")
    parser.add_argument(
        "--output-dir",
        default=SCRIPT_DIR,
        help="Directory to save .tflite files (default: script directory)",
    )
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    f32_path = os.path.join(args.output_dir, "tiny_muaalem_float32.tflite")
    int8_path = os.path.join(args.output_dir, "tiny_muaalem_int8.tflite")
    int4_path = os.path.join(args.output_dir, "tiny_muaalem_int4.tflite")

    print("Loading model...")
    model, processor = load_tiny_model()
    wrapped = TinyMuaalemWrapper(model).eval().float()

    dummy_input = processor(16000 * [0], sampling_rate=16000, return_tensors="pt")
    sample = (dummy_input["input_features"].float(),)

    # 1. float32
    print("[1/3] Converting float32...")
    tflite_f32 = litert_torch.convert(wrapped, sample)
    tflite_f32.export(f32_path)
    print(f"Saved {f32_path} ({os.path.getsize(f32_path) / 1e6:.1f} MB)")

    # 2. int8 dynamic
    print("[2/3] Quantizing to int8...")
    qt = quantizer.Quantizer(f32_path)
    qt.load_quantization_recipe(recipe.dynamic_wi8_afp32())
    qt.quantize().export_model(int8_path)
    print(f"Saved {int8_path} ({os.path.getsize(int8_path) / 1e6:.1f} MB)")

    # 3. int4 dynamic
    print("[3/3] Quantizing to int4...")
    qt = quantizer.Quantizer(f32_path)
    qt.load_quantization_recipe(recipe.dynamic_wi4_afp32())
    qt.quantize().export_model(int4_path)
    print(f"Saved {int4_path} ({os.path.getsize(int4_path) / 1e6:.1f} MB)")

    print("\nDone!")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Convert tiny Muaalem model to LiteRT (.tflite) format.
Produces float32, int8, and int4 quantized versions.

Usage:
    python convert_tiny_to_tflite.py [--architecture {w2v-conformer,w2v}] [--output-dir OUTPUT_DIR]
"""

import argparse
import os
import torch
import litert_torch
from transformers import AutoFeatureExtractor
from ai_edge_quantizer import quantizer, recipe
from conformer_python.muaalem_offline import vocab

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def _get_base_config(model_id):
    from conformer_python.muaalem_offline import (
        Wav2Vec2BertForMultilevelCTCConfig,
    )

    return Wav2Vec2BertForMultilevelCTCConfig.from_pretrained(model_id)


def load_conformer_model(model_id):
    from conformer_python.muaalem_offline import (
        Wav2Vec2BertForMultilevelCTC,
        Wav2Vec2BertForMultilevelCTCConfig,
    )

    processor = AutoFeatureExtractor.from_pretrained(model_id)
    config = Wav2Vec2BertForMultilevelCTCConfig.from_pretrained(
        model_id,
        pad_token_id=vocab.PAD_TOKEN_IDX,
        attention_dropout=0.0,
        hidden_dropout=0.0,
        feat_proj_dropout=0.0,
        mask_time_prob=0.0,
        layerdrop=0.0,
        ctc_loss_reduction="mean",
        add_adapter=False,
        num_hidden_layers=12,
        hidden_size=512,
        output_hidden_size=512,
        intermediate_size=2048,
        num_attention_heads=12,
        adapter_stride=1,
    )
    model = Wav2Vec2BertForMultilevelCTC.from_pretrained(
        model_id,
        config=config,
        ignore_mismatched_sizes=True,
    )
    model.eval()
    return model, processor, "input_features"


def create_w2v_model(model_id):
    from conformer_python.muaalem_offline_w2v import (
        Wav2Vec2ForMultilevelCTC,
        Wav2Vec2ForMultilevelCTCConfig,
    )

    base_config = _get_base_config(model_id)
    config = Wav2Vec2ForMultilevelCTCConfig(
        level_to_vocab_size=base_config.level_to_vocab_size,
        level_to_loss_weight=base_config.level_to_loss_weight,
        pad_token_id=vocab.PAD_TOKEN_IDX,
        hidden_size=384,
        num_hidden_layers=24,
        num_attention_heads=4,
        intermediate_size=1536,
        attention_dropout=0.0,
        hidden_dropout=0.0,
        feat_proj_dropout=0.0,
        mask_time_prob=0.0,
        layerdrop=0.0,
        ctc_loss_reduction="mean",
        add_adapter=False,
    )
    model = Wav2Vec2ForMultilevelCTC(config)
    model.eval()
    return model, None, "input_values"


def create_whisper_enc_model(model_id):
    from conformer_python.muaalem_offline_whisper_enc import (
        WhisperEncoderForMultilevelCTC,
        WhisperEncoderForMultilevelCTCConfig,
    )

    base_config = _get_base_config(model_id)
    config = WhisperEncoderForMultilevelCTCConfig(
        level_to_vocab_size=base_config.level_to_vocab_size,
        level_to_loss_weight=base_config.level_to_loss_weight,
        pad_token_id=vocab.PAD_TOKEN_IDX,
        d_model=768,
        encoder_layers=12,
        encoder_attention_heads=12,
        encoder_ffn_dim=3072,
        max_source_positions=35,
        dropout=0.0,
        attention_dropout=0.0,
        activation_dropout=0.0,
        ctc_loss_reduction="mean",
        encoder_layerdrop=0.0,
    )
    model = WhisperEncoderForMultilevelCTC(config)
    model.eval()
    processor = AutoFeatureExtractor.from_pretrained("openai/whisper-small")
    return model, processor, "input_features"


def _make_sample(arch_key, model, processor):
    if arch_key == "w2v-conformer":
        dummy_input = processor(
            int(16000 * 0.7) * [0], sampling_rate=16000, return_tensors="pt"
        )
        return (dummy_input["input_features"].float(),)
    elif arch_key == "whisper-enc":
        dummy_input = processor(
            int(16000 * 0.7) * [0],
            sampling_rate=16000,
            return_tensors="pt",
            padding=False,
        )
        return (dummy_input["input_features"].float(),)
    else:
        return (torch.zeros(1, int(16000 * 0.7)).float(),)


class MuaalemWrapper(torch.nn.Module):
    def __init__(self, model, input_key):
        super().__init__()
        self.model = model
        self.input_key = input_key
        self.level_names = list(model.level_to_lm_head.keys())

    def forward(self, x):
        kwargs = {self.input_key: x, "return_dict": True}
        out = self.model(**kwargs)
        return tuple(out["logits"][name] for name in self.level_names)


def main():
    parser = argparse.ArgumentParser(description="Convert tiny Muaalem to TFLite")
    parser.add_argument(
        "--architecture",
        choices=["w2v-conformer", "w2v", "whisper-enc"],
        default="w2v-conformer",
        help="Model architecture (default: w2v-conformer)",
    )
    parser.add_argument(
        "--model-id",
        default="obadx/muaalem-model-v3_2",
        help="HuggingFace model ID (default: obadx/muaalem-model-v3_2)",
    )
    parser.add_argument(
        "--output-dir",
        default=SCRIPT_DIR,
        help="Directory to save .tflite files (default: script directory)",
    )
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    arch = args.architecture
    prefix = (
        "tiny_muaalem"
        if arch == "w2v-conformer"
        else "tiny_muaalem_w2v"
        if arch == "w2v"
        else "tiny_muaalem_whisper_enc"
    )
    f32_path = os.path.join(args.output_dir, f"{prefix}_float32.tflite")
    int8_path = os.path.join(args.output_dir, f"{prefix}_int8.tflite")
    int4_path = os.path.join(args.output_dir, f"{prefix}_int4.tflite")

    print(f"Loading {arch} model...")
    if arch == "w2v-conformer":
        model, processor, input_key = load_conformer_model(args.model_id)
    elif arch == "whisper-enc":
        model, processor, input_key = create_whisper_enc_model(args.model_id)
    else:
        model, processor, input_key = create_w2v_model(args.model_id)

    wrapped = MuaalemWrapper(model, input_key).eval().float()
    sample = _make_sample(arch, model, processor)

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

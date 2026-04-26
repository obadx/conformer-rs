import time
from pathlib import Path

import numpy as np
import onnx
import torch
import torch.nn as nn
from onnxsim import simplify
from transformers import AutoFeatureExtractor

from conformer_python.muaalem_offline import (
    Wav2Vec2BertForMultilevelCTC,
    Wav2Vec2BertForMultilevelCTCConfig,
    vocab,
)

MODEL_ID = "obadx/muaalem-model-v3_2"
OUTPUT_DIR = Path(__file__).parent.parent / "models"
OUTPUT_DIR.mkdir(exist_ok=True)


class MultiLevelCTCWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.level_names = list(model.level_to_lm_head.keys())

    def forward(self, input_features):
        with torch.no_grad():
            outputs = self.model(input_features, return_dict=True)
            return tuple(outputs["logits"][name] for name in self.level_names)


def build_model_and_processor():
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


def get_dummy_input(processor, sampling_rate=16000):
    return processor(
        sampling_rate * [0], sampling_rate=sampling_rate, return_tensors="pt"
    )


def export_float32(model, dummy_input, output_path):
    print("\n[1/3] Exporting float32...")
    start = time.perf_counter()

    wrapper = MultiLevelCTCWrapper(model)
    wrapper.eval()

    torch.onnx.export(
        wrapper,
        (dummy_input["input_features"],),
        str(output_path),
        input_names=["input_features"],
        output_names=wrapper.level_names,
        dynamic_shapes={"input_features": {1: None}},
        opset_version=18,
        do_constant_folding=True,
    )

    m = onnx.load(output_path)
    m_simp, check = simplify(m)
    onnx.save(m_simp, output_path)

    elapsed = time.perf_counter() - start
    print(f"  -> Saved to {output_path}")
    print(f"  -> Time: {elapsed:.1f}s, check passed: {check}")
    return check


def export_float16(model, dummy_input, output_path):
    print("\n[2/3] Exporting float16...")
    start = time.perf_counter()

    wrapper = MultiLevelCTCWrapper(model)
    wrapper = wrapper.half()

    torch.onnx.export(
        wrapper,
        (dummy_input["input_features"].half(),),
        str(output_path),
        input_names=["input_features"],
        output_names=wrapper.level_names,
        dynamic_shapes={"input_features": {1: None}},
        opset_version=18,
        do_constant_folding=True,
    )

    m = onnx.load(output_path)
    m_simp, check = simplify(m)
    onnx.save(m_simp, output_path)

    elapsed = time.perf_counter() - start
    print(f"  -> Saved to {output_path}")
    print(f"  -> Time: {elapsed:.1f}s, check passed: {check}")
    return check





def export_int8(model, dummy_input, output_path, float32_model_path):
    print("\n[3/3] Exporting int8 with onnxruntime quantization...")
    print("  -> Strategy: Hybrid int8 (MatMul/Conv → int8, LayerNorm → fp32)")
    start = time.perf_counter()

    try:
        from onnxruntime.quantization import quantize_dynamic, QuantType

        quantize_dynamic(
            model_input=str(float32_model_path),
            model_output=str(output_path),
            weight_type=QuantType.QInt8,
            op_types_to_quantize=["MatMul", "Conv"],
            extra_options={
                "MatMulConstBOnly": True,
            },
        )

        elapsed = time.perf_counter() - start
        print(f"  -> Saved to {output_path}")
        print(f"  -> Time: {elapsed:.1f}s")
        return True

    except Exception as e:
        elapsed = time.perf_counter() - start
        print(f"  -> int8 export failed: {type(e).__name__}: {e}")
        print(f"  -> Falling back to float16...")
        return False


if __name__ == "__main__":
    print("=" * 50)
    print("Muaalem Model → ONNX Export")
    print("=" * 50)

    model, processor = build_model_and_processor()
    dummy_input = get_dummy_input(processor)
    level_names = list(model.level_to_lm_head.keys())
    print(f"\nLevels ({len(level_names)}): {level_names}")

    print("\nModel info:")
    print(f"  - Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  - Input shape: {dummy_input['input_features'].shape}")

    float32_path = OUTPUT_DIR / "tiny_muaalem_float32.onnx"
    float16_path = OUTPUT_DIR / "tiny_muaalem_float16.onnx"
    int8_path = OUTPUT_DIR / "tiny_muaalem_int8.onnx"

    export_float32(model, dummy_input, float32_path)
    export_float16(model, dummy_input, float16_path)
    int8_success = export_int8(model, dummy_input, int8_path, float32_path)

    if not int8_success:
        import shutil
        shutil.copy(float16_path, int8_path)
        print(f"  -> Copied float16 as fallback to {int8_path}")

    print("\n" + "=" * 50)
    print("Export complete!")
    print("=" * 50)
    print("\nGenerated files:")
    print(f"  - {float32_path}")
    print(f"  - {float16_path}")
    print(f"  - {int8_path}")
#!/usr/bin/env python3
"""
Compile a model for Qualcomm NPU using QAI Hub.

Usage:
    python compile_for_npu.py --model MODEL_PATH [--device DEVICE] [--os OS] [--output OUTPUT]
"""

import argparse
import os
import qai_hub

# Increase timeout for slow connections
import qai_hub.util.session as _session

_session.EXTERNAL_RESPONSE_TIMEOUT_SECONDS = 120

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def main():
    parser = argparse.ArgumentParser(
        description="Compile model for Qualcomm NPU via QAI Hub"
    )
    parser.add_argument(
        "--model", required=True, help="Path to input model (ONNX or TFLite)"
    )
    parser.add_argument(
        "--device", default="Samsung Galaxy S24 (Family)", help="Target device name"
    )
    parser.add_argument("--os", default="14", help="Target Android OS version")
    parser.add_argument("--output", default=None, help="Output path for compiled model")
    parser.add_argument(
        "--runtime",
        default="tflite",
        choices=["tflite", "onnx", "qnn"],
        help="Target runtime",
    )
    args = parser.parse_args()

    if args.output is None:
        base = os.path.splitext(os.path.basename(args.model))[0]
        args.output = os.path.join(SCRIPT_DIR, f"{base}_qnn.tflite")

    print(f"Uploading model: {args.model}")
    model = qai_hub.upload_model(args.model)
    print(f"Uploaded model ID: {model.model_id}")

    print(f"\nCompiling for {args.device} (Android {args.os})...")
    job = qai_hub.submit_compile_job(
        model=model,
        device=qai_hub.Device(args.device, os=args.os),
        options=f"--target_runtime {args.runtime}",
    )
    print(f"Compile job ID: {job.job_id}")
    print("Waiting for compilation...")
    job.wait()
    job.download_target_model(args.output)
    print(f"\nSaved to: {args.output}")
    print(f"Size: {os.path.getsize(args.output) / 1e6:.2f} MB")


if __name__ == "__main__":
    main()

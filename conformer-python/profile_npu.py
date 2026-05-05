#!/usr/bin/env python3
"""
Profile a compiled model on a real Qualcomm device via QAI Hub.

Usage:
    python profile_npu.py --model MODEL_PATH [--device DEVICE] [--os OS] [--output-dir DIR]
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
        description="Profile model on Qualcomm device via QAI Hub"
    )
    parser.add_argument("--model", required=True, help="Path to compiled .tflite model")
    parser.add_argument(
        "--device", default="Samsung Galaxy S24 (Family)", help="Target device name"
    )
    parser.add_argument("--os", default="14", help="Target Android OS version")
    parser.add_argument(
        "--output-dir",
        default=os.path.join(SCRIPT_DIR, "qai_hub_results"),
        help="Directory to save results",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Uploading model: {args.model}")
    model = qai_hub.upload_model(args.model)
    print(f"Model ID: {model.model_id}")

    print(f"\nSubmitting profile job on {args.device} (Android {args.os})...")
    job = qai_hub.submit_profile_job(
        model=model,
        device=qai_hub.Device(args.device, os=args.os),
    )
    print(f"Profile job ID: {job.job_id}")
    print("Waiting for profiling to complete...")
    job.wait()

    print("\nProfiling complete!")
    job.download_results(args.output_dir)
    print(f"Results saved to: {args.output_dir}")
    for f in os.listdir(args.output_dir):
        path = os.path.join(args.output_dir, f)
        print(f"  {f} ({os.path.getsize(path) / 1024:.1f} KB)")


if __name__ == "__main__":
    main()

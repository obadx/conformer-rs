#!/usr/bin/env python3
"""
Download a compiled model from QAI Hub by job ID.

Usage:
    python download_compiled.py --job-id JOB_ID [--output OUTPUT]
"""

import argparse
import os
import qai_hub

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def main():
    parser = argparse.ArgumentParser(description="Download compiled model from QAI Hub")
    parser.add_argument("--job-id", required=True, help="QAI Hub compile job ID")
    parser.add_argument("--output", default=None, help="Output path for compiled model")
    args = parser.parse_args()

    if args.output is None:
        args.output = os.path.join(SCRIPT_DIR, f"compiled_{args.job_id}.tflite")

    print(f"Fetching job {args.job_id}...")
    job = qai_hub.get_job(args.job_id)
    job.download_target_model(args.output)
    print(f"Saved to: {args.output}")
    print(f"Size: {os.path.getsize(args.output) / 1e6:.2f} MB")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Download profile results from QAI Hub by job ID.

Usage:
    python download_profile_results.py --job-id JOB_ID [--output-dir DIR]
"""

import argparse
import os
import qai_hub

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def main():
    parser = argparse.ArgumentParser(
        description="Download profile results from QAI Hub"
    )
    parser.add_argument("--job-id", required=True, help="QAI Hub profile job ID")
    parser.add_argument(
        "--output-dir",
        default=os.path.join(SCRIPT_DIR, "qai_hub_results"),
        help="Directory to save results",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Fetching job {args.job_id}...")
    job = qai_hub.get_job(args.job_id)
    job.download_results(args.output_dir)
    print(f"Results saved to: {args.output_dir}")
    for f in os.listdir(args.output_dir):
        path = os.path.join(args.output_dir, f)
        print(f"  {f} ({os.path.getsize(path) / 1024:.1f} KB)")


if __name__ == "__main__":
    main()

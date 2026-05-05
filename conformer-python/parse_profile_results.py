#!/usr/bin/env python3
"""
Parse and display QAI Hub profile results from a JSON file.

Usage:
    python parse_profile_results.py --results RESULTS_JSON [--device DEVICE_NAME]
"""

import argparse
import json
import os
import statistics


def main():
    parser = argparse.ArgumentParser(description="Parse QAI Hub profile results")
    parser.add_argument(
        "--results", required=True, help="Path to profile results JSON file"
    )
    parser.add_argument(
        "--device", default="Unknown Device", help="Device name for display"
    )
    args = parser.parse_args()

    with open(args.results) as f:
        data = json.load(f)

    summary = data["execution_summary"]
    details = data.get("execution_detail", [])

    all_times = [t / 1000 for t in summary["all_inference_times"]]

    print("=== QAI Hub NPU Profile Results ===\n")
    print(f"Device:  {args.device}")
    print(f"Model:   {os.path.basename(args.results)}\n")

    print("Inference Time:")
    print(f"  Estimated:  {summary['estimated_inference_time'] / 1000:.3f} ms")
    print(f"  First:      {all_times[0]:.3f} ms")
    print(f"  Min (warm): {min(all_times[1:]):.3f} ms")
    print(f"  Max (warm): {max(all_times[1:]):.3f} ms")
    print(f"  Mean (warm):{statistics.mean(all_times[1:]):.3f} ms")
    print(f"  Median:     {statistics.median(all_times[1:]):.3f} ms\n")

    print("Memory:")
    print(
        f"  Peak inference: {summary['estimated_inference_peak_memory'] / 1e6:.2f} MB"
    )
    print(f"  First load:     {summary['first_load_time'] / 1000:.1f} ms")
    print(f"  Warm load:      {summary['warm_load_time'] / 1000:.1f} ms\n")

    npu_ops = [d for d in details if d.get("compute_unit") == "NPU"]
    cpu_ops = [d for d in details if d.get("compute_unit") == "CPU"]
    gpu_ops = [d for d in details if d.get("compute_unit") == "GPU"]

    print("Operator Breakdown:")
    print(f"  NPU: {len(npu_ops)}")
    print(f"  CPU: {len(cpu_ops)}")
    print(f"  GPU: {len(gpu_ops)}")
    print(f"  Total: {len(details)}")

    if npu_ops:
        npu_time = sum(d.get("execution_time", 0) for d in npu_ops)
        print(f"\n  NPU execution time: {npu_time / 1000:.3f} ms")
    if cpu_ops:
        cpu_time = sum(d.get("execution_time", 0) for d in cpu_ops)
        print(f"  CPU execution time: {cpu_time / 1000:.3f} ms")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Benchmark tiny Muaalem .tflite models using the LiteRT Python API.

Usage:
    python benchmark_litert.py [--model-dir MODEL_DIR] [--warmup N] [--runs N]
"""

import argparse
import os
import time
import numpy as np
from ai_edge_litert.compiled_model import CompiledModel

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

MODELS = [
    ("float32", "tiny_muaalem_float32.tflite"),
    ("int8", "tiny_muaalem_int8.tflite"),
    ("int4", "tiny_muaalem_int4.tflite"),
]


def benchmark(model_path: str, warmup: int, runs: int):
    model = CompiledModel.from_file(model_path)
    inputs = model.create_input_buffers(0)
    outputs = model.create_output_buffers(0)

    # Fill inputs with zeros (all models take float32 input)
    for i, inp in enumerate(inputs):
        req = model.get_input_buffer_requirements(i, 0)
        arr = np.zeros(req["buffer_size"] // 4, dtype=np.float32)
        inp.write(arr)

    # Warmup
    for _ in range(warmup):
        model.run_by_index(0, inputs, outputs)

    # Benchmark
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        model.run_by_index(0, inputs, outputs)
        times.append((time.perf_counter() - t0) * 1000)

    return times


def main():
    parser = argparse.ArgumentParser(description="Benchmark LiteRT .tflite models")
    parser.add_argument(
        "--model-dir", default=SCRIPT_DIR, help="Directory containing .tflite files"
    )
    parser.add_argument("--warmup", type=int, default=5, help="Number of warmup runs")
    parser.add_argument(
        "--runs", type=int, default=100, help="Number of benchmark runs"
    )
    args = parser.parse_args()

    print(
        f"{'Model':<10} {'Avg (ms)':>10} {'P50 (ms)':>10} {'P99 (ms)':>10} {'Size (MB)':>10}"
    )
    print("-" * 55)

    for name, filename in MODELS:
        path = os.path.join(args.model_dir, filename)
        if not os.path.exists(path):
            print(f"{name:<10} {'NOT FOUND':>10}")
            continue
        try:
            times = benchmark(path, args.warmup, args.runs)
            size = os.path.getsize(path) / 1e6
            print(
                f"{name:<10} {np.mean(times):>10.2f} {np.percentile(times, 50):>10.2f} {np.percentile(times, 99):>10.2f} {size:>10.1f}"
            )
        except Exception as e:
            print(f"{name:<10} FAILED: {e}")


if __name__ == "__main__":
    main()

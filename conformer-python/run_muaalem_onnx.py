import time
from pathlib import Path

import numpy as np
import onnx
import onnxruntime

PRECISIONS = ["float32", "float16", "int8"]
N_ITERATIONS = 10
WARMUP = 2
INPUT_SHAPE = (1, 49, 160)  # Matches model's actual input shape

OUTPUT_DIR = Path(__file__).parent.parent / "models"


def get_session_options():
    sess_options = onnxruntime.SessionOptions()
    sess_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
    return sess_options


def run_benchmark(model_path, precision):
    print(f"\n{'='*40}")
    print(f"Precision: {precision}")
    print(f"{'='*40}")

    if not model_path.exists():
        print(f"  Skipping: file not found at {model_path}")
        return None

    # Load time
    load_start = time.perf_counter()
    try:
        sess = onnxruntime.InferenceSession(str(model_path), get_session_options())
    except Exception as e:
        print(f"  Failed to load model: {e}")
        return None
    load_time = time.perf_counter() - load_start
    print(f"  Load time: {load_time*1000:.1f}ms")

    # Get I/O
    input_name = sess.get_inputs()[0].name
    output_names = [o.name for o in sess.get_outputs()]
    print(f"  Input: {input_name} {sess.get_inputs()[0].shape}")
    print(f"  Outputs: {output_names}")

    # Prepare input
    dtype = np.float16 if precision == "float16" else np.float32
    x = np.random.randn(*INPUT_SHAPE).astype(dtype)

    # Warmup
    for _ in range(WARMUP):
        sess.run(output_names, {input_name: x})

    # Benchmark
    times = []
    for _ in range(N_ITERATIONS):
        start = time.perf_counter()
        outputs = sess.run(output_names, {input_name: x})
        times.append(time.perf_counter() - start)

    avg_time = np.mean(times) * 1000
    std_time = np.std(times) * 1000
    min_time = np.min(times) * 1000
    max_time = np.max(times) * 1000

    print(f"  Inference times (ms): avg={avg_time:.2f}±{std_time:.2f}, min={min_time:.2f}, max={max_time:.2f}")
    print(f"  Output shapes: {[o.shape for o in outputs]}")

    return {
        "precision": precision,
        "load_time_ms": load_time * 1000,
        "avg_inference_ms": avg_time,
        "std_ms": std_time,
        "min_ms": min_time,
        "max_ms": max_time,
    }


def main():
    print("=" * 50)
    print("Muaalem ONNX Benchmark")
    print("=" * 50)
    print(f"Input shape: {INPUT_SHAPE}")
    print(f"Iterations: {N_ITERATIONS} (warmup: {WARMUP})")

    results = []
    for precision in PRECISIONS:
        model_path = OUTPUT_DIR / f"tiny_muaalem_{precision}.onnx"
        result = run_benchmark(model_path, precision)
        if result:
            results.append(result)

    # Summary
    print("\n" + "=" * 50)
    print("Summary")
    print("=" * 50)
    print(f"{'Precision':<10} {'Load (ms)':<12} {'Inference (ms)':<15} {'Std (ms)':<10}")
    print("-" * 50)
    for r in results:
        print(f"{r['precision']:<10} {r['load_time_ms']:<12.1f} {r['avg_inference_ms']:<15.2f} {r['std_ms']:<10.2f}")


if __name__ == "__main__":
    main()
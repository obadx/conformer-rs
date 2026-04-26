use ndarray::Array3;
use ort::{
    session::Session,
    value::{DynValue, Tensor},
};
use std::time::Instant;

const MODELS_DIR: &str = "/home/abdullah/Documents/courses/quran-works/conformer-rs/models";
const INPUT_SHAPE: (usize, usize, usize) = (1, 49, 160);
const N_ITERATIONS: usize = 10;
const WARMUP: usize = 2;

struct BenchmarkResult {
    precision: String,
    load_time_ms: f64,
    avg_inference_ms: f64,
}

fn create_input() -> DynValue {
    let arr: Array3<f32> = Array3::from_elem(INPUT_SHAPE, 0.1);
    Tensor::from_array(arr).unwrap().into_dyn()
}

fn run_benchmark(model_path: &str, precision: &str) -> Option<BenchmarkResult> {
    println!("\n{}", "=".repeat(40));
    println!("Precision: {precision}");
    println!("{}", "=".repeat(40));

    let load_start = Instant::now();
    let builder = Session::builder().expect("Failed to create session builder");
    let mut session = builder
        .with_optimization_level(ort::session::builder::GraphOptimizationLevel::Level3)
        .expect("Failed to set optimization level")
        .commit_from_file(model_path)
        .expect("Failed to load model");
    let load_time_ms = load_start.elapsed().as_secs_f64() * 1000.0;
    println!("  Load time: {load_time_ms:.1}ms");

    let input_info = &session.inputs()[0];
    let input_name = input_info.name().to_string();
    let input_dtype = input_info.dtype();
    let output_names: Vec<_> = session.outputs().iter().map(|o| o.name().to_string()).collect();

    println!("  Input: {input_name} [{}, {}, {}]", INPUT_SHAPE.0, INPUT_SHAPE.1, INPUT_SHAPE.2);
    println!("  Input dtype: {input_dtype:?}");
    println!("  Outputs: {:?}", output_names);

    let input_tensor = create_input();

    for _ in 0..WARMUP {
        let _ = session.run(ort::inputs! { input_name.clone() => &input_tensor });
    }

    let mut times = Vec::with_capacity(N_ITERATIONS);
    for _ in 0..N_ITERATIONS {
        let start = Instant::now();
        let _outputs = session.run(ort::inputs! { input_name.clone() => &input_tensor }).unwrap();
        let elapsed = start.elapsed().as_secs_f64() * 1000.0;
        times.push(elapsed);
    }

    let avg_time_ms = times.iter().sum::<f64>() / times.len() as f64;
    let std_ms = (times.iter().map(|t| (t - avg_time_ms).powi(2)).sum::<f64>() / times.len() as f64).sqrt();
    let min_ms = times.iter().cloned().fold(f64::INFINITY, f64::min);
    let max_ms = times.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

    println!(
        "  Inference times (ms): avg={avg_time_ms:.2}±{std_ms:.2}, min={min_ms:.2}, max={max_ms:.2}"
    );

    println!("  Output count: {}", output_names.len());

    Some(BenchmarkResult {
        precision: precision.to_string(),
        load_time_ms,
        avg_inference_ms: avg_time_ms,
    })
}

fn main() -> ort::Result<()> {
    println!("{}", "=".repeat(50));
    println!("Muaalem ONNX Benchmark (Rust/ort)");
    println!("{}", "=".repeat(50));
    println!("Input shape: {:?}", INPUT_SHAPE);
    println!("Iterations: {N_ITERATIONS} (warmup: {WARMUP})");
    println!("Note: Skipping float16 (requires native f16 support in Rust)");

    let precisions = ["float32", "int8"];
    let mut results = Vec::new();

    for precision in precisions {
        let model_path = format!("{MODELS_DIR}/tiny_muaalem_{precision}.onnx");
        if let Some(result) = run_benchmark(&model_path, precision) {
            results.push(result);
        }
    }

    println!("\n{}", "=".repeat(50));
    println!("Summary");
    println!("{}", "=".repeat(50));
    println!("{:10} {:12} {:15}", "Precision", "Load (ms)", "Inference (ms)");
    println!("{}", "-".repeat(50));
    for r in &results {
        println!(
            "{:10} {:12.1} {:15.2}",
            r.precision, r.load_time_ms, r.avg_inference_ms
        );
    }

    Ok(())
}
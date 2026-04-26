use clap::Parser;
use ndarray::Array3;
use ort::{
    session::Session,
    value::{DynValue, Tensor},
};
use ort_tract;
use std::{path::PathBuf, time::Instant};

#[derive(Parser, Debug)]
#[command(name = "conformer-rs")]
#[command(about = "Muaalem ONNX Benchmark - Rust/ort with tract backend")]
struct Args {
    /// Directory containing ONNX model files
    #[arg(short, long, default_value = "models")]
    models: PathBuf,

    /// Number of benchmark iterations
    #[arg(short, long, default_value = "10")]
    iterations: usize,

    /// Number of warmup iterations
    #[arg(short, long, default_value = "2")]
    warmup: usize,

    /// Input shape (batch, time, features)
    #[arg(long, default_value = "1,49,160")]
    shape: String,
}

struct BenchmarkResult {
    precision: String,
    load_time_ms: f64,
    avg_inference_ms: f64,
    std_ms: f64,
    min_ms: f64,
    max_ms: f64,
}

fn parse_shape(s: &str) -> (usize, usize, usize) {
    let parts: Vec<usize> = s
        .split(',')
        .map(|p| p.trim().parse().unwrap())
        .collect();
    if parts.len() != 3 {
        panic!("Invalid shape: expected format 'batch,time,features'");
    }
    (parts[0], parts[1], parts[2])
}

fn find_onnx_models(models_dir: &PathBuf) -> Vec<(String, PathBuf)> {
    let mut models = Vec::new();
    
    if !models_dir.exists() {
        eprintln!("Error: Models directory '{}' does not exist", models_dir.display());
        return models;
    }

    let entries = match std::fs::read_dir(models_dir) {
        Ok(e) => e,
        Err(e) => {
            eprintln!("Error reading directory '{}': {}", models_dir.display(), e);
            return models;
        }
    };

    for entry in entries.flatten() {
        let path = entry.path();
        if path.extension().map_or(false, |ext| ext == "onnx") {
            if let Some(name) = path.file_stem() {
                let name_str = name.to_string_lossy().to_string();
                models.push((name_str, path));
            }
        }
    }

    models.sort_by(|a, b| a.0.cmp(&b.0));
    models
}

fn create_input(shape: (usize, usize, usize)) -> DynValue {
    let arr: Array3<f32> = Array3::from_elem(shape, 0.1);
    Tensor::from_array(arr).unwrap().into_dyn()
}

fn run_benchmark(
    model_path: &PathBuf,
    precision: &str,
    input_shape: (usize, usize, usize),
    n_iterations: usize,
    warmup: usize,
) -> Option<BenchmarkResult> {
    println!("\n{}", "=".repeat(40));
    println!("Precision: {precision}");
    println!("{}", "=".repeat(40));

    let load_start = Instant::now();
    let builder = match Session::builder() {
        Ok(b) => b,
        Err(e) => {
            println!("  Failed to create session builder: {e}");
            return None;
        }
    };
    let mut session = match builder
        .with_optimization_level(ort::session::builder::GraphOptimizationLevel::Level3)
    {
        Ok(mut b) => match b.commit_from_file(model_path) {
            Ok(s) => s,
            Err(e) => {
                println!("  Failed to load model: {e}");
                return None;
            }
        },
        Err(e) => {
            println!("  Failed to set optimization level: {e}");
            return None;
        }
    };
    let load_time_ms = load_start.elapsed().as_secs_f64() * 1000.0;
    println!("  Load time: {load_time_ms:.1}ms");

    let input_info = &session.inputs()[0];
    let input_name = input_info.name().to_string();
    let input_dtype = input_info.dtype();
    let output_names: Vec<_> = session.outputs().iter().map(|o| o.name().to_string()).collect();

    println!(
        "  Input: {} {:?}",
        input_name,
        input_info.dtype().tensor_shape()
    );
    println!("  Input dtype: {input_dtype:?}");
    println!("  Outputs: {output_names:?}");

    let input_tensor = create_input(input_shape);

    for _ in 0..warmup {
        let _ = session.run(ort::inputs! { input_name.clone() => &input_tensor });
    }

    let mut times = Vec::with_capacity(n_iterations);
    for _ in 0..n_iterations {
        let start = Instant::now();
        let _outputs = session
            .run(ort::inputs! { input_name.clone() => &input_tensor })
            .unwrap();
        let elapsed = start.elapsed().as_secs_f64() * 1000.0;
        times.push(elapsed);
    }

    let avg_time_ms = times.iter().sum::<f64>() / times.len() as f64;
    let std_ms = (times
        .iter()
        .map(|t| (t - avg_time_ms).powi(2))
        .sum::<f64>()
        / times.len() as f64)
        .sqrt();
    let min_ms = times.iter().cloned().fold(f64::INFINITY, f64::min);
    let max_ms = times.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

    println!(
        "  Inference (ms): avg={avg_time_ms:.2}±{std_ms:.2}, min={min_ms:.2}, max={max_ms:.2}"
    );
    println!("  Output count: {}", output_names.len());

    Some(BenchmarkResult {
        precision: precision.to_string(),
        load_time_ms,
        avg_inference_ms: avg_time_ms,
        std_ms,
        min_ms,
        max_ms,
    })
}

fn main() -> ort::Result<()> {
    ort::set_api(ort_tract::api());

    let args = Args::parse();
    let input_shape = parse_shape(&args.shape);

    println!("{}", "=".repeat(50));
    println!("Muaalem ONNX Benchmark (Rust/ort-tract)");
    println!("{}", "=".repeat(50));
    println!("Models dir: {}", args.models.display());
    println!("Input shape: {:?}", input_shape);
    println!("Iterations: {} (warmup: {})", args.iterations, args.warmup);

    let models = find_onnx_models(&args.models);
    if models.is_empty() {
        eprintln!("No .onnx files found in '{}'", args.models.display());
        return Ok(());
    }

    println!("Found {} model(s)", models.len());

    let mut results = Vec::new();

    for (name, path) in models {
        if let Some(result) = run_benchmark(&path, &name, input_shape, args.iterations, args.warmup)
        {
            results.push(result);
        }
    }

    if results.is_empty() {
        println!("\nNo models succeeded!");
        return Ok(());
    }

    println!("\n{}", "=".repeat(50));
    println!("Summary");
    println!("{}", "=".repeat(50));
    println!(
        "{:20} {:12} {:15} {:10}",
        "Model", "Load (ms)", "Inference (ms)", "Std (ms)"
    );
    println!("{}", "-".repeat(50));
    for r in &results {
        println!(
            "{:20} {:12.1} {:15.2} {:10.2}",
            r.precision, r.load_time_ms, r.avg_inference_ms, r.std_ms
        );
    }

    Ok(())
}
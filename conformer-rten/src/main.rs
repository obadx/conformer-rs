use clap::Parser;
use rten::{Model, RunOptions, Value};
use std::path::PathBuf;
use std::time::Instant;

#[derive(Parser, Debug)]
#[command(name = "conformer-rten")]
#[command(about = "Muaalem ONNX Benchmark - Rust/rten backend")]
struct Args {
    #[arg(short, long, default_value = "models")]
    models: PathBuf,

    #[arg(short, long, default_value = "10")]
    iterations: usize,

    #[arg(short, long, default_value = "2")]
    warmup: usize,

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
    let parts: Vec<usize> = s.split(',').map(|p| p.trim().parse().unwrap()).collect();
    if parts.len() != 3 {
        panic!("Invalid shape: expected format 'batch,time,features'");
    }
    (parts[0], parts[1], parts[2])
}

fn find_onnx_models(models_dir: &PathBuf) -> Vec<(String, PathBuf)> {
    let mut models = Vec::new();

    if !models_dir.exists() {
        eprintln!(
            "Error: Models directory '{}' does not exist",
            models_dir.display()
        );
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

fn create_input(shape: (usize, usize, usize)) -> Value {
    let (b, t, f) = shape;
    let data: Vec<f32> = vec![0.1; b * t * f];
    Value::from_shape(&[b, t, f], data).expect("failed to create input tensor")
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
    let model = match Model::load_file(model_path) {
        Ok(m) => m,
        Err(e) => {
            println!("  Failed to load model: {e}");
            return None;
        }
    };
    let load_time_ms = load_start.elapsed().as_secs_f64() * 1000.0;
    println!("  Load time: {load_time_ms:.1}ms");

    let input_ids = model.input_ids();
    let output_ids = model.output_ids();

    if input_ids.is_empty() {
        println!("  No input nodes found");
        return None;
    }

    let input_id = input_ids[0];
    let input_info = match model.node_info(input_id) {
        Some(info) => info,
        None => {
            println!("  No info for input node");
            return None;
        }
    };

    let input_name = input_info.name().unwrap_or("input");
    let input_shape_info = model.input_shape(0);

    println!("  Input: {input_name} {:?}", input_shape_info);
    println!("  Outputs: {} outputs", output_ids.len());
    if output_ids.len() <= 10 {
        for &oid in output_ids {
            if let Some(info) = model.node_info(oid) {
                println!("    - {}", info.name().unwrap_or("?"));
            }
        }
    } else {
        for &oid in output_ids.iter().take(3) {
            if let Some(info) = model.node_info(oid) {
                println!("    - {}", info.name().unwrap_or("?"));
            }
        }
        println!("    ... and {} more", output_ids.len() - 3);
    }

    let input_tensor = create_input(input_shape);

    let inputs = vec![(input_id, input_tensor.into())];
    let outputs: Vec<_> = output_ids.to_vec();

    let opts = RunOptions::default();

    for _ in 0..warmup {
        let _ = model.run(inputs.clone(), &outputs, Some(opts.clone()));
    }

    let mut times = Vec::with_capacity(n_iterations);
    for _ in 0..n_iterations {
        let start = Instant::now();
        match model.run(inputs.clone(), &outputs, Some(opts.clone())) {
            Ok(_) => {}
            Err(e) => {
                println!("  Inference error: {e}");
                return None;
            }
        }
        let elapsed = start.elapsed().as_secs_f64() * 1000.0;
        times.push(elapsed);
    }

    let avg_time_ms = times.iter().sum::<f64>() / times.len() as f64;
    let std_ms =
        (times.iter().map(|t| (t - avg_time_ms).powi(2)).sum::<f64>() / times.len() as f64).sqrt();
    let min_ms = times.iter().cloned().fold(f64::INFINITY, f64::min);
    let max_ms = times.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

    println!(
        "  Inference (ms): avg={avg_time_ms:.2}±{std_ms:.2}, min={min_ms:.2}, max={max_ms:.2}"
    );
    println!("  Output count: {}", output_ids.len());

    Some(BenchmarkResult {
        precision: precision.to_string(),
        load_time_ms,
        avg_inference_ms: avg_time_ms,
        std_ms,
        min_ms,
        max_ms,
    })
}

fn main() {
    let args = Args::parse();
    let input_shape = parse_shape(&args.shape);

    println!("{}", "=".repeat(50));
    println!("Muaalem ONNX Benchmark (Rust/rten)");
    println!("{}", "=".repeat(50));
    println!("Models dir: {}", args.models.display());
    println!("Input shape: {:?}", input_shape);
    println!("Iterations: {} (warmup: {})", args.iterations, args.warmup);

    let models = find_onnx_models(&args.models);
    if models.is_empty() {
        eprintln!("No .onnx files found in '{}'", args.models.display());
        return;
    }

    println!("Found {} model(s)", models.len());

    let mut results = Vec::new();

    for (name, path) in &models {
        if let Some(result) = run_benchmark(path, name, input_shape, args.iterations, args.warmup) {
            results.push(result);
        }
    }

    if results.is_empty() {
        println!("\nNo models succeeded!");
        return;
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
}

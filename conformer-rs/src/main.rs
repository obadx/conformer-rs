use ndarray::Array3;
use std::time::Instant;
use tract_nnef::prelude::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model_path = std::env::args_os()
        .nth(1)
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| {
            let exe_path = std::env::current_exe().unwrap();
            exe_path.parent().unwrap().join("model.nnef")
        });

    println!("Using model: {:?}", model_path);

    let load_start = Instant::now();
    let model = tract_nnef::nnef()
        .model_for_path(&model_path)?
        .into_optimized()?
        .into_runnable()?;
    let load_time_ms = load_start.elapsed().as_millis();

    println!("Model loaded in {} ms", load_time_ms);

    let input: Array3<f32> = Array3::zeros((1, 50, 144));
    let input: Tensor = input.into();

    let infer_start = Instant::now();
    let result = model.run(tvec![input.into()])?;
    let infer_time_ms = infer_start.elapsed().as_millis();

    let output = result[0].to_array_view::<f32>()?;

    println!("Output shape: {:?}", output.shape());
    println!("Inference time: {} ms", infer_time_ms);
    println!("Total time: {} ms", load_time_ms + infer_time_ms);

    Ok(())
}


use std::path::PathBuf;
use tract_ndarray::Array3;
use tract_onnx::prelude::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model_path = std::env::args_os()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            let exe_path = std::env::current_exe().unwrap();
            exe_path.parent().unwrap().join("model.onnx")
        });

    println!("Using model: {:?}", model_path);

    let model = tract_onnx::onnx()
        .model_for_path(&model_path)?
        .into_runnable()?;

    println!("Model loaded sucessfully!");

    let input: Array3<f32> = Array3::zeros((1, 50, 144));
    let input: Tensor = input.into();

    let result = model.run(tvec![input.into()])?;

    let output = result[0].to_array_view::<f32>()?;

    println!("Output shape: {:?}", output.shape());

    Ok(())
}


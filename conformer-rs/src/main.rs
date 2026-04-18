use tract_onnx::prelude::*;
use tract_ndarray::Array3;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model_path = concat!(env!("CARGO_MANIFEST_DIR"), "/model.onnx");
    
    let model = tract_onnx::onnx()
        .model_for_path(model_path)?
        .into_runnable()?;
    
    let input: Array3<f32> = Array3::zeros((1, 50, 144));
    let input: Tensor = input.into();
    
    let result = model.run(tvec![input.into()])?;
    
    let output = result[0].to_array_view::<f32>()?;
    
    println!("Output shape: {:?}", output.shape());
    
    Ok(())
}
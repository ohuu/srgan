use burn_import::onnx::ModelGen;

fn main() {
    if std::env::var("CARGO_FEATURE_BUILD_VGG").is_ok() {
        println!("builing vgg19!");
        ModelGen::new()
            .input("models/vgg19.onnx")
            .out_dir("model/")
            .run_from_script();
    }
}

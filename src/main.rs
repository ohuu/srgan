mod data;
mod model;
mod training;
mod utils;

use burn::{
    backend::{Autodiff, Wgpu},
    optim::AdamConfig,
    tensor::{Distribution, Tensor},
};
use model::ModelConfig;
use std::{error::Error, path::Path};
use training::{train, TrainingConfig};
use utils::save_image;

fn main() -> Result<(), Box<dyn Error>> {
    type MyBackend = Wgpu<f32>;
    type MyAutodiffBackend = Autodiff<MyBackend>;

    let device = burn::backend::wgpu::WgpuDevice::default();
    let artifact_dir = "/tmp/srgan";

    // initialise the model
    let model = ModelConfig::new().init::<MyBackend>(&device);

    // initialise training params and train
    let config = TrainingConfig::new(ModelConfig::new(), AdamConfig::new());
    train::<MyAutodiffBackend>(artifact_dir, config, &device);

    // print out some images
    // create a [32, 10] noise tensor so the generator can make 32 new images
    let batch_size = 32;
    let noise = Tensor::<MyBackend, 2>::empty([batch_size, 10], &device);
    let noise = noise.random_like(Distribution::Uniform(0.0, 1.0));
    let images = model.generator.forward(noise);

    // save a slice of the images
    save_image(
        images,
        4,
        Path::new(&format!("{}/generated.png", artifact_dir)),
    )
    .unwrap();

    Ok(())
}

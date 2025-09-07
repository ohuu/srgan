mod model;
mod utils;

#[cfg(feature = "ndarray")]
use burn::backend::ndarray::NdArray;
#[cfg(not(feature = "ndarray"))]
use burn::backend::wgpu::Wgpu;

use burn::{
    module::Module,
    record::{FullPrecisionSettings, NamedMpkFileRecorder, Recorder},
    tensor::{Device, Tensor},
};
use image::RgbImage;
use std::{error::Error, path::PathBuf, str::FromStr};

use crate::{
    model::generator::{Generator, GeneratorConfig, GeneratorRecord},
    utils::{image_to_tensor, tensor_to_image},
};

#[cfg(feature = "ndarray")]
type MyBackend = NdArray<f32>;
#[cfg(not(feature = "ndarray"))]
type MyBackend = Wgpu<f32>;

pub struct Model {
    pub generator: Generator<MyBackend>,
    pub device: Device<MyBackend>,
}

impl Model {
    pub fn new() -> Self {
        let device = Default::default();

        // load model
        let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::new();
        let path = PathBuf::from_str("/home/ohuu/Documents/srgan_out/generator").expect("");
        let record: GeneratorRecord<MyBackend> = recorder
            .load(path.into(), &device)
            .expect("Should be able to load generator model.");

        let generator = GeneratorConfig::new().init(&device).load_record(record);

        Self { generator, device }
    }

    pub fn generate(&self, image: RgbImage) -> Result<RgbImage, Box<dyn Error>> {
        let image_tensor = image_to_tensor(image, &self.device).unsqueeze();

        let sr_tensor: Tensor<MyBackend, 3> = self.generator.forward(image_tensor).squeeze(0);
        let sr_image = tensor_to_image(sr_tensor);

        Ok(sr_image)
    }
}

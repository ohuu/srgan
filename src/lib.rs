mod model;

use crate::model::generator::{Generator, GeneratorConfig};
use burn::{
    backend::NdArray,
    tensor::{Device, Tensor, TensorData},
};
use wasm_bindgen::prelude::*;

type MyBackend = NdArray<f32>;

#[wasm_bindgen]
pub struct Model {
    generator: Generator<MyBackend>,
    device: Device<MyBackend>,
}

#[wasm_bindgen]
impl Model {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        let device = Default::default();
        let generator = GeneratorConfig::new().init(&device);

        Self { generator, device }
    }

    #[wasm_bindgen]
    pub fn generate(
        &self,
        image: Vec<f32>,
        width: usize,
        height: usize,
    ) -> Result<Vec<f32>, JsValue> {
        let image_tensor = image.into_iter().map(|c| c as u8).collect::<Vec<_>>();
        let image_tensor = Tensor::<MyBackend, 3>::from_data(
            TensorData::new(image_tensor, [3, width, height]),
            &self.device,
        )
        .unsqueeze();

        let sr_image: Tensor<MyBackend, 3> = self.generator.forward(image_tensor).squeeze(0);
        let data = sr_image
            .to_data()
            .to_vec::<f32>()
            .unwrap()
            .iter()
            .map(|c| (255.0 * ((c + 1.0) / 2.0)))
            .collect::<Vec<_>>();

        Ok(data)
    }
}

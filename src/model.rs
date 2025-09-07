pub mod discriminator;
pub mod generator;
mod layers;
pub mod vgg19;

use burn::{module::Module, prelude::*};

use crate::model::{
    discriminator::{Discriminator, DiscriminatorConfig},
    generator::{Generator, GeneratorConfig},
};

#[derive(Module, Debug)]
pub struct Model<B: Backend> {
    pub generator: Generator<B>,
    pub discriminator: Discriminator<B>,
}

#[derive(Config, Debug)]
pub struct ModelConfig {
    pub generator_config: GeneratorConfig,
    pub discriminator_config: DiscriminatorConfig,
}
impl ModelConfig {
    #[allow(unused)]
    pub fn init<B: Backend>(&self, device: &B::Device) -> Model<B> {
        let generator = self.generator_config.init(device);
        let discriminator = self.discriminator_config.init(device);

        Model {
            generator,
            discriminator,
        }
    }
}

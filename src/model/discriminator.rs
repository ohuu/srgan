use burn::{
    module::Module,
    nn::{
        conv::{Conv2d, Conv2dConfig},
        LeakyRelu, LeakyReluConfig, Linear, LinearConfig, PaddingConfig2d, Sigmoid,
    },
    prelude::*,
};

use crate::model::layers::DiscBlock;

#[derive(Module, Debug)]
pub struct Discriminator<B: Backend> {
    in_layer: Conv2d<B>,
    disc_layer_1: DiscBlock<B>,
    disc_layer_2: DiscBlock<B>,
    disc_layer_3: DiscBlock<B>,
    disc_layer_4: DiscBlock<B>,
    disc_layer_5: DiscBlock<B>,
    disc_layer_6: DiscBlock<B>,
    disc_layer_7: DiscBlock<B>,
    fc_layer: Linear<B>,
    out_layer: Linear<B>,
    lrelu: LeakyRelu,
    sig: Sigmoid,
}
impl<B: Backend> Discriminator<B> {
    #[allow(unused)]
    pub fn forward(&self, input: Tensor<B, 4>) -> Tensor<B, 2> {
        let output = self.in_layer.forward(input);
        let output = self.lrelu.forward(output);

        let output = self.disc_layer_1.forward(output);
        let output = self.disc_layer_2.forward(output);
        let output = self.disc_layer_3.forward(output);
        let output = self.disc_layer_4.forward(output);
        let output = self.disc_layer_5.forward(output);
        let output = self.disc_layer_6.forward(output);
        let output = self.disc_layer_7.forward(output);

        let output = output.flatten(1, 3);

        let output = self.fc_layer.forward(output);
        let output = self.lrelu.forward(output);
        let output = self.out_layer.forward(output);

        self.sig.forward(output).clamp(0.00001, 0.99999)
    }
}

#[derive(Config, Debug)]
pub struct DiscriminatorConfig {
    pub hr_size: usize,
}
impl DiscriminatorConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> Discriminator<B> {
        let in_layer = Conv2dConfig::new([3, 64], [3, 3])
            .with_padding(PaddingConfig2d::Same)
            .init(device);

        let disc_layer_1 = DiscBlock::new([64, 64], 2, device);
        let disc_layer_2 = DiscBlock::new([64, 128], 1, device);
        let disc_layer_3 = DiscBlock::new([128, 128], 2, device);
        let disc_layer_4 = DiscBlock::new([128, 256], 1, device);
        let disc_layer_5 = DiscBlock::new([256, 256], 2, device);
        let disc_layer_6 = DiscBlock::new([256, 512], 1, device);
        let disc_layer_7 = DiscBlock::new([512, 512], 2, device);
        let channels = 512;
        let pixels = self.hr_size / 16;

        let fc_layer = LinearConfig::new(channels * pixels * pixels, 1024).init(device);
        let out_layer = LinearConfig::new(1024, 1).init(device);
        let lrelu = LeakyReluConfig::new().init();
        let sig = Sigmoid::new();

        Discriminator {
            in_layer,
            disc_layer_1,
            disc_layer_2,
            disc_layer_3,
            disc_layer_4,
            disc_layer_5,
            disc_layer_6,
            disc_layer_7,
            fc_layer,
            out_layer,
            lrelu,
            sig,
        }
    }
}

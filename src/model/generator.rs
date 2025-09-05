use burn::{
    module::Module,
    nn::{
        conv::{Conv2d, Conv2dConfig},
        BatchNorm, BatchNormConfig, PRelu, PReluConfig, PaddingConfig2d,
    },
    prelude::*,
};

use crate::model::layers::{ResidualBlock, UpscaleBlock};

#[derive(Module, Debug)]
pub struct Generator<B: Backend> {
    in_layer: Conv2d<B>,
    res_layer_01: ResidualBlock<B>,
    res_layer_02: ResidualBlock<B>,
    res_layer_03: ResidualBlock<B>,
    res_layer_04: ResidualBlock<B>,
    res_layer_05: ResidualBlock<B>,
    res_layer_06: ResidualBlock<B>,
    res_layer_07: ResidualBlock<B>,
    res_layer_08: ResidualBlock<B>,
    res_layer_09: ResidualBlock<B>,
    res_layer_10: ResidualBlock<B>,
    res_layer_11: ResidualBlock<B>,
    res_layer_12: ResidualBlock<B>,
    res_layer_13: ResidualBlock<B>,
    res_layer_14: ResidualBlock<B>,
    res_layer_15: ResidualBlock<B>,
    res_layer_16: ResidualBlock<B>,
    post_res_layer: Conv2d<B>,
    batch_norm: BatchNorm<B, 2>,
    up_layer_1: UpscaleBlock<B>,
    up_layer_2: UpscaleBlock<B>,
    out_layer: Conv2d<B>,
    prelu: PRelu<B>,
}
impl<B: Backend> Generator<B> {
    pub fn forward(&self, input: Tensor<B, 4>) -> Tensor<B, 4> {
        let in_output = self.in_layer.forward(input);
        let in_output = self.prelu.forward(in_output);

        let res_output = self.res_layer_01.forward(in_output.clone());
        let res_output = self.res_layer_02.forward(res_output);
        let res_output = self.res_layer_03.forward(res_output);
        let res_output = self.res_layer_04.forward(res_output);
        let res_output = self.res_layer_05.forward(res_output);
        let res_output = self.res_layer_06.forward(res_output);
        let res_output = self.res_layer_07.forward(res_output);
        let res_output = self.res_layer_08.forward(res_output);
        let res_output = self.res_layer_09.forward(res_output);
        let res_output = self.res_layer_10.forward(res_output);
        let res_output = self.res_layer_11.forward(res_output);
        let res_output = self.res_layer_12.forward(res_output);
        let res_output = self.res_layer_13.forward(res_output);
        let res_output = self.res_layer_14.forward(res_output);
        let res_output = self.res_layer_15.forward(res_output);
        let res_output = self.res_layer_16.forward(res_output);

        let post_res_output = self.post_res_layer.forward(res_output);
        let post_res_output = self.batch_norm.forward(post_res_output);
        let post_res_output = in_output.add(post_res_output);

        let up_output = self.up_layer_1.forward(post_res_output);
        let up_output = self.up_layer_2.forward(up_output);

        let out = self.out_layer.forward(up_output);

        burn::tensor::activation::tanh(out)
    }
}

#[derive(Config, Debug)]
pub struct GeneratorConfig;
impl GeneratorConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> Generator<B> {
        let in_layer = Conv2dConfig::new([3, 64], [9, 9])
            .with_padding(PaddingConfig2d::Same)
            .init(device);

        let res_layer_01 = ResidualBlock::new([64, 64], device);
        let res_layer_02 = ResidualBlock::new([64, 64], device);
        let res_layer_03 = ResidualBlock::new([64, 64], device);
        let res_layer_04 = ResidualBlock::new([64, 64], device);
        let res_layer_05 = ResidualBlock::new([64, 64], device);
        let res_layer_06 = ResidualBlock::new([64, 64], device);
        let res_layer_07 = ResidualBlock::new([64, 64], device);
        let res_layer_08 = ResidualBlock::new([64, 64], device);
        let res_layer_09 = ResidualBlock::new([64, 64], device);
        let res_layer_10 = ResidualBlock::new([64, 64], device);
        let res_layer_11 = ResidualBlock::new([64, 64], device);
        let res_layer_12 = ResidualBlock::new([64, 64], device);
        let res_layer_13 = ResidualBlock::new([64, 64], device);
        let res_layer_14 = ResidualBlock::new([64, 64], device);
        let res_layer_15 = ResidualBlock::new([64, 64], device);
        let res_layer_16 = ResidualBlock::new([64, 64], device);

        let post_res_layer = Conv2dConfig::new([64, 64], [3, 3])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .init(device);
        let batch_norm = BatchNormConfig::new(64)
            .with_momentum(0.5)
            .init::<B, 2>(device);

        let up_layer_1 = UpscaleBlock::new([64, 256], device);
        let up_layer_2 = UpscaleBlock::new([64, 256], device);

        let out_layer = Conv2dConfig::new([64, 3], [9, 9])
            .with_padding(PaddingConfig2d::Same)
            .init(device);
        let prelu = PReluConfig::new().init(device);

        Generator {
            in_layer,
            res_layer_01,
            res_layer_02,
            res_layer_03,
            res_layer_04,
            res_layer_05,
            res_layer_06,
            res_layer_07,
            res_layer_08,
            res_layer_09,
            res_layer_10,
            res_layer_11,
            res_layer_12,
            res_layer_13,
            res_layer_14,
            res_layer_15,
            res_layer_16,
            post_res_layer,
            batch_norm,
            up_layer_1,
            up_layer_2,
            out_layer,
            prelu,
        }
    }
}

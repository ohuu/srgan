use burn::{
    nn::{
        conv::{Conv2d, Conv2dConfig},
        BatchNorm, BatchNormConfig, LeakyRelu, LeakyReluConfig, PRelu, PReluConfig,
        PaddingConfig2d,
    },
    prelude::*,
};

#[derive(Module, Debug)]
pub struct DiscBlock<B: Backend> {
    conv: Conv2d<B>,
    bn: BatchNorm<B, 2>,
    lrelu: LeakyRelu,
}
impl<B: Backend> DiscBlock<B> {
    pub fn new(channels: [usize; 2], stride: usize, device: &B::Device) -> Self {
        let conv = Conv2dConfig::new(channels, [3, 3])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .with_stride([stride, stride])
            .init(device);
        let bn = BatchNormConfig::new(channels[1])
            .with_momentum(0.8)
            .init(device);
        let lrelu = LeakyReluConfig::new().with_negative_slope(0.2).init();

        Self { conv, bn, lrelu }
    }

    #[allow(unused)]
    pub fn forward(&self, input: Tensor<B, 4>) -> Tensor<B, 4> {
        let output = self.conv.forward(input);
        let output = self.bn.forward(output);
        self.lrelu.forward(output)
    }
}

#[derive(Module, Debug)]
pub struct UpscaleBlock<B: Backend> {
    conv: Conv2d<B>,
    shuffle: PixelShuffler,
    prelu: PRelu<B>,
}
impl<B: Backend> UpscaleBlock<B> {
    pub fn new(channels: [usize; 2], device: &B::Device) -> Self {
        let conv = Conv2dConfig::new(channels, [3, 3])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .init(device);
        let shuffle = PixelShuffler::new();
        let prelu = PReluConfig::new().init(device);

        Self {
            conv,
            shuffle,
            prelu,
        }
    }

    pub fn forward(&self, input: Tensor<B, 4>) -> Tensor<B, 4> {
        let output = self.conv.forward(input);
        let output = self.shuffle.forward(output);
        self.prelu.forward(output)
    }
}

#[derive(Module, Debug, Clone)]
pub struct PixelShuffler {}
impl PixelShuffler {
    pub fn new() -> Self {
        Self {}
    }

    pub fn forward<B: Backend>(&self, input: Tensor<B, 4>) -> Tensor<B, 4> {
        let [batch_size, channels, height, width] = input.dims();

        let output_channels = channels / (2 * 2);
        let output_height = height * 2;
        let output_width = width * 2;

        // Reshape: [B, C, H, W] -> [B, output_channels, factor, factor, H, W]
        let reshaped = input.reshape([batch_size, output_channels, 2, 2, height, width]);

        // Permute dimensions: [B, output_channels, factor_h, factor_w, H, W]
        //                  -> [B, output_channels, H, factor_w, factor_h, W]
        //                  -> [B, output_channels, H, W, factor_h, factor_w]
        //                  -> [B, output_channels, H, factor_h, W, factor_w]
        let permuted = reshaped.swap_dims(2, 4).swap_dims(3, 5).swap_dims(3, 4);

        // Final reshape to get output: [B, output_channels, H*factor, W*factor]
        permuted.reshape([batch_size, output_channels, output_height, output_width])
    }
}

#[derive(Module, Debug)]
pub struct ResidualBlock<B: Backend> {
    conv1: Conv2d<B>,
    bn1: BatchNorm<B, 2>,
    conv2: Conv2d<B>,
    bn2: BatchNorm<B, 2>,
    prelu: PRelu<B>,
}
impl<B: Backend> ResidualBlock<B> {
    pub fn new(channels: [usize; 2], device: &B::Device) -> Self {
        let conv1 = Conv2dConfig::new(channels, [3, 3])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .init(device);
        let bn1 = nn::BatchNormConfig::new(channels[1])
            .with_momentum(0.5)
            .init(device);
        let conv2 = Conv2dConfig::new([channels[1], channels[1]], [3, 3])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .init(device);
        let bn2 = nn::BatchNormConfig::new(channels[1])
            .with_momentum(0.5)
            .init(device);
        let prelu = PReluConfig::new().init(device);

        Self {
            conv1,
            bn1,
            conv2,
            bn2,
            prelu,
        }
    }

    pub fn forward(&self, input: Tensor<B, 4>) -> Tensor<B, 4> {
        let output = self.conv1.forward(input.clone());
        let output = self.bn1.forward(output);
        let output = self.prelu.forward(output);
        let output = self.conv2.forward(output);
        let output = self.bn2.forward(output);

        input.add(output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::MyBackend;

    #[test]
    fn test_pixel_shuffle_values() {
        #[cfg(not(feature = "ndarray"))]
        let device = burn::backend::wgpu::WgpuDevice::default();
        #[cfg(feature = "ndarray")]
        let device = burn::backend::ndarray::NdArray::default();

        // Input: [1, 4, 3, 3] -> Output should be [1, 1, 4, 4]
        // We'll put different values in each channel to track them
        let input_data = [[
            [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]],
            [[2.0, 2.0, 2.0], [2.0, 2.0, 2.0], [2.0, 2.0, 2.0]],
            [[3.0, 3.0, 3.0], [3.0, 3.0, 3.0], [3.0, 3.0, 3.0]],
            [[4.0, 4.0, 4.0], [4.0, 4.0, 4.0], [4.0, 4.0, 4.0]],
        ]];

        let input = Tensor::<MyBackend, 4>::from_data(input_data, &device);
        let pixel_shuffle = PixelShuffler::new();
        let output = pixel_shuffle.forward(input);

        println!("Output dims: {:?}", output.dims());
        let data = output.to_data();
        let [_, _, rows, cols] = output.dims();
        let values = data.to_vec::<f32>().unwrap();
        for i in 0..rows {
            print!("[");
            for j in 0..cols {
                let idx = i * cols + j;
                print!("{:8.4}", values[idx]);
                if j < cols - 1 {
                    print!(", ");
                }
            }
            println!("]");
        }
        // Expected output should be:
        // [[[
        //   [1.0, 2.0],
        //   [3.0, 4.0]
        // ]]]
    }
}

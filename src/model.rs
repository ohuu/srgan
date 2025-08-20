use burn::{
    module::Module,
    nn::{
        conv::{Conv2d, Conv2dConfig},
        BatchNorm, BatchNormConfig, LeakyRelu, LeakyReluConfig, Linear, LinearConfig, PRelu,
        PReluConfig, PaddingConfig2d, Sigmoid,
    },
    prelude::*,
};

/// Layer block of generator model
#[derive(Module, Debug)]
pub struct ResidualBlock<B: Backend> {
    conv1: Conv2d<B>,
    bn1: BatchNorm<B, 0>,
    conv2: Conv2d<B>,
    bn2: BatchNorm<B, 0>,
    prelu: PRelu<B>,
}

impl<B: Backend> ResidualBlock<B> {
    pub fn new(channels: [usize; 2], device: &B::Device) -> Self {
        let conv1 = Conv2dConfig::new(channels, [3, 3])
            .with_padding(PaddingConfig2d::Same)
            .init(device);
        let bn1: BatchNorm<B, 0> = nn::BatchNormConfig::new(channels[1])
            .with_momentum(0.5)
            .init(device);
        let conv2 = Conv2dConfig::new([channels[1], channels[1]], [3, 3])
            .with_padding(PaddingConfig2d::Same)
            .init(device);
        let bn2: BatchNorm<B, 0> = nn::BatchNormConfig::new(channels[1])
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

#[derive(Module, Debug, Clone)]
pub struct PixelShuffler {
    pub upscale_factor: usize,
}

impl PixelShuffler {
    pub fn new(upscale_factor: usize) -> Self {
        Self { upscale_factor }
    }

    pub fn forward<B: Backend>(&self, input: Tensor<B, 4>) -> Tensor<B, 4> {
        let [batch_size, channels, height, width] = input.dims();
        let factor = self.upscale_factor;
        let factor_squared = factor * factor;

        // Ensure channels is divisible by upscale_factor²
        assert_eq!(
            channels % factor_squared,
            0,
            "Input channels must be divisible by upscale_factor²"
        );

        let output_channels = channels / factor_squared;
        let output_height = height * factor;
        let output_width = width * factor;

        // Reshape: [B, C, H, W] -> [B, output_channels, factor, factor, H, W]
        let reshaped = input.reshape([batch_size, output_channels, factor, factor, height, width]);

        // Permute dimensions: [B, output_channels, factor, factor, H, W]
        // -> [B, output_channels, H, factor, W, factor]
        let permuted = reshaped.swap_dims(2, 4).swap_dims(3, 5);

        // Final reshape to get output: [B, output_channels, H*factor, W*factor]
        permuted.reshape([batch_size, output_channels, output_height, output_width])
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
            .with_padding(PaddingConfig2d::Same)
            .init(device);
        let shuffle = PixelShuffler::new(2);
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

#[derive(Module, Debug)]
pub struct DiscBlock<B: Backend> {
    conv: Conv2d<B>,
    bn: BatchNorm<B, 2>,
    lrelu: LeakyRelu,
}

impl<B: Backend> DiscBlock<B> {
    pub fn new(channels: [usize; 2], stride: usize, device: &B::Device) -> Self {
        let conv = Conv2dConfig::new(channels, [3, 3])
            .with_padding(PaddingConfig2d::Same)
            .with_stride([stride, stride])
            .init(device);
        let bn = BatchNormConfig::new(channels[1])
            .with_momentum(0.8)
            .init(device);
        let lrelu = LeakyReluConfig::new().with_negative_slope(0.2).init();

        Self { conv, bn, lrelu }
    }

    pub fn forward(&self, input: Tensor<B, 4>) -> Tensor<B, 4> {
        let output = self.conv.forward(input);
        let output = self.bn.forward(output);
        self.lrelu.forward(output)
    }
}

/// Generator model
#[derive(Module, Debug)]
pub struct Generator<B: Backend> {
    in_layer: Conv2d<B>,
    res_layers: Vec<ResidualBlock<B>>,
    post_res_layer: Conv2d<B>,
    batch_norm: BatchNorm<B, 0>,
    up_layers: Vec<UpscaleBlock<B>>,
    out_layer: Conv2d<B>,
    prelu: PRelu<B>,
}

impl<B: Backend> Generator<B> {
    pub fn init(device: &B::Device) -> Self {
        let in_layer = Conv2dConfig::new([3, 64], [9, 9]).init(device);
        let mut res_layers = Vec::<ResidualBlock<B>>::new();
        for _ in 0..16 {
            res_layers.push(ResidualBlock::new([64, 64], device));
        }
        let post_res_layer = Conv2dConfig::new([64, 64], [3, 3])
            .with_padding(PaddingConfig2d::Same)
            .init(device);
        let batch_norm = BatchNormConfig::new(64).with_momentum(0.5).init(device);
        let mut up_layers = Vec::<UpscaleBlock<B>>::new();
        for _ in 0..2 {
            up_layers.push(UpscaleBlock::new([64, 64], device));
        }
        let out_layer = Conv2dConfig::new([64, 3], [9, 9])
            .with_padding(PaddingConfig2d::Same)
            .init(device);
        let prelu = PReluConfig::new().init(device);

        Generator {
            in_layer,
            res_layers,
            post_res_layer,
            batch_norm,
            up_layers,
            out_layer,
            prelu,
        }
    }

    pub fn forward(&self, input: Tensor<B, 4>) -> Tensor<B, 4> {
        let in_output = self.in_layer.forward(input);
        let in_output = self.prelu.forward(in_output);

        let mut res_output = in_output.clone();
        for res_layer in self.res_layers.iter() {
            res_output = res_layer.forward(res_output);
        }

        let post_res_output = self.post_res_layer.forward(res_output);
        let post_res_output = self.batch_norm.forward(post_res_output);
        let post_res_output = in_output.add(post_res_output);

        let mut up_output = post_res_output.clone();
        for up_layer in self.up_layers.iter() {
            up_output = up_layer.forward(up_output);
        }

        self.out_layer.forward(up_output)
    }
}

#[derive(Module, Debug)]
pub struct Discriminator<B: Backend> {
    in_layer: Conv2d<B>,
    disc_layers: Vec<DiscBlock<B>>,
    fc_layer: Linear<B>,
    out_layer: Linear<B>,
    lrelu: LeakyRelu,
    sig: Sigmoid,
}

impl<B: Backend> Discriminator<B> {
    pub fn new(device: &B::Device) -> Self {
        let in_layer = Conv2dConfig::new([3, 64], [3, 3])
            .with_padding(PaddingConfig2d::Same)
            .init(device);
        let mut disc_layers = Vec::<DiscBlock<B>>::new();
        let ns = [1, 2, 2, 4, 4, 8, 8, 8];
        for i in 0..7 {
            let stride = match i % 2 {
                0 => 2,
                _ => 1,
            };
            disc_layers.push(DiscBlock::new([ns[i], ns[i + 1]], stride, device));
        }
        let fc_layer = LinearConfig::new(64 * 8, 1024).init(device);
        let out_layer = LinearConfig::new(1024, 1).init(device);
        let lrelu = LeakyReluConfig::new().init();
        let sig = Sigmoid::new();

        Discriminator {
            in_layer,
            disc_layers,
            fc_layer,
            out_layer,
            lrelu,
            sig,
        }
    }

    pub fn forward(&self, input: Tensor<B, 4>) -> Tensor<B, 2> {
        let output = self.in_layer.forward(input);
        let output = self.lrelu.forward(output);

        let mut disc_output = output.clone();
        for disc_layer in self.disc_layers.iter() {
            disc_output = disc_layer.forward(disc_output);
        }

        let [batch_size, channels, image_h, image_w] = disc_output.dims();
        let fc_input = disc_output.reshape([batch_size, channels * image_h * image_w]);

        let output = self.fc_layer.forward(fc_input);
        let output = self.lrelu.forward(output);
        let output = self.out_layer.forward(output);

        self.sig.forward(output)
    }
}

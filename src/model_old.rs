use burn::{
    module::{Module, ModuleMapper, ParamId},
    nn::BatchNorm,
    prelude::*,
    tensor::backend::AutodiffBackend,
};

/// Layer block of generator model
#[derive(Module, Debug)]
pub struct LayerBlock<B: Backend> {
    fc: nn::Linear<B>,
    bn: nn::BatchNorm<B, 0>,
    leakyrelu: nn::LeakyRelu,
}

impl<B: Backend> LayerBlock<B> {
    pub fn new(input: usize, output: usize, device: &B::Device) -> Self {
        let fc = nn::LinearConfig::new(input, output)
            .with_bias(true)
            .init(device);
        let bn: BatchNorm<B, 0> = nn::BatchNormConfig::new(output)
            .with_epsilon(0.8)
            .init(device);
        let leakyrelu = nn::LeakyReluConfig::new().with_negative_slope(0.2).init();

        Self { fc, bn, leakyrelu }
    }

    pub fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
        let output = self.fc.forward(input); // output: [Batch, x]
        let output = self.bn.forward(output); // output: [Batch, x]

        self.leakyrelu.forward(output) // output: [Batch, x]
    }
}

#[derive(Module, Debug)]
pub struct ConvLayerBlock<B: Backend> {
    conv: nn::conv::Conv2d<B>,
    relu: nn::LeakyRelu,
    bn: nn::BatchNorm<B, 2>,
}

impl<B: Backend> ConvLayerBlock<B> {
    pub fn new(
        channels: [usize; 2],
        kernel: [usize; 2],
        stride: [usize; 2],
        device: &B::Device,
    ) -> Self {
        let conv = nn::conv::Conv2dConfig::new(channels, kernel)
            .with_padding(nn::PaddingConfig2d::Same)
            .with_stride(stride)
            .init(device);
        let relu = nn::LeakyReluConfig::new().init();
        let bn = nn::BatchNormConfig::new(channels[1]).init(device);

        Self { conv, relu, bn }
    }

    pub fn forward(&self, input: Tensor<B, 4>) -> Tensor<B, 4> {
        let output = self.conv.forward(input);
        let output = self.relu.forward(output);
        self.bn.forward(output)
    }
}

#[derive(Module, Debug)]
pub struct DeconvLayerBlock<B: Backend> {
    deconv: nn::conv::ConvTranspose2d<B>,
    relu: nn::LeakyRelu,
    bn: nn::BatchNorm<B, 2>,
}

impl<B: Backend> DeconvLayerBlock<B> {
    pub fn new(
        channels: [usize; 2],
        kernel: [usize; 2],
        stride: [usize; 2],
        device: &B::Device,
    ) -> Self {
        let deconv = nn::conv::ConvTranspose2dConfig::new(channels, kernel)
            .with_stride(stride)
            .init(device);
        let relu = nn::LeakyReluConfig::new().init();
        let bn = nn::BatchNormConfig::new(channels[1]).init(device);

        Self { deconv, relu, bn }
    }

    pub fn forward(&self, input: Tensor<B, 4>) -> Tensor<B, 4> {
        let output = self.deconv.forward(input);
        let output = self.relu.forward(output);
        self.bn.forward(output)
    }
}

/// Generator model
#[derive(Module, Debug)]
pub struct Generator<B: Backend> {
    layer1: LayerBlock<B>,
    layer2: LayerBlock<B>,
    layer3: LayerBlock<B>,
    layer4: LayerBlock<B>,
    fc: nn::Linear<B>,
    tanh: nn::Tanh,
}

impl<B: Backend> Generator<B> {
    /// Applies the forward pass on the input tensor by specified order
    pub fn forward(&self, noise: Tensor<B, 2>) -> Tensor<B, 2> {
        let output = self.layer1.forward(noise);
        let output = self.layer2.forward(output);
        let output = self.layer3.forward(output);
        let output = self.layer4.forward(output);
        let output = self.fc.forward(output);

        self.tanh.forward(output) // [batch_size, channels*height*width]
    }
}

#[derive(Module, Debug)]
pub struct Generator2<B: Backend> {
    layer1: LayerBlock<B>,
    layer2: LayerBlock<B>,
    layer3: DeconvLayerBlock<B>,
    conv: nn::conv::ConvTranspose2d<B>,
    tanh: nn::Tanh,
}

impl<B: Backend> Generator2<B> {
    pub fn forward(&self, noise: Tensor<B, 2>) -> Tensor<B, 4> {
        let output = self.layer1.forward(noise);
        let output = self.layer2.forward(output);

        // reshape
        let [batch_size, _] = output.dims();
        let output = output.reshape([batch_size, 8, 8, 8]);

        let output = self.layer3.forward(output);

        // reshape
        // add 2 pixels of padding around the edge because fuck MNIST
        let output = output.pad((2, 2, 2, 2), 0.0);

        let output = self.conv.forward(output);
        self.tanh.forward(output)
    }
}

#[derive(Module, Debug)]
pub struct Discriminator<B: Backend> {
    fc1: nn::Linear<B>,
    leakyrelu1: nn::LeakyRelu,
    fc2: nn::Linear<B>,
    leakyrelu2: nn::LeakyRelu,
    fc3: nn::Linear<B>,
}

impl<B: Backend> Discriminator<B> {
    /// Applies the forward pass on the input tensor by specified order.
    /// The input image shape is [batch, channels, height, width]
    pub fn forward(&self, images: Tensor<B, 4>) -> Tensor<B, 2> {
        // Full connection for each batch
        let output = images.flatten(1, 3); // output: [batch, channels*height*width]
        let output = self.fc1.forward(output); // output: [batch, 512]
        let output = self.leakyrelu1.forward(output); // output: [batch, 512]
        let output = self.fc2.forward(output); // output: [batch, 256]
        let output = self.leakyrelu2.forward(output); // output: [batch, 256]

        self.fc3.forward(output) // output: [batch, 1]
    }
}

#[derive(Module, Debug)]
pub struct Discriminator2<B: Backend> {
    layer1: ConvLayerBlock<B>,
    layer2: ConvLayerBlock<B>,
    layer3: LayerBlock<B>,
    fc: nn::Linear<B>,
    pool: nn::pool::MaxPool2d,
    sig: nn::Sigmoid,
}

impl<B: Backend> Discriminator2<B> {
    pub fn forward(&self, images: Tensor<B, 4>) -> Tensor<B, 2> {
        let output = self.layer1.forward(images);
        let output = self.layer2.forward(output);
        let output = self.pool.forward(output);

        let [batch_size, channels, img_h, img_w] = output.dims();
        let output = output.reshape([batch_size, channels * img_h * img_w]);

        let output = self.layer3.forward(output);
        let output = self.fc.forward(output);

        self.sig.forward(output)
    }
}

#[derive(Module, Debug)]
pub struct Model<B: Backend> {
    pub generator: Generator<B>,
    pub discriminator: Discriminator<B>,
}

// Config for constructing and saving my Neural Network
#[derive(Config, Debug)]
pub struct ModelConfig {
    #[config(default = 100)]
    pub latent_dims: usize,
    #[config(default = 28)]
    pub image_height: usize,
    #[config(default = 28)]
    pub image_width: usize,
    #[config(default = 1)]
    pub channels: usize,
}
impl ModelConfig {
    /// Returns the initialized model.
    pub fn init<B: Backend>(&self, device: &B::Device) -> (Generator2<B>, Discriminator2<B>) {
        // // Construct the initialized generator
        // let layer1 = LayerBlock::new(self.latent_dims, 128, device);
        // let layer2 = LayerBlock::new(128, 256, device);
        // let layer3 = LayerBlock::new(256, 512, device);
        // let layer4 = LayerBlock::new(512, 1024, device);
        // let fc = nn::LinearConfig::new(1024, self.channels * self.image_height * self.image_width)
        //     .with_bias(true)
        //     .init(device);

        // let generator = Generator {
        //     layer1,
        //     layer2,
        //     layer3,
        //     layer4,
        //     fc,
        //     tanh: nn::Tanh::new(),
        // };

        // Construct the initialized generator
        let layer1 = LayerBlock::new(self.latent_dims, 256, device);
        let layer2 = LayerBlock::new(256, 512, device);
        let layer3 = DeconvLayerBlock::new([8, 4], [3, 3], [3, 3], device);
        let conv = nn::conv::ConvTranspose2dConfig::new([4, 1], [3, 3])
            .with_padding([1, 1])
            .init(device);
        let tanh = nn::Tanh::new();

        let generator = Generator2 {
            layer1,
            layer2,
            layer3,
            conv,
            tanh,
        };

        // // Construct the initialized discriminator
        // let fc1 = nn::LinearConfig::new(self.channels * self.image_height * self.image_width, 512)
        //     .init(device);
        // let leakyrelu1 = nn::LeakyReluConfig::new().with_negative_slope(0.2).init();
        // let fc2 = nn::LinearConfig::new(512, 256).init(device);
        // let leakyrelu2 = nn::LeakyReluConfig::new().with_negative_slope(0.2).init();
        // let fc3 = nn::LinearConfig::new(256, 1).init(device);

        // let discriminator = Discriminator {
        //     fc1,
        //     leakyrelu1,
        //     fc2,
        //     leakyrelu2,
        //     fc3,
        // };

        // Construct the initialized discriminator
        let layer1 = ConvLayerBlock::new([1, 4], [3, 3], [1, 1], device);
        let layer2 = ConvLayerBlock::new([4, 8], [3, 3], [1, 1], device);
        let layer3 = LayerBlock::new(8 * 14 * 14, 512, device);
        let fc = nn::LinearConfig::new(512, 1).init(device);
        let pool = nn::pool::MaxPool2dConfig::new([3, 3])
            .with_strides([2, 2])
            .with_padding(nn::PaddingConfig2d::Explicit(1, 1))
            .init();
        let sig = nn::Sigmoid::new();

        let discriminator = Discriminator2 {
            layer1,
            layer2,
            layer3,
            fc,
            pool,
            sig,
        };

        (generator, discriminator)
    }
}

/// Clip module mapper to clip all module parameters between a range of values
#[derive(Module, Clone, Debug)]
pub struct Clip {
    pub min: f32,
    pub max: f32,
}

impl<B: AutodiffBackend> ModuleMapper<B> for Clip {
    fn map_float<const D: usize>(&mut self, _id: ParamId, tensor: Tensor<B, D>) -> Tensor<B, D> {
        let is_require_grad = tensor.is_require_grad();

        let mut tensor = Tensor::from_inner(tensor.inner().clamp(self.min, self.max));

        if is_require_grad {
            tensor = tensor.require_grad();
        }
        tensor
    }
}

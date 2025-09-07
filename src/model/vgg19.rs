use std::env;

// Generated from ONNX "models/vgg19.onnx" by burn-import
use burn::nn::conv::Conv2d;
use burn::nn::conv::Conv2dConfig;
use burn::nn::pool::MaxPool2d;
use burn::nn::pool::MaxPool2dConfig;
use burn::nn::Dropout;
use burn::nn::DropoutConfig;
use burn::nn::Linear;
use burn::nn::LinearConfig;
use burn::nn::PaddingConfig2d;
use burn::record::FullPrecisionSettings;
use burn::record::Recorder;
use burn::{
    module::Module,
    tensor::{backend::Backend, Tensor},
};

#[derive(Module, Debug)]
pub struct Model<B: Backend> {
    conv2d1: Conv2d<B>,
    conv2d2: Conv2d<B>,
    maxpool2d1: MaxPool2d,
    conv2d3: Conv2d<B>,
    conv2d4: Conv2d<B>,
    maxpool2d2: MaxPool2d,
    conv2d5: Conv2d<B>,
    conv2d6: Conv2d<B>,
    conv2d7: Conv2d<B>,
    conv2d8: Conv2d<B>,
    maxpool2d3: MaxPool2d,
    conv2d9: Conv2d<B>,
    conv2d10: Conv2d<B>,
    conv2d11: Conv2d<B>,
    conv2d12: Conv2d<B>,
    maxpool2d4: MaxPool2d,
    conv2d13: Conv2d<B>,
    conv2d14: Conv2d<B>,
    conv2d15: Conv2d<B>,
    conv2d16: Conv2d<B>,
    maxpool2d5: MaxPool2d,
    gemm1: Linear<B>,
    dropout1: Dropout,
    gemm2: Linear<B>,
    dropout2: Dropout,
    gemm3: Linear<B>,
    phantom: core::marker::PhantomData<B>,
    device: burn::module::Ignored<B::Device>,
}

impl<B: Backend> Default for Model<B> {
    fn default() -> Self {
        let current_dir = env::current_dir().expect("Failed to get current directory");
        let current_dir_str = current_dir
            .to_str()
            .expect("Failed to convert path to &str");
        println!("Loading VGG from: {:?}/models/vgg19", current_dir);
        Self::from_file(
            &format!("{}/models/vgg19", current_dir_str),
            &Default::default(),
        )
    }
}

impl<B: Backend> Model<B> {
    pub fn from_file(file: &str, device: &B::Device) -> Self {
        let record = burn::record::NamedMpkFileRecorder::<FullPrecisionSettings>::new()
            .load(file.into(), device)
            .expect("Record file to exist.");
        Self::new(device).load_record(record)
    }
}

impl<B: Backend> Model<B> {
    #[allow(unused_variables)]
    pub fn new(device: &B::Device) -> Self {
        let conv2d1 = Conv2dConfig::new([3, 64], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d2 = Conv2dConfig::new([64, 64], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let maxpool2d1 = MaxPool2dConfig::new([2, 2])
            .with_strides([2, 2])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .init();
        let conv2d3 = Conv2dConfig::new([64, 128], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d4 = Conv2dConfig::new([128, 128], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let maxpool2d2 = MaxPool2dConfig::new([2, 2])
            .with_strides([2, 2])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .init();
        let conv2d5 = Conv2dConfig::new([128, 256], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d6 = Conv2dConfig::new([256, 256], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d7 = Conv2dConfig::new([256, 256], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d8 = Conv2dConfig::new([256, 256], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let maxpool2d3 = MaxPool2dConfig::new([2, 2])
            .with_strides([2, 2])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .init();
        let conv2d9 = Conv2dConfig::new([256, 512], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d10 = Conv2dConfig::new([512, 512], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d11 = Conv2dConfig::new([512, 512], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d12 = Conv2dConfig::new([512, 512], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let maxpool2d4 = MaxPool2dConfig::new([2, 2])
            .with_strides([2, 2])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .init();
        let conv2d13 = Conv2dConfig::new([512, 512], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d14 = Conv2dConfig::new([512, 512], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d15 = Conv2dConfig::new([512, 512], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d16 = Conv2dConfig::new([512, 512], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let maxpool2d5 = MaxPool2dConfig::new([2, 2])
            .with_strides([2, 2])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .init();
        let gemm1 = LinearConfig::new(25088, 4096).with_bias(true).init(device);
        let dropout1 = DropoutConfig::new(0.5).init();
        let gemm2 = LinearConfig::new(4096, 4096).with_bias(true).init(device);
        let dropout2 = DropoutConfig::new(0.5).init();
        let gemm3 = LinearConfig::new(4096, 1000).with_bias(true).init(device);
        Self {
            conv2d1,
            conv2d2,
            maxpool2d1,
            conv2d3,
            conv2d4,
            maxpool2d2,
            conv2d5,
            conv2d6,
            conv2d7,
            conv2d8,
            maxpool2d3,
            conv2d9,
            conv2d10,
            conv2d11,
            conv2d12,
            maxpool2d4,
            conv2d13,
            conv2d14,
            conv2d15,
            conv2d16,
            maxpool2d5,
            gemm1,
            dropout1,
            gemm2,
            dropout2,
            gemm3,
            phantom: core::marker::PhantomData,
            device: burn::module::Ignored(device.clone()),
        }
    }

    #[allow(clippy::let_and_return, clippy::approx_constant, unused)]
    pub fn forward(&self, input1: Tensor<B, 4>) -> Tensor<B, 2> {
        let conv2d1_out1 = self.conv2d1.forward(input1);
        let relu1_out1 = burn::tensor::activation::relu(conv2d1_out1);
        let conv2d2_out1 = self.conv2d2.forward(relu1_out1);
        let relu2_out1 = burn::tensor::activation::relu(conv2d2_out1);
        let maxpool2d1_out1 = self.maxpool2d1.forward(relu2_out1);
        let conv2d3_out1 = self.conv2d3.forward(maxpool2d1_out1);
        let relu3_out1 = burn::tensor::activation::relu(conv2d3_out1);
        let conv2d4_out1 = self.conv2d4.forward(relu3_out1);
        let relu4_out1 = burn::tensor::activation::relu(conv2d4_out1);
        let maxpool2d2_out1 = self.maxpool2d2.forward(relu4_out1);
        let conv2d5_out1 = self.conv2d5.forward(maxpool2d2_out1);
        let relu5_out1 = burn::tensor::activation::relu(conv2d5_out1);
        let conv2d6_out1 = self.conv2d6.forward(relu5_out1);
        let relu6_out1 = burn::tensor::activation::relu(conv2d6_out1);
        let conv2d7_out1 = self.conv2d7.forward(relu6_out1);
        let relu7_out1 = burn::tensor::activation::relu(conv2d7_out1);
        let conv2d8_out1 = self.conv2d8.forward(relu7_out1);
        let relu8_out1 = burn::tensor::activation::relu(conv2d8_out1);
        let maxpool2d3_out1 = self.maxpool2d3.forward(relu8_out1);
        let conv2d9_out1 = self.conv2d9.forward(maxpool2d3_out1);
        let relu9_out1 = burn::tensor::activation::relu(conv2d9_out1);
        let conv2d10_out1 = self.conv2d10.forward(relu9_out1);
        let relu10_out1 = burn::tensor::activation::relu(conv2d10_out1);
        let conv2d11_out1 = self.conv2d11.forward(relu10_out1);
        let relu11_out1 = burn::tensor::activation::relu(conv2d11_out1);
        let conv2d12_out1 = self.conv2d12.forward(relu11_out1);
        let relu12_out1 = burn::tensor::activation::relu(conv2d12_out1);
        let maxpool2d4_out1 = self.maxpool2d4.forward(relu12_out1);
        let conv2d13_out1 = self.conv2d13.forward(maxpool2d4_out1);
        let relu13_out1 = burn::tensor::activation::relu(conv2d13_out1);
        let conv2d14_out1 = self.conv2d14.forward(relu13_out1);
        let relu14_out1 = burn::tensor::activation::relu(conv2d14_out1);
        let conv2d15_out1 = self.conv2d15.forward(relu14_out1);
        let relu15_out1 = burn::tensor::activation::relu(conv2d15_out1);
        let conv2d16_out1 = self.conv2d16.forward(relu15_out1);
        let relu16_out1 = burn::tensor::activation::relu(conv2d16_out1);
        let maxpool2d5_out1 = self.maxpool2d5.forward(relu16_out1);
        let flatten1_out1 = {
            let leading_dim = maxpool2d5_out1.shape().dims[..1].iter().product::<usize>() as i32;
            maxpool2d5_out1.reshape::<2, _>([leading_dim, -1])
        };
        let gemm1_out1 = self.gemm1.forward(flatten1_out1);
        let relu17_out1 = burn::tensor::activation::relu(gemm1_out1);
        let dropout1_out1 = self.dropout1.forward(relu17_out1);
        let flatten2_out1 = {
            let leading_dim = dropout1_out1.shape().dims[..1].iter().product::<usize>() as i32;
            dropout1_out1.reshape::<2, _>([leading_dim, -1])
        };
        let gemm2_out1 = self.gemm2.forward(flatten2_out1);
        let relu18_out1 = burn::tensor::activation::relu(gemm2_out1);
        let dropout2_out1 = self.dropout2.forward(relu18_out1);
        let flatten3_out1 = {
            let leading_dim = dropout2_out1.shape().dims[..1].iter().product::<usize>() as i32;
            dropout2_out1.reshape::<2, _>([leading_dim, -1])
        };
        let gemm3_out1 = self.gemm3.forward(flatten3_out1);
        gemm3_out1
    }

    #[allow(unused)]
    pub fn extract_features(&self, input1: Tensor<B, 4>) -> Tensor<B, 4> {
        let conv2d1_out1 = self.conv2d1.forward(input1);
        let relu1_out1 = burn::tensor::activation::relu(conv2d1_out1);
        let conv2d2_out1 = self.conv2d2.forward(relu1_out1);
        let relu2_out1 = burn::tensor::activation::relu(conv2d2_out1);
        let maxpool2d1_out1 = self.maxpool2d1.forward(relu2_out1);
        let conv2d3_out1 = self.conv2d3.forward(maxpool2d1_out1);
        let relu3_out1 = burn::tensor::activation::relu(conv2d3_out1);
        let conv2d4_out1 = self.conv2d4.forward(relu3_out1);
        let relu4_out1 = burn::tensor::activation::relu(conv2d4_out1);
        let maxpool2d2_out1 = self.maxpool2d2.forward(relu4_out1);
        let conv2d5_out1 = self.conv2d5.forward(maxpool2d2_out1);
        let relu5_out1 = burn::tensor::activation::relu(conv2d5_out1);
        let conv2d6_out1 = self.conv2d6.forward(relu5_out1);
        let relu6_out1 = burn::tensor::activation::relu(conv2d6_out1);
        let conv2d7_out1 = self.conv2d7.forward(relu6_out1);
        let relu7_out1 = burn::tensor::activation::relu(conv2d7_out1);
        let conv2d8_out1 = self.conv2d8.forward(relu7_out1);
        let relu8_out1 = burn::tensor::activation::relu(conv2d8_out1);
        relu8_out1
    }
}

use burn::{
    config::Config,
    data::dataloader::DataLoader,
    nn::loss::{BinaryCrossEntropyLoss, BinaryCrossEntropyLossConfig, MseLoss, Reduction},
    optim::{AdamConfig, GradientsParams, Optimizer},
    tensor::{
        backend::{AutodiffBackend, Backend},
        Int, Tensor,
    },
};
use image::RgbImage;
use std::{error::Error, sync::Arc};

use crate::{
    data_srgan::SrganBatch,
    model::{discriminator::Discriminator, generator::Generator, vgg19::Model as Vgg, ModelConfig},
    utils::{save_sample, tensor_to_image},
};

// ////////////////////////////////////////////////////////////////////////////
// Helpers
#[derive(Clone, Debug)]
pub struct DiscOutput<B: Backend> {
    loss: Tensor<B, 1>,
}
#[derive(Clone, Debug)]
pub struct GenOutput<B: Backend> {
    loss: Tensor<B, 1>,
}

fn imagenet_norm<B: Backend>(input: Tensor<B, 4>) -> Tensor<B, 4> {
    let device = &input.device();
    let scaled = (input + 1.0) / 2.0;
    let mean = Tensor::<B, 1>::from_floats([0.485, 0.456, 0.406], device).reshape([1, 3, 1, 1]);
    let std = Tensor::<B, 1>::from_floats([0.229, 0.224, 0.225], device).reshape([1, 3, 1, 1]);

    (scaled - mean) / std
}

fn calc_disc_loss<B: AutodiffBackend>(
    batch: &SrganBatch<B>,
    generator: &Generator<B>,
    discriminator: &Discriminator<B>,
    bce: &BinaryCrossEntropyLoss<B>,
) -> DiscOutput<B> {
    let fake_imgs = generator.forward(batch.lr_data.clone());

    let fake_out = discriminator.forward(fake_imgs);
    let fake_targets = Tensor::<B, 2, Int>::zeros([batch.size, 1], &fake_out.device());
    let fake_loss = bce.forward(fake_out.clone(), fake_targets.clone());

    let real_out = discriminator.forward(batch.hr_data.clone());
    let real_targets = Tensor::<B, 2, Int>::ones([batch.size, 1], &real_out.device());
    let real_loss = bce.forward(real_out.clone(), real_targets.clone());

    let loss = (fake_loss + real_loss) * 0.5;

    DiscOutput { loss }
}

fn calc_gen_loss<B: AutodiffBackend>(
    batch: &SrganBatch<B>,
    generator: &Generator<B>,
    discriminator: &Discriminator<B>,
    vgg: &Vgg<B::InnerBackend>,
    adversarial_loss_weight: f32,
    bce: &BinaryCrossEntropyLoss<B>,
    mse: &MseLoss,
) -> GenOutput<B> {
    let generated_images = generator.forward(batch.lr_data.clone());
    // DEBUG: Check generator output
    // println!(
    //     "Generated images - min: {:.6}, max: {:.6}, mean: {:.6}",
    //     generated_images.clone().min().into_scalar(),
    //     generated_images.clone().max().into_scalar(),
    //     generated_images.clone().mean().into_scalar()
    // );

    let adversarial_out = discriminator.forward(generated_images.clone());
    let adversarial_targets = Tensor::<B, 2, Int>::ones([batch.size, 1], &adversarial_out.device());
    let adversarial_loss =
        bce.forward(adversarial_out.clone(), adversarial_targets) * adversarial_loss_weight;

    // DEBUG: Adversarial out
    // println!(
    //     "Adversarial out - min: {:.6}, max: {:.6}, mean: {:.6}",
    //     adversarial_out.clone().min().into_scalar(),
    //     adversarial_out.clone().max().into_scalar(),
    //     adversarial_out.clone().mean().into_scalar()
    // );

    let fake_imagenet_norm = imagenet_norm(generated_images.clone());
    let real_imagenet_norm = imagenet_norm(batch.hr_data.clone());

    // DEBUG: Check normalized image ranges
    // println!(
    //     "Normalized fake - min: {:.6}, max: {:.6}, mean: {:.6}",
    //     fake_imagenet_norm.clone().min().into_scalar(),
    //     fake_imagenet_norm.clone().max().into_scalar(),
    //     fake_imagenet_norm.clone().mean().into_scalar()
    // );

    // println!(
    //     "Normalized real - min: {:.6}, max: {:.6}, mean: {:.6}",
    //     real_imagenet_norm.clone().min().into_scalar(),
    //     real_imagenet_norm.clone().max().into_scalar(),
    //     fake_imagenet_norm.clone().max().into_scalar()
    // );

    let feature_space_fake_no_grads = vgg.extract_features(fake_imagenet_norm.inner());
    let feature_space_real_no_grads = vgg.extract_features(real_imagenet_norm.inner());
    let feature_space_fake = Tensor::<B, 4>::from_inner(feature_space_fake_no_grads);
    let feature_space_real = Tensor::<B, 4>::from_inner(feature_space_real_no_grads);

    // DEBUG: Check feature magnitudes
    // println!(
    //     "Fake features - min: {:.6}, max: {:.6}, mean: {:.6}",
    //     feature_space_fake.clone().min().into_scalar(),
    //     feature_space_fake.clone().max().into_scalar(),
    //     feature_space_fake.clone().mean().into_scalar()
    // );

    // println!(
    //     "Real features - min: {:.6}, max: {:.6}, mean: {:.6}",
    //     feature_space_real.clone().min().into_scalar(),
    //     feature_space_real.clone().max().into_scalar(),
    //     feature_space_real.clone().mean().into_scalar()
    // );

    let content_loss = mse.forward(feature_space_real, feature_space_fake, Reduction::Mean);

    // DEBUG: Check loss components
    // println!(
    //     "Content loss - min: {:.6}, max: {:.6}, mean: {:.6}",
    //     content_loss.clone().min().into_scalar(),
    //     content_loss.clone().max().into_scalar(),
    //     content_loss.clone().mean().into_scalar()
    // );

    // println!(
    //     "Adversarial loss - min: {:.6}, max: {:.6}, mean: {:.6}",
    //     adversarial_loss.clone().min().into_scalar(),
    //     adversarial_loss.clone().max().into_scalar(),
    //     adversarial_loss.clone().mean().into_scalar()
    // );

    let loss = content_loss + adversarial_loss.clone();

    GenOutput { loss }
}

// ////////////////////////////////////////////////////////////////////////////
// Training
#[derive(Config)]
pub struct TrainingConfig {
    pub model_config: ModelConfig,
    pub gen_optimizer: AdamConfig,
    pub disc_optimizer: AdamConfig,
    pub outdir: String,

    #[config(default = 1000)]
    pub epochs: usize,
    #[config(default = 1e-4)]
    pub gen_lr: f64,
    #[config(default = 1e-4)]
    pub disc_lr: f64,
    #[config(default = 5)]
    pub sample_interval: usize,
}

pub fn train<B: AutodiffBackend>(
    config: TrainingConfig,
    dataloader_train: Arc<dyn DataLoader<B, SrganBatch<B>>>,
    dataloader_valid: Arc<dyn DataLoader<B, SrganBatch<B>>>,
    device: &B::Device,
) -> Result<(), Box<dyn Error>> {
    let mut generator = config.model_config.generator_config.init::<B>(device);
    let mut discriminator = config.model_config.discriminator_config.init::<B>(device);
    let vgg = Vgg::default();

    let mut gen_optimizer = config.gen_optimizer.init();
    let mut disc_optimizer = config.disc_optimizer.init();

    let bce = BinaryCrossEntropyLossConfig::new().init::<B>(device);
    let mse = MseLoss::new();

    for epoch in 0..config.epochs {
        for (iteration, batch) in dataloader_train.iter().enumerate() {
            // train discriminator
            let disc_out = calc_disc_loss(&batch, &generator, &discriminator, &bce);
            let grads = disc_out.loss.backward();
            let grads = GradientsParams::from_grads(grads, &discriminator);
            discriminator = disc_optimizer.step(config.disc_lr, discriminator, grads);

            // train generator
            let gen_out =
                calc_gen_loss(&batch, &generator, &discriminator, &vgg, 0.001, &bce, &mse);
            let grads = gen_out.loss.backward();
            let grads = GradientsParams::from_grads(grads, &generator);
            generator = gen_optimizer.step(config.gen_lr, generator, grads);

            // print progress
            let batch_num = dataloader_train.num_items() / batch.size;
            println!(
                "[Epoch {:4}/{:4}] [Batch {:3}/{:3}] [D loss: {:+.5}] [G loss: {:+.5}]",
                epoch + 1,
                config.epochs,
                iteration,
                batch_num,
                disc_out.loss.into_scalar(),
                gen_out.loss.into_scalar()
            );

            // save sample
            if epoch % config.sample_interval == 0 && iteration == (batch.size - 1) {
                println!("saving sample");
                match dataloader_valid.iter().next() {
                    Some(batch) => {
                        let mut images: Vec<(RgbImage, RgbImage, RgbImage)> = vec![];
                        for i in 0..3 {
                            // get batch item
                            let item = batch.from_index(i);

                            // generate sr_data
                            let lr_data: Tensor<B, 4> = item.lr_data.clone().unsqueeze();
                            let sr_data: Tensor<B, 4> = generator.forward(lr_data);
                            let sr_data = sr_data.squeeze(0);

                            // generate images
                            images.push((
                                tensor_to_image(item.lr_data),
                                tensor_to_image(sr_data),
                                tensor_to_image(item.hr_data),
                            ));
                        }
                        let comp_image = save_sample(images, 128, 4);

                        // save composite image
                        let path = format!("{}/image-{epoch}.png", config.outdir);
                        comp_image.save(path)?;
                    }

                    _ => println!("error getting validation batch!"),
                }
            }
        }
    }

    Ok(())
}

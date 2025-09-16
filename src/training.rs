use burn::{
    config::Config,
    data::dataloader::DataLoader,
    module::Module,
    nn::loss::{BinaryCrossEntropyLoss, BinaryCrossEntropyLossConfig, MseLoss, Reduction},
    optim::{adaptor::OptimizerAdaptor, Adam, AdamConfig, GradientsParams, Optimizer},
    record::{FullPrecisionSettings, NamedMpkFileRecorder, Recorder},
    tensor::{
        backend::{AutodiffBackend, Backend},
        cast::ToElement,
        Int, Tensor,
    },
};
use image::RgbImage;
use std::{
    error::Error,
    path::PathBuf,
    str::FromStr,
    sync::Arc,
    time::{SystemTime, UNIX_EPOCH},
};

use crate::{
    data_srgan::SrganBatch,
    model::{
        discriminator::{Discriminator, DiscriminatorRecord},
        generator::{Generator, GeneratorRecord},
        vgg19::Model as Vgg,
        ModelConfig,
    },
    utils::{save_mosaic, tensor_to_image},
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
    let fake_loss = bce.forward(fake_out, fake_targets);

    let real_out = discriminator.forward(batch.hr_data.clone());
    let real_targets = Tensor::<B, 2, Int>::ones([batch.size, 1], &real_out.device());
    let real_loss = bce.forward(real_out, real_targets);

    let loss = (fake_loss + real_loss) * 0.5;

    DiscOutput { loss }
}

fn calc_gen_loss<B: AutodiffBackend>(
    batch: &SrganBatch<B>,
    generator: &Generator<B>,
    discriminator: &Discriminator<B>,
    vgg: &Vgg<B>,
    adversarial_loss_weight: f32,
    bce: &BinaryCrossEntropyLoss<B>,
    mse: &MseLoss,
) -> GenOutput<B> {
    let generated_images = generator.forward(batch.lr_data.clone());

    let adversarial_out = discriminator.forward(generated_images.clone());
    let adversarial_targets = Tensor::<B, 2, Int>::ones([batch.size, 1], &adversarial_out.device());
    let adversarial_loss = bce.forward(adversarial_out, adversarial_targets);

    let fake_imagenet_norm = imagenet_norm(generated_images);
    let real_imagenet_norm = imagenet_norm(batch.hr_data.clone());

    let feature_space_fake = vgg.extract_features(fake_imagenet_norm);
    let feature_space_real = vgg.extract_features(real_imagenet_norm);

    let content_loss = mse.forward(feature_space_real, feature_space_fake, Reduction::Mean);

    let loss = content_loss + (adversarial_loss * adversarial_loss_weight);

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

    #[config(default = 10)]
    pub epochs: usize,
    #[config(default = 1e-4)]
    pub gen_lr: f64,
    #[config(default = 1e-4)]
    pub disc_lr: f64,
}

pub fn save_sample<B: Backend>(
    dataloader_valid: Arc<dyn DataLoader<B, SrganBatch<B>>>,
    generator: &Generator<B>,
    outdir: &str,
) -> Result<(), Box<dyn Error>> {
    match dataloader_valid.iter().next() {
        Some(batch) => {
            let mut images: Vec<(RgbImage, RgbImage, RgbImage)> = vec![];
            for i in 0..3 {
                // get batch item
                let item = batch.from_index(i);

                // generate sr_data
                let lr_data: Tensor<B, 4> = item.lr_data.clone().unsqueeze();
                let sr_data: Tensor<B, 4> = generator.forward(lr_data).detach();
                let sr_data = sr_data.squeeze(0);

                // generate images
                images.push((
                    tensor_to_image(item.lr_data),
                    tensor_to_image(sr_data),
                    tensor_to_image(item.hr_data),
                ));
            }
            let comp_image = save_mosaic(images, 128, 4);

            // save composite image
            let time = SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs();
            let path = format!("{}/{}.png", outdir, time);
            comp_image.save(path)?;
        }

        _ => println!("error getting validation batch!"),
    }

    Ok(())
}

pub fn train<B: AutodiffBackend>(
    config: TrainingConfig,
    dataloader_train: Arc<dyn DataLoader<B, SrganBatch<B>>>,
    dataloader_valid: Arc<dyn DataLoader<B, SrganBatch<B>>>,
    recorder: NamedMpkFileRecorder<FullPrecisionSettings>,
    device: &B::Device,
    should_continue: bool,
) -> Result<(), Box<dyn Error>> {
    let mut generator = config.model_config.generator_config.init::<B>(device);
    let mut discriminator = config.model_config.discriminator_config.init::<B>(device);
    let vgg = Vgg::default();

    let mut gen_optimizer = config.gen_optimizer.init();
    let mut disc_optimizer = config.disc_optimizer.init();

    // Continue where you left off?
    if should_continue {
        let gen_path = PathBuf::from_str(format!("{}/gen.mpk", &config.outdir).as_str());
        let disc_path = PathBuf::from_str(format!("{}/disc.mpk", &config.outdir).as_str());

        match (gen_path, disc_path) {
            (Ok(gen_path), Ok(disc_path)) if gen_path.exists() && disc_path.exists() => {
                println!("Continuing from previous run");

                let record = recorder
                    .load::<DiscriminatorRecord<B>>(disc_path.into(), &device)
                    .expect("Should be able to load discriminator model.");
                discriminator = discriminator.load_record(record);
                let record = recorder
                    .load::<GeneratorRecord<B>>(gen_path.into(), &device)
                    .expect("Should be able to load generator model.");
                generator = generator.load_record(record);
            }

            _ => {
                println!("Unable to load saved model files. Starting new session...");
            }
        }
    }

    let bce = BinaryCrossEntropyLossConfig::new().init::<B>(device);
    let mse = MseLoss::new();

    let mut d_loss = 0.0;
    let mut g_loss = 0.0;
    for epoch in 0..config.epochs {
        for (it, batch) in dataloader_train.iter().enumerate() {
            let batch_num = dataloader_train.num_items() / batch.size;

            let disc_out = calc_disc_loss(&batch, &generator, &discriminator, &bce);
            let disc_loss = disc_out.loss.clone().into_scalar().to_f32();
            let grads = disc_out.loss.backward();
            let grads = GradientsParams::from_grads(grads, &discriminator);
            discriminator = disc_optimizer.step(config.disc_lr, discriminator, grads);
            d_loss += disc_loss;

            // train generator
            let gen_out =
                calc_gen_loss(&batch, &generator, &discriminator, &vgg, 0.001, &bce, &mse);
            let gen_loss = gen_out.loss.clone().into_scalar().to_f32();
            let grads = gen_out.loss.backward();
            let grads = GradientsParams::from_grads(grads, &generator);
            generator = gen_optimizer.step(config.gen_lr, generator, grads);
            g_loss += gen_loss;

            // print progress
            if it > 0 && it % (batch_num / 10) == 0 {
                println!(
                    "[Epoch: {:2}/{:2}, Batch: {:4}/{:4}][D loss: {:+.5}, G loss: {:+.5}]",
                    epoch,
                    config.epochs,
                    it * batch.size,
                    batch_num,
                    d_loss / (it as f32),
                    g_loss / (it as f32)
                );

                // save sample
                save_sample(dataloader_valid.clone(), &generator, &config.outdir)?;
            }
        }
        d_loss = 0.0;
        g_loss = 0.0;
    }

    // save sample
    save_sample(dataloader_valid.clone(), &generator, &config.outdir)?;

    // save model
    recorder.record(
        discriminator.into_record(),
        format!("{}/disc", config.outdir).into(),
    )?;
    recorder.record(
        generator.into_record(),
        format!("{}/gen", config.outdir).into(),
    )?;

    Ok(())
}

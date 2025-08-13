use crate::data::MnistBatcher;
use crate::model::ModelConfig;
use crate::utils::save_image;
use burn::optim::{AdamConfig, GradientsParams, Optimizer};
use burn::{
    data::{dataloader::DataLoaderBuilder, dataset::vision::MnistDataset},
    prelude::*,
    record::CompactRecorder,
    tensor::{backend::AutodiffBackend, Distribution},
};

// pub struct Trainer<B: AutodiffBackend> {
//     pub model: Model<B>,
//     pub device: B::Device,
// }
// impl<B: AutodiffBackend> TrainStep<MnistBatch<B>, ClassificationOutput<B>> for Trainer<B> {
//     fn step(&self, batch: MnistBatch<B>) -> TrainOutput<ClassificationOutput<B>> {
//         let device = self.device;

//         // Generate a batch of fake images from noise (standarded normal distribution)
//         let noise = Tensor::<B, 2>::random(
//             [batch.batch_size, self.model.generator.latent_dims],
//             Distribution::Normal(0.0, 1.0),
//             &device,
//         );

//         // detach: do not update gerenator, only discriminator is updated
//         let fake_images = self.model.generator.forward(noise.clone()).detach(); // [batch_size, channels*height*width]
//         let fake_images = fake_images.reshape([
//             batch.batch_size,
//             batch.channels,
//             batch.image_height,
//             batch.image_width,
//         ]);

//         // get real images
//         let real_images = batch.images.reshape([
//             batch.batch_size,
//             batch.channels,
//             batch.image_height,
//             batch.image_width,
//         ]);

//         let item = self.forward_classification(batch.images, batch.targets);

//         TrainOutput::new(self, item.loss.backward(), item)
//     }
// }

// impl<B: Backend> ValidStep<MnistBatch<B>, ClassificationOutput<B>> for Trainer<B> {
//     fn step(&self, batch: MnistBatch<B>) -> ClassificationOutput<B> {
//         self.forward_classification(batch.images, batch.targets)
//     }
// }

#[derive(Config)]
pub struct TrainingConfig {
    pub model: ModelConfig,
    pub optimizer: AdamConfig,

    #[config(default = 2000)]
    pub num_epochs: usize,
    #[config(default = 60)]
    pub batch_size: usize,
    #[config(default = 10)]
    pub num_workers: usize,
    #[config(default = 42)]
    pub seed: u64,
    #[config(default = 3e-4)]
    pub lr: f64,

    /// Number of training steps for discriminator before generator is trained per iteration
    #[config(default = 20)]
    pub num_critic: usize,
    /// Save a sample of images every `sample_interval` epochs
    #[config(default = 20)]
    pub sample_interval: usize,
}

// Create the directory to save the model and model config
fn create_artifact_dir(artifact_dir: &str) {
    // Remove existing artifacts
    std::fs::remove_dir_all(artifact_dir).ok();
    std::fs::create_dir_all(artifact_dir).ok();
}

pub fn train<B: AutodiffBackend>(artifact_dir: &str, config: TrainingConfig, device: &B::Device) {
    create_artifact_dir(artifact_dir);

    // Save training config
    config
        .save(format!("{artifact_dir}/config.json"))
        .expect("Config should be saved successfully");
    B::seed(config.seed);

    // Create the model and optimizer
    let model = config.model.init::<B>(&device);
    let mut generator = model.generator;
    let mut discriminator = model.discriminator;
    let mut optimizer_g = config.optimizer.init();
    let mut optimizer_d = config.optimizer.init();

    // Create the dataset batcher
    let batcher_train = MnistBatcher {};

    // Create the dataloaders
    let dataloader_train = DataLoaderBuilder::new(batcher_train.clone())
        .batch_size(config.batch_size)
        .shuffle(config.seed)
        .num_workers(config.num_workers)
        .build(MnistDataset::train());

    // Iterate over our training for X epochs
    for epoch in 0..config.num_epochs {
        // Implement our training loop
        for (iteration, batch) in dataloader_train.iter().enumerate() {
            // Generate a batch of fake images from noise (standarded normal distribution)
            let noise = Tensor::<B, 2>::random(
                [config.batch_size, config.model.latent_dims],
                Distribution::Normal(0.0, 1.0),
                &device,
            );

            // detach: do not update gerenator, only discriminator is updated
            let fake_images = generator.forward(noise.clone()).detach(); // [batch_size, channels*height*width]
            let fake_images = fake_images.reshape([
                config.batch_size,
                config.model.channels,
                config.model.image_height,
                config.model.image_width,
            ]);
            // println!("made {:?} fake images", fake_images.dims()[0]);

            // get real images
            let batch_size = batch.images.dims()[0];
            let real_images = batch.images.reshape([batch_size, 1, 28, 28]);
            // println!("got {:?} real images", real_images.dims()[0]);
            // save_image(real_images.clone(), 4, "./poop.png").unwrap();

            // Adversarial loss
            let loss_d = -discriminator.forward(real_images).mean()
                + discriminator.forward(fake_images.clone()).mean();
            // println!("{loss_d}");

            // Gradients for the current backward pass
            let grads = loss_d.backward();
            // Gradients linked to each parameter of the discriminator
            let grads = GradientsParams::from_grads(grads, &discriminator);
            // Update the discriminator using the optimizer
            discriminator = optimizer_d.step(config.lr, discriminator, grads);

            // Train the generator every num_critic iterations
            if iteration != 0 && iteration % config.num_critic == 0 {
                // Generate a batch of images again without detaching
                let critic_fake_images = generator.forward(noise.clone());
                let critic_fake_images = critic_fake_images.reshape([
                    config.batch_size,
                    config.model.channels,
                    config.model.image_height,
                    config.model.image_width,
                ]);
                // Adversarial loss. Minimize it to make the fake images as truth
                let loss_g = -discriminator.forward(critic_fake_images).mean();

                let grads = loss_g.backward();
                let grads = GradientsParams::from_grads(grads, &generator);
                generator = optimizer_g.step(config.lr, generator, grads);

                // Print the progression
                let batch_num = (dataloader_train.num_items() as f32 / config.batch_size as f32)
                    .ceil() as usize;
                println!(
                    "[Epoch {}/{}] [Batch {}/{}] [D loss: {}] [G loss: {}]",
                    epoch + 1,
                    config.num_epochs,
                    iteration,
                    batch_num,
                    loss_d.into_scalar(),
                    loss_g.into_scalar()
                );
            }

            //  If at save interval => save the first 25 generated images
            if epoch % config.sample_interval == 0 && iteration == 0 {
                println!("asdfsadffs");
                // // [B, C, H, W] to [B, H, C, W] to [B, H, W, C]
                // let fake_images = fake_images.swap_dims(2, 1).swap_dims(3, 2).slice(0..25);
                // Normalize the images. The Rgb32 images should be in range 0.0-1.0
                // let fake_images = (fake_images.clone()
                //     - fake_images.clone().min().reshape([1, 1, 1, 1]))
                //     / (fake_images.clone().max().reshape([1, 1, 1, 1])
                //         - fake_images.clone().min().reshape([1, 1, 1, 1]));
                // Add 0.5/255.0 to the images, refer to pytorch save_image source
                let fake_images = (fake_images + 0.5 / 255.0).clamp(0.0, 1.0);
                // Save images in artifact directory
                let path = format!("{artifact_dir}/image-{epoch}.png");
                save_image::<B, _>(fake_images, 4, path).unwrap();
            }
        }
    }

    // Save the trained models
    generator
        .save_file(format!("{artifact_dir}/generator"), &CompactRecorder::new())
        .expect("Generator should be saved successfully");
    discriminator
        .save_file(
            format!("{artifact_dir}/discriminator"),
            &CompactRecorder::new(),
        )
        .expect("Discriminator should be saved successfully");
}

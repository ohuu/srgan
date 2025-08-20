mod data_srgan;
mod model;
mod training;
mod utils;

use burn::{
    backend::{Autodiff, Wgpu},
    lr_scheduler::constant::ConstantLr,
    optim::AdamConfig,
    record::CompactRecorder,
    train::{metric::AccuracyMetric, LearnerBuilder},
};
use model::ModelConfig;
use std::error::Error;
use training::{train, TrainingConfig};

fn main() -> Result<(), Box<dyn Error>> {
    type MyBackend = Wgpu<f32>;
    type MyAutodiffBackend = Autodiff<MyBackend>;

    let device = burn::backend::wgpu::WgpuDevice::default();
    let artifact_dir = "/tmp/srgan";

    // // Create dataset
    //     let dataset = SrganDataset::new(
    //         "./srgan_dataset",
    //         (256, 256), // HR size
    //         4           // Scale factor (4x upscaling)
    //     )?;

    //     // Create dataloader
    //     let device = Default::default();
    //     let batcher = SrganBatcher::new(device);

    //     let dataloader = DataLoaderBuilder::new(batcher)
    //         .batch_size(8)
    //         .shuffle(42)
    //         .num_workers(4)
    //         .build(dataset);

    //     // Use in training loop
    //     for batch in dataloader {
    //         let hr_images = batch.hr_images; // [batch_size, 3, 256, 256]
    //         let lr_images = batch.lr_images; // [batch_size, 3, 64, 64]

    //         // Your SRGAN training logic here
    //         println!("HR batch shape: {:?}", hr_images.shape());
    //         println!("LR batch shape: {:?}", lr_images.shape());
    //     }

    //     Ok(())

    let learner = LearnerBuilder::new(artifact_dir)
        .metric_train_numeric(AccuracyMetric::new())
        .metric_valid_numeric(AccuracyMetric::new())
        .with_file_checkpointer(CompactRecorder::new())
        .devices(vec![device.clone()])
        .num_epochs(1000)
        .summary()
        .build(
            ModelConfig::init(&device),
            AdamConfig::new().init(),
            ConstantLr::new(3e-4),
        );

    // initialise training params and train
    let config = TrainingConfig::new(ModelConfig::new(), AdamConfig::new());
    train::<MyAutodiffBackend>(artifact_dir, config.clone(), &device);

    Ok(())
}

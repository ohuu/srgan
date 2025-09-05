#![recursion_limit = "256"]

mod data_srgan;
mod model;
mod training;
mod utils;

use burn::{
    backend::{Autodiff, NdArray, Wgpu},
    data::dataloader::DataLoaderBuilder,
    optim::AdamConfig,
};
use std::{error::Error, time::Instant};

use crate::{
    data_srgan::{SrganBatcher, SrganDataset},
    model::{discriminator::DiscriminatorConfig, generator::GeneratorConfig, ModelConfig},
    training::{train, TrainingConfig},
    utils::tensor_to_image,
};

type MyBackend = Wgpu<f32>;
// type MyBackend = NdArray<f32>;
type MyAutodiffBackend = Autodiff<MyBackend>;

fn main() -> Result<(), Box<dyn Error>> {
    let device = burn::backend::wgpu::WgpuDevice::default();
    // let device = burn::backend::ndarray::NdArrayDevice::default();
    let datadir = "/home/ohuu/Downloads/DIV2K/hr128";

    // Create training data
    // let datadir_train = format!("{}/training/scale_x4", datadir);
    // let dataset_train = SrganDataset::<MyAutodiffBackend>::new(&datadir_train, &device)?;
    // let dataloader_train = DataLoaderBuilder::new(SrganBatcher::new())
    //     .batch_size(1)
    //     .shuffle(42)
    //     .num_workers(30)
    //     .build(dataset_train);

    // Create validation data
    let datadir_valid = format!("{}/validation/scale_x4", datadir);
    let dataset_valid = SrganDataset::<MyAutodiffBackend>::new(&datadir_valid, &device)?;
    let dataloader_valid = DataLoaderBuilder::new(SrganBatcher::new())
        .batch_size(1)
        .shuffle(42)
        .num_workers(10)
        .build(dataset_valid);

    // Train
    let model_config = ModelConfig::new(GeneratorConfig::new(), DiscriminatorConfig::new(128));
    let gen_optimizer = AdamConfig::new();
    let disc_optimizer = AdamConfig::new();
    let outdir = "/tmp/srgan".to_string();
    let training_config = TrainingConfig::new(model_config, gen_optimizer, disc_optimizer, outdir)
        .with_epochs(200)
        .with_gen_lr(1e-5)
        .with_disc_lr(1e-5)
        .with_sample_interval(1);

    // Infer the whole validation set
    let start = Instant::now();
    for (_, batch) in dataloader_valid.iter().enumerate() {
        let generator = training_config
            .model_config
            .generator_config
            .init::<MyBackend>(&device);

        let gen_image = generator.forward(batch.lr_data.inner());
        tensor_to_image(gen_image.squeeze(0));
    }
    let duration = start.elapsed();
    println!("Loop took: {:.6} seconds", duration.as_secs_f64());

    // train::<MyAutodiffBackend>(
    //     training_config,
    //     dataloader_train.clone(),
    //     dataloader_valid.clone(),
    //     &device,
    // )?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utils::tensor_to_image;
    use burn::{data::dataset::Dataset, module::Module, tensor::s};

    fn assert_vectors_close(actual: &[f32], expected: &[f32], tolerance: f32) {
        assert_eq!(
            actual.len(),
            expected.len(),
            "Vectors must have the same length"
        );

        for (a, e) in actual.iter().zip(expected.iter()) {
            assert!(
                (a - e).abs() < tolerance,
                "Values differ by more than tolerance: {} vs {} (diff: {})",
                a,
                e,
                (a - e).abs()
            );
        }
    }

    #[test]
    pub fn can_load_image() -> Result<(), Box<dyn Error>> {
        let device = burn::backend::wgpu::WgpuDevice::default();
        // let device = burn::backend::ndarray::NdArrayDevice::default();
        let datadir: &str = "/home/ohuu/Downloads/DIV2K/test/hr128";

        // Create dataset
        let datadir = format!("{}/training/scale_x4", datadir);
        let dataset = SrganDataset::<MyAutodiffBackend>::new(&datadir, &device)?;

        // Get first item in dataset
        let item = dataset.get(0).unwrap();

        // Test
        let dims = item.lr_data.clone().dims();
        let pixel_y0x0 = item
            .lr_data
            .clone()
            .slice([s![..], s![0], s![0]])
            .to_data()
            .to_vec::<f32>()
            .unwrap();
        let pixel_y0x1 = item
            .lr_data
            .clone()
            .slice([s![..], s![0], s![1]])
            .to_data()
            .to_vec::<f32>()
            .unwrap();
        let pixel_y0x2 = item
            .lr_data
            .clone()
            .slice([s![..], s![0], s![2]])
            .to_data()
            .to_vec::<f32>()
            .unwrap();

        assert_eq!(dims, [3, 32, 32]);

        assert_vectors_close(&pixel_y0x0, &vec![-0.5372, -0.5686, -0.6], 0.001);
        assert_vectors_close(&pixel_y0x1, &vec![-0.2784, -0.4196, -0.3098], 0.001);
        assert_vectors_close(&pixel_y0x2, &vec![0.4274, 0.1529, 0.2470], 0.001);

        println!("image -> tensor");
        println!("---------------");
        println!("dims = {:?}", item.lr_data.clone().dims());
        println!("pixel y0,x0 = {:?}", pixel_y0x0);
        println!("pixel y0,x1 = {:?}", pixel_y0x1);
        println!("pixel y0,x2 = {:?}", pixel_y0x2);

        Ok(())
    }

    #[test]
    pub fn can_save_image() -> Result<(), Box<dyn Error>> {
        let device = burn::backend::wgpu::WgpuDevice::default();
        // let device = burn::backend::ndarray::NdArrayDevice::default();
        let datadir: &str = "/home/ohuu/Downloads/DIV2K/test/hr128";

        // Create dataset
        let datadir = format!("{}/training/scale_x4", datadir);
        let dataset = SrganDataset::<MyAutodiffBackend>::new(&datadir, &device)?;

        // Get first item in dataset
        let item = dataset.get(0).unwrap();

        // Convert tensor to image
        let lr_image = tensor_to_image(item.lr_data.clone());

        let expected_pixels = [[59, 55, 51], [92, 74, 88], [182, 147, 159]];
        for i in 0..3 {
            let pixel = lr_image.get_pixel(i as u32, 0);
            let r = pixel[0];
            let g = pixel[1];
            let b = pixel[2];

            assert_eq!([r, g, b], expected_pixels[i]);
        }

        // Save image
        lr_image.clone().save("/home/ohuu/test.png").unwrap();

        Ok(())
    }
}

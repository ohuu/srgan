use burn::prelude::*;
use image::{imageops, Rgb, RgbImage};
use std::{error::Error, path::Path};

#[allow(unused)]
pub fn load_image(path: &Path) -> Result<RgbImage, Box<dyn Error>> {
    let img = image::open(path)?;
    let rgb_img = img.to_rgb8();
    Ok(rgb_img)
}

pub fn image_to_tensor<B: Backend>(img: RgbImage, device: &B::Device) -> Tensor<B, 3> {
    let (width, height) = img.dimensions();
    let mut data = Vec::with_capacity((3 * width * height) as usize);

    // Convert RGB to tensor format [C, H, W] with values in [-1, 1]
    for channel in 0..3 {
        for y in 0..height {
            for x in 0..width {
                let pixel = img.get_pixel(x, y);
                let value = ((pixel[channel] as f32 / 255.0) * 2.0) - 1.0;

                data.push(value);
            }
        }
    }

    Tensor::from_data(
        TensorData::new(data, [3, height as usize, width as usize]),
        device,
    )
}

pub fn tensor_to_image<B: Backend>(tensor: Tensor<B, 3>) -> RgbImage {
    // get dims
    let [_, height, width] = tensor.dims();

    // Move tensor data to CPU and convert to Vec<f32>
    let data: Vec<f32> = tensor.to_data().to_vec().unwrap();

    // Create RgbImage
    let mut img = RgbImage::new(width as u32, height as u32);

    // Convert and copy pixel data
    for y in 0..height {
        for x in 0..width {
            // Calculate indices for each channel (CHW format)
            let r_idx = 0 * height * width + y * width + x; // Red channel
            let g_idx = 1 * height * width + y * width + x; // Green channel
            let b_idx = 2 * height * width + y * width + x; // Blue channel

            // Get float values and convert to u8 (0-255)
            let r = (((data[r_idx].clamp(-1.0, 1.0) + 1.0) / 2.0) * 255.0).round() as u8;
            let g = (((data[g_idx].clamp(-1.0, 1.0) + 1.0) / 2.0) * 255.0).round() as u8;
            let b = (((data[b_idx].clamp(-1.0, 1.0) + 1.0) / 2.0) * 255.0).round() as u8;

            // Set pixel in image
            img.put_pixel(x as u32, y as u32, Rgb([r as u8, g, b]));
        }
    }

    img
}

#[allow(unused)]
pub fn save_mosaic(
    images: Vec<(RgbImage, RgbImage, RgbImage)>,
    hr_image_size: u32,
    gap: u32,
) -> RgbImage {
    // create new composite image
    let width: u32 = 3 * hr_image_size + 2 * gap;
    let height: u32 = images.len() as u32 * hr_image_size + 2 * gap;
    let mut comp_image = RgbImage::new(width, height);

    let mut row = 0;
    for (lr_image, sr_image, hr_image) in images {
        // resize lr_image
        let resized_lr_image = imageops::resize(
            &lr_image,
            hr_image_size,
            hr_image_size,
            imageops::FilterType::Nearest,
        );

        // write resized lr image to composite
        for (x, y, pixel) in resized_lr_image.enumerate_pixels() {
            comp_image.put_pixel(x, row * hr_image_size + row * gap + y, *pixel);
        }

        // write the sr image to composite
        for (x, y, pixel) in sr_image.enumerate_pixels() {
            comp_image.put_pixel(
                hr_image_size + gap + x,
                row * hr_image_size + row * gap + y,
                *pixel,
            );
        }

        // write hr image to composite
        for (x, y, pixel) in hr_image.enumerate_pixels() {
            comp_image.put_pixel(
                2 * hr_image_size + 2 * gap + x,
                row * hr_image_size + row * gap + y,
                *pixel,
            );
        }

        row += 1;
    }

    comp_image
}

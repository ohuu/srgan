use burn::prelude::*;
use image::{buffer::ConvertBuffer, ImageResult, Rgb32FImage, RgbImage};
use std::path::Path;

/// Save the generated images
/// The images format is [B, C, H, W]
pub fn save_image<B: Backend, Q: AsRef<Path>>(
    images: Tensor<B, 4>,
    nrows: u32,
    path: Q,
) -> ImageResult<()> {
    let ncols = (images.dims()[0] as f32 / nrows as f32).ceil() as u32;

    let width = images.dims()[3] as u32;
    let height = images.dims()[2] as u32;

    // Supports both 1 and 3 channels image
    let channels = match images.dims()[1] {
        1 => 3,
        3 => 1,
        _ => panic!("Wrong channels number"),
    };

    let mut imgbuf = RgbImage::new(ncols * width, nrows * height);
    // Write images into a nrow*ncol grid layout
    for row in 0..nrows {
        for col in 0..ncols {
            let image: Tensor<B, 3> = images
                .clone()
                .slice((row * ncols + col) as usize..(row * ncols + col + 1) as usize)
                .squeeze(0);
            // The Rgb32 should be in range 0.0-1.0
            let image = image.into_data().iter::<f32>().collect::<Vec<f32>>();
            // Supports both 1 and 3 channels image
            let image = image
                .into_iter()
                .flat_map(|n| std::iter::repeat_n(n, channels))
                .collect();

            let image = Rgb32FImage::from_vec(width, height, image).unwrap();
            let image: RgbImage = image.convert();
            for (x, y, pixel) in image.enumerate_pixels() {
                imgbuf.put_pixel((col * width) + x, (row * height) + y, *pixel);
            }
        }
    }
    imgbuf.save(path)
}

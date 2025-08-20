use burn::{
    data::{dataloader::batcher::Batcher, dataset::Dataset},
    prelude::*,
};
use image::RgbImage;
use std::path::Path;
use walkdir::WalkDir;

// ////////////////////////////////////////////////////////////////////////////
// Helpers
fn load_image(path: &Path) -> Result<RgbImage, Box<dyn std::error::Error>> {
    let img = image::open(path)?;
    let rgb_img = img.to_rgb8();
    Ok(rgb_img)
}

fn image_to_tensor<B: Backend>(img: RgbImage, device: &B::Device) -> Tensor<B, 3> {
    let (width, height) = img.dimensions();
    let mut data = Vec::with_capacity((3 * width * height) as usize);

    // Convert RGB to tensor format [C, H, W] with values in [0, 1]
    for channel in 0..3 {
        for y in 0..height {
            for x in 0..width {
                let pixel = img.get_pixel(x, y);
                let value = pixel[channel] as f32 / 255.0;
                data.push(value);
            }
        }
    }

    Tensor::from_data(
        TensorData::new(data, [3, height as usize, width as usize]),
        device,
    )
}

// ////////////////////////////////////////////////////////////////////////////
// Dataset
#[derive(Debug, Clone)]
pub struct SrganDataset<B: Backend> {
    pub hr_images: Vec<Tensor<B, 3>>,
    pub lr_images: Vec<Tensor<B, 3>>,
}

impl<B: Backend> SrganDataset<B> {
    pub fn new<P: AsRef<Path>>(
        root: P,
        device: &B::Device,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        // Load HR images
        let hr_root = root.as_ref().join("hr");

        let mut hr_images = Vec::new();
        for entry in WalkDir::new(hr_root).into_iter().filter_map(|e| e.ok()) {
            let path = entry.path();
            if let Some(ext) = path.extension() {
                if matches!(ext.to_str(), Some("jpg") | Some("jpeg") | Some("png")) {
                    let hr_image = load_image(path)?;
                    let hr_tensor = image_to_tensor(hr_image, device);
                    hr_images.push(hr_tensor);
                }
            }
        }

        // Load LR images
        let lr_root = root.as_ref().join("lr");

        let mut lr_images = Vec::new();
        for entry in WalkDir::new(lr_root).into_iter().filter_map(|e| e.ok()) {
            let path = entry.path();
            if let Some(ext) = path.extension() {
                if matches!(ext.to_str(), Some("jpg") | Some("jpeg") | Some("png")) {
                    let lr_image = load_image(path)?;
                    let lr_tensor = image_to_tensor(lr_image, device);
                    lr_images.push(lr_tensor);
                }
            }
        }

        Ok(Self {
            hr_images,
            lr_images,
        })
    }
}

#[derive(Debug, Clone)]
pub struct SrganItem<B: Backend> {
    pub hr_image: Tensor<B, 3>,
    pub lr_image: Tensor<B, 3>,
}

impl<B: Backend> Dataset<SrganItem<B>> for SrganDataset<B> {
    fn get(&self, index: usize) -> Option<SrganItem<B>> {
        let hr_image = self.hr_images.get(index)?;
        let lr_image = self.lr_images.get(index)?;

        Some(SrganItem {
            hr_image: hr_image.clone(),
            lr_image: lr_image.clone(),
        })
    }

    fn len(&self) -> usize {
        self.hr_images.len()
    }
}

// ////////////////////////////////////////////////////////////////////////////
// Batcher
#[derive(Debug, Clone)]
pub struct SrganBatch<B: Backend> {
    pub hr_images: Tensor<B, 4>,
    pub lr_images: Tensor<B, 4>,
}

#[derive(Debug, Clone)]
pub struct SrganBatcher {}

impl<B: Backend> Batcher<B, SrganItem<B>, SrganBatch<B>> for SrganBatcher {
    fn batch(&self, items: Vec<SrganItem<B>>, device: &B::Device) -> SrganBatch<B> {
        let hr_images = items
            .iter()
            .map(|item| item.hr_image.clone().unsqueeze_dim(0))
            .collect::<Vec<_>>();

        let lr_images = items
            .iter()
            .map(|item| item.lr_image.clone().unsqueeze_dim(0))
            .collect::<Vec<_>>();

        let hr_batch = Tensor::cat(hr_images, 0).to_device(device);
        let lr_batch = Tensor::cat(lr_images, 0).to_device(device);

        SrganBatch {
            hr_images: hr_batch,
            lr_images: lr_batch,
        }
    }
}

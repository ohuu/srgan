use burn::{
    data::{dataloader::batcher::Batcher, dataset::Dataset},
    prelude::*,
};
use std::{error::Error, path::Path};
use walkdir::WalkDir;

use crate::utils::{image_to_tensor, load_image};

// ////////////////////////////////////////////////////////////////////////////
// Dataset
#[derive(Debug, Clone)]
pub struct SrganDataset<B: Backend> {
    pub hr_data: Vec<Tensor<B, 3>>,
    pub lr_data: Vec<Tensor<B, 3>>,
}

impl<B: Backend> SrganDataset<B> {
    pub fn new<P: AsRef<Path>>(root: P, device: &B::Device) -> Result<Self, Box<dyn Error>> {
        // Load HR images
        let hr_root = root.as_ref().join("hr");

        let mut hr_data = Vec::new();
        for entry in WalkDir::new(hr_root).into_iter().filter_map(|e| e.ok()) {
            let path = entry.path();
            if let Some(ext) = path.extension() {
                if matches!(ext.to_str(), Some("jpg") | Some("jpeg") | Some("png")) {
                    let hr_image = load_image(path)?;
                    let hr_tensor = image_to_tensor(hr_image, device);
                    hr_data.push(hr_tensor);
                }
            }
        }

        // Load LR images
        let lr_root = root.as_ref().join("lr");

        let mut lr_data = Vec::new();
        for entry in WalkDir::new(lr_root).into_iter().filter_map(|e| e.ok()) {
            let path = entry.path();
            if let Some(ext) = path.extension() {
                if matches!(ext.to_str(), Some("jpg") | Some("jpeg") | Some("png")) {
                    let lr_image = load_image(path)?;
                    let lr_tensor = image_to_tensor(lr_image, device);
                    lr_data.push(lr_tensor);
                }
            }
        }

        Ok(Self { hr_data, lr_data })
    }
}

#[derive(Debug, Clone)]
pub struct SrganItem<B: Backend> {
    pub hr_data: Tensor<B, 3>,
    pub lr_data: Tensor<B, 3>,
}

impl<B: Backend> Dataset<SrganItem<B>> for SrganDataset<B> {
    fn get(&self, index: usize) -> Option<SrganItem<B>> {
        let hr_data = self.hr_data.get(index)?;
        let lr_data = self.lr_data.get(index)?;

        Some(SrganItem {
            hr_data: hr_data.clone(),
            lr_data: lr_data.clone(),
        })
    }

    fn len(&self) -> usize {
        self.hr_data.len()
    }
}

// ////////////////////////////////////////////////////////////////////////////
// Batcher
#[derive(Debug, Clone)]
pub struct SrganBatch<B: Backend> {
    pub hr_data: Tensor<B, 4>,
    pub lr_data: Tensor<B, 4>,
    pub size: usize,
}
impl<B: Backend> SrganBatch<B> {
    pub fn from_index(&self, index: usize) -> SrganItem<B> {
        let lr_data = self.lr_data.clone().slice_dim(0, index).squeeze(0);
        let hr_data = self.hr_data.clone().slice_dim(0, index).squeeze(0);

        SrganItem { lr_data, hr_data }
    }
}

#[derive(Debug, Clone)]
pub struct SrganBatcher {}
impl SrganBatcher {
    pub fn new() -> Self {
        Self {}
    }
}
impl<B: Backend> Batcher<B, SrganItem<B>, SrganBatch<B>> for SrganBatcher {
    fn batch(&self, items: Vec<SrganItem<B>>, device: &B::Device) -> SrganBatch<B> {
        let hr = items
            .iter()
            .map(|item| item.hr_data.clone().unsqueeze_dim(0))
            .collect::<Vec<_>>();

        let lr = items
            .iter()
            .map(|item| item.lr_data.clone().unsqueeze_dim(0))
            .collect::<Vec<_>>();

        let hr_batch = Tensor::cat(hr, 0).to_device(device);
        let lr_batch = Tensor::cat(lr, 0).to_device(device);

        SrganBatch {
            hr_data: hr_batch,
            lr_data: lr_batch,
            size: items.len(),
        }
    }
}

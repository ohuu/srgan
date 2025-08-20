use burn::{
    optim::{adaptor::OptimizerAdaptor, Adam, AdamConfig},
    tensor::{
        backend::{AutodiffBackend, Backend},
        Tensor,
    },
};

use crate::{
    data_srgan::SrganBatch,
    model::{Discriminator, Generator},
};

#[derive(Clone, Debug)]
pub struct DiscLoss<B: Backend> {
    loss: Tensor<B, 1>,
}
#[derive(Clone, Debug)]
pub struct GenLoss<B: Backend> {
    loss: Tensor<B, 1>,
    content_loss: Tensor<B, 1>,
    adversarial_loss: Tensor<B, 1>,
}
#[derive(Clone, Debug)]
pub struct SrganOutput<B: Backend> {
    disc_loss: DiscLoss<B>,
    gen_loss: GenLoss<B>,
    generated_images: Tensor<B, 4>,
}

pub struct SrganTrainer<B: AutodiffBackend> {
    pub generator: Generator<B>,
    pub discriminator: Discriminator<B>,
    pub gen_opt: OptimizerAdaptor<Adam, Generator<B>, B>,
    pub disc_opt: OptimizerAdaptor<Adam, Discriminator<B>, B>,
}
impl<B: AutodiffBackend> SrganTrainer<B> {
    pub fn new(generator: Generator<B>, discriminator: Discriminator<B>) -> Self {
        let gen_opt = AdamConfig::new().init();
        let disc_opt = AdamConfig::new().init();

        Self {
            generator,
            discriminator,
            gen_opt,
            disc_opt,
        }
    }

    pub fn calc_disc_loss(&self, batch: SrganBatch<B>) -> DiscLoss<B> {
        unimplemented!("")
    }

    pub fn calc_gen_loss(&self, batch: SrganBatch<B>) -> GenLoss<B> {
        unimplemented!("")
    }

    /// Custom training loop
    /// This is required because it's not possible to train two networks
    /// simultaneously using the TrainStep trait
    pub fn train(&self, dataloader_train: usize /*replace with actual type*/, epochs: usize) {
        for epoch in 0..epochs {
            // get next batch from dataloader
            // for (iteration, batch) in dataloader_train.iter().enumerate() {
            let disc_loss = self.calc_disc_loss(batch);
            let gen_loss = self.calc_gen_loss(batch);
        }
        unimplemented!()
    }
}

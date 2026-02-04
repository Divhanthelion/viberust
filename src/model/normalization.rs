//! RMSNorm - Root Mean Square Layer Normalization
//!
//! Used in Gemma instead of LayerNorm for efficiency

use burn::{
    module::{Module, Param},
    tensor::{backend::Backend, Tensor},
};

/// RMSNorm module
#[derive(Module, Debug)]
pub struct RmsNorm<B: Backend> {
    weight: Param<Tensor<B, 1>>,
    #[module(skip)]
    eps: f64,
}

impl<B: Backend> RmsNorm<B> {
    pub fn new(hidden_size: usize, eps: f64, device: &B::Device) -> Self {
        let weight = Tensor::ones([hidden_size], device);
        Self {
            weight: Param::from_tensor(weight),
            eps,
        }
    }

    pub fn load_weights(&mut self, weight: Tensor<B, 1>) {
        self.weight = Param::from_tensor(weight);
    }

    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        // Compute RMS: sqrt(mean(x^2) + eps)
        let variance = x.clone().powf_scalar(2.0).mean_dim(2);
        let rms = (variance + self.eps).sqrt();

        // Normalize and scale
        let normalized = x / rms;

        // Broadcast weight for multiplication
        normalized * self.weight.val().clone().unsqueeze::<2>().unsqueeze::<3>()
    }
}

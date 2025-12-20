//! RMSNorm - Root Mean Square Layer Normalization
//!
//! Used in Gemma instead of LayerNorm for efficiency

use burn::{
    module::{Module, Param},
    prelude::*,
    tensor::{backend::Backend, Tensor},
};

/// RMSNorm configuration
#[derive(Config)]
pub struct RmsNormConfig {
    /// Hidden dimension
    pub hidden_size: usize,
    /// Epsilon for numerical stability
    #[config(default = 1e-6)]
    pub eps: f64,
}

impl RmsNormConfig {
    /// Initialize RMSNorm module
    pub fn init<B: Backend>(&self, device: &B::Device) -> RmsNorm<B> {
        let weight: Tensor<B, 1> = Tensor::ones([self.hidden_size], device);
        RmsNorm {
            weight: Param::from_tensor(weight),
            eps: self.eps,
        }
    }
}

/// RMSNorm module
#[derive(Module, Debug)]
pub struct RmsNorm<B: Backend> {
    /// Learnable scale parameter
    weight: Param<Tensor<B, 1>>,
    /// Epsilon for numerical stability
    #[module(skip)]
    eps: f64,
}

impl<B: Backend> RmsNorm<B> {
    /// Forward pass
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        // Compute RMS: sqrt(mean(x^2))
        let variance = x.clone().powf_scalar(2.0).mean_dim(2);
        let rms = (variance + self.eps).sqrt();
        
        // Normalize and scale
        let normalized = x / rms;
        normalized * self.weight.val().clone().unsqueeze()
    }
}

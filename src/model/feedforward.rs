//! GeGLU Feed-Forward Network
//!
//! Gemma uses Gated Linear Units with GELU activation
//! instead of standard ReLU FFN for better expressivity

use burn::{
    module::Module,
    nn::{Linear, LinearConfig},
    prelude::*,
    tensor::{activation::gelu, backend::Backend, Tensor},
};

/// GeGLU FFN configuration
#[derive(Config)]
pub struct GeGluFeedForwardConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
}

impl GeGluFeedForwardConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> GeGluFeedForward<B> {
        // Gate and up projections (combined for efficiency)
        let gate_proj = LinearConfig::new(self.hidden_size, self.intermediate_size)
            .with_bias(false)
            .init(device);
        let up_proj = LinearConfig::new(self.hidden_size, self.intermediate_size)
            .with_bias(false)
            .init(device);
        let down_proj = LinearConfig::new(self.intermediate_size, self.hidden_size)
            .with_bias(false)
            .init(device);

        GeGluFeedForward {
            gate_proj,
            up_proj,
            down_proj,
        }
    }
}

/// GeGLU Feed-Forward module
#[derive(Module, Debug)]
pub struct GeGluFeedForward<B: Backend> {
    gate_proj: Linear<B>,
    up_proj: Linear<B>,
    down_proj: Linear<B>,
}

impl<B: Backend> GeGluFeedForward<B> {
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        // GeGLU: GELU(gate(x)) * up(x)
        let gate = gelu(self.gate_proj.forward(x.clone()));
        let up = self.up_proj.forward(x);
        let hidden = gate * up;
        
        // Project back to hidden size
        self.down_proj.forward(hidden)
    }
}

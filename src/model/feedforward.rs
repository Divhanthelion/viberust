//! GeGLU Feed-Forward Network
//!
//! Gemma uses Gated Linear Units with GELU activation

use burn::{
    module::{Module, Param},
    nn::{Linear, LinearConfig},
    tensor::{activation::gelu, backend::Backend, Tensor},
};

/// GeGLU Feed-Forward module
#[derive(Module, Debug)]
pub struct GeGluFeedForward<B: Backend> {
    gate_proj: Linear<B>,
    up_proj: Linear<B>,
    down_proj: Linear<B>,
}

impl<B: Backend> GeGluFeedForward<B> {
    pub fn new(hidden_size: usize, intermediate_size: usize, device: &B::Device) -> Self {
        let gate_proj = LinearConfig::new(hidden_size, intermediate_size)
            .with_bias(false)
            .init(device);
        let up_proj = LinearConfig::new(hidden_size, intermediate_size)
            .with_bias(false)
            .init(device);
        let down_proj = LinearConfig::new(intermediate_size, hidden_size)
            .with_bias(false)
            .init(device);

        Self {
            gate_proj,
            up_proj,
            down_proj,
        }
    }

    pub fn load_weights(
        &mut self,
        gate_weight: Tensor<B, 2>,
        up_weight: Tensor<B, 2>,
        down_weight: Tensor<B, 2>,
    ) {
        // Transpose weights for Burn's linear layer
        self.gate_proj.weight = Param::from_tensor(gate_weight.transpose());
        self.up_proj.weight = Param::from_tensor(up_weight.transpose());
        self.down_proj.weight = Param::from_tensor(down_weight.transpose());
    }

    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        // GeGLU: GELU(gate(x)) * up(x)
        let gate = gelu(self.gate_proj.forward(x.clone()));
        let up = self.up_proj.forward(x);
        let hidden = gate * up;

        // Project back to hidden size
        self.down_proj.forward(hidden)
    }
}

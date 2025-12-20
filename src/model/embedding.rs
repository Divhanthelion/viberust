//! Token and positional embeddings with RoPE

use burn::{
    module::Module,
    nn::{Embedding, EmbeddingConfig},
    prelude::*,
    tensor::{backend::Backend, Tensor},
};

/// Embedding layer configuration
#[derive(Config)]
pub struct TokenEmbeddingConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
}

impl TokenEmbeddingConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> TokenEmbedding<B> {
        let embedding = EmbeddingConfig::new(self.vocab_size, self.hidden_size).init(device);
        TokenEmbedding { embedding }
    }
}

/// Token embedding module
#[derive(Module, Debug)]
pub struct TokenEmbedding<B: Backend> {
    embedding: Embedding<B>,
}

impl<B: Backend> TokenEmbedding<B> {
    pub fn forward(&self, input_ids: Tensor<B, 2, burn::tensor::Int>) -> Tensor<B, 3> {
        self.embedding.forward(input_ids)
    }
}

/// Apply Rotary Positional Embeddings (RoPE)
///
/// RoPE encodes position by rotating pairs of dimensions
/// Simplified implementation for clarity
pub fn apply_rope<B: Backend>(
    x: Tensor<B, 4>,  // [batch, heads, seq, head_dim]
    seq_len: usize,
    head_dim: usize,
    device: &B::Device,
) -> Tensor<B, 4> {
    let [batch, heads, _seq, _head_dim] = x.dims();
    
    // Generate position indices
    let positions: Vec<f32> = (0..seq_len).map(|i| i as f32).collect();
    let pos: Tensor<B, 1> = Tensor::from_floats(positions.as_slice(), device);
    
    // Generate frequency bands (inverse frequencies)
    let dim_pairs = head_dim / 2;
    let inv_freq: Vec<f32> = (0..dim_pairs)
        .map(|i| 1.0 / 10000_f32.powf(2.0 * i as f32 / head_dim as f32))
        .collect();
    let freq: Tensor<B, 1> = Tensor::from_floats(inv_freq.as_slice(), device);
    
    // Compute angles: [seq_len, dim_pairs]
    let pos_2d: Tensor<B, 2> = pos.unsqueeze_dim(1);
    let freq_2d: Tensor<B, 2> = freq.unsqueeze_dim(0);
    let angles: Tensor<B, 2> = pos_2d.matmul(freq_2d);
    
    // Compute sin and cos
    let cos: Tensor<B, 2> = angles.clone().cos();
    let sin: Tensor<B, 2> = angles.sin();
    
    // Expand cos/sin for broadcasting: [1, 1, seq, dim_pairs]
    let cos_3d: Tensor<B, 3> = cos.unsqueeze_dim::<3>(0);
    let cos_expanded: Tensor<B, 4> = cos_3d.unsqueeze_dim::<4>(0);
    let sin_3d: Tensor<B, 3> = sin.unsqueeze_dim::<3>(0);
    let sin_expanded: Tensor<B, 4> = sin_3d.unsqueeze_dim::<4>(0);
    
    // Split x into first and second halves along head_dim
    let x_first: Tensor<B, 4> = x.clone().slice([0..batch, 0..heads, 0..seq_len, 0..dim_pairs]);
    let x_second: Tensor<B, 4> = x.slice([0..batch, 0..heads, 0..seq_len, dim_pairs..head_dim]);
    
    // Apply rotation: [x1*cos - x2*sin, x1*sin + x2*cos]
    let rotated_first = x_first.clone() * cos_expanded.clone() - x_second.clone() * sin_expanded.clone();
    let rotated_second = x_first * sin_expanded + x_second * cos_expanded;
    
    // Concatenate back
    Tensor::cat(vec![rotated_first, rotated_second], 3)
}

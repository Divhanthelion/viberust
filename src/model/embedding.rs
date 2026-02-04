//! Token embeddings and Rotary Positional Embeddings (RoPE)

use burn::{
    module::{Module, Param},
    nn::{Embedding, EmbeddingConfig},
    tensor::{backend::Backend, Int, Tensor},
};

/// Token embedding layer
#[derive(Module, Debug)]
pub struct TokenEmbedding<B: Backend> {
    pub embedding: Embedding<B>,
    #[module(skip)]
    pub hidden_size: usize,
}

impl<B: Backend> TokenEmbedding<B> {
    pub fn new(vocab_size: usize, hidden_size: usize, device: &B::Device) -> Self {
        let embedding = EmbeddingConfig::new(vocab_size, hidden_size).init(device);
        Self {
            embedding,
            hidden_size,
        }
    }

    pub fn forward(&self, input_ids: Tensor<B, 2, Int>) -> Tensor<B, 3> {
        // Gemma normalizes embeddings by sqrt(hidden_size)
        let embeddings = self.embedding.forward(input_ids);
        let scale = (self.hidden_size as f32).sqrt();
        embeddings * scale
    }

    pub fn load_weights(&mut self, weight: Tensor<B, 2>) {
        self.embedding.weight = Param::from_tensor(weight);
    }
}

/// Rotary Positional Embeddings
///
/// RoPE encodes position by rotating pairs of dimensions.
/// This allows the model to generalize to longer sequences.
#[derive(Debug, Clone)]
pub struct RotaryEmbedding<B: Backend> {
    cos_cache: Tensor<B, 2>,
    sin_cache: Tensor<B, 2>,
    head_dim: usize,
}

impl<B: Backend> RotaryEmbedding<B> {
    pub fn new(
        head_dim: usize,
        max_seq_len: usize,
        theta: f64,
        device: &B::Device,
    ) -> Self {
        let dim_pairs = head_dim / 2;

        // Generate inverse frequencies
        let inv_freq: Vec<f32> = (0..dim_pairs)
            .map(|i| 1.0 / (theta as f32).powf(2.0 * i as f32 / head_dim as f32))
            .collect();

        // Generate position indices
        let positions: Vec<f32> = (0..max_seq_len).map(|i| i as f32).collect();

        // Compute angles: [max_seq_len, dim_pairs]
        let pos_tensor: Tensor<B, 1> = Tensor::from_floats(positions.as_slice(), device);
        let freq_tensor: Tensor<B, 1> = Tensor::from_floats(inv_freq.as_slice(), device);

        let pos_2d: Tensor<B, 2> = pos_tensor.unsqueeze_dim(1);
        let freq_2d: Tensor<B, 2> = freq_tensor.unsqueeze_dim(0);
        let angles: Tensor<B, 2> = pos_2d.matmul(freq_2d);

        // Precompute cos and sin
        let cos_cache = angles.clone().cos();
        let sin_cache = angles.sin();

        Self {
            cos_cache,
            sin_cache,
            head_dim,
        }
    }

    /// Apply RoPE to Q or K tensor
    /// Input shape: [batch, heads, seq_len, head_dim]
    pub fn apply(&self, x: Tensor<B, 4>, start_pos: usize) -> Tensor<B, 4> {
        let [batch, heads, seq_len, head_dim] = x.dims();
        let dim_pairs = head_dim / 2;
        let device = x.device();

        // Get cos/sin for this sequence range
        let cos = self.cos_cache.clone()
            .slice([start_pos..(start_pos + seq_len), 0..dim_pairs]);
        let sin = self.sin_cache.clone()
            .slice([start_pos..(start_pos + seq_len), 0..dim_pairs]);

        // Expand for broadcasting: [1, 1, seq_len, dim_pairs]
        let cos: Tensor<B, 4> = cos.unsqueeze_dim::<3>(0).unsqueeze_dim::<4>(0);
        let sin: Tensor<B, 4> = sin.unsqueeze_dim::<3>(0).unsqueeze_dim::<4>(0);

        // Split x into first and second halves along head_dim
        let x_first: Tensor<B, 4> = x.clone().slice([0..batch, 0..heads, 0..seq_len, 0..dim_pairs]);
        let x_second: Tensor<B, 4> = x.slice([0..batch, 0..heads, 0..seq_len, dim_pairs..head_dim]);

        // Apply rotation: [x1*cos - x2*sin, x1*sin + x2*cos]
        let rotated_first = x_first.clone() * cos.clone() - x_second.clone() * sin.clone();
        let rotated_second = x_first * sin + x_second * cos;

        // Concatenate back
        Tensor::cat(vec![rotated_first, rotated_second], 3)
    }
}

//! Multi-Query Attention (MQA) implementation
//!
//! MQA shares K/V heads across all Q heads, reducing memory bandwidth
//! requirements - critical for the Ryzen 5900HX's limited APU bandwidth

use burn::{
    module::Module,
    nn::{Linear, LinearConfig},
    prelude::*,
    tensor::{backend::Backend, Tensor},
};

use super::embedding::apply_rope;

/// Multi-Query Attention configuration
#[derive(Config)]
pub struct MultiQueryAttentionConfig {
    pub hidden_size: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
}

impl MultiQueryAttentionConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> MultiQueryAttention<B> {
        let head_dim = self.hidden_size / self.num_heads;
        
        let q_proj = LinearConfig::new(self.hidden_size, self.hidden_size)
            .with_bias(false)
            .init(device);
        let k_proj = LinearConfig::new(self.hidden_size, self.num_kv_heads * head_dim)
            .with_bias(false)
            .init(device);
        let v_proj = LinearConfig::new(self.hidden_size, self.num_kv_heads * head_dim)
            .with_bias(false)
            .init(device);
        let o_proj = LinearConfig::new(self.hidden_size, self.hidden_size)
            .with_bias(false)
            .init(device);

        MultiQueryAttention {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            num_heads: self.num_heads,
            num_kv_heads: self.num_kv_heads,
            head_dim,
        }
    }
}

/// Multi-Query Attention module
#[derive(Module, Debug)]
pub struct MultiQueryAttention<B: Backend> {
    q_proj: Linear<B>,
    k_proj: Linear<B>,
    v_proj: Linear<B>,
    o_proj: Linear<B>,
    #[module(skip)]
    num_heads: usize,
    #[module(skip)]
    num_kv_heads: usize,
    #[module(skip)]
    head_dim: usize,
}

impl<B: Backend> MultiQueryAttention<B> {
    pub fn forward(&self, x: Tensor<B, 3>, mask: Option<Tensor<B, 4>>) -> Tensor<B, 3> {
        let [batch, seq_len, _hidden] = x.dims();
        let device = x.device();

        // Project Q, K, V
        let q = self.q_proj.forward(x.clone());
        let k = self.k_proj.forward(x.clone());
        let v = self.v_proj.forward(x);

        // Reshape for multi-head attention
        let q: Tensor<B, 4> = q
            .reshape([batch, seq_len, self.num_heads, self.head_dim])
            .swap_dims(1, 2);
        
        let k: Tensor<B, 4> = k
            .reshape([batch, seq_len, self.num_kv_heads, self.head_dim])
            .swap_dims(1, 2);
        let v: Tensor<B, 4> = v
            .reshape([batch, seq_len, self.num_kv_heads, self.head_dim])
            .swap_dims(1, 2);

        // Apply RoPE to Q and K
        let q = apply_rope(q, seq_len, self.head_dim, &device);
        let k = apply_rope(k, seq_len, self.head_dim, &device);

        // Expand K, V for MQA (repeat for each query head group)
        let k = self.repeat_kv(k, self.num_heads / self.num_kv_heads);
        let v = self.repeat_kv(v, self.num_heads / self.num_kv_heads);

        // Scaled dot-product attention
        let scale = (self.head_dim as f32).sqrt();
        let attn_weights = q.matmul(k.transpose()) / scale;

        // Apply causal mask
        let attn_weights = if let Some(m) = mask {
            attn_weights + m
        } else {
            let causal_mask = self.create_causal_mask(seq_len, &device);
            attn_weights + causal_mask
        };

        // Softmax
        let attn_weights = burn::tensor::activation::softmax(attn_weights, 3);

        // Apply attention to values
        let attn_output = attn_weights.matmul(v);

        // Reshape back
        let attn_output: Tensor<B, 3> = attn_output
            .swap_dims(1, 2)
            .reshape([batch, seq_len, self.num_heads * self.head_dim]);

        self.o_proj.forward(attn_output)
    }

    fn repeat_kv(&self, x: Tensor<B, 4>, n_rep: usize) -> Tensor<B, 4> {
        if n_rep == 1 {
            return x;
        }
        let [batch, num_kv_heads, seq_len, head_dim] = x.dims();
        let x: Tensor<B, 5> = x.unsqueeze_dim(2);
        let x = x.repeat_dim(2, n_rep);
        x.reshape([batch, num_kv_heads * n_rep, seq_len, head_dim])
    }

    fn create_causal_mask(&self, seq_len: usize, device: &B::Device) -> Tensor<B, 4> {
        let mask_val = -1e9_f32;
        let mut mask_data = vec![0.0_f32; seq_len * seq_len];
        
        for i in 0..seq_len {
            for j in (i + 1)..seq_len {
                mask_data[i * seq_len + j] = mask_val;
            }
        }
        
        let mask_1d: Tensor<B, 1> = Tensor::from_floats(mask_data.as_slice(), device);
        let mask_2d: Tensor<B, 2> = mask_1d.reshape([seq_len, seq_len]);
        let mask_3d: Tensor<B, 3> = mask_2d.unsqueeze_dim::<3>(0);
        mask_3d.unsqueeze_dim::<4>(0)
    }
}

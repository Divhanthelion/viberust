//! Multi-Query Attention (MQA) with KV-cache
//!
//! Gemma uses MQA which shares K/V heads across all Q heads,
//! reducing memory bandwidth - critical for APU inference

use burn::{
    module::{Module, Param},
    nn::{Linear, LinearConfig},
    tensor::{activation::softmax, backend::Backend, Tensor},
};

use super::embedding::RotaryEmbedding;

/// KV-cache for a single layer
#[derive(Debug, Clone)]
pub struct KvCache<B: Backend> {
    pub k: Option<Tensor<B, 4>>,
    pub v: Option<Tensor<B, 4>>,
    pub max_seq_len: usize,
}

impl<B: Backend> KvCache<B> {
    pub fn new(max_seq_len: usize) -> Self {
        Self {
            k: None,
            v: None,
            max_seq_len,
        }
    }

    pub fn clear(&mut self) {
        self.k = None;
        self.v = None;
    }

    pub fn update(&mut self, k: Tensor<B, 4>, v: Tensor<B, 4>) -> (Tensor<B, 4>, Tensor<B, 4>) {
        let (new_k, new_v) = match (&self.k, &self.v) {
            (Some(cached_k), Some(cached_v)) => {
                // Concatenate along sequence dimension
                let new_k = Tensor::cat(vec![cached_k.clone(), k], 2);
                let new_v = Tensor::cat(vec![cached_v.clone(), v], 2);
                (new_k, new_v)
            }
            _ => (k, v),
        };

        self.k = Some(new_k.clone());
        self.v = Some(new_v.clone());

        (new_k, new_v)
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
    pub fn new(
        hidden_size: usize,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        device: &B::Device,
    ) -> Self {
        let q_proj = LinearConfig::new(hidden_size, num_heads * head_dim)
            .with_bias(false)
            .init(device);
        let k_proj = LinearConfig::new(hidden_size, num_kv_heads * head_dim)
            .with_bias(false)
            .init(device);
        let v_proj = LinearConfig::new(hidden_size, num_kv_heads * head_dim)
            .with_bias(false)
            .init(device);
        let o_proj = LinearConfig::new(num_heads * head_dim, hidden_size)
            .with_bias(false)
            .init(device);

        Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            num_heads,
            num_kv_heads,
            head_dim,
        }
    }

    /// Load weights from loaded tensors
    pub fn load_weights(
        &mut self,
        q_weight: Tensor<B, 2>,
        k_weight: Tensor<B, 2>,
        v_weight: Tensor<B, 2>,
        o_weight: Tensor<B, 2>,
    ) {
        // Transpose weights (HuggingFace stores as [out, in], Burn expects [in, out] for forward)
        self.q_proj.weight = Param::from_tensor(q_weight.transpose());
        self.k_proj.weight = Param::from_tensor(k_weight.transpose());
        self.v_proj.weight = Param::from_tensor(v_weight.transpose());
        self.o_proj.weight = Param::from_tensor(o_weight.transpose());
    }

    pub fn forward(
        &self,
        x: Tensor<B, 3>,
        rope: &RotaryEmbedding<B>,
        cache: &mut KvCache<B>,
        start_pos: usize,
    ) -> Tensor<B, 3> {
        let [batch, seq_len, _hidden] = x.dims();
        let device = x.device();

        // Project Q, K, V
        let q = self.q_proj.forward(x.clone());
        let k = self.k_proj.forward(x.clone());
        let v = self.v_proj.forward(x);

        // Reshape for multi-head attention: [batch, seq, heads, head_dim]
        let q: Tensor<B, 4> = q
            .reshape([batch, seq_len, self.num_heads, self.head_dim])
            .swap_dims(1, 2); // [batch, heads, seq, head_dim]

        let k: Tensor<B, 4> = k
            .reshape([batch, seq_len, self.num_kv_heads, self.head_dim])
            .swap_dims(1, 2);

        let v: Tensor<B, 4> = v
            .reshape([batch, seq_len, self.num_kv_heads, self.head_dim])
            .swap_dims(1, 2);

        // Apply RoPE to Q and K
        let q = rope.apply(q, start_pos);
        let k = rope.apply(k, start_pos);

        // Update KV cache
        let (k, v) = cache.update(k, v);
        let kv_seq_len = k.dims()[2];

        // Expand K, V for MQA (repeat for each query head group)
        let k = self.repeat_kv(k);
        let v = self.repeat_kv(v);

        // Scaled dot-product attention
        let scale = (self.head_dim as f32).sqrt();
        let attn_weights: Tensor<B, 4> = q.matmul(k.transpose()) / scale;

        // Apply causal mask
        let attn_weights = self.apply_causal_mask(attn_weights, seq_len, kv_seq_len, &device);

        // Softmax
        let attn_weights = softmax(attn_weights, 3);

        // Apply attention to values
        let attn_output = attn_weights.matmul(v);

        // Reshape back: [batch, heads, seq, head_dim] -> [batch, seq, hidden]
        let attn_output: Tensor<B, 3> = attn_output
            .swap_dims(1, 2)
            .reshape([batch, seq_len, self.num_heads * self.head_dim]);

        self.o_proj.forward(attn_output)
    }

    /// Repeat K/V for MQA
    fn repeat_kv(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let n_rep = self.num_heads / self.num_kv_heads;
        if n_rep == 1 {
            return x;
        }

        let [batch, num_kv_heads, seq_len, head_dim] = x.dims();
        let x: Tensor<B, 5> = x.unsqueeze_dim(2);
        let x = x.repeat_dim(2, n_rep);
        x.reshape([batch, num_kv_heads * n_rep, seq_len, head_dim])
    }

    /// Apply causal mask for autoregressive generation
    fn apply_causal_mask(
        &self,
        attn_weights: Tensor<B, 4>,
        q_len: usize,
        kv_len: usize,
        device: &B::Device,
    ) -> Tensor<B, 4> {
        // Create causal mask: mask positions where q_pos < k_pos
        let mask_val = -1e9_f32;
        let mut mask_data = vec![0.0_f32; q_len * kv_len];

        for q_pos in 0..q_len {
            let actual_q_pos = kv_len - q_len + q_pos; // Position in full sequence
            for k_pos in 0..kv_len {
                if k_pos > actual_q_pos {
                    mask_data[q_pos * kv_len + k_pos] = mask_val;
                }
            }
        }

        let mask_1d: Tensor<B, 1> = Tensor::from_floats(mask_data.as_slice(), device);
        let mask_2d: Tensor<B, 2> = mask_1d.reshape([q_len, kv_len]);
        let mask_3d: Tensor<B, 3> = mask_2d.unsqueeze_dim::<3>(0);
        let mask_4d: Tensor<B, 4> = mask_3d.unsqueeze_dim::<4>(0);

        attn_weights + mask_4d
    }
}

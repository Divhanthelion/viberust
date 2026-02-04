//! Gemma model implementation using Burn
//!
//! Implements a decoder-only transformer with:
//! - RoPE (Rotary Positional Embeddings)
//! - RMSNorm (Root Mean Square Normalization)
//! - GeGLU activations
//! - Multi-Query Attention (MQA) for bandwidth efficiency
//! - KV-cache for efficient autoregressive generation

mod attention;
mod embedding;
mod feedforward;
mod normalization;
mod transformer;

pub use transformer::{GemmaModel, GemmaModelConfig};

use burn::tensor::{backend::Backend, Int, Tensor, TensorData};
use tracing::debug;

/// Sampling parameters for text generation
#[derive(Debug, Clone)]
pub struct SamplingParams {
    pub temperature: f32,
    pub top_p: f32,
    pub max_tokens: usize,
}

impl Default for SamplingParams {
    fn default() -> Self {
        Self {
            temperature: 0.7,
            top_p: 0.95,
            max_tokens: 512,
        }
    }
}

/// Generate tokens autoregressively with KV-cache
pub fn generate<B: Backend>(
    model: &mut GemmaModel<B>,
    input_ids: Vec<u32>,
    params: &SamplingParams,
    eos_token_id: u32,
    device: &B::Device,
) -> Vec<u32> {
    let mut generated = Vec::new();
    let mut all_tokens = input_ids.clone();
    let mut next_token;

    // Clear KV cache for new generation
    model.clear_cache();

    // Process prompt (prefill)
    let input_i32: Vec<i32> = input_ids.iter().map(|&x| x as i32).collect();
    let prompt_data = TensorData::new(input_i32, [1, input_ids.len()]);
    let prompt_tensor: Tensor<B, 2, Int> = Tensor::from_data(prompt_data, device);

    let logits = model.forward(prompt_tensor, 0);
    let [_batch, seq_len, vocab_size] = logits.dims();

    // Get last token logits
    let last_logits = logits
        .slice([0..1, (seq_len - 1)..seq_len, 0..vocab_size])
        .reshape([vocab_size]);

    // Sample next token
    next_token = sample_token::<B>(last_logits, params.temperature);

    if next_token == eos_token_id {
        return generated;
    }

    generated.push(next_token);
    all_tokens.push(next_token);

    // Autoregressive generation (decode)
    for _ in 1..params.max_tokens {
        let pos = all_tokens.len() - 1;

        // Only feed the last token (KV cache handles context)
        let token_data = TensorData::new(vec![next_token as i32], [1, 1]);
        let token_tensor: Tensor<B, 2, Int> = Tensor::from_data(token_data, device);

        let logits = model.forward(token_tensor, pos);
        let [_batch, _seq, vocab_size] = logits.dims();

        let last_logits = logits.reshape([vocab_size]);
        next_token = sample_token::<B>(last_logits, params.temperature);

        if next_token == eos_token_id {
            break;
        }

        generated.push(next_token);
        all_tokens.push(next_token);
    }

    debug!("Generated {} tokens", generated.len());
    generated
}

/// Sample a token from logits using temperature scaling
fn sample_token<B: Backend>(
    logits: Tensor<B, 1>,
    temperature: f32,
) -> u32 {
    // Apply temperature
    let scaled = if temperature > 0.0 && temperature != 1.0 {
        logits / temperature
    } else {
        logits
    };

    // Greedy sampling (argmax) for now
    // TODO: Add top-p/nucleus sampling
    let token_idx = scaled.argmax(0);
    let token_data = token_idx.into_data();
    token_data.to_vec::<i64>().unwrap()[0] as u32
}

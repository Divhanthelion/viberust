//! Transformer model implementation using Burn
//!
//! Implements a decoder-only transformer with:
//! - RoPE (Rotary Positional Embeddings)
//! - RMSNorm (Root Mean Square Normalization)
//! - GeGLU activations
//! - Multi-Query Attention (MQA) for bandwidth efficiency

mod attention;
mod embedding;
mod feedforward;
mod normalization;
mod transformer;

pub use transformer::CodeGemmaModel;

use burn::tensor::{backend::Backend, Tensor};

/// Generate text tokens autoregressively
pub fn generate<B: Backend>(
    model: &CodeGemmaModel<B>,
    input_ids: Tensor<B, 2, burn::tensor::Int>,
    max_new_tokens: usize,
    temperature: f32,
    device: &B::Device,
) -> Vec<u32> {
    let mut tokens = input_ids.clone();
    let mut generated = Vec::new();

    for _ in 0..max_new_tokens {
        // Forward pass
        let logits = model.forward(tokens.clone());
        
        // Get last token logits
        let [batch, seq, vocab] = logits.dims();
        let last_logits = logits.slice([0..batch, (seq - 1)..seq, 0..vocab]);
        let last_logits: Tensor<B, 2> = last_logits.reshape([batch, vocab]);

        // Apply temperature
        let scaled_logits = last_logits / temperature;
        
        // Softmax and sample (greedy for now)
        let probs = burn::tensor::activation::softmax(scaled_logits, 1);
        let next_token = probs.argmax(1);
        
        // Get token value
        let token_val = next_token.clone().into_data().to_vec::<i64>().unwrap()[0] as u32;
        
        // Check for EOS
        if token_val == 1 || token_val == 2 {
            break;
        }
        
        generated.push(token_val);
        
        // Append to sequence
        let next_token_1d: Tensor<B, 1, burn::tensor::Int> = 
            Tensor::from_ints([token_val as i32], device);
        let next_token_2d: Tensor<B, 2, burn::tensor::Int> = next_token_1d.reshape([1, 1]);
        tokens = Tensor::cat(vec![tokens, next_token_2d], 1);
    }

    generated
}

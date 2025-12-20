//! Full Transformer model (CodeGemma-style decoder)

use burn::{
    module::Module,
    nn::{Linear, LinearConfig},
    prelude::*,
    tensor::{backend::Backend, Tensor},
};

use super::{
    attention::{MultiQueryAttention, MultiQueryAttentionConfig},
    embedding::{TokenEmbedding, TokenEmbeddingConfig},
    feedforward::{GeGluFeedForward, GeGluFeedForwardConfig},
    normalization::{RmsNorm, RmsNormConfig},
};
use crate::config::ModelConfig;

/// Transformer decoder block
#[derive(Module, Debug)]
pub struct TransformerBlock<B: Backend> {
    input_layernorm: RmsNorm<B>,
    self_attn: MultiQueryAttention<B>,
    post_attention_layernorm: RmsNorm<B>,
    mlp: GeGluFeedForward<B>,
}

impl<B: Backend> TransformerBlock<B> {
    pub fn new(config: &ModelConfig, device: &B::Device) -> Self {
        let input_layernorm: RmsNorm<B> = RmsNormConfig::new(config.hidden_size).init(device);
        
        let self_attn: MultiQueryAttention<B> = MultiQueryAttentionConfig::new(
            config.hidden_size,
            config.num_heads,
            config.num_heads / 4,  // MQA: fewer KV heads
        )
        .init(device);
        
        let post_attention_layernorm: RmsNorm<B> = RmsNormConfig::new(config.hidden_size).init(device);
        
        let mlp: GeGluFeedForward<B> = GeGluFeedForwardConfig::new(config.hidden_size, config.intermediate_size)
            .init(device);

        Self {
            input_layernorm,
            self_attn,
            post_attention_layernorm,
            mlp,
        }
    }

    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        // Pre-norm architecture with residual connections
        let residual = x.clone();
        let x = self.input_layernorm.forward(x);
        let x = self.self_attn.forward(x, None);
        let x = residual + x;

        let residual = x.clone();
        let x = self.post_attention_layernorm.forward(x);
        let x = self.mlp.forward(x);
        residual + x
    }
}

/// CodeGemma model configuration
#[derive(Config)]
pub struct CodeGemmaConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_heads: usize,
    pub num_layers: usize,
    pub intermediate_size: usize,
}

impl From<&ModelConfig> for CodeGemmaConfig {
    fn from(config: &ModelConfig) -> Self {
        Self {
            vocab_size: config.vocab_size,
            hidden_size: config.hidden_size,
            num_heads: config.num_heads,
            num_layers: config.num_layers,
            intermediate_size: config.intermediate_size,
        }
    }
}

/// Full CodeGemma model
#[derive(Module, Debug)]
pub struct CodeGemmaModel<B: Backend> {
    embed_tokens: TokenEmbedding<B>,
    layers: Vec<TransformerBlock<B>>,
    norm: RmsNorm<B>,
    lm_head: Linear<B>,
}

impl<B: Backend> CodeGemmaModel<B> {
    /// Create a new model from configuration
    pub fn new(config: &ModelConfig, device: &B::Device) -> Self {
        let embed_tokens: TokenEmbedding<B> = TokenEmbeddingConfig::new(config.vocab_size, config.hidden_size)
            .init(device);

        let layers: Vec<TransformerBlock<B>> = (0..config.num_layers)
            .map(|_| TransformerBlock::new(config, device))
            .collect();

        let norm: RmsNorm<B> = RmsNormConfig::new(config.hidden_size).init(device);

        let lm_head: Linear<B> = LinearConfig::new(config.hidden_size, config.vocab_size)
            .with_bias(false)
            .init(device);

        Self {
            embed_tokens,
            layers,
            norm,
            lm_head,
        }
    }

    /// Forward pass returning logits
    pub fn forward(&self, input_ids: Tensor<B, 2, burn::tensor::Int>) -> Tensor<B, 3> {
        // Token embeddings
        let mut hidden_states = self.embed_tokens.forward(input_ids);

        // Apply transformer layers
        for layer in &self.layers {
            hidden_states = layer.forward(hidden_states);
        }

        // Final normalization
        hidden_states = self.norm.forward(hidden_states);

        // Project to vocabulary
        self.lm_head.forward(hidden_states)
    }
}

//! Full Gemma Transformer model

use burn::{
    module::Param,
    nn::{Linear, LinearConfig},
    tensor::{backend::Backend, Int, Tensor, TensorData},
};

use super::{
    attention::{KvCache, MultiQueryAttention},
    embedding::{RotaryEmbedding, TokenEmbedding},
    feedforward::GeGluFeedForward,
    normalization::RmsNorm,
};
use crate::loader::{GemmaConfig, LoadedWeights};

/// Gemma model configuration
#[derive(Debug, Clone)]
pub struct GemmaModelConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub max_position_embeddings: usize,
    pub rms_norm_eps: f64,
    pub rope_theta: f64,
}

impl From<&GemmaConfig> for GemmaModelConfig {
    fn from(config: &GemmaConfig) -> Self {
        Self {
            vocab_size: config.vocab_size,
            hidden_size: config.hidden_size,
            intermediate_size: config.intermediate_size,
            num_hidden_layers: config.num_hidden_layers,
            num_attention_heads: config.num_attention_heads,
            num_key_value_heads: config.num_key_value_heads,
            head_dim: config.head_dim,
            max_position_embeddings: config.max_position_embeddings,
            rms_norm_eps: config.rms_norm_eps,
            rope_theta: config.rope_theta,
        }
    }
}

/// A single transformer decoder block
pub struct TransformerBlock<B: Backend> {
    input_layernorm: RmsNorm<B>,
    self_attn: MultiQueryAttention<B>,
    post_attention_layernorm: RmsNorm<B>,
    mlp: GeGluFeedForward<B>,
    cache: KvCache<B>,
}

impl<B: Backend> TransformerBlock<B> {
    pub fn new(config: &GemmaModelConfig, device: &B::Device) -> Self {
        let input_layernorm = RmsNorm::new(config.hidden_size, config.rms_norm_eps, device);
        let self_attn = MultiQueryAttention::new(
            config.hidden_size,
            config.num_attention_heads,
            config.num_key_value_heads,
            config.head_dim,
            device,
        );
        let post_attention_layernorm = RmsNorm::new(config.hidden_size, config.rms_norm_eps, device);
        let mlp = GeGluFeedForward::new(config.hidden_size, config.intermediate_size, device);
        let cache = KvCache::new(config.max_position_embeddings);

        Self {
            input_layernorm,
            self_attn,
            post_attention_layernorm,
            mlp,
            cache,
        }
    }

    pub fn forward(
        &mut self,
        x: Tensor<B, 3>,
        rope: &RotaryEmbedding<B>,
        start_pos: usize,
    ) -> Tensor<B, 3> {
        // Pre-norm architecture with residual connections
        let residual = x.clone();
        let x = self.input_layernorm.forward(x);
        let x = self.self_attn.forward(x, rope, &mut self.cache, start_pos);
        let x = residual + x;

        let residual = x.clone();
        let x = self.post_attention_layernorm.forward(x);
        let x = self.mlp.forward(x);
        residual + x
    }

    pub fn clear_cache(&mut self) {
        self.cache.clear();
    }
}

/// Full Gemma model
pub struct GemmaModel<B: Backend> {
    embed_tokens: TokenEmbedding<B>,
    layers: Vec<TransformerBlock<B>>,
    norm: RmsNorm<B>,
    lm_head: Linear<B>,
    rope: RotaryEmbedding<B>,
    config: GemmaModelConfig,
}

impl<B: Backend> GemmaModel<B> {
    /// Create a new model from configuration (random weights)
    pub fn new(config: GemmaModelConfig, device: &B::Device) -> Self {
        let embed_tokens = TokenEmbedding::new(config.vocab_size, config.hidden_size, device);

        let layers: Vec<TransformerBlock<B>> = (0..config.num_hidden_layers)
            .map(|_| TransformerBlock::new(&config, device))
            .collect();

        let norm = RmsNorm::new(config.hidden_size, config.rms_norm_eps, device);

        let lm_head = LinearConfig::new(config.hidden_size, config.vocab_size)
            .with_bias(false)
            .init(device);

        let rope = RotaryEmbedding::new(
            config.head_dim,
            config.max_position_embeddings,
            config.rope_theta,
            device,
        );

        Self {
            embed_tokens,
            layers,
            norm,
            lm_head,
            rope,
            config,
        }
    }

    /// Load weights from LoadedWeights
    pub fn load_weights(&mut self, weights: &LoadedWeights, device: &B::Device) -> anyhow::Result<()> {
        use anyhow::Context;
        use tracing::info;

        info!("Loading model weights...");

        // Helper to create tensor from weight data
        let make_tensor_2d = |name: &str| -> anyhow::Result<Tensor<B, 2>> {
            let data = weights.tensors.get(name)
                .with_context(|| format!("Weight '{}' not found", name))?;
            let shape = weights.shapes.get(name)
                .with_context(|| format!("Shape for '{}' not found", name))?;

            if shape.len() != 2 {
                anyhow::bail!("Expected 2D tensor for '{}', got {}D", name, shape.len());
            }

            let tensor_data = TensorData::new(data.clone(), [shape[0], shape[1]]);
            Ok(Tensor::from_data(tensor_data, device))
        };

        let make_tensor_1d = |name: &str| -> anyhow::Result<Tensor<B, 1>> {
            let data = weights.tensors.get(name)
                .with_context(|| format!("Weight '{}' not found", name))?;
            let shape = weights.shapes.get(name)
                .with_context(|| format!("Shape for '{}' not found", name))?;

            if shape.len() != 1 {
                anyhow::bail!("Expected 1D tensor for '{}', got {}D", name, shape.len());
            }

            let tensor_data = TensorData::new(data.clone(), [shape[0]]);
            Ok(Tensor::from_data(tensor_data, device))
        };

        // Load embeddings
        let embed_weight = make_tensor_2d("model.embed_tokens.weight")?;
        self.embed_tokens.load_weights(embed_weight);
        info!("Loaded embedding weights");

        // Load each transformer layer
        for (i, layer) in self.layers.iter_mut().enumerate() {
            let prefix = format!("model.layers.{}", i);

            // Input layernorm
            let norm_weight = make_tensor_1d(&format!("{}.input_layernorm.weight", prefix))?;
            layer.input_layernorm.load_weights(norm_weight);

            // Attention weights
            let q_weight = make_tensor_2d(&format!("{}.self_attn.q_proj.weight", prefix))?;
            let k_weight = make_tensor_2d(&format!("{}.self_attn.k_proj.weight", prefix))?;
            let v_weight = make_tensor_2d(&format!("{}.self_attn.v_proj.weight", prefix))?;
            let o_weight = make_tensor_2d(&format!("{}.self_attn.o_proj.weight", prefix))?;
            layer.self_attn.load_weights(q_weight, k_weight, v_weight, o_weight);

            // Post-attention layernorm
            let norm_weight = make_tensor_1d(&format!("{}.post_attention_layernorm.weight", prefix))?;
            layer.post_attention_layernorm.load_weights(norm_weight);

            // MLP weights
            let gate_weight = make_tensor_2d(&format!("{}.mlp.gate_proj.weight", prefix))?;
            let up_weight = make_tensor_2d(&format!("{}.mlp.up_proj.weight", prefix))?;
            let down_weight = make_tensor_2d(&format!("{}.mlp.down_proj.weight", prefix))?;
            layer.mlp.load_weights(gate_weight, up_weight, down_weight);

            if i % 6 == 0 {
                info!("Loaded layer {}/{}", i + 1, self.config.num_hidden_layers);
            }
        }

        // Final norm
        let norm_weight = make_tensor_1d("model.norm.weight")?;
        self.norm.load_weights(norm_weight);

        // LM head (often tied with embeddings, but load separately if present)
        if let Ok(lm_head_weight) = make_tensor_2d("lm_head.weight") {
            self.lm_head.weight = Param::from_tensor(lm_head_weight.transpose());
        } else {
            // Tie with embedding weights
            let embed_weight = make_tensor_2d("model.embed_tokens.weight")?;
            self.lm_head.weight = Param::from_tensor(embed_weight.transpose());
        }

        info!("Model weights loaded successfully!");
        Ok(())
    }

    /// Forward pass returning logits
    pub fn forward(&mut self, input_ids: Tensor<B, 2, Int>, start_pos: usize) -> Tensor<B, 3> {
        // Token embeddings
        let mut hidden_states = self.embed_tokens.forward(input_ids);

        // Apply transformer layers
        for layer in &mut self.layers {
            hidden_states = layer.forward(hidden_states, &self.rope, start_pos);
        }

        // Final normalization
        hidden_states = self.norm.forward(hidden_states);

        // Project to vocabulary
        self.lm_head.forward(hidden_states)
    }

    /// Clear all KV caches
    pub fn clear_cache(&mut self) {
        for layer in &mut self.layers {
            layer.clear_cache();
        }
    }

    /// Get model config
    pub fn config(&self) -> &GemmaModelConfig {
        &self.config
    }
}

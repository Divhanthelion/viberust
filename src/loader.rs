//! Model weight loader for Gemma/CodeGemma models
//!
//! Handles downloading from HuggingFace and loading safetensors weights

use anyhow::{Context, Result};
use burn::tensor::{backend::Backend, Tensor, TensorData};
use safetensors::SafeTensors;
use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;
use tracing::{info, warn};

/// Gemma model variants
#[derive(Debug, Clone, Copy)]
pub enum GemmaModel {
    /// CodeGemma 2B - fast, good for code completion
    CodeGemma2B,
    /// Gemma 2B - general purpose
    Gemma2B,
    /// Gemma 2B Instruct - instruction tuned
    Gemma2BInstruct,
}

impl GemmaModel {
    pub fn repo_id(&self) -> &'static str {
        match self {
            GemmaModel::CodeGemma2B => "google/codegemma-2b",
            GemmaModel::Gemma2B => "google/gemma-2b",
            GemmaModel::Gemma2BInstruct => "google/gemma-2b-it",
        }
    }

    pub fn config(&self) -> GemmaConfig {
        match self {
            GemmaModel::CodeGemma2B | GemmaModel::Gemma2B | GemmaModel::Gemma2BInstruct => {
                GemmaConfig {
                    vocab_size: 256000,
                    hidden_size: 2048,
                    intermediate_size: 16384,
                    num_hidden_layers: 18,
                    num_attention_heads: 8,
                    num_key_value_heads: 1, // MQA
                    head_dim: 256,
                    max_position_embeddings: 8192,
                    rms_norm_eps: 1e-6,
                    rope_theta: 10000.0,
                }
            }
        }
    }
}

/// Gemma model configuration
#[derive(Debug, Clone)]
pub struct GemmaConfig {
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

/// Loaded weights from safetensors
pub struct LoadedWeights {
    pub tensors: HashMap<String, Vec<f32>>,
    pub shapes: HashMap<String, Vec<usize>>,
    pub config: GemmaConfig,
}

impl LoadedWeights {
    /// Get a tensor by name
    pub fn get_tensor<B: Backend>(
        &self,
        name: &str,
        device: &B::Device,
    ) -> Result<Tensor<B, 2>> {
        let data = self.tensors.get(name)
            .with_context(|| format!("Weight '{}' not found", name))?;
        let shape = self.shapes.get(name)
            .with_context(|| format!("Shape for '{}' not found", name))?;

        if shape.len() != 2 {
            anyhow::bail!("Expected 2D tensor for '{}', got {}D", name, shape.len());
        }

        let tensor_data = TensorData::new(data.clone(), [shape[0], shape[1]]);
        Ok(Tensor::from_data(tensor_data, device))
    }

    /// Get a 1D tensor (for norms, biases)
    pub fn get_tensor_1d<B: Backend>(
        &self,
        name: &str,
        device: &B::Device,
    ) -> Result<Tensor<B, 1>> {
        let data = self.tensors.get(name)
            .with_context(|| format!("Weight '{}' not found", name))?;
        let shape = self.shapes.get(name)
            .with_context(|| format!("Shape for '{}' not found", name))?;

        if shape.len() != 1 {
            anyhow::bail!("Expected 1D tensor for '{}', got {}D", name, shape.len());
        }

        let tensor_data = TensorData::new(data.clone(), [shape[0]]);
        Ok(Tensor::from_data(tensor_data, device))
    }

    /// List all available weight names
    pub fn list_weights(&self) -> Vec<&String> {
        self.tensors.keys().collect()
    }
}

/// Weight loader for Gemma models
pub struct WeightLoader {
    cache_dir: PathBuf,
}

impl WeightLoader {
    pub fn new() -> Result<Self> {
        let cache_dir = dirs::cache_dir()
            .unwrap_or_else(|| PathBuf::from("."))
            .join("viberust");

        fs::create_dir_all(&cache_dir)?;

        Ok(Self { cache_dir })
    }

    /// Download model from HuggingFace Hub
    pub fn download_model(&self, model: GemmaModel) -> Result<PathBuf> {
        let repo_id = model.repo_id();
        info!("Downloading model from HuggingFace: {}", repo_id);

        let api = hf_hub::api::sync::Api::new()?;
        let repo = api.model(repo_id.to_string());

        // Download the safetensors file
        let model_path = repo.get("model.safetensors")
            .with_context(|| format!("Failed to download model from {}", repo_id))?;

        info!("Model downloaded to: {:?}", model_path);
        Ok(model_path)
    }

    /// Download tokenizer from HuggingFace Hub
    pub fn download_tokenizer(&self, model: GemmaModel) -> Result<PathBuf> {
        let repo_id = model.repo_id();
        info!("Downloading tokenizer from HuggingFace: {}", repo_id);

        let api = hf_hub::api::sync::Api::new()?;
        let repo = api.model(repo_id.to_string());

        let tokenizer_path = repo.get("tokenizer.json")
            .with_context(|| format!("Failed to download tokenizer from {}", repo_id))?;

        info!("Tokenizer downloaded to: {:?}", tokenizer_path);
        Ok(tokenizer_path)
    }

    /// Load weights from a safetensors file
    pub fn load_safetensors(&self, path: &PathBuf, config: GemmaConfig) -> Result<LoadedWeights> {
        info!("Loading weights from: {:?}", path);

        let file_data = fs::read(path)
            .with_context(|| format!("Failed to read file: {:?}", path))?;

        let tensors = SafeTensors::deserialize(&file_data)
            .context("Failed to deserialize safetensors")?;

        let mut loaded_tensors = HashMap::new();
        let mut shapes = HashMap::new();

        for (name, tensor_view) in tensors.tensors() {
            let shape: Vec<usize> = tensor_view.shape().to_vec();

            // Convert to f32
            let data: Vec<f32> = match tensor_view.dtype() {
                safetensors::Dtype::F32 => {
                    let bytes = tensor_view.data();
                    bytes.chunks_exact(4)
                        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                        .collect()
                }
                safetensors::Dtype::F16 => {
                    let bytes = tensor_view.data();
                    bytes.chunks_exact(2)
                        .map(|b| {
                            let bits = u16::from_le_bytes([b[0], b[1]]);
                            half::f16::from_bits(bits).to_f32()
                        })
                        .collect()
                }
                safetensors::Dtype::BF16 => {
                    let bytes = tensor_view.data();
                    bytes.chunks_exact(2)
                        .map(|b| {
                            let bits = u16::from_le_bytes([b[0], b[1]]);
                            half::bf16::from_bits(bits).to_f32()
                        })
                        .collect()
                }
                dtype => {
                    warn!("Unsupported dtype {:?} for tensor {}, skipping", dtype, name);
                    continue;
                }
            };

            loaded_tensors.insert(name.to_string(), data);
            shapes.insert(name.to_string(), shape);
        }

        info!("Loaded {} tensors", loaded_tensors.len());

        Ok(LoadedWeights {
            tensors: loaded_tensors,
            shapes,
            config,
        })
    }

    /// Load a model (download if needed)
    pub fn load_model(&self, model: GemmaModel) -> Result<LoadedWeights> {
        let model_path = self.download_model(model)?;
        let config = model.config();
        self.load_safetensors(&model_path, config)
    }
}

/// Print weight statistics for debugging
pub fn print_weight_stats(weights: &LoadedWeights) {
    info!("=== Weight Statistics ===");
    for (name, data) in &weights.tensors {
        let shape = weights.shapes.get(name).unwrap();
        let min = data.iter().cloned().fold(f32::INFINITY, f32::min);
        let max = data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mean: f32 = data.iter().sum::<f32>() / data.len() as f32;
        info!("{}: shape={:?}, min={:.4}, max={:.4}, mean={:.4}", name, shape, min, max, mean);
    }
}

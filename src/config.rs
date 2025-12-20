//! Server and model configuration

use serde::{Deserialize, Serialize};

/// Server configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerConfig {
    pub host: String,
    pub port: u16,
    pub model: ModelConfig,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            host: "127.0.0.1".to_string(),
            port: 3000,
            model: ModelConfig::default(),
        }
    }
}

/// Model configuration optimized for Ryzen 5900HX
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    /// Model name/path
    pub name: String,
    /// Maximum context window (tokens)
    pub max_context: usize,
    /// Maximum tokens to generate
    pub max_tokens: usize,
    /// Temperature for sampling
    pub temperature: f32,
    /// Top-p nucleus sampling
    pub top_p: f32,
    /// Vocabulary size
    pub vocab_size: usize,
    /// Hidden dimension
    pub hidden_size: usize,
    /// Number of attention heads
    pub num_heads: usize,
    /// Number of transformer layers
    pub num_layers: usize,
    /// Intermediate (FFN) size
    pub intermediate_size: usize,
}

impl Default for ModelConfig {
    fn default() -> Self {
        // CodeGemma 2B-like configuration optimized for 5900HX
        Self {
            name: "codegemma-2b".to_string(),
            max_context: 4096,  // Limited for APU memory bandwidth
            max_tokens: 2048,
            temperature: 0.7,
            top_p: 0.95,
            vocab_size: 256000,
            hidden_size: 2048,
            num_heads: 8,
            num_layers: 18,
            intermediate_size: 16384,
        }
    }
}

/// System prompt for Rust/Wasm vibecoding
pub const SYSTEM_PROMPT: &str = r#"You are an expert Rust engineer specializing in systems programming and WebAssembly.
You write high-performance, memory-safe, idiomatic Rust code.
You prefer functional programming paradigms and leverage Rust's type system effectively.
When writing WebAssembly, you prioritize wasm-bindgen compatibility and optimize for size.
You understand the borrow checker deeply and write code that compiles on the first try.
You provide concise, working code with minimal boilerplate."#;

//! Server and model configuration

use serde::{Deserialize, Serialize};

/// Server configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerConfig {
    pub host: String,
    pub port: u16,
    pub max_tokens: usize,
    pub temperature: f32,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            host: "127.0.0.1".to_string(),
            port: 3000,
            max_tokens: 512,
            temperature: 0.7,
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

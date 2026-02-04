//! VibeRust Protocol - Local LLM inference server for Rust/Wasm vibecoding
//!
//! Uses Burn framework with WGPU backend for hardware-accelerated inference
//! on AMD Ryzen 5900HX (Vega 8 iGPU via DirectX 12/Vulkan)

mod config;
mod loader;
mod model;
mod server;
mod tokenizer;

use anyhow::Result;
use burn::backend::wgpu::{Wgpu, WgpuDevice};
use tracing::info;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

use loader::{GemmaModel as GemmaVariant, WeightLoader};
use model::{GemmaModel, GemmaModelConfig};

/// Backend type: WGPU for GPU acceleration
pub type Backend = Wgpu;

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize tracing
    tracing_subscriber::registry()
        .with(tracing_subscriber::EnvFilter::new(
            std::env::var("RUST_LOG").unwrap_or_else(|_| "viberust=info".into()),
        ))
        .with(tracing_subscriber::fmt::layer())
        .init();

    info!("🔥 VibeRust Protocol - Starting up...");

    // Initialize WGPU device (auto-selects best available GPU)
    let device = WgpuDevice::DefaultDevice;
    info!("📡 WGPU Device initialized: {:?}", device);

    // Load configuration
    let config = config::ServerConfig::default();
    info!("⚙️  Server config: {:?}", config);

    // Initialize weight loader
    let loader = WeightLoader::new()?;

    // Download and load model weights
    info!("📦 Loading Gemma 2B model...");
    let gemma_variant = GemmaVariant::Gemma2B;
    let weights = loader.load_model(gemma_variant)?;

    // Download tokenizer
    info!("📝 Loading tokenizer...");
    let tokenizer_path = loader.download_tokenizer(gemma_variant)?;
    let tokenizer = tokenizer::Tokenizer::from_file(tokenizer_path.to_str().unwrap())?;

    // Create model
    let model_config = GemmaModelConfig::from(&weights.config);
    info!("🏗️  Creating model with config: {:?}", model_config);
    let mut model: GemmaModel<Backend> = GemmaModel::new(model_config, &device);

    // Load weights into model
    model.load_weights(&weights, &device)?;

    // Start the inference server
    info!("🚀 Starting server on http://{}:{}", config.host, config.port);
    server::run(config, model, tokenizer, device).await?;

    Ok(())
}

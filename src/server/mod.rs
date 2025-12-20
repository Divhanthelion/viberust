//! Axum-based HTTP server with OpenAI-compatible API

mod handlers;
mod types;

pub use handlers::*;

use anyhow::Result;
use axum::{
    routing::{get, post},
    Router,
};
use burn::backend::wgpu::WgpuDevice;
use std::sync::Arc;
use tower_http::cors::{Any, CorsLayer};
use tracing::info;

use crate::config::{ModelConfig, ServerConfig};
use crate::tokenizer::Tokenizer;

/// Shared application state (model-free for thread safety)
#[derive(Clone)]
pub struct AppState {
    pub tokenizer: Arc<Tokenizer>,
    pub config: ModelConfig,
    pub device: WgpuDevice,
}

/// Run the inference server
pub async fn run(config: ServerConfig, device: WgpuDevice) -> Result<()> {
    info!("🔧 Initializing model: {}", config.model.name);
    info!("✅ Model config ready with {} layers", config.model.num_layers);

    // Initialize tokenizer
    info!("📝 Initializing tokenizer...");
    let tokenizer = Arc::new(Tokenizer::simple());
    info!("✅ Tokenizer ready");

    // Create shared state (without model for thread safety)
    let state = AppState {
        tokenizer,
        config: config.model.clone(),
        device,
    };

    // Configure CORS for IDE integration
    let cors = CorsLayer::new()
        .allow_origin(Any)
        .allow_methods(Any)
        .allow_headers(Any);

    // Build router with OpenAI-compatible endpoints
    let app = Router::new()
        .route("/health", get(health))
        .route("/v1/chat/completions", post(chat_completions))
        .route("/v1/completions", post(completions))
        .route("/v1/models", get(list_models))
        .with_state(state)
        .layer(cors);

    // Start server
    let addr = format!("{}:{}", config.host, config.port);
    let listener = tokio::net::TcpListener::bind(&addr).await?;
    
    info!("🎯 VibeRust server listening on http://{}", addr);
    info!("📡 OpenAI-compatible API available at:");
    info!("   POST /v1/chat/completions");
    info!("   POST /v1/completions");
    info!("   GET  /v1/models");

    axum::serve(listener, app).await?;

    Ok(())
}

//! Axum-based HTTP server with OpenAI-compatible API

mod handlers;
mod types;

pub use handlers::*;
pub use types::*;

use anyhow::Result;
use axum::{
    routing::{get, post},
    Router,
};
use burn::backend::wgpu::{Wgpu, WgpuDevice};
use std::sync::Arc;
use tokio::sync::Mutex;
use tower_http::cors::{Any, CorsLayer};
use tracing::info;

use crate::config::ServerConfig;
use crate::model::GemmaModel;
use crate::tokenizer::Tokenizer;

/// Backend type alias
pub type Backend = Wgpu;

/// Shared application state
pub struct AppState {
    pub model: Arc<Mutex<GemmaModel<Backend>>>,
    pub tokenizer: Arc<Tokenizer>,
    pub config: ServerConfig,
    pub device: WgpuDevice,
}

/// Run the inference server
pub async fn run(
    config: ServerConfig,
    model: GemmaModel<Backend>,
    tokenizer: Tokenizer,
    device: WgpuDevice,
) -> Result<()> {
    info!("✅ Model loaded with {} layers", model.config().num_hidden_layers);

    // Create shared state
    let state = Arc::new(AppState {
        model: Arc::new(Mutex::new(model)),
        tokenizer: Arc::new(tokenizer),
        config: config.clone(),
        device,
    });

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

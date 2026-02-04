//! HTTP request handlers with real model inference

use axum::{
    extract::State,
    http::StatusCode,
    response::{sse::{Event, Sse}, IntoResponse, Json},
};
use futures::stream::{self, Stream};
use std::{convert::Infallible, sync::Arc, time::{SystemTime, UNIX_EPOCH}};
use tracing::{info, warn};

use super::types::*;
use super::AppState;
use crate::config::SYSTEM_PROMPT;
use crate::model::{generate, SamplingParams};

/// Health check endpoint
pub async fn health(State(state): State<Arc<AppState>>) -> Json<HealthResponse> {
    let model = state.model.lock().await;
    Json(HealthResponse {
        status: "ok".to_string(),
        model: "gemma-2b".to_string(),
        device: format!("{:?}", state.device),
    })
}

/// List available models
pub async fn list_models(State(_state): State<Arc<AppState>>) -> Json<ModelsResponse> {
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs();

    Json(ModelsResponse {
        object: "list".to_string(),
        data: vec![ModelInfo {
            id: "gemma-2b".to_string(),
            object: "model".to_string(),
            created: now,
            owned_by: "viberust".to_string(),
        }],
    })
}

/// Chat completions endpoint (OpenAI-compatible)
pub async fn chat_completions(
    State(state): State<Arc<AppState>>,
    Json(req): Json<ChatCompletionRequest>,
) -> impl IntoResponse {
    let stream = req.stream.unwrap_or(false);

    if stream {
        // Streaming not yet implemented with real inference
        let stream = generate_stream_placeholder().await;
        Sse::new(stream).into_response()
    } else {
        match generate_complete(state, req).await {
            Ok(response) => Json(response).into_response(),
            Err(e) => {
                warn!("Generation error: {}", e);
                (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()).into_response()
            }
        }
    }
}

/// Legacy completions endpoint
pub async fn completions(
    State(state): State<Arc<AppState>>,
    Json(req): Json<CompletionRequest>,
) -> impl IntoResponse {
    let chat_req = ChatCompletionRequest {
        model: req.model,
        messages: vec![Message {
            role: Role::User,
            content: req.prompt,
        }],
        temperature: req.temperature,
        max_tokens: req.max_tokens,
        stream: req.stream,
        top_p: None,
    };

    match generate_complete(state, chat_req).await {
        Ok(chat_resp) => {
            let response = CompletionResponse {
                id: chat_resp.id,
                object: "text_completion".to_string(),
                created: chat_resp.created,
                model: chat_resp.model,
                choices: chat_resp
                    .choices
                    .into_iter()
                    .map(|c| CompletionChoice {
                        index: c.index,
                        text: c.message.content,
                        finish_reason: c.finish_reason,
                    })
                    .collect(),
                usage: chat_resp.usage,
            };
            Json(response).into_response()
        }
        Err(e) => {
            warn!("Generation error: {}", e);
            (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()).into_response()
        }
    }
}

/// Generate complete response using actual model inference
async fn generate_complete(
    state: Arc<AppState>,
    req: ChatCompletionRequest,
) -> anyhow::Result<ChatCompletionResponse> {
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)?
        .as_secs();

    // Format prompt with system message
    let prompt = format_prompt(&req.messages);
    info!("Generating response for prompt ({} chars)", prompt.len());

    // Tokenize input
    let input_ids = state.tokenizer.encode(&prompt)?;
    let prompt_tokens = input_ids.len();
    info!("Tokenized to {} tokens", prompt_tokens);

    // Get sampling params
    let params = SamplingParams {
        temperature: req.temperature.unwrap_or(state.config.temperature),
        top_p: req.top_p.unwrap_or(0.95),
        max_tokens: req.max_tokens.unwrap_or(state.config.max_tokens),
    };

    // Run inference
    let mut model = state.model.lock().await;

    // Get EOS token ID (Gemma uses 1 for EOS)
    let eos_token_id = state.tokenizer.eos_token_id().unwrap_or(1);

    let generated_ids = generate(
        &mut model,
        input_ids,
        &params,
        eos_token_id,
        &state.device,
    );

    let completion_tokens = generated_ids.len();
    info!("Generated {} tokens", completion_tokens);

    // Decode output
    let output_text = state.tokenizer.decode(&generated_ids)?;

    Ok(ChatCompletionResponse {
        id: format!("chatcmpl-{}", now),
        object: "chat.completion".to_string(),
        created: now,
        model: "gemma-2b".to_string(),
        choices: vec![ChatChoice {
            index: 0,
            message: Message {
                role: Role::Assistant,
                content: output_text,
            },
            finish_reason: "stop".to_string(),
        }],
        usage: Usage {
            prompt_tokens,
            completion_tokens,
            total_tokens: prompt_tokens + completion_tokens,
        },
    })
}

/// Placeholder streaming (to be replaced with real streaming)
async fn generate_stream_placeholder() -> impl Stream<Item = Result<Event, Infallible>> {
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs();

    let chunks = vec![
        ChatCompletionChunk {
            id: format!("chatcmpl-{}", now),
            object: "chat.completion.chunk".to_string(),
            created: now,
            model: "gemma-2b".to_string(),
            choices: vec![ChunkChoice {
                index: 0,
                delta: Delta {
                    role: Some(Role::Assistant),
                    content: Some("Streaming inference coming soon...".to_string()),
                },
                finish_reason: None,
            }],
        },
        ChatCompletionChunk {
            id: format!("chatcmpl-{}", now),
            object: "chat.completion.chunk".to_string(),
            created: now,
            model: "gemma-2b".to_string(),
            choices: vec![ChunkChoice {
                index: 0,
                delta: Delta {
                    role: None,
                    content: None,
                },
                finish_reason: Some("stop".to_string()),
            }],
        },
    ];

    stream::iter(chunks.into_iter().map(|chunk| {
        let data = serde_json::to_string(&chunk).unwrap();
        Ok(Event::default().data(data))
    }))
}

/// Format messages into a prompt string (Gemma format)
fn format_prompt(messages: &[Message]) -> String {
    let mut prompt = String::new();

    // Inject system prompt for Rust/Wasm expertise
    prompt.push_str("<start_of_turn>user\n");
    prompt.push_str(SYSTEM_PROMPT);
    prompt.push_str("<end_of_turn>\n");
    prompt.push_str("<start_of_turn>model\nUnderstood. I'm ready to help with Rust and WebAssembly development.<end_of_turn>\n");

    for msg in messages {
        let role = match msg.role {
            Role::System => continue, // Already handled above
            Role::User => "user",
            Role::Assistant => "model",
        };
        prompt.push_str(&format!("<start_of_turn>{}\n{}<end_of_turn>\n", role, msg.content));
    }

    prompt.push_str("<start_of_turn>model\n");
    prompt
}

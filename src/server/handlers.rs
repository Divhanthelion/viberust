//! HTTP request handlers

use axum::{
    extract::State,
    http::StatusCode,
    response::{
        sse::{Event, Sse},
        IntoResponse, Json,
    },
};
use futures::stream::{self, Stream};
use std::{convert::Infallible, time::{SystemTime, UNIX_EPOCH}};
use tracing::{info, warn};

use super::types::*;
use super::AppState;
use crate::config::SYSTEM_PROMPT;

/// Health check endpoint
pub async fn health(State(state): State<AppState>) -> Json<HealthResponse> {
    Json(HealthResponse {
        status: "ok".to_string(),
        model: state.config.name.clone(),
        device: format!("{:?}", state.device),
    })
}

/// List available models
pub async fn list_models(State(state): State<AppState>) -> Json<ModelsResponse> {
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs();

    Json(ModelsResponse {
        object: "list".to_string(),
        data: vec![ModelInfo {
            id: state.config.name.clone(),
            object: "model".to_string(),
            created: now,
            owned_by: "viberust".to_string(),
        }],
    })
}

/// Chat completions endpoint (OpenAI-compatible)
pub async fn chat_completions(
    State(state): State<AppState>,
    Json(req): Json<ChatCompletionRequest>,
) -> impl IntoResponse {
    let stream = req.stream.unwrap_or(false);
    
    if stream {
        let stream = generate_stream(state, req).await;
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
    State(state): State<AppState>,
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

/// Generate complete response (non-streaming)
/// Note: Full model inference would happen here with proper weight loading
async fn generate_complete(
    state: AppState,
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
    
    // Placeholder response - in production, this would run actual inference
    // The model would be initialized per-request or via a dedicated inference thread
    let output_text = generate_placeholder_response(&req.messages);
    let completion_tokens = state.tokenizer.encode(&output_text)?.len();

    Ok(ChatCompletionResponse {
        id: format!("chatcmpl-{}", now),
        object: "chat.completion".to_string(),
        created: now,
        model: state.config.name.clone(),
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

/// Generate streaming response
async fn generate_stream(
    state: AppState,
    _req: ChatCompletionRequest,
) -> impl Stream<Item = Result<Event, Infallible>> {
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs();

    let model_name = state.config.name.clone();

    // Simulated streaming chunks
    let response_parts = vec![
        "// Generated by VibeRust\n",
        "fn example() {\n",
        "    println!(\"Hello from VibeRust!\");\n",
        "}\n",
    ];

    let mut chunks = vec![
        ChatCompletionChunk {
            id: format!("chatcmpl-{}", now),
            object: "chat.completion.chunk".to_string(),
            created: now,
            model: model_name.clone(),
            choices: vec![ChunkChoice {
                index: 0,
                delta: Delta {
                    role: Some(Role::Assistant),
                    content: None,
                },
                finish_reason: None,
            }],
        },
    ];

    for part in response_parts {
        chunks.push(ChatCompletionChunk {
            id: format!("chatcmpl-{}", now),
            object: "chat.completion.chunk".to_string(),
            created: now,
            model: model_name.clone(),
            choices: vec![ChunkChoice {
                index: 0,
                delta: Delta {
                    role: None,
                    content: Some(part.to_string()),
                },
                finish_reason: None,
            }],
        });
    }

    chunks.push(ChatCompletionChunk {
        id: format!("chatcmpl-{}", now),
        object: "chat.completion.chunk".to_string(),
        created: now,
        model: model_name,
        choices: vec![ChunkChoice {
            index: 0,
            delta: Delta {
                role: None,
                content: None,
            },
            finish_reason: Some("stop".to_string()),
        }],
    });

    stream::iter(chunks.into_iter().map(|chunk| {
        let data = serde_json::to_string(&chunk).unwrap();
        Ok(Event::default().data(data))
    }))
}

/// Format messages into a prompt string
fn format_prompt(messages: &[Message]) -> String {
    let mut prompt = String::new();
    
    // Inject system prompt for Rust/Wasm expertise
    prompt.push_str("<start_of_turn>system\n");
    prompt.push_str(SYSTEM_PROMPT);
    prompt.push_str("<end_of_turn>\n");

    for msg in messages {
        let role = match msg.role {
            Role::System => "system",
            Role::User => "user",
            Role::Assistant => "model",
        };
        prompt.push_str(&format!("<start_of_turn>{}\n{}<end_of_turn>\n", role, msg.content));
    }

    prompt.push_str("<start_of_turn>model\n");
    prompt
}

/// Generate a placeholder response for demonstration
fn generate_placeholder_response(messages: &[Message]) -> String {
    let last_message = messages.last().map(|m| m.content.as_str()).unwrap_or("");
    
    if last_message.to_lowercase().contains("rust") {
        r#"```rust
// Example Rust code generated by VibeRust
fn main() {
    println!("Hello from VibeRust!");
}
```

This is a placeholder response. To enable full inference:
1. Download CodeGemma weights
2. Convert to Burn format using burn-import
3. Load weights in the model initialization"#.to_string()
    } else if last_message.to_lowercase().contains("wasm") {
        r#"```rust
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub fn greet(name: &str) -> String {
    format!("Hello, {}!", name)
}
```

This WebAssembly module can be compiled with:
```bash
wasm-pack build --target web
```"#.to_string()
    } else {
        "VibeRust is ready! Ask me about Rust or WebAssembly development.".to_string()
    }
}

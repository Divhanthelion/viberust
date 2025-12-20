# VibeRust Protocol

Local LLM inference server for Rust/WebAssembly vibecoding using the Burn deep learning framework.

## Features

- **Pure Rust** - Zero Python dependencies, single binary deployment
- **WGPU Backend** - Hardware acceleration via DirectX 12/Vulkan on AMD APUs
- **OpenAI-Compatible API** - Works with VS Code extensions (Continue, etc.)
- **CodeGemma Architecture** - Decoder-only transformer with:
  - RoPE (Rotary Positional Embeddings)
  - RMSNorm
  - GeGLU activations
  - Multi-Query Attention (MQA) for bandwidth efficiency

## Requirements

- Rust 1.75+
- Windows 11 with DirectX 12 or Vulkan support
- AMD Ryzen 5900HX (or similar APU) recommended

## Quick Start

```bash
# Build release binary
cargo build --release

# Run server
cargo run --release
```

Server starts at `http://127.0.0.1:3000`

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/v1/chat/completions` | POST | Chat completions (OpenAI-compatible) |
| `/v1/completions` | POST | Text completions |
| `/v1/models` | GET | List available models |

## Example Usage

```bash
curl http://localhost:3000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "Write a Rust function to parse JSON"}],
    "max_tokens": 500
  }'
```

## IDE Integration

Configure your IDE extension to use `http://localhost:3000` as the API base URL.

### VS Code (Continue)

```json
{
  "models": [{
    "title": "VibeRust Local",
    "provider": "openai",
    "model": "codegemma-2b",
    "apiBase": "http://localhost:3000/v1"
  }]
}
```

## Architecture

```
viberust/
├── src/
│   ├── main.rs          # Entry point, WGPU initialization
│   ├── config.rs        # Server and model configuration
│   ├── tokenizer.rs     # Text tokenization
│   ├── model/
│   │   ├── mod.rs       # Model exports and generation
│   │   ├── transformer.rs   # Full CodeGemma model
│   │   ├── attention.rs     # Multi-Query Attention
│   │   ├── feedforward.rs   # GeGLU FFN
│   │   ├── embedding.rs     # Token embeddings + RoPE
│   │   └── normalization.rs # RMSNorm
│   └── server/
│       ├── mod.rs       # Axum server setup
│       ├── handlers.rs  # API request handlers
│       └── types.rs     # OpenAI-compatible types
```

## Performance Notes

On AMD Ryzen 5900HX with Vega 8 iGPU:
- **2B model (int8)**: ~35 tokens/sec
- **7B model (int8)**: ~10 tokens/sec

Memory bandwidth is the primary bottleneck on APUs. The MQA architecture reduces KV-cache bandwidth requirements.

## License

MIT

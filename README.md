<p align="center">
  <img src="assets/banner.png" alt="locallm-router" width="600" />
</p>

<h1 align="center">locallm-router</h1>

<p align="center">
  A single OpenAI-compatible endpoint that lazily manages multiple local LLM backends across your GPUs.
</p>

<p align="center">
  <strong>vLLM</strong> &bull; <strong>llama.cpp</strong> &bull; <strong>SGLang</strong> &bull; <strong>FreeToken</strong> &mdash; one port, any model, on demand.
</p>

---

## What it does

You configure a list of models. When a request comes in with `"model": "my-model"`, locallm-router:

1. **Downloads** the model from HuggingFace (if not cached)
2. **Evicts** idle models from the GPU if VRAM is tight
3. **Starts** the right backend (vLLM, llama-server, SGLang, or FreeToken)
4. **Proxies** the request to the running backend
5. **Idles out** the model after a configurable timeout

No model runs until something asks for it. No VRAM wasted on models nobody is using.

## Features

- **Multi-backend** &mdash; vLLM, llama.cpp (GGUF), SGLang, and FreeToken (`ft`, edge MoE) behind one unified API
- **Lazy lifecycle** &mdash; models download, start, and stop automatically
- **VRAM-aware scheduling** &mdash; GPU lock tracks estimated VRAM per model, evicts LRU when needed
- **Multi-GPU** &mdash; tensor parallel for large models, or spread different models across GPUs
- **Load balancing** &mdash; run the same model on multiple GPUs, route with `:balance`, `:gpu0`, `:gpu1`
- **Hot config reload** &mdash; change model args without restarting the whole system
- **Dashboard** &mdash; SvelteKit web UI for monitoring, chat playground, and config editing
- **Idle sweep** &mdash; models that go unused get stopped and their VRAM reclaimed

## Quick start

### CLI (native)

Runs the router directly on your machine — needs [Bun](https://bun.sh), plus
`llama-server` on your PATH for GGUF models (and/or `vllm` for vLLM models):

```bash
bunx locallm-router init      # scaffold ~/.config/locallm-router/config.json
bunx locallm-router doctor    # check backends, GPUs, config
bunx locallm-router up        # run in the foreground

# Run automatically at login (systemd user unit on Linux, launchd on macOS):
bunx locallm-router service install
# Optional on Linux — start at boot without logging in:
loginctl enable-linger $USER
```

`locallm-router status` and `locallm-router stop-all` work from any terminal
while the service runs.

### Docker

```bash
git clone https://github.com/Bewinxed/locallm-router.git
cd locallm-router

# Copy and edit your model config
cp config.example.json config.json
cp .env.example .env

# Build and run
docker compose up --build
```

The proxy listens on **:30000**, the dashboard on **:30001**, and the management API on **:30099**.

## Configuration

### `config.json`

```jsonc
{
  "port": 30000,
  "managementPort": 30099,
  "models": [
    {
      "name": "qwen-7b",
      "modelPath": "Qwen/Qwen2.5-7B-Instruct",  // HuggingFace repo ID
      "parameterCount": 7,
      "gpu": 0,
      "idleTimeout": 300000,    // 5 minutes
      "dtype": "bfloat16",
      "extraArgs": ["--gpu-memory-utilization", "0.85"]
    },
    {
      "name": "qwythos-27b",
      "backend": "llama-server",
      "modelPath": "empero-ai/Qwythos-27B-v1-GGUF", // GGUF repo…
      "modelFile": "Qwythos-27B-Q4_K_M.gguf",       // …download ONLY this quant
      "gpu": 0,
      "idleTimeout": 600000,
      "extraArgs": ["-ngl", "999", "-c", "16384", "-fa", "on", "--jinja"]
    },
    {
      "name": "my-gguf",
      "backend": "llama-server",
      "modelPath": "/models/my-model.gguf",       // Local GGUF file
      "gpu": 1,
      "idleTimeout": 300000,
      "aliases": ["default"],
      "balanceGpus": [0, 1],    // Run on both GPUs, route with :balance
      "extraArgs": ["-ngl", "999", "-c", "32768"]
    }
  ]
}
```

### Environment variables

| Variable | Default | Description |
|---|---|---|
| `HF_TOKEN` | | HuggingFace token for gated models |
| `HF_HOME` | `./models` | Local directory for downloaded models |
| `LLM_ROUTER_CONFIG` | `/app/config.json` | Path to config file |
| `LLM_IDLE_TIMEOUT_MINUTES` | | Override idle timeout for all models |
| `NVIDIA_VISIBLE_DEVICES` | `all` | Which GPUs to expose |

## API

locallm-router is fully OpenAI-compatible. Point any client at `http://localhost:30000`:

```bash
curl http://localhost:30000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "qwen-7b", "messages": [{"role": "user", "content": "Hello"}]}'
```

### Endpoints

| Endpoint | Description |
|---|---|
| `GET /v1/models` | List all configured models and their status |
| `POST /v1/chat/completions` | Chat completions (routes by `model` field) |
| `POST /v1/completions` | Text completions |
| `POST /v1/embeddings` | Embeddings |

### Load balancing

For models with `balanceGpus` configured:

```bash
# Auto-route to the least loaded GPU
curl ... -d '{"model": "my-gguf:balance", ...}'

# Pin to a specific GPU
curl ... -d '{"model": "my-gguf:gpu0", ...}'
curl ... -d '{"model": "my-gguf:gpu1", ...}'
```

### Management API (port 30099)

| Endpoint | Description |
|---|---|
| `GET /status` | Status of all models and GPUs |
| `POST /start/<model>` | Force start a model |
| `POST /stop/<model>` | Force stop a model |
| `POST /stop-all` | Stop all running models |
| `POST /restart/<model>` | Restart with fresh config. `?context=N` or `{ "overrides": { "-c": "N" } }` |
| `POST /reload-config/<model>` | Hot reload config from disk |

The same endpoints are also served under `/manager/*` on the proxy port
(`:30000/manager/status`, …). Because they start and stop processes, port
30099 binds to `127.0.0.1` — set `ROUTER_MGMT_HOST=0.0.0.0` to expose it
(docker-compose.yml already does).

## Architecture

```
                    ┌─────────────┐
   :30000  ───────▶ │   Proxy     │ ◀── config.json
                    │  (Bun.serve)│
                    └──────┬──────┘
                           │  routes by "model" field
              ┌────────────┼────────────┐
              ▼            ▼            ▼
        ┌──────────┐ ┌──────────┐ ┌──────────┐
        │  vLLM    │ │ llama    │ │ SGLang   │
        │ :40000   │ │ :40001   │ │ :40002   │
        └──────────┘ └──────────┘ └──────────┘
              │            │            │
         GPU 0         GPU 1        GPU 0
              │            │            │
        ┌─────┴────────────┴────────────┴─────┐
        │          GPU Lock (VRAM scheduler)   │
        │     Tracks usage, evicts LRU models  │
        └─────────────────────────────────────┘
```

## Project structure

```
locallm-router/
├── docker-compose.yml
├── Dockerfile              # 3-stage: llama.cpp build, vLLM/torch, runtime
├── config.json             # Your model configuration
├── src/
│   ├── index.ts            # Entrypoint, management API, dashboard launcher
│   ├── proxy.ts            # Unified OpenAI proxy with model routing
│   ├── process.ts          # Backend process lifecycle (vLLM, llama, SGLang)
│   ├── config.ts           # Config loading and hot reload
│   ├── download.ts         # HuggingFace model downloader
│   ├── gpu-lock.ts         # VRAM-aware GPU scheduler
│   └── model-size.ts       # VRAM estimation from model metadata
└── dashboard/              # SvelteKit monitoring UI
    └── src/
```

## Requirements

- NVIDIA GPU(s) with CUDA support
- Docker with [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
- `docker compose` v2+

## License

MIT

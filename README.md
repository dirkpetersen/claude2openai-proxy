# claude-to-openai proxy

## Overview
FastAPI proxy translating Anthropic-style /v1/messages to OpenAI via LiteLLM. Handles model mapping, content block conversion, streaming (SSE), tools, and token counting with reduced logging noise.

- Original code came from: https://github.com/1rgs/claude-code-proxy
- fixed for generic usages

## Features
- Model mapping via env and prefixes (anthropic/* passthrough, openai/* normalized)
- Content-block to OpenAI messages conversion (text/image/tool_use/tool_result)
- SSE streaming compatible with Anthropic clients
- Tool calls passthrough; converts to Anthropic tool_use blocks on response
- Token counting endpoint using LiteLLM token_counter
- Minimal logging with filtered noise and pretty request logs
- OpenAPI schema and Swagger UI with Anthropic-compatible models and endpoint docs

## Run
```
uv run uvicorn server:app --host 0.0.0.0 --port 8082 --reload
```

## API Docs (OpenAPI / Swagger)
- OpenAPI JSON: GET `/_openapi.json` or `/openapi.json` (FastAPI default)
- Swagger UI: GET `/docs`
- ReDoc: GET `/redoc`

App metadata (title/description/version) and schema descriptions are embedded so the docs reflect the Anthropic request/response shapes and streaming behavior.

## Anthropic Compatibility Notes
- Endpoints accept Anthropic Messages API-shaped payloads.
- Streaming when `stream=true` returns Anthropic-style SSE events in this order:
  - `message_start` → `content_block_start` (text) → `content_block_delta`(n) → `content_block_stop`
  - optional tool call blocks: `content_block_start` (tool_use) → `input_json_delta`(n) → `content_block_stop`
  - `message_delta` (includes `stop_reason`, `usage`) → `message_stop` → final `data: [DONE]`
- Tooling:
  - Request tools follow Anthropic `tools[]` with `name`, `description`, `input_schema` (JSON Schema).
  - Responses may include `tool_use` blocks for Claude-family models; for non-Claude models, tool info is appended as text.

## Container

### Build
```
docker build -t claude-proxy .
```

### Run
```
docker run -d \
  --name claude-proxy \
  -p 8082:8082 \
  -e OPENAI_API_BASE="http://10.2.2.10:4000/v1" \
  -e BIG_MODEL=gpt-4.1 \
  -e SMALL_MODEL=gpt-4.1-mini \
  -e MAX_TOKENS=65535 \
  -e BIG_PREFIXES="opus,sonnet" \
  -e SMALL_PREFIXES="haiku" \
  claude-proxy
```

### Docker Compose
docker-compose.yml:
```
version: "3.8"
services:
  claude-proxy:
    image: claude-proxy
    container_name: claude-proxy
    ports:
      - "8082:8082"
    environment:
      OPENAI_API_BASE: "http://10.2.2.10:4000/v1"
      BIG_MODEL: "gpt-4.1"
      SMALL_MODEL: "gpt-4.1-mini"
      MAX_TOKENS: "65535"
      BIG_PREFIXES: "opus,sonnet"
      SMALL_PREFIXES: "haiku"
    restart: unless-stopped
```
Run with:
```
docker compose up -d
```

## Usages

- Claude Code
```
ANTHROPIC_BASE_URL=http://localhost:8082 ANTHROPIC_API_KEY=sk-keyhere ANTHROPIC_MODEL="big-size-model" ANTHROPIC_SMALL_FAST_MODEL="small-size-model" claude
```

## Environment
- OPENAI_API_KEY: OpenAI key (or send via X-API-Key header)
- OPENAI_API_BASE: upstream OpenAI-compatible base URL (e.g., https://api.openai.com/v1)
- BASE_URL: external public prefix for this proxy behind LB/reverse-proxy (e.g., https://foo.bar/claude-proxy/v1). Only affects returned links/OpenAPI servers; not used for upstream calls
- BIG_MODEL: default big model (default: gpt-4.1)
- SMALL_MODEL: default small model (default: gpt-4.1-mini)
- BIG_PREFIXES: map prefixes to BIG_MODEL (default: opus,sonnet)
- SMALL_PREFIXES: map prefixes to SMALL_MODEL (default: haiku)
- MAX_TOKENS: cap for OpenAI models (default: 65535)

## Model Resolution
- If body.model exists: used as-is; provider prefix added if missing
- Else: when body.size=="small" use openai/SMALL_MODEL, otherwise openai/BIG_MODEL
- If model is an OpenAI known id without prefix, server prefixes as openai/<model>

## Endpoints
- POST /v1/messages
  - Request body: Anthropic-like { model, max_tokens, messages, system, tools, tool_choice, temperature, stream }
  - Behavior: converts to OpenAI format for LiteLLM, caps max_tokens for OpenAI models to MAX_TOKENS
  - Streaming: when stream=true, emits Anthropic SSE events (message_start, content_block_start/delta/stop, message_delta, message_stop, final [DONE])
  - Auth: Header `X-API-Key: <OpenAI key>` (OpenAPI 문서에 노출됨)
- POST /v1/messages/count_tokens
  - Returns { input_tokens }
  - Auth: Header `X-API-Key: <OpenAI key>`
- GET /
  - Returns { message, docs, redoc, openapi }

## Content Conversion Notes
- User/assistant messages may be string or content blocks; lists are flattened to text for OpenAI
- tool_result blocks in user messages are converted to plain text segments
- image blocks emit a placeholder text when targeting OpenAI text endpoints
- Unsupported fields in messages are removed to satisfy OpenAI API

## Headers/Auth
- Send `X-API-Key: <OpenAI key>`
- If missing, request is rejected with 401

## Examples
Simple completion:
```
curl -sS -X POST http://localhost:8082/v1/messages \
  -H 'Content-Type: application/json' \
  -H "X-API-Key: $OPENAI_API_KEY" \
  -d '{
    "model": "openai/gpt-4o-mini",
    "max_tokens": 128,
    "messages": [
      {"role": "user", "content": "Say hi"}
    ]
  }'
```

Streaming response:
```
curl -N -sS -X POST http://localhost:8082/v1/messages \
  -H 'Content-Type: application/json' \
  -H "X-API-Key: $OPENAI_API_KEY" \
  -d '{
    "model": "openai/gpt-4o-mini",
    "stream": true,
    "max_tokens": 64,
    "messages": [
      {"role": "user", "content": "Stream a short sentence"}
    ]
  }'
```

Token count:
```
curl -sS -X POST http://localhost:8082/v1/messages/count_tokens \
  -H 'Content-Type: application/json' \
  -H "X-API-Key: $OPENAI_API_KEY" \
  -d '{
    "model": "openai/gpt-4o-mini",
    "messages": [
      {"role": "user", "content": "How many tokens?"}
    ]
  }'
```

## Logging
- Global WARN level, uvicorn access/error suppressed
- Filters noisy LiteLLM/internal messages
- Pretty per-request log showing Anthropic→OpenAI mapping and counts

## Security Notes
- Do not log secrets; API key is read from header and assigned to process env for the call, then restored
- Validate inputs where possible; unsupported fields are stripped before upstream calls

## Install as user level system service 

- use the `start-service.sh` and `stop-service.sh` scripts if you prefer systemd over containers.
- make sure uv is isntalled (or run `curl -LsSf https://astral.sh/uv/install.sh | sh`)

run start-service.sh

```
$ ./start-service.sh
[INFO] Installing dependencies...
Using CPython 3.12.3 interpreter at: /usr/bin/python3
Creating virtual environment at: .venv
Resolved 67 packages in 1.10s
warning: `aiohttp==3.11.14` is yanked (reason: "Regression: https://github.com/aio-libs/aiohttp/issues/10617")
Installed 63 packages in 398ms
 + aiohappyeyeballs==2.6.1
 + aiohttp==3.11.14
 + aiosignal==1.3.2
 + annotated-types==0.7.0
 + anyio==4.9.0
 + attrs==25.3.0
 + certifi==2025.1.31
 + charset-normalizer==3.4.1
 + click==8.1.8
 + distro==1.9.0
 + dnspython==2.7.0
 + email-validator==2.2.0
 + fastapi==0.115.11
 + fastapi-cli==0.0.7
 + filelock==3.18.0
 + frozenlist==1.5.0
 + fsspec==2025.3.0
 + h11==0.14.0
 + httpcore==1.0.7
 + httptools==0.6.4
 + httpx==0.28.1
 + huggingface-hub==0.29.3
 + idna==3.10
 + importlib-metadata==8.6.1
 + jinja2==3.1.6
 + jiter==0.9.0
 + jsonschema==4.23.0
 + jsonschema-specifications==2024.10.1
 + litellm==1.63.11
 + markdown-it-py==3.0.0
 + markupsafe==3.0.2
 + mdurl==0.1.2
 + multidict==6.2.0
 + openai==1.66.3
 + packaging==24.2
 + propcache==0.3.0
 + pydantic==2.10.6
 + pydantic-core==2.27.2
 + pygments==2.19.1
 + python-dotenv==1.0.1
 + python-multipart==0.0.20
 + pyyaml==6.0.2
 + referencing==0.36.2
 + regex==2024.11.6
 + requests==2.32.3
 + rich==13.9.4
 + rich-toolkit==0.13.2
 + rpds-py==0.23.1
 + shellingham==1.5.4
 + sniffio==1.3.1
 + starlette==0.46.1
 + tiktoken==0.9.0
 + tokenizers==0.21.1
 + tqdm==4.67.1
 + typer==0.15.2
 + typing-extensions==4.12.2
 + urllib3==2.3.0
 + uvicorn==0.34.0
 + uvloop==0.21.0
 + watchfiles==1.0.4
 + websockets==15.0.1
 + yarl==1.18.3
 + zipp==3.21.0
[INFO] Creating systemd service file...
[INFO] Service file created at /home/ochat/.config/systemd/user/claude-proxy.service
[INFO] Stopping any existing service...
[INFO] Reloading systemd user daemon...
[INFO] Enabling claude-proxy service...
Created symlink /home/ochat/.config/systemd/user/default.target.wants/claude-proxy.service → /home/ochat/.config/systemd/user/claude-proxy.service.
[INFO] Starting claude-proxy service...
[INFO] ✅ Service claude-proxy is running successfully!
[INFO] Service status:
● claude-proxy.service - Claude to OpenAI Proxy Server
     Loaded: loaded (/home/ochat/.config/systemd/user/claude-proxy.service; enabled; preset: enabled)
     Active: active (running) since Mon 2025-09-15 14:04:28 PDT; 3s ago
   Main PID: 432858 (uv)
      Tasks: 5 (limit: 2267)
     Memory: 82.2M (peak: 82.4M)
        CPU: 2.042s
     CGroup: /user.slice/user-996.slice/user@996.service/app.slice/claude-proxy.service
             ├─432858 uv run uvicorn server:app --host 0.0.0.0 --port 8088
             └─432862 /home/ochat/claude2openai-proxy/.venv/bin/python /home/ochat/claude2openai-proxy/.venv/bin/uvicorn server:app --host 0.0.0.0 --port 8088

Sep 15 14:04:28 ochat systemd[1268]: Started claude-proxy.service - Claude to OpenAI Proxy Server.
[INFO] 🎉 Setup complete!
[INFO]
[INFO] Your service is now running on http://0.0.0.0:8088
[INFO]
[INFO] Useful commands:
[INFO]   Check status:    systemctl --user status claude-proxy
[INFO]   View logs:       journalctl --user -u claude-proxy -f
[INFO]   Restart service: systemctl --user restart claude-proxy
[INFO]   Stop service:    systemctl --user stop claude-proxy
[INFO]   Disable service: systemctl --user disable claude-proxy
[INFO]
[INFO] To enable linger (start service on boot without login):
[INFO]   sudo loginctl enable-linger ochat
```

and then stop-service 

```
$ ./stop-service.sh
[INFO] Stopping claude-proxy service...
[INFO] Disabling claude-proxy service...
[INFO] Removing service file...
[INFO] Reloading systemd user daemon...
[INFO] ✅ Service claude-proxy has been stopped and removed!
```

# Anthropic Proxy (with LiteLLM)

This FastAPI server acts as a proxy to translate Anthropic-format API requests (Claude-compatible) into OpenAI-compatible requests, forwarding them to any OpenAI-style backend and remapping streamed/completed responses.

## Usage

### 1. Setup (one-time)

Run the setup script to create a virtual environment and install all dependencies for both proxy and LiteLLM:

```bash
./setup_litellm_copilot.sh
```

The script will:

- Create and activate a Python venv
- Install all dependencies for proxying and LiteLLM
- Patch for Copilot debug (if used)
- Remind you to edit `config.yaml` for LiteLLM (if custom config needed)
- Remind you to set your `OPENAI_API_KEY` and ensure `OPENAI_API_BASE` points to the correct backend

### 2. Launch services

To start both the LiteLLM backend and this proxy together:

```bash
./start_servers.sh
```

This launches both litellm and the proxy in sequence, listening by default on ports 4000 (backend) and 3000 (proxy).

---

## Inspiration & References

- [[Feature]: Add GitHub Copilot as model provider](https://github.com/BerriAI/litellm/issues/6564#issuecomment-2894574403)
- [Request: Custom Claude API Endpoint Support](https://github.com/anthropics/claude-code/issues/216#issuecomment-2765752730)
- [maxnowack/anthropic-proxy](https://github.com/maxnowack/anthropic-proxy)

## API

- POST `/v1/messages`: Accepts Anthropic-style messages and tools payload. Returns responses in Anthropic format (sync or streamed/SSE).
- Payloads and responses are automatically translated between protocols.

---

## Example usage

### Call the Anthropic API via the proxy with `curl`

```bash
curl -X POST "http://localhost:3000/v1/messages" \
  -H "Content-Type: application/json" \
  -H "x-api-key: sk-..." \
  -d '{
        "model": "my-model",
        "max_tokens": 100,
        "messages": [
          {"role": "user", "content": "Hello, Claude!"}
        ]
      }'
```

### Use Claude Code with a local proxy

```bash
ANTHROPIC_BASE_URL="http://localhost:3000" claude
```

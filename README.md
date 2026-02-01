# anthropic-proxy-litellm

In-process LiteLLM proxy for Anthropic-compatible API access to any LLM.

Wraps [LiteLLM](https://github.com/BerriAI/litellm)'s proxy server in a
Python context manager (`LiteLLMProxy`), so you can spin up an
Anthropic-compatible endpoint in a single `with` block — no separate process,
no shell scripts, no port management.

Works with any LiteLLM-supported backend: **AWS Bedrock**, **GitHub Copilot**,
**Ollama**, **OpenAI**, **Azure**, and
[many more](https://docs.litellm.ai/docs/providers).

## Install

```bash
uv pip install -e .
```

## Quickstart

### With Claude Agent SDK

```python
from anthropic_proxy_litellm import LiteLLMProxy

with LiteLLMProxy(config="config.yaml", model="my-model") as proxy:
    async for msg in proxy.query("Say hello"):
        print(msg)
```

### With Claude Code CLI

Write a script that starts the proxy and prints env vars:

```python
# start_proxy.py
from anthropic_proxy_litellm import LiteLLMProxy

models = [
    {"model_name": "gpt-4.1", "litellm_params": {"model": "github_copilot/gpt-4.1"}},
]

with LiteLLMProxy(model_list=models, model="gpt-4.1") as proxy:
    for key, value in proxy.env.items():
        print(f"{key}={value}", flush=True)
    print("", flush=True)  # blank line = ready

    import signal, threading
    stop = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    try:
        stop.wait()
    except KeyboardInterrupt:
        pass
```

Then drive it from a shell script:

```bash
#!/usr/bin/env bash
set -euo pipefail
PIPE=$(mktemp -u); mkfifo "$PIPE"
cleanup() { kill "$PROXY_PID" 2>/dev/null; rm -f "$PIPE"; }
trap cleanup EXIT

uv run start_proxy.py > "$PIPE" &
PROXY_PID=$!

while IFS= read -r line; do
    [ -z "$line" ] && break
    export "$line"
done < "$PIPE"

claude -p "Say hello"
```

See [`examples/copilot/`](examples/copilot/) for a complete working version.

## API

### `LiteLLMProxy`

| Parameter          | Env var fallback             | Description                               |
|--------------------|------------------------------|-------------------------------------------|
| `config`           | `LITELLM_CONFIG`             | Path to LiteLLM YAML config file          |
| `model_list`       | —                            | List of model dicts (generates temp YAML) |
| `port`             | `LITELLM_PORT`               | Proxy port (default: 4000)                |
| `model`            | `ANTHROPIC_MODEL`            | Model name for SDK / env export           |
| `small_fast_model` | `ANTHROPIC_SMALL_FAST_MODEL` | Small/fast model name (default: `model`)  |
| `verbose`          | `LITELLM_VERBOSE`            | Enable LiteLLM debug logging              |
| `log_requests`     | `LITELLM_LOG_REQUESTS`       | Log HTTP requests with method/path/status |
| `env_file`         | —                            | Path to `.env` file to load               |

Provide exactly one of `config` or `model_list` (or set `LITELLM_CONFIG`).

### Methods

- **`proxy.options(**kwargs)`** — Returns a `ClaudeAgentOptions` pre-configured
  for this proxy. Pass any SDK option as a kwarg.
- **`proxy.query(prompt, **kwargs)`** — Async generator: sends a prompt through
  the Claude Agent SDK and yields raw messages.
- **`proxy.env`** — Dict of environment variables (`ANTHROPIC_BASE_URL`, etc.)
  for passing to subprocesses.

## Examples

| Directory                                              | Backend        | Proxy? | Description                                      |
|--------------------------------------------------------|----------------|--------|--------------------------------------------------|
| [`examples/bedrock-native/`](examples/bedrock-native/) | AWS Bedrock    | No     | Anthropic models via `CLAUDE_CODE_USE_BEDROCK=1` |
| [`examples/bedrock/`](examples/bedrock/)               | AWS Bedrock    | Yes    | Non-Anthropic models (DeepSeek, Qwen, etc.)      |
| [`examples/copilot/`](examples/copilot/)               | GitHub Copilot | Yes    | Inline models, proxy + `claude` CLI              |

## References

- [LiteLLM `/v1/messages` endpoint](https://docs.litellm.ai/docs/anthropic_unified/)
- [Claude Agent SDK](https://platform.claude.com/docs/en/agent-sdk/overview)
- [Claude Code CLI](https://docs.anthropic.com/en/docs/claude-code)

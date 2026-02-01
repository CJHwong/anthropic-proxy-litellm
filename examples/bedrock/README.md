# Bedrock Example (Non-Anthropic Models)

Use the LiteLLM proxy to run Claude Code against non-Anthropic models on
AWS Bedrock (DeepSeek, Qwen, OpenAI, etc.).

For **Anthropic models** on Bedrock (Claude Sonnet, Haiku), use
[`examples/bedrock-native/`](../bedrock-native/) instead — no proxy needed.

## Prerequisites

- AWS credentials configured (`aws configure` or environment variables)
- Models enabled in your AWS account via the Bedrock console

## Setup

```bash
cp .env.example .env
# Edit .env with your AWS profile / region
```

## Run

```bash
# Interactive session:
./demo.sh

# Non-interactive prompt:
./demo.sh "What is the capital of France?"
```

No prior `pip install` needed — PEP 723 inline metadata lets `uv run` resolve
all dependencies automatically.

## How it works

- **`demo.py`** starts the LiteLLM proxy with `config.yaml`, prints env vars
  to stdout, then blocks until terminated.
- **`demo.sh`** backgrounds `demo.py`, reads the env vars from its output,
  exports them, runs `claude`, and kills the proxy on exit.

## Configuration

Edit `config.yaml` to add or remove Bedrock models. Each entry maps a
`model_name` to a `bedrock/<model-id>` backend.

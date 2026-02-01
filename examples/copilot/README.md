# GitHub Copilot + Claude Code Example

Start a LiteLLM proxy with Copilot models and run Claude Code against it.

## Prerequisites

- `claude` CLI installed (`npm i -g @anthropic-ai/claude-code`)
- GitHub Copilot subscription
- Copilot authentication configured for LiteLLM
  (see [LiteLLM Copilot docs](https://docs.litellm.ai/docs/providers/github_copilot))

## Run

```bash
# Interactive session:
./demo.sh

# Non-interactive prompt:
./demo.sh "What files are in this directory?"
```

No prior `pip install` needed — PEP 723 inline metadata lets `uv run` resolve
all dependencies automatically.

## How it works

- **`demo.py`** starts the LiteLLM proxy with inline Copilot models, prints
  env vars (`ANTHROPIC_BASE_URL`, `ANTHROPIC_MODEL`, etc.) to stdout, then
  blocks until terminated.
- **`demo.sh`** backgrounds `demo.py`, reads the env vars from its output,
  exports them, runs `claude`, and kills the proxy on exit.

## Configuration

Edit the `models` list in `demo.py` to add or change Copilot-backed models.

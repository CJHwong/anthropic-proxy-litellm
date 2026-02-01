# Bedrock Native Example (Anthropic Models)

Run Claude Code with Anthropic models on AWS Bedrock — no proxy needed.

Claude Code supports Bedrock natively via `CLAUDE_CODE_USE_BEDROCK=1`.
This is the simplest path when using Anthropic models (Claude Sonnet,
Haiku, etc.) on Bedrock.

For **non-Anthropic** models on Bedrock (DeepSeek, Qwen, etc.), see
[`examples/bedrock/`](../bedrock/) which uses the LiteLLM proxy.

## Prerequisites

- AWS credentials configured (`aws configure` or environment variables)
- Anthropic models enabled in your AWS account via the Bedrock console

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

## Configuration

Override model selection via environment variables:

```bash
ANTHROPIC_MODEL=anthropic.claude-3-5-haiku-20241022-v1:0 ./demo.sh
```

#!/usr/bin/env bash
# Run Claude Code with Anthropic models on Bedrock — no proxy needed.
#
# Claude Code supports Bedrock natively via CLAUDE_CODE_USE_BEDROCK=1.
# This only works with Anthropic models (Claude Sonnet, Haiku, etc.).
# For non-Anthropic models on Bedrock (DeepSeek, Qwen, etc.), see
# examples/bedrock/ which uses the LiteLLM proxy.
#
# Usage:
#   ./demo.sh                    # interactive session
#   ./demo.sh "Say hello"        # non-interactive prompt
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Load AWS credentials from .env if present.
if [ -f "$SCRIPT_DIR/.env" ]; then
    set -a
    source "$SCRIPT_DIR/.env"
    set +a
fi

export CLAUDE_CODE_USE_BEDROCK=1
export ANTHROPIC_MODEL="${ANTHROPIC_MODEL:-us.anthropic.claude-sonnet-4-20250514-v1:0}"
export ANTHROPIC_SMALL_FAST_MODEL="${ANTHROPIC_SMALL_FAST_MODEL:-anthropic.claude-3-5-haiku-20241022-v1:0}"

if [ $# -gt 0 ]; then
    claude -p "$@"
else
    claude
fi

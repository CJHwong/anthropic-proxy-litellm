#!/usr/bin/env bash
# Run the Bedrock demo using Claude Agent SDK through LiteLLM proxy.
#
# Usage:
#   ./demo.sh                                     # default prompt
#   ./demo.sh "Explain quantum computing briefly"  # custom prompt
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

uv run "$SCRIPT_DIR/demo.py" "$@"

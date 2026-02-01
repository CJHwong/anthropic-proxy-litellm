#!/usr/bin/env bash
# Start LiteLLM proxy with Copilot models, then run Claude Code.
#
# Usage:
#   ./demo.sh                    # interactive session
#   ./demo.sh "Say hello"        # non-interactive prompt
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

PIPE=$(mktemp -u)
mkfifo "$PIPE"

cleanup() {
    [ -n "${PROXY_PID:-}" ] && kill "$PROXY_PID" 2>/dev/null
    rm -f "$PIPE"
}
trap cleanup EXIT

uv run "$SCRIPT_DIR/demo.py" > "$PIPE" &
PROXY_PID=$!

# Read env vars until blank line (proxy ready signal).
GOT_READY=false
while IFS= read -r line; do
    if [ -z "$line" ]; then
        GOT_READY=true
        break
    fi
    export "$line"
done < "$PIPE"

# If demo.py died before signalling ready, bail out.
if ! $GOT_READY || ! kill -0 "$PROXY_PID" 2>/dev/null; then
    echo "error: proxy failed to start" >&2
    exit 1
fi

if [ $# -gt 0 ]; then
    claude -p "$@"
else
    claude
fi

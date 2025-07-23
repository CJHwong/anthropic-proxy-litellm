#!/bin/bash
set -e

# Usage: MODEL="model-name" ./start_servers.sh
# If MODEL is not set, default will be used by litellm and proxy.py

.venv/bin/litellm --config config.yaml &
LITELLM_PID=$!
OPENAI_API_BASE=http://localhost:4000 PORT=3000 MODEL="$MODEL" .venv/bin/python proxy.py &
PROXY_PID=$!

trap 'kill $LITELLM_PID $PROXY_PID; exit' SIGINT
wait $LITELLM_PID $PROXY_PID

#!/bin/bash
set -e

.venv/bin/litellm --config config.yaml &
OPENAI_API_BASE=http://localhost:4000 PORT=3000 .venv/bin/python proxy.py

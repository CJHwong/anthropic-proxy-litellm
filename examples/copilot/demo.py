# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "anthropic-proxy-litellm",
# ]
#
# [tool.uv.sources]
# anthropic-proxy-litellm = { path = "../../" }
# ///
"""Start LiteLLM proxy with GitHub Copilot models and print env vars.

Prints KEY=VALUE lines to stdout (one per env var), followed by a blank
line to signal readiness, then blocks until terminated.

Designed to be backgrounded by demo.sh.

Usage:
    uv run demo.py
"""

from __future__ import annotations

import signal
import threading

from anthropic_proxy_litellm import LiteLLMProxy

COPILOT_HEADERS = {
    'Editor-Version': 'Neovim/0.9.0',
    'Copilot-Integration-Id': 'vscode-chat',
}

models = [
    {
        'model_name': 'gpt-4.1',
        'litellm_params': {
            'model': 'github_copilot/gpt-4.1',
            'extra_headers': COPILOT_HEADERS,
            'drop_params': True,
        },
    },
    {
        'model_name': 'claude-sonnet-4',
        'litellm_params': {
            'model': 'github_copilot/claude-sonnet-4',
            'extra_headers': COPILOT_HEADERS,
        },
    },
]

stop = threading.Event()
signal.signal(signal.SIGTERM, lambda *_: stop.set())

with LiteLLMProxy(model_list=models, model='gpt-4.1') as proxy:
    for key, value in proxy.env.items():
        print(f'{key}={value}', flush=True)
    print('', flush=True)

    try:
        stop.wait()
    except KeyboardInterrupt:
        pass

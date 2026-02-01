# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "anthropic-proxy-litellm",
#     "anyio",
# ]
#
# [tool.uv.sources]
# anthropic-proxy-litellm = { path = "../../" }
# ///
"""Demo: use Claude Agent SDK through LiteLLM proxy with Bedrock models.

Usage:
    uv run demo.py
    uv run demo.py "Explain quantum computing in one sentence"
"""

from __future__ import annotations

import logging
import sys

import anyio
from claude_agent_sdk import AssistantMessage, TextBlock

from anthropic_proxy_litellm import LiteLLMProxy

logging.basicConfig(format='%(message)s', level=logging.INFO)

prompt = sys.argv[1] if len(sys.argv) > 1 else 'Say hello!'

# Option 1: Load models from a YAML config file.
# proxy = LiteLLMProxy(
#     config='config.yaml',
#     model='openai.gpt-oss-120b-1:0',
#     small_fast_model='openai.gpt-oss-20b-1:0',
#     env_file='.env',
# )

# Option 2: Specify models inline.
proxy = LiteLLMProxy(
    model_list=[
        {
            'model_name': 'openai.gpt-oss-120b-1:0',
            'litellm_params': {
                'model': 'bedrock/openai.gpt-oss-120b-1:0',
                'drop_params': True,
            },
        },
        {
            'model_name': 'openai.gpt-oss-20b-1:0',
            'litellm_params': {
                'model': 'bedrock/openai.gpt-oss-20b-1:0',
                'drop_params': True,
            },
        },
    ],
    model='openai.gpt-oss-120b-1:0',
    small_fast_model='openai.gpt-oss-20b-1:0',
    env_file='.env',
)

with proxy:

    async def _main() -> None:
        async for message in proxy.query(prompt):
            if isinstance(message, AssistantMessage):
                for block in message.content:
                    if isinstance(block, TextBlock):
                        print(block.text)

    try:
        anyio.run(_main)
    except KeyboardInterrupt:
        pass

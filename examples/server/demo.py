# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "anthropic-proxy-litellm",
# ]
#
# [tool.uv.sources]
# anthropic-proxy-litellm = { path = "../../" }
# ///
"""Demo: use start()/stop() to embed a LiteLLM proxy in a long-lived server.

Shows the explicit lifecycle API — useful when a context manager doesn't fit
(Django AppConfig.ready(), FastAPI lifespan, etc.).

The example runs a minimal stdlib HTTP server that returns the proxy env as
JSON, proving the proxy is up and the env is available for SDK calls.

Usage:
    uv run demo.py                  # http://localhost:8080/
    curl http://localhost:8080/     # returns proxy env as JSON

Django equivalent::

    # myapp/apps.py
    from django.apps import AppConfig
    from anthropic_proxy_litellm import LiteLLMProxy

    proxy = LiteLLMProxy(config='litellm.yaml', model='my-model')

    class MyAppConfig(AppConfig):
        def ready(self):
            proxy.start()

FastAPI equivalent::

    from contextlib import asynccontextmanager
    from fastapi import FastAPI
    from anthropic_proxy_litellm import LiteLLMProxy

    proxy = LiteLLMProxy(config='litellm.yaml', model='my-model')

    @asynccontextmanager
    async def lifespan(app):
        proxy.start()
        yield
        proxy.stop()

    app = FastAPI(lifespan=lifespan)
"""

from __future__ import annotations

import atexit
import json
import logging
from http.server import BaseHTTPRequestHandler, HTTPServer

from anthropic_proxy_litellm import LiteLLMProxy

logging.basicConfig(format='%(message)s', level=logging.INFO)

# -- Proxy setup (explicit start/stop) ----------------------------------------

proxy = LiteLLMProxy(
    model_list=[
        {
            'model_name': 'openai.gpt-oss-120b-1:0',
            'litellm_params': {
                'model': 'bedrock/openai.gpt-oss-120b-1:0',
                'drop_params': True,
            },
        },
    ],
    model='openai.gpt-oss-120b-1:0',
)

proxy.start()
atexit.register(proxy.stop)

# -- HTTP handler --------------------------------------------------------------


class Handler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        body = json.dumps(proxy.env, indent=2).encode()
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: object) -> None:  # noqa: A002
        logging.info(format, *args)


if __name__ == '__main__':
    server = HTTPServer(('127.0.0.1', 8080), Handler)
    logging.info('Serving on http://127.0.0.1:8080 — Ctrl+C to stop')
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass

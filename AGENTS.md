# AGENTS.md

This file provides guidance to AI coding agents when working with code in this repository.

## What This Is

In-process LiteLLM proxy (`LiteLLMProxy`). Use as a context manager or via explicit `start()`/`stop()` calls. Starts a uvicorn server on a daemon thread, polls `/health` until ready, tears down on exit. No subprocess management. LiteLLM's native `/v1/messages` endpoint handles Anthropic protocol translation — there is no custom message format converter.

## Commands

```bash
uv sync --group dev          # install deps + dev tools
uv run ruff check src/ examples/   # lint
uv run ruff format src/ examples/  # format
uv run ty check src/               # type check (Python 3.10 target)
```

## Architecture

```
User code  ──▶  LiteLLMProxy (context manager)
                    │
                    │ localhost:{port} via daemon thread (uvicorn)
                    ▼
                LiteLLM /v1/messages
                    │
                    ▼
                Backend (Bedrock, Copilot, Ollama, ...)
```

Single public class in `src/anthropic_proxy_litellm/proxy.py`. Config resolution: `model_list` param → temp YAML (cleaned on exit) > `config` param > `LITELLM_CONFIG` env var. Temp YAML workaround exists because LiteLLM's `initialize(config=dict)` is broken upstream.

## Key Design Decisions

- **Temp YAML for `model_list`**: LiteLLM only reliably reads config from `CONFIG_FILE_PATH` pointing at a file. The temp file is created in `_resolve_config()` and deleted in `__exit__`.
- **stdout suppression**: LiteLLM prints startup banners asynchronously on the daemon thread. `contextlib.redirect_stdout` redirects to `/dev/null` during the entire startup+healthcheck window, not just `_start_proxy()`.
- **No default model or config**: The library requires explicit configuration. Defaults belong in examples, not the library.
- **Single instance per process**: A class-level `threading.Lock` enforces that only one `LiteLLMProxy` context may be active at a time. The lock is acquired in `__enter__` (non-blocking) and released in `__exit__`.
- **Request logging**: Optional `_RequestLoggingMiddleware` (ASGI middleware) wraps the LiteLLM app when `log_requests=True` or `LITELLM_LOG_REQUESTS=1`. Logs `METHOD /path - STATUS (0.123s)` at `logger.info` level; only wraps HTTP scope.
- **`start()`/`stop()` API**: For long-lived processes (Django, FastAPI) that cannot use `with`. `__enter__`/`__exit__` delegate to these methods. `start()` guards against double-start; `stop()` is a safe no-op when not running.

## Conventions

- Python 3.10+ (`from __future__ import annotations` everywhere, `typing_extensions.Self` under `TYPE_CHECKING`)
- Single quotes, ruff-formatted
- Ruff rules: E, F, I, UP, B, SIM, RUF

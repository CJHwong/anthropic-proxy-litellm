"""LiteLLMProxy — in-process LiteLLM proxy with managed lifecycle.

Works with any LiteLLM-supported backend (AWS Bedrock, GitHub Copilot,
Ollama, etc.).

Usage::

    from anthropic_proxy_litellm import LiteLLMProxy

    # Context manager (scripts, short-lived)
    with LiteLLMProxy(config='config.yaml', model='my-model') as proxy:
        async for msg in proxy.query('Say hello'):
            print(msg)

    # Explicit start/stop (Django, FastAPI, long-lived servers)
    proxy = LiteLLMProxy(config='config.yaml', model='my-model')
    proxy.start()
    # ... use proxy.env, proxy.query(), etc. ...
    proxy.stop()
"""

from __future__ import annotations

import contextlib
import logging
import os
import tempfile
import threading
import time
import types
import urllib.request
from collections.abc import AsyncIterator
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from claude_agent_sdk import ClaudeAgentOptions
    from typing_extensions import Self

import yaml

__all__ = ['LiteLLMProxy']

logger = logging.getLogger(__name__)


class _RequestLoggingMiddleware:
    """Lightweight ASGI middleware that logs HTTP requests.

    Logs ``METHOD /path - STATUS (0.123s)`` at ``logger.info`` level.
    Only wraps HTTP scope; passes through lifespan/websocket unchanged.
    """

    def __init__(self, app: Any) -> None:
        self._app = app

    async def __call__(self, scope: dict, receive: Any, send: Any) -> None:
        if scope['type'] != 'http':
            await self._app(scope, receive, send)
            return

        method = scope.get('method', '?')
        path = scope.get('path', '?')
        start = time.monotonic()
        status_code: int | None = None

        async def send_wrapper(message: dict) -> None:
            nonlocal status_code
            if message['type'] == 'http.response.start':
                status_code = message.get('status', 0)
            await send(message)

        await self._app(scope, receive, send_wrapper)
        elapsed = time.monotonic() - start
        logger.info('%s %s - %s (%.3fs)', method, path, status_code, elapsed)


class LiteLLMProxy:
    """In-process LiteLLM proxy with managed lifecycle.

    Use as a context manager (``with LiteLLMProxy(...) as proxy:``) or call
    :meth:`start` / :meth:`stop` explicitly for long-lived processes like
    Django or FastAPI servers.

    Provide model configuration via *one* of:

    - ``config``: path to a LiteLLM YAML config file.
    - ``model_list``: a list of model dicts (written to a temp YAML at startup).

    Falls back to the ``LITELLM_CONFIG`` env var if neither is given.

    Only one ``LiteLLMProxy`` may be active per process at a time.  Attempting
    to start a second instance while one is already active raises ``RuntimeError``.

    Parameters fall back to environment variables:

    ==================  ============================  ===========
    Constructor         Env var                       Default
    ==================  ============================  ===========
    config              LITELLM_CONFIG                —
    model_list          —                             —
    port                LITELLM_PORT                  4000
    model               ANTHROPIC_MODEL               *(required)*
    small_fast_model    ANTHROPIC_SMALL_FAST_MODEL    model
    verbose             LITELLM_VERBOSE               false
    log_requests        LITELLM_LOG_REQUESTS          false
    env_file            —                             *(none)*
    ==================  ============================  ===========
    """

    _DEFAULT_PORT = 4000
    _PROBE_TIMEOUT = 60.0
    _PROBE_INITIAL_INTERVAL = 1.0
    _PROBE_MAX_INTERVAL = 5.0
    _instance_lock = threading.Lock()

    def __init__(
        self,
        *,
        config: str | Path | None = None,
        model_list: list[dict] | None = None,
        port: int | None = None,
        model: str | None = None,
        small_fast_model: str | None = None,
        verbose: bool | None = None,
        log_requests: bool | None = None,
        env_file: str | Path | None = None,
    ) -> None:
        if config is not None and model_list is not None:
            raise ValueError('Provide config or model_list, not both.')

        self._config_path = Path(config) if config is not None else None
        self._model_list = model_list
        self._port = port if port is not None else int(os.environ.get('LITELLM_PORT', str(self._DEFAULT_PORT)))
        # Store raw args; env var resolution deferred to start() (after load_dotenv).
        self._model_raw = model
        self._small_fast_model_raw = small_fast_model
        self._model: str | None = None
        self._small_fast_model: str | None = None

        if verbose is not None:
            self._verbose = verbose
        else:
            self._verbose = os.environ.get('LITELLM_VERBOSE', '').lower() in (
                '1',
                'true',
                'yes',
            )

        if log_requests is not None:
            self._log_requests = log_requests
        else:
            self._log_requests = os.environ.get('LITELLM_LOG_REQUESTS', '').lower() in (
                '1',
                'true',
                'yes',
            )

        self._env_file = Path(env_file) if env_file else None
        self._server: Any = None  # uvicorn.Server, typed as Any to keep import lazy
        self._tmp_config: Path | None = None  # temp YAML created from model_list
        self._saved_config_file_path: str | None = None
        self._started: bool = False

    # -- Context manager -------------------------------------------------------

    def __enter__(self) -> Self:
        return self.start()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: types.TracebackType | None,
    ) -> None:
        self.stop()

    # -- Public API: lifecycle -------------------------------------------------

    def start(self) -> Self:
        """Start the proxy.  Returns ``self`` for chaining.

        If a healthy proxy is already listening on the configured port, this
        instance becomes *client-only* — it exposes the same :attr:`env` and
        :meth:`options` interface but does **not** start a uvicorn server.
        :meth:`stop` on a client-only instance simply releases the lock.

        Raises ``RuntimeError`` if this instance is already started or another
        ``LiteLLMProxy`` is active in the same process.
        """
        if self._started:
            raise RuntimeError('This LiteLLMProxy is already started.')

        if not self._instance_lock.acquire(blocking=False):
            raise RuntimeError(
                'Another LiteLLMProxy is already active in this process. Only one instance may be active at a time.'
            )

        try:
            if self._env_file is not None:
                from dotenv import load_dotenv

                load_dotenv(self._env_file)

            # Resolve model after load_dotenv so .env values are available.
            self._model = self._model_raw or os.environ.get('ANTHROPIC_MODEL') or None
            if self._model is None:
                raise ValueError('No model specified. Pass model= or set ANTHROPIC_MODEL.')

            self._small_fast_model = (
                self._small_fast_model_raw or os.environ.get('ANTHROPIC_SMALL_FAST_MODEL') or self._model
            )

            # If a proxy is already healthy on our port, become client-only.
            if self._probe_health():
                logger.info(
                    'Proxy already healthy on port %d — reusing existing server (client-only).',
                    self._port,
                )
                self._started = True
                return self

            self._saved_config_file_path = os.environ.get('CONFIG_FILE_PATH')
            os.environ['LITELLM_DONT_SHOW_FEEDBACK_BOX'] = 'true'

            resolved_config = self._resolve_config()
            if resolved_config is None:
                raise ValueError('No config provided. Pass config=, model_list=, or set LITELLM_CONFIG.')

            logger.info('Starting LiteLLM proxy on port %d...', self._port)

            if not self._verbose:
                with open(os.devnull, 'w') as devnull, contextlib.redirect_stdout(devnull):
                    self._server = self._start_proxy(resolved_config)
                    self._wait_for_proxy()
            else:
                self._server = self._start_proxy(resolved_config)
                self._wait_for_proxy()

            logger.info('Proxy ready.')
            self._started = True
        except BaseException:
            # Clean up resources created during the failed start attempt.
            if self._tmp_config is not None:
                with contextlib.suppress(OSError):
                    self._tmp_config.unlink(missing_ok=True)
                self._tmp_config = None

            if self._saved_config_file_path is not None:
                os.environ['CONFIG_FILE_PATH'] = self._saved_config_file_path
            else:
                os.environ.pop('CONFIG_FILE_PATH', None)

            self._instance_lock.release()
            raise

        return self

    def stop(self) -> None:
        """Stop the proxy and clean up resources.

        Safe to call when the proxy is not running (no-op).  Client-only
        instances (where :meth:`start` detected an existing healthy proxy)
        skip server shutdown and only release the instance lock.
        """
        if not self._started:
            return

        try:
            if self._server is not None:
                self._shutdown_proxy()
                self._server = None

                if self._tmp_config is not None:
                    try:
                        self._tmp_config.unlink(missing_ok=True)
                    except OSError:
                        pass  # Best-effort cleanup; temp file may already be gone.
                    self._tmp_config = None

                # Restore CONFIG_FILE_PATH to its previous state.
                if self._saved_config_file_path is not None:
                    os.environ['CONFIG_FILE_PATH'] = self._saved_config_file_path
                else:
                    os.environ.pop('CONFIG_FILE_PATH', None)

            self._started = False
        finally:
            self._instance_lock.release()

    # -- Public API: SDK helpers -----------------------------------------------

    def options(self, **kwargs: Any) -> ClaudeAgentOptions:
        """Return a ``ClaudeAgentOptions`` pre-configured for this proxy.

        The model, ``ANTHROPIC_BASE_URL``, and related env vars are set
        automatically.  Pass any ``ClaudeAgentOptions`` field as a kwarg to
        override (e.g. ``max_turns=5``, ``system_prompt='...'``).
        """
        from claude_agent_sdk import ClaudeAgentOptions

        defaults: dict[str, Any] = {
            'model': self._model,
            'env': self.env,
        }
        defaults.update(kwargs)
        return ClaudeAgentOptions(**defaults)

    async def query(self, prompt: str, **kwargs: Any) -> AsyncIterator[Any]:
        """Convenience wrapper: ``options()`` + ``claude_agent_sdk.query()``.

        *prompt* is sent to the agent.  Any extra *kwargs* are forwarded to
        :meth:`options`, so you can pass ``max_turns``, ``system_prompt``,
        ``allowed_tools``, etc.

        Yields raw SDK messages (``AssistantMessage``, ``ToolUseMessage``, ...)
        so callers can inspect or filter as needed.
        """
        from claude_agent_sdk import query as sdk_query

        opts = self.options(**kwargs)
        async for message in sdk_query(prompt=prompt, options=opts):
            yield message

    @property
    def env(self) -> dict[str, str]:
        """Raw env dict for the Claude Agent SDK.

        Useful for passing to subprocesses or custom ``ClaudeAgentOptions``.
        """
        return {
            'ANTHROPIC_BASE_URL': f'http://localhost:{self._port}',
            # Disables experimental Claude Code features that may use APIs
            # unsupported by LiteLLM or the target backend model.
            'CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS': '1',
            'ANTHROPIC_MODEL': self._model or '',
            'ANTHROPIC_SMALL_FAST_MODEL': self._small_fast_model or self._model or '',
        }

    # -- Private: config resolution --------------------------------------------

    def _resolve_config(self) -> Path | None:
        """Return the config file path, creating a temp YAML from model_list if needed."""
        if self._model_list is not None:
            data = {'model_list': self._model_list}
            fd, path = tempfile.mkstemp(suffix='.yaml', prefix='litellm-')
            with os.fdopen(fd, 'w') as f:
                yaml.safe_dump(data, f)
            self._tmp_config = Path(path)
            return self._tmp_config

        if self._config_path is not None:
            return self._config_path

        env_config = os.environ.get('LITELLM_CONFIG')
        if env_config:
            return Path(env_config)

        return None

    def _probe_health(self) -> bool:
        """Check whether a proxy is already listening on our port.

        Two-phase probe:

        1. **Quick check** — a single request.  ``ConnectionRefusedError``
           (nothing bound) returns ``False`` immediately so cold starts pay
           no penalty.
        2. **Retry loop** — entered only when the first attempt gets a
           *non-refused* error (timeout, HTTP 503, etc.), meaning something
           *is* listening but not yet healthy.  Retries with exponential
           back-off (capped at ``_PROBE_MAX_INTERVAL``) until
           ``_PROBE_TIMEOUT`` seconds have elapsed.
        """
        url = f'http://localhost:{self._port}/health'

        # Phase 1: quick single check.
        try:
            with urllib.request.urlopen(url, timeout=2) as resp:  # nosec B310
                if resp.status == 200:
                    return True
        except OSError as e:
            # urllib wraps socket errors: URLError(reason=ConnectionRefusedError).
            cause = getattr(e, 'reason', e)
            if isinstance(cause, ConnectionRefusedError):
                return False
            # Timeout, HTTP error, etc. — enter phase 2.

        # Phase 2: something is listening; retry until healthy or deadline.
        deadline = time.monotonic() + self._PROBE_TIMEOUT
        interval = self._PROBE_INITIAL_INTERVAL
        attempt = 1

        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break

            sleep_time = min(interval, remaining)
            logger.debug(
                'Health probe attempt %d failed, retrying in %.1fs (%.0fs remaining)...',
                attempt,
                sleep_time,
                remaining,
            )
            time.sleep(sleep_time)
            interval = min(interval * 2, self._PROBE_MAX_INTERVAL)

            attempt += 1
            try:
                with urllib.request.urlopen(url, timeout=2) as resp:  # nosec B310
                    if resp.status == 200:
                        return True
            except OSError:
                pass

        return False

    # -- Private: proxy lifecycle ----------------------------------------------

    def _start_proxy(self, config_path: Path) -> Any:
        """Start the LiteLLM proxy in-process on a daemon thread."""
        import uvicorn

        os.environ['CONFIG_FILE_PATH'] = str(config_path)

        if not self._verbose:
            for name in (
                'LiteLLM',
                'LiteLLM Proxy',
                'LiteLLM Router',
                'litellm',
                'litellm.proxy',
                'litellm.proxy.health_check',
            ):
                logging.getLogger(name).setLevel(logging.CRITICAL)

        from litellm.proxy.proxy_server import app

        target_app: Any = app
        if self._log_requests:
            target_app = _RequestLoggingMiddleware(app)

        log_level = 'info' if self._verbose else 'warning'
        uvi_config = uvicorn.Config(
            target_app,
            host='127.0.0.1',
            port=self._port,
            log_level=log_level,
        )
        server = uvicorn.Server(uvi_config)
        thread = threading.Thread(target=server.run, daemon=True)
        thread.start()
        return server

    def _wait_for_proxy(self, timeout: int = 60) -> None:
        """Poll the proxy health endpoint until it responds or timeout expires.

        The LiteLLM ``/health`` endpoint probes configured backends on the
        first call, which can easily take 10-20 seconds.  A short per-request
        timeout (e.g. 2 s) causes every attempt to time out before the
        endpoint finishes, so we cap each attempt at the lesser of 15 s or
        the remaining budget.
        """
        url = f'http://localhost:{self._port}/health'
        deadline = time.monotonic() + timeout

        while time.monotonic() < deadline:
            per_req = min(deadline - time.monotonic(), 15)
            if per_req <= 0:
                break
            try:
                with urllib.request.urlopen(url, timeout=per_req) as resp:  # nosec B310
                    if resp.status == 200:
                        return
            except OSError as exc:
                logger.debug('Health check failed: %s', exc)
            time.sleep(0.5)

        raise RuntimeError(f'LiteLLM proxy did not become healthy within {timeout}s on port {self._port}')

    def _shutdown_proxy(self) -> None:
        """Signal uvicorn to exit and wait for the socket to close."""
        self._server.should_exit = True
        deadline = time.monotonic() + 5
        while self._server.started and time.monotonic() < deadline:
            time.sleep(0.1)
        if self._server.started:
            logger.warning(
                'LiteLLM proxy server did not stop within 5s on port %d',
                self._port,
            )

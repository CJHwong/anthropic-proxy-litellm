"""BifrostProxy — managed Bifrost gateway with subprocess lifecycle.

**Experimental.**  This module uses ``npx @maximhq/bifrost`` to download and
run the Bifrost binary, which requires Node.js 18+ on the host.  This is not
a preferred distribution method for production use — it exists solely for
benchmarking and evaluation against :class:`LiteLLMProxy`.

Works with any Bifrost-supported backend (AWS Bedrock, Anthropic, OpenAI,
Ollama, etc.).  See https://github.com/maximhq/bifrost for the upstream
project.

Usage::

    from anthropic_proxy_litellm.bifrost import BifrostProxy

    # Context manager (scripts, short-lived)
    with BifrostProxy(
        providers={'bedrock': {'keys': [{'name': 'default', 'weight': 1,
            'bedrock_key_config': {'region': 'us-west-2'}}]}},
        model='us.anthropic.claude-sonnet-4-20250514-v1:0',
    ) as proxy:
        async for msg in proxy.query('Say hello'):
            print(msg)

    # Explicit start/stop (Django, FastAPI, long-lived servers)
    proxy = BifrostProxy(config='bifrost.json', model='my-model')
    proxy.start()
    # ... use proxy.env, proxy.query(), etc. ...
    proxy.stop()

Benchmark comparison (vs LiteLLMProxy)
--------------------------------------

Methodology: both proxies were started on the same machine against the same
Bedrock backend (``openai.gpt-oss-20b-1:0``, ``us-west-2``).  Two benchmark
suites were run:

**Proxy overhead** — measured by sending 200 sequential HTTP requests to a
lightweight liveness endpoint (``/health/liveliness`` for LiteLLM,
``/health`` for Bifrost) after a 30-request warmup.  This isolates the HTTP
stack overhead of each proxy without involving any LLM backend.  Startup time
is wall-clock from ``start()`` to the first successful health probe.  Shutdown
time is wall-clock from ``stop()`` to full resource cleanup.  Memory is
process RSS (``ps -o rss=``) — RSS delta for LiteLLM (in-process thread),
absolute subprocess RSS for Bifrost.  Bifrost binary was pre-cached by npx
to avoid penalising startup with a download.

**Real invocation** — 3 sequential ``/v1/messages`` requests with a short
prompt (``"Reply with exactly one word: hello"``, ``max_tokens=64``) through
each proxy to the same Bedrock model, measuring wall-clock round-trip time.

Results (macOS, Apple Silicon, 2026-03-18)::

    Proxy overhead
    ──────────────────────────────────────────────────────
    Metric           LiteLLM            Bifrost
    ──────────────────────────────────────────────────────
    Startup          5.08 s             5.05 s       Tie
    Shutdown         5.08 s             0.004 s      Bifrost 1270x
    Liveness p50     1.06 ms            475 µs       Bifrost 2.2x
    Liveness p99     2.65 ms            2.13 ms      Bifrost 1.2x
    Memory           +291 MB            103 MB       Bifrost 2.8x

    Real invocation (Bedrock openai.gpt-oss-20b-1:0)
    ──────────────────────────────────────────────────────
    Metric           LiteLLM            Bifrost
    ──────────────────────────────────────────────────────
    Invoke avg       843 ms             943 ms       Same (backend-dominated)
    Invoke min       492 ms             787 ms
    Invoke max       1.11 s             1.06 s

Takeaway: proxy overhead is negligible relative to backend latency.  Bifrost
uses less memory and shuts down instantly; LiteLLM is Python-native with no
Node.js dependency and simpler model routing (no ``provider/model`` prefix).
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import signal
import subprocess
import tempfile
import threading
import time
import types
from collections.abc import AsyncIterator
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.error import URLError
from urllib.request import urlopen

if TYPE_CHECKING:
    from claude_agent_sdk import ClaudeAgentOptions
    from typing_extensions import Self

__all__ = ['BifrostProxy']

logger = logging.getLogger(__name__)


class BifrostProxy:
    """Managed Bifrost AI gateway with subprocess lifecycle.

    Use as a context manager (``with BifrostProxy(...) as proxy:``) or call
    :meth:`start` / :meth:`stop` explicitly for long-lived processes.

    Provide backend configuration via *one* of:

    - ``config``: path to a Bifrost JSON config file.
    - ``providers``: a dict of provider configs (written to a temp JSON at startup).

    Only one ``BifrostProxy`` may be active per process at a time.

    Parameters fall back to environment variables:

    ==================  ============================  ===========
    Constructor         Env var                       Default
    ==================  ============================  ===========
    config              BIFROST_CONFIG                —
    providers           —                             —
    port                BIFROST_PORT                  8080
    model               ANTHROPIC_MODEL               *(required)*
    small_fast_model    ANTHROPIC_SMALL_FAST_MODEL    model
    verbose             BIFROST_VERBOSE               false
    env_file            —                             *(none)*
    ==================  ============================  ===========
    """

    _DEFAULT_PORT = 8080
    _PROBE_TIMEOUT = 30.0
    _PROBE_INITIAL_INTERVAL = 0.2
    _PROBE_MAX_INTERVAL = 2.0
    _instance_lock = threading.Lock()

    def __init__(
        self,
        *,
        config: str | Path | None = None,
        providers: dict[str, Any] | None = None,
        port: int | None = None,
        model: str | None = None,
        small_fast_model: str | None = None,
        verbose: bool | None = None,
        env_file: str | Path | None = None,
    ) -> None:
        if config is not None and providers is not None:
            raise ValueError('Provide config or providers, not both.')

        self._config_path = Path(config) if config is not None else None
        self._providers = providers
        self._port = port if port is not None else int(os.environ.get('BIFROST_PORT', str(self._DEFAULT_PORT)))
        self._model_raw = model
        self._small_fast_model_raw = small_fast_model
        self._model: str | None = None
        self._small_fast_model: str | None = None

        if verbose is not None:
            self._verbose = verbose
        else:
            self._verbose = os.environ.get('BIFROST_VERBOSE', '').lower() in ('1', 'true', 'yes')

        self._env_file = Path(env_file) if env_file else None
        self._process: subprocess.Popen[bytes] | None = None
        self._app_dir: Path | None = None  # temp dir for config + data
        self._tmp_config: Path | None = None
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
        """Start the Bifrost gateway.  Returns ``self`` for chaining.

        If a healthy gateway is already listening on the configured port, this
        instance becomes *client-only* — it does **not** start a subprocess.

        Raises ``RuntimeError`` if this instance is already started, another
        ``BifrostProxy`` is active, or ``npx`` is not found.
        """
        if self._started:
            raise RuntimeError('This BifrostProxy is already started.')

        if not self._instance_lock.acquire(blocking=False):
            raise RuntimeError(
                'Another BifrostProxy is already active in this process. Only one instance may be active at a time.'
            )

        try:
            if self._env_file is not None:
                from dotenv import load_dotenv

                load_dotenv(self._env_file)

            self._model = self._model_raw or os.environ.get('ANTHROPIC_MODEL') or None
            if self._model is None:
                raise ValueError('No model specified. Pass model= or set ANTHROPIC_MODEL.')

            self._small_fast_model = (
                self._small_fast_model_raw or os.environ.get('ANTHROPIC_SMALL_FAST_MODEL') or self._model
            )

            # If a gateway is already healthy on our port, become client-only.
            if self._probe_health():
                logger.info(
                    'Bifrost already healthy on port %d — reusing existing gateway (client-only).',
                    self._port,
                )
                self._started = True
                return self

            self._ensure_npx()

            config_path = self._resolve_config()
            if config_path is None:
                raise ValueError('No config provided. Pass config=, providers=, or set BIFROST_CONFIG.')

            logger.info('Starting Bifrost gateway on port %d...', self._port)
            self._start_gateway(config_path)
            self._wait_for_gateway()
            logger.info('Bifrost ready.')
            self._started = True
        except BaseException:
            self._cleanup_on_failure()
            self._instance_lock.release()
            raise

        return self

    def stop(self) -> None:
        """Stop the gateway and clean up resources.

        Safe to call when the gateway is not running (no-op).
        """
        if not self._started:
            return

        try:
            if self._process is not None:
                self._shutdown_gateway()
                self._process = None

            # Clean up temp app dir (contains config + any bifrost data).
            if self._app_dir is not None:
                try:
                    import shutil as _shutil

                    _shutil.rmtree(self._app_dir, ignore_errors=True)
                except OSError:
                    pass
                self._app_dir = None
                self._tmp_config = None

            self._started = False
        finally:
            self._instance_lock.release()

    # -- Public API: SDK helpers -----------------------------------------------

    def options(self, **kwargs: Any) -> ClaudeAgentOptions:
        """Return a ``ClaudeAgentOptions`` pre-configured for this proxy."""
        from claude_agent_sdk import ClaudeAgentOptions

        defaults: dict[str, Any] = {
            'model': self._model,
            'env': self.env,
        }
        defaults.update(kwargs)
        return ClaudeAgentOptions(**defaults)

    async def query(self, prompt: str, **kwargs: Any) -> AsyncIterator[Any]:
        """Convenience wrapper: ``options()`` + ``claude_agent_sdk.query()``."""
        from claude_agent_sdk import query as sdk_query

        opts = self.options(**kwargs)
        async for message in sdk_query(prompt=prompt, options=opts):
            yield message

    @property
    def env(self) -> dict[str, str]:
        """Raw env dict for the Claude Agent SDK."""
        return {
            # Bifrost exposes Anthropic-compatible endpoints under /anthropic/
            'ANTHROPIC_BASE_URL': f'http://localhost:{self._port}/anthropic',
            'CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS': '1',
            'ANTHROPIC_MODEL': self._model or '',
            'ANTHROPIC_SMALL_FAST_MODEL': self._small_fast_model or self._model or '',
        }

    # -- Private: binary discovery ---------------------------------------------

    @staticmethod
    def _ensure_npx() -> None:
        """Verify that npx is available on PATH."""
        if shutil.which('npx') is None:
            raise RuntimeError(
                'npx not found on PATH. Bifrost requires npx (Node.js 18+) to run.\n'
                'Install Node.js: https://nodejs.org/ or use a version manager like nvm.'
            )

    # -- Private: config resolution --------------------------------------------

    def _resolve_config(self) -> Path | None:
        """Return the config file path, creating a temp JSON from providers if needed."""
        if self._providers is not None:
            self._app_dir = Path(tempfile.mkdtemp(prefix='bifrost-'))
            config_data: dict[str, Any] = {
                '$schema': 'https://www.getbifrost.ai/schema',
                'config_store': {'enabled': False},
                'logs_store': {'enabled': False},
                'providers': self._providers,
            }
            config_file = self._app_dir / 'config.json'
            config_file.write_text(json.dumps(config_data, indent=2))
            self._tmp_config = config_file
            return self._app_dir

        if self._config_path is not None:
            # app-dir is the directory containing the config file.
            return self._config_path.parent

        env_config = os.environ.get('BIFROST_CONFIG')
        if env_config:
            return Path(env_config).parent

        return None

    def _probe_health(self) -> bool:
        """Check whether a gateway is already listening on our port."""
        url = f'http://localhost:{self._port}/health'
        try:
            with urlopen(url, timeout=2) as resp:  # nosec B310
                if resp.status == 200:
                    return True
        except OSError as exc:
            cause = getattr(exc, 'reason', exc)
            if isinstance(cause, ConnectionRefusedError):
                return False
        return False

    # -- Private: gateway lifecycle --------------------------------------------

    def _start_gateway(self, app_dir: Path) -> None:
        """Start Bifrost as a subprocess via npx."""
        cmd = [
            'npx',
            '@maximhq/bifrost',
            '-port',
            str(self._port),
            '-host',
            '127.0.0.1',
            '-app-dir',
            str(app_dir),
        ]

        stdout_target = None if self._verbose else subprocess.DEVNULL
        stderr_target = None if self._verbose else subprocess.DEVNULL

        self._process = subprocess.Popen(
            cmd,
            stdout=stdout_target,
            stderr=stderr_target,
            # Start in a new process group so we can kill cleanly.
            preexec_fn=os.setsid if os.name != 'nt' else None,
        )

    def _wait_for_gateway(self) -> None:
        """Poll the health endpoint until Bifrost responds or timeout expires."""
        url = f'http://localhost:{self._port}/health'
        deadline = time.monotonic() + self._PROBE_TIMEOUT
        interval = self._PROBE_INITIAL_INTERVAL

        while time.monotonic() < deadline:
            # Check if the process died.
            if self._process is not None and self._process.poll() is not None:
                raise RuntimeError(f'Bifrost process exited with code {self._process.returncode} during startup.')

            try:
                with urlopen(url, timeout=5) as resp:  # nosec B310
                    if resp.status == 200:
                        return
            except (OSError, URLError):
                pass

            time.sleep(interval)
            interval = min(interval * 2, self._PROBE_MAX_INTERVAL)

        raise RuntimeError(f'Bifrost did not become healthy within {self._PROBE_TIMEOUT}s on port {self._port}')

    def _shutdown_gateway(self) -> None:
        """Terminate the Bifrost subprocess."""
        if self._process is None:
            return

        # Send SIGTERM to the process group for clean shutdown.
        try:
            if os.name != 'nt':
                os.killpg(os.getpgid(self._process.pid), signal.SIGTERM)
            else:
                self._process.terminate()
        except (OSError, ProcessLookupError):
            pass

        try:
            self._process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            logger.warning('Bifrost did not stop within 5s, sending SIGKILL...')
            try:
                if os.name != 'nt':
                    os.killpg(os.getpgid(self._process.pid), signal.SIGKILL)
                else:
                    self._process.kill()
            except (OSError, ProcessLookupError):
                pass
            self._process.wait(timeout=3)

    def _cleanup_on_failure(self) -> None:
        """Clean up resources after a failed start attempt."""
        if self._process is not None:
            self._shutdown_gateway()
            self._process = None

        if self._app_dir is not None:
            try:
                shutil.rmtree(self._app_dir, ignore_errors=True)
            except OSError:
                pass
            self._app_dir = None
            self._tmp_config = None

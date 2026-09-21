"""Background `vllm-sr serve` orchestration shared by CLI integration tests.

A serve session is a one-shot command that returns once the runtime is up, so
every test that needs a live stack drives the same three steps: start it in the
background, require that it completed startup, and stop it again on the way
out. Keeping them here lets more than one integration module share the
orchestration instead of restating it.

Signed-off-by: vLLM-SR Team
"""

import os
import subprocess
from contextlib import contextmanager
from pathlib import Path

from cli.commands.runtime_support import sensitive_env_names

STARTUP_DIAGNOSTIC_TAIL_CHARS = 4000


def startup_diagnostics(stdout, stderr, returncode, secret_values=()):
    """Keep the failure at the end of both streams without exposing credentials."""
    sections = [f"Serve exit code: {returncode}"]
    for name, output in (("stdout", stdout), ("stderr", stderr)):
        text = output or ""
        for value in sorted(set(secret_values), key=len, reverse=True):
            if value:
                text = text.replace(value, "[redacted]")
        if len(text) > STARTUP_DIAGNOSTIC_TAIL_CHARS:
            text = "[earlier output omitted]\n" + text[-STARTUP_DIAGNOSTIC_TAIL_CHARS:]
        sections.append(f"{name}:\n{text or '[empty]'}")
    return "\n".join(sections)


class ServeSessionMixin:
    """Start, await, and stop a background `vllm-sr serve` for one test."""

    def _start_serve_background(
        self,
        env: dict[str, str] | None = None,
        arguments: tuple[str, ...] = (),
    ) -> subprocess.Popen:
        """Start vllm-sr serve in background (non-blocking)."""
        cmd = [
            "vllm-sr",
            "serve",
            *arguments,
            "--image-pull-policy",
            "ifnotpresent",
        ]
        print(f"\nStarting in background: {' '.join(cmd)}")

        environment = os.environ if env is None else env
        names = sensitive_env_names(Path(self.test_dir) / "config.yaml")
        self._serve_secret_values = tuple(
            environment[name] for name in names if environment.get(name)
        )

        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=self.test_dir,
            env=env,
        )
        return process

    def _stop_serve_process(
        self, serve_process: subprocess.Popen | None
    ) -> tuple[str, str]:
        """Terminate a background serve process and drain its output pipes."""
        if serve_process is None:
            return "", ""
        if serve_process.poll() is None:
            serve_process.terminate()
        try:
            stdout, stderr = serve_process.communicate(timeout=10)
        except subprocess.TimeoutExpired:
            serve_process.kill()
            stdout, stderr = serve_process.communicate(timeout=10)
        return stdout or "", stderr or ""

    def _wait_for_serve_success(self, serve_process: subprocess.Popen) -> None:
        """Drain the one-shot serve command and require successful startup."""
        try:
            stdout, stderr = serve_process.communicate(
                timeout=self.HEALTH_CHECK_TIMEOUT
            )
        except subprocess.TimeoutExpired:
            serve_process.terminate()
            try:
                stdout, stderr = serve_process.communicate(timeout=10)
            except subprocess.TimeoutExpired:
                serve_process.kill()
                stdout, stderr = serve_process.communicate(timeout=10)
            self.fail(
                "Serve did not complete startup before the timeout:\n"
                + startup_diagnostics(
                    stdout,
                    stderr,
                    serve_process.returncode,
                    getattr(self, "_serve_secret_values", ()),
                )
            )
        if serve_process.returncode != 0:
            self.fail(
                "Serve failed before completing runtime startup:\n"
                + startup_diagnostics(
                    stdout,
                    stderr,
                    serve_process.returncode,
                    getattr(self, "_serve_secret_values", ()),
                )
            )
        print("  ✓ Serve command completed runtime startup")

    @contextmanager
    def _running_serve(
        self,
        *,
        env: dict[str, str] | None = None,
        endpoint: str = "host.docker.internal:8000",
        base_url: str | None = None,
        provider: str | None = None,
        api_key_env: str | None = None,
        api_only: bool = False,
        managed_storage: bool = False,
        ensure_models_dir: bool = False,
        skip_processing: bool = False,
        looper: bool = False,
    ):
        """Start one background serve session and clean it up automatically."""
        self.write_minimal_canonical_config(
            endpoint=endpoint,
            base_url=base_url,
            provider=provider,
            api_key_env=api_key_env,
            api_only=api_only,
            managed_storage=managed_storage,
            skip_processing=skip_processing,
            looper=looper,
        )
        if ensure_models_dir:
            os.makedirs(os.path.join(self.test_dir, "models"), exist_ok=True)

        full_env = os.environ.copy()
        if env:
            full_env.update(env)

        serve_process = self._start_serve_background(env=full_env)
        try:
            self._wait_for_serve_success(serve_process)
            yield serve_process
        finally:
            self._stop_serve_process(serve_process)

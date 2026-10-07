"""A model runtime started by a test, and the JSON calls the tests send it.

`ServeProcess` runs a serving command (`vllm-sr serve MODEL ...` in engine
mode, or `vllm-srun serve ...`) on a free local port and stops it with
SIGINT, as a reader would with Ctrl-C. `page_requests` reads the requests a
docs page tells readers to send, so the tests send exactly those.
`write_fixture` writes a tiny random-weight package with the runtime of the
router image, so the host needs no runtime of its own.
"""

import json
import os
import re
import signal
import socket
import subprocess
import time
from pathlib import Path
from urllib import error as urllib_error
from urllib import request as urllib_request

READY_TIMEOUT_SECONDS = 180
STOP_TIMEOUT_SECONDS = 30
HTTP_TIMEOUT_SECONDS = 60
FIXTURE_TIMEOUT_SECONDS = 600
HTTP_OK = 200
DEFAULT_IMAGE = "ghcr.io/vllm-project/semantic-router/vllm-sr:latest"
# A runtime request on a page: the path a curl command calls and its JSON body.
CURL_REQUEST = re.compile(
    r"curl[^\n]*?(/v1/(?:decisions|classify|embeddings|rerank|bundle))"
    r"(?:(?!\ncurl).)*?-d '(\{.*?\})'",
    re.S,
)


def page_requests(page: Path) -> dict[str, dict]:
    """The runtime requests a docs page tells readers to send, by path."""
    text = page.read_text(encoding="utf-8")
    return {path: json.loads(body) for path, body in CURL_REQUEST.findall(text)}


def container_runtime() -> str:
    return os.environ.get("CONTAINER_RUNTIME", "").strip() or "docker"


def router_image() -> str:
    return os.environ.get("VLLM_SR_IMAGE", "").strip() or DEFAULT_IMAGE


def write_fixture(output: Path, family: str, variant: str, seed: int = 0) -> Path:
    """`vllm-srun fixture` in the router image, writing *output* as this user."""
    output.parent.mkdir(parents=True, exist_ok=True)
    runtime = container_runtime()
    identity = (
        ["--userns=keep-id"]
        if runtime == "podman" and os.geteuid() != 0
        else ["--user", f"{os.getuid()}:{os.getgid()}"]
    )
    result = subprocess.run(
        [
            runtime,
            "run",
            "--rm",
            *identity,
            "-v",
            f"{output.parent.resolve()}:/out:z",
            "--entrypoint",
            "vllm-srun",
            router_image(),
            "fixture",
            f"/out/{output.name}",
            "--family",
            family,
            "--variant",
            variant,
            "--seed",
            str(seed),
        ],
        capture_output=True,
        text=True,
        timeout=FIXTURE_TIMEOUT_SECONDS,
        check=False,
    )
    if result.returncode != 0:
        raise AssertionError(f"vllm-srun fixture failed: {result.stderr}")
    return output


def free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def call(base: str, path: str, body: dict | None = None) -> tuple[int, object]:
    """GET (no body) or POST a JSON body; the status and the decoded answer."""
    data = None if body is None else json.dumps(body).encode()
    request = urllib_request.Request(
        base + path,
        data=data,
        method="GET" if body is None else "POST",
        headers={"Content-Type": "application/json"} if body is not None else {},
    )
    try:
        with urllib_request.urlopen(request, timeout=HTTP_TIMEOUT_SECONDS) as response:
            payload = response.read()
            status = response.status
    except urllib_error.HTTPError as failure:
        payload, status = failure.read(), failure.code
    text = payload.decode()
    try:
        return status, json.loads(text)
    except json.JSONDecodeError:
        return status, text


class ServeProcess:
    """One serving command on a free local port, with its output in a log file."""

    def __init__(
        self, command: list[str], log_path: Path, env: dict[str, str] | None = None
    ):
        self.name = " ".join(command[:2])
        self.port = free_port()
        self.base = f"http://127.0.0.1:{self.port}"
        self.log = log_path.open("w")
        self.process = subprocess.Popen(
            [*command, "--port", str(self.port)],
            stdout=self.log,
            stderr=subprocess.STDOUT,
            env={**os.environ, "HF_HUB_OFFLINE": "1", **(env or {})},
        )

    def wait_ready(self) -> dict:
        deadline = time.monotonic() + READY_TIMEOUT_SECONDS
        interval = 0.25
        last = None
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                raise AssertionError(
                    f"{self.name} exited with {self.process.returncode}"
                )
            try:
                status, last = call(self.base, "/health")
                if status == HTTP_OK:
                    return last
            except (urllib_error.URLError, ConnectionError):
                pass
            time.sleep(interval)
            interval = min(interval * 2, 2.0)
        raise AssertionError(f"not ready within {READY_TIMEOUT_SECONDS}s: {last}")

    def stop(self) -> int:
        if self.process.poll() is None:
            self.process.send_signal(signal.SIGINT)
            try:
                self.process.wait(timeout=STOP_TIMEOUT_SECONDS)
            except subprocess.TimeoutExpired as error:
                self.process.kill()
                self.process.wait()
                raise AssertionError(f"{self.name} did not stop on SIGINT") from error
        self.log.close()
        return self.process.returncode

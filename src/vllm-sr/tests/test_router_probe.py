"""The router probe runs as the CLI runs it inside the container: python3 -c."""

import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
from cli.router_probe import router_probe_command

TOKEN = "secret-value"


class _Router(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        if self.headers.get("Authorization", f"Bearer {TOKEN}") != f"Bearer {TOKEN}":
            self.send_error(401)
        elif self.path == "/ready":
            self.send_response(200)
            self.end_headers()
        else:
            self.send_error(503)

    def log_message(self, format: str, *args: object) -> None:
        pass


@pytest.fixture(scope="module")
def router():
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Router)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_port}"
    server.shutdown()


def _probe(command: list[str], **env: str) -> subprocess.CompletedProcess[str]:
    assert command[0] == "python3"
    return subprocess.run(
        [sys.executable, *command[1:]],
        capture_output=True,
        text=True,
        env=env,
        timeout=30,
    )


def test_the_probe_passes_only_on_a_2xx_answer(router):
    ready = _probe(router_probe_command(f"{router}/ready"))
    unavailable = _probe(router_probe_command(f"{router}/health"))

    assert (ready.returncode, ready.stderr) == (0, "")
    assert (unavailable.returncode, unavailable.stderr) == (1, "")


def test_the_probe_fails_when_nothing_listens():
    with ThreadingHTTPServer(("127.0.0.1", 0), _Router) as closed:
        url = f"http://127.0.0.1:{closed.server_port}/ready"

    assert _probe(router_probe_command(url, timeout=2)).returncode == 1


def test_the_bearer_comes_from_the_named_environment_variable(router):
    command = router_probe_command(f"{router}/ready", 5.0, "MANAGEMENT_TOKEN")

    assert TOKEN not in repr(command)
    assert _probe(command, MANAGEMENT_TOKEN=TOKEN).returncode == 0
    assert _probe(command, MANAGEMENT_TOKEN="wrong").returncode == 1
    assert _probe(command).returncode == 1

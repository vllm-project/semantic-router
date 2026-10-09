"""Private host supervisor and authenticated proxy for Router model processes.

The model HTTP/JSON API is forwarded unchanged. The private process lease API
lets the Router retain ownership of grouping, reload, and generation drain.
"""

from __future__ import annotations

import argparse
import hmac
import http.client
import json
import logging
import logging.handlers
import os
import re
import signal
import socket
import subprocess
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

MAX_BYTES = 64 << 20
LEASE_SECONDS = 45
HOP_HEADERS = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailer",
    "transfer-encoding",
    "upgrade",
    "content-length",
}
KEY = re.compile(r"[0-9a-f]{64}\Z")
log = logging.getLogger("apple_runtime")


class Processes:
    def __init__(self, config: dict):
        self.config = config
        self.lock = threading.RLock()
        self.children: dict[str, dict] = {}

    def launch(self, key: str, models: list[dict], *, leased: bool = True) -> int:
        if not KEY.fullmatch(key) or not models or len(models) > 256:
            raise ValueError("invalid process identity or model inventory")
        normalized = []
        for model in models:
            artifact = model.get("model", "")
            # Image filesystem paths cannot be interpreted on the Mac host.
            engine_local = (
                self.config.get("engine_mode") and Path(artifact).expanduser().is_dir()
            )
            if engine_local:
                artifact = str(Path(artifact).expanduser().resolve())
            if not engine_local and not re.fullmatch(
                r"[A-Za-z0-9_.-]+(?:/[A-Za-z0-9_.-]+)?", artifact
            ):
                raise ValueError(
                    "Apple router models require pinned Hub artifacts; image paths are unavailable on the host"
                )
            if model.get("device", "mps") != "mps":
                raise ValueError("Apple models must explicitly select mps")
            normalized.append({**model, "model": artifact, "device": "mps"})
        with self.lock:
            previous = self.children.get(key)
            if previous and previous["models"] != normalized:
                raise ValueError("process identity conflicts with its model inventory")
            if previous and previous["process"].poll() is None:
                previous["expires"] = time.monotonic() + LEASE_SECONDS
                return previous["port"]
            with socket.socket() as probe:
                probe.bind(("127.0.0.1", self.config.get("engine_port") or 0))
                port = probe.getsockname()[1]
            models_path = Path(self.config["directory"]) / f"{key}.models.json"
            models_path.write_text(json.dumps({"models": normalized}))
            models_path.chmod(0o600)
            command = [
                self.config["python"],
                "-m",
                "vllm_srun",
                "serve",
                "--models",
                str(models_path),
                "--host",
                "127.0.0.1",
                "--port",
                str(port),
                "--max-request-bytes",
                str(MAX_BYTES),
                "--max-bundle-tasks",
                "1024",
                "--cache-dir",
                self.config["cache"],
            ]
            command += ["--log-level", self.config.get("log_level", "info")]
            env = {
                **os.environ,
                "PYTHONUNBUFFERED": "1",
                "PYTORCH_ENABLE_MPS_FALLBACK": "0",
            }
            child = subprocess.Popen(
                command,
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )
            self.children[key] = {
                "process": child,
                "models": normalized,
                "port": port,
                "leased": leased,
                "expires": time.monotonic() + LEASE_SECONDS,
            }
            threading.Thread(target=self._logs, args=(key, child), daemon=True).start()
            log.info(
                "Started MPS process %s pid=%s models=%s",
                key[:12],
                child.pid,
                [m.get("name") for m in normalized],
            )
            return port

    @staticmethod
    def _logs(key: str, child: subprocess.Popen) -> None:
        assert child.stdout is not None
        for line in iter(child.stdout.readline, b""):
            log.info("[%s] %s", key[:12], line.decode(errors="replace").rstrip())
        child.stdout.close()

    def stop(self, key: str, *, expired_child: dict | None = None) -> None:
        with self.lock:
            if expired_child is not None:
                # A renewal or replacement may have happened since reap took
                # its snapshot. Decide and remove under the same process lock.
                current = self.children.get(key)
                if (
                    current is not expired_child
                    or not current["leased"]
                    or current["expires"] >= time.monotonic()
                ):
                    return
            child = self.children.pop(key, None)
            if child:
                process = child["process"]
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait(timeout=5)
                log.info("Stopped MPS process %s", key[:12])

    def reap(self) -> None:
        with self.lock:
            expired = [
                (key, child)
                for key, child in self.children.items()
                if child["leased"] and child["expires"] < time.monotonic()
            ]
        for key, child in expired:
            self.stop(key, expired_child=child)

    def close(self) -> None:
        with self.lock:
            for key in list(self.children):
                self.stop(key)


class Handler(BaseHTTPRequestHandler):
    server: "BridgeServer"

    def log_message(self, *_args) -> None:
        pass  # Requests can contain prompts or credentials; never access-log them.

    def reply(self, status: int, body: dict) -> None:
        data = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def dispatch(self) -> None:
        expected = "Bearer " + self.server.config["token"]
        if not hmac.compare_digest(self.headers.get("Authorization", ""), expected):
            self.reply(401, {"error": "unauthorized"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if self.headers.get("Transfer-Encoding") or not 0 <= length <= MAX_BYTES:
                self.reply(413, {"error": "request body exceeds bridge limit"})
                return
            body = self.rfile.read(length)
            if self.path == "/status" and self.command == "GET":
                self.reply(
                    200,
                    {
                        "pid": os.getpid(),
                        "identity": self.server.config["identity"],
                        "processes": len(self.server.processes.children),
                    },
                )
                return
            if self.path == "/shutdown" and self.command == "POST":
                self.reply(200, {"stopping": True})
                threading.Thread(target=self.server.shutdown, daemon=True).start()
                return
            if (
                self.path == "/engine"
                and self.command == "POST"
                and self.server.config.get("engine_mode")
            ):
                payload = json.loads(body)
                port = self.server.processes.launch(
                    payload["key"], payload["models"], leased=False
                )
                self.reply(200, {"port": port})
                return
            parts = self.path.split("/", 3)
            if len(parts) < 3 or parts[1] != "processes" or not KEY.fullmatch(parts[2]):
                self.reply(404, {"error": "unknown bridge route"})
                return
            key = parts[2]
            if len(parts) == 3 and self.command == "POST":
                port = self.server.processes.launch(key, json.loads(body)["models"])
                self.reply(200, {"port": port})
                return
            if len(parts) == 3 and self.command == "DELETE":
                self.server.processes.stop(key)
                self.reply(200, {"stopped": True})
                return
            with self.server.processes.lock:
                child = self.server.processes.children.get(key)
                if not child or child["process"].poll() is not None:
                    self.reply(503, {"error": "host runtime unavailable"})
                    return
                port = child["port"]
            connection = http.client.HTTPConnection("127.0.0.1", port, timeout=1800)
            try:
                excluded = (
                    HOP_HEADERS
                    | {"host", "authorization"}
                    | {
                        value.strip().lower()
                        for value in self.headers.get("Connection", "").split(",")
                    }
                )
                headers = {
                    name: value
                    for name, value in self.headers.items()
                    if name.lower() not in excluded
                }
                connection.request(self.command, "/" + parts[3], body, headers)
                response = connection.getresponse()
                data = response.read(MAX_BYTES + 1)
                if len(data) > MAX_BYTES:
                    self.reply(502, {"error": "runtime response exceeds bridge limit"})
                    return
                self.send_response(response.status)
                excluded = HOP_HEADERS | {
                    value.strip().lower()
                    for value in response.getheader("Connection", "").split(",")
                }
                for name, value in response.getheaders():
                    if name.lower() not in excluded:
                        self.send_header(name, value)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
            finally:
                connection.close()
        except (
            OSError,
            ValueError,
            KeyError,
            TypeError,
            http.client.HTTPException,
        ) as error:
            log.warning("Bridge request failed: %s", type(error).__name__)
            self.reply(503, {"error": str(error)})

    do_GET = dispatch
    do_POST = dispatch
    do_DELETE = dispatch


class BridgeServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, config: dict):
        self.config = config
        self.processes = Processes(config)
        super().__init__(("127.0.0.1", config["port"]), Handler)

    def service_actions(self) -> None:
        self.processes.reap()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--state", required=True, type=Path)
    args = parser.parse_args()
    config = json.loads(args.state.read_text())
    config["pid"] = os.getpid()
    config["process_identity"] = subprocess.run(
        ["ps", "-ww", "-p", str(os.getpid()), "-o", "lstart=", "-o", "command="],
        capture_output=True,
        text=True,
        check=True,
        timeout=5,
    ).stdout.strip()
    # Publish ownership before any model child can start, including when the
    # launching CLI crashes immediately after Popen.
    fd, filename = tempfile.mkstemp(prefix=".state-", dir=args.state.parent)
    temporary = Path(filename)
    with os.fdopen(fd, "w") as output:
        json.dump(config, output)
    temporary.replace(args.state)
    handler = logging.handlers.RotatingFileHandler(
        config["log"], maxBytes=10 << 20, backupCount=3
    )
    logging.basicConfig(
        level=logging.INFO,
        handlers=[handler],
        format="%(asctime)s %(levelname)s %(message)s",
    )
    server = BridgeServer(config)

    # Handler must return promptly; shutdown waits for serve_forever's thread.
    def terminate(_signum, _frame):
        threading.Thread(target=server.shutdown, daemon=True).start()

    signal.signal(signal.SIGTERM, terminate)
    signal.signal(signal.SIGINT, terminate)
    try:
        server.serve_forever(poll_interval=1)
    finally:
        server.processes.close()
        server.server_close()


if __name__ == "__main__":
    main()

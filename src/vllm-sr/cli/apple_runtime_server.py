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
import shutil
import signal
import socket
import subprocess
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path, PurePosixPath

MAX_BYTES = 64 << 20
LEASE_SECONDS = 45
MAX_MODELS_PER_PROCESS = 256
PROCESS_ROUTE_PARTS = 3
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
        self.socket_root = Path(tempfile.mkdtemp(prefix="vsr-", dir="/tmp"))
        self.closed = False
        self.probes: set[subprocess.Popen] = set()
        self.terminations = ThreadPoolExecutor(
            max_workers=4, thread_name_prefix="apple-retire"
        )

    def launch(
        self,
        key: str,
        models: list[dict],
        *,
        lease: str | None = None,
    ) -> None:
        if not KEY.fullmatch(key) or not models or len(models) > MAX_MODELS_PER_PROCESS:
            raise ValueError("invalid process identity or model inventory")
        if not isinstance(lease, str) or not KEY.fullmatch(lease):
            raise ValueError("managed host processes require a group lease identity")
        normalized = []
        for model in models:
            artifact = model.get("model", "")
            local = Path(artifact).is_absolute()
            if local:
                root = Path(self.config["models_directory"]).resolve()
                container_path = PurePosixPath(artifact)
                if container_path.is_relative_to("/app/models"):
                    path = root / container_path.relative_to("/app/models")
                else:
                    path = Path(artifact)
                path = path.resolve()
                if not path.is_relative_to(root) or not path.is_dir():
                    raise ValueError(
                        "local Apple models must be inside the stack's mounted models directory"
                    )
                artifact = str(path)
            if not local and not re.fullmatch(
                r"[A-Za-z0-9_.-]+(?:/[A-Za-z0-9_.-]+)?", artifact
            ):
                raise ValueError(
                    "Apple models require a Hub artifact or a directory under /app/models"
                )
            if model.get("device", "mps") != "mps":
                raise ValueError("Apple models must explicitly select mps")
            normalized.append({**model, "model": artifact, "device": "mps"})
        with self.lock:
            if self._renew(key, normalized, lease):
                return
        # Placement checks may download and verify the package. Never hold the
        # process lock while doing that, and never load weights onto the GPU here.
        fd, filename = tempfile.mkstemp(
            dir=self.config["directory"], prefix=key + ".", suffix=".models.json"
        )
        models_path = Path(filename)
        with os.fdopen(fd, "w") as output:
            json.dump({"models": normalized}, output)
        retained = False
        try:
            self._check_models(models_path)
            with self.lock:
                # Another acquire may have completed while placement was checked.
                if self._renew(key, normalized, lease):
                    return
                self._start_locked(key, normalized, lease, models_path)
                retained = True
        finally:
            if not retained:
                models_path.unlink(missing_ok=True)

    def _start_locked(
        self, key: str, models: list[dict], lease: str, models_path: Path
    ) -> None:
        """Publish one worker under the lock; its files belong to this lifetime."""
        directory = Path(tempfile.mkdtemp(prefix="p-", dir=self.socket_root))
        worker_socket = directory / "r.sock"
        command = [
            self.config["python"],
            "-m",
            "vllm_srun",
            "serve",
            "--models",
            str(models_path),
            "--uds",
            str(worker_socket),
            "--max-request-bytes",
            str(MAX_BYTES),
            "--max-bundle-tasks",
            "1024",
            "--cache-dir",
            self.config["cache"],
            "--log-level",
            self.config.get("log_level", "info"),
        ]
        env = {
            **os.environ,
            "PYTHONUNBUFFERED": "1",
            "PYTORCH_ENABLE_MPS_FALLBACK": "0",
        }
        try:
            child = subprocess.Popen(
                command,
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )
        except BaseException:
            directory.rmdir()
            raise
        deadline = time.monotonic() + LEASE_SECONDS
        self.children[key] = {
            "process": child,
            "models": models,
            "models_path": models_path,
            "socket": str(worker_socket),
            "directory": directory,
            "leases": {lease: deadline},
            "expires": deadline,
        }
        threading.Thread(target=self._logs, args=(key, child), daemon=True).start()
        log.info(
            "Started MPS process %s pid=%s models=%s",
            key[:12],
            child.pid,
            [model.get("name") for model in models],
        )

    def _renew(self, key: str, models: list[dict], lease: str) -> bool:
        """Called under the process lock, including the post-preflight recheck."""
        if self.closed:
            raise ValueError("host supervisor is shutting down")
        previous = self.children.get(key)
        if previous and previous["models"] != models:
            raise ValueError("process identity conflicts with its model inventory")
        if previous and previous["process"].poll() is None:
            previous["leases"][lease] = time.monotonic() + LEASE_SECONDS
            previous["expires"] = max(previous["leases"].values())
            return True
        if previous:
            self.children.pop(key)
            shutil.rmtree(previous["directory"])
            previous["models_path"].unlink(missing_ok=True)
        return False

    def _check_models(self, path: Path) -> None:
        with self.lock:
            if self.closed:
                raise ValueError("host supervisor is shutting down")
            probe = subprocess.Popen(
                [
                    self.config["python"],
                    str(Path(__file__).with_name("apple_runtime_probe.py")),
                    str(path),
                    self.config["cache"],
                ],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            self.probes.add(probe)
        try:
            _, errors = probe.communicate(timeout=1800)
        except subprocess.TimeoutExpired as error:
            probe.kill()
            probe.communicate(timeout=5)
            raise ValueError("Apple model placement preparation timed out") from error
        finally:
            with self.lock:
                self.probes.discard(probe)
        if probe.returncode:
            raise ValueError("Apple model placement failed: " + errors[-4096:].strip())

    @staticmethod
    def _logs(key: str, child: subprocess.Popen) -> None:
        assert child.stdout is not None
        for line in iter(child.stdout.readline, b""):
            log.info("[%s] %s", key[:12], line.decode(errors="replace").rstrip())
        child.stdout.close()

    def stop(
        self, key: str, *, expired_child: dict | None = None, lease: str | None = None
    ) -> bool:
        if lease is not None and (
            not isinstance(lease, str) or not KEY.fullmatch(lease)
        ):
            raise ValueError("invalid group lease identity")
        with self.lock:
            current = self.children.get(key)
            if lease is not None:
                if not current or lease not in current["leases"]:
                    return False
                del current["leases"][lease]
                if current["leases"]:
                    current["expires"] = max(current["leases"].values())
                    return False
            # A renewal or replacement may have happened since reap took
            # its snapshot. Decide and remove under the same process lock.
            if expired_child is not None and (
                current is not expired_child or current["expires"] >= time.monotonic()
            ):
                return False
            child = self.children.pop(key, None)
            if child:
                retirement = self.terminations.submit(self._terminate, key, child)
        if child:
            retirement.result()
            return True
        return False

    @staticmethod
    def _stop_process(process: subprocess.Popen) -> None:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)

    @classmethod
    def _terminate(cls, key: str, child: dict) -> None:
        cls._stop_process(child["process"])
        shutil.rmtree(child["directory"])
        child["models_path"].unlink(missing_ok=True)
        log.info("Stopped MPS process %s", key[:12])

    def reap(self) -> None:
        with self.lock:
            now = time.monotonic()
            for child in self.children.values():
                child["leases"] = {
                    owner: deadline
                    for owner, deadline in child["leases"].items()
                    if deadline >= now
                }
                child["expires"] = max(child["leases"].values(), default=0)
            expired = [
                (key, child)
                for key, child in self.children.items()
                if child["expires"] < now
            ]
        for key, child in expired:
            self.stop(key, expired_child=child)

    def close(self) -> None:
        with self.lock:
            if self.closed:
                return
            self.closed = True
            keys = list(self.children)
            probes = list(self.probes)
        errors = []
        for probe in probes:
            try:
                self._stop_process(probe)
            except (OSError, subprocess.TimeoutExpired) as error:
                errors.append(error)
        for key in keys:
            try:
                self.stop(key)
            except (OSError, subprocess.TimeoutExpired) as error:
                errors.append(error)
        # Include retirements already removed from the live inventory.
        self.terminations.shutdown(wait=True)
        if errors:
            raise RuntimeError(
                "Host processes could not be stopped cleanly"
            ) from errors[0]
        self.socket_root.rmdir()


class Handler(BaseHTTPRequestHandler):
    server: BridgeServer

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
            parts = self.path.split("/", PROCESS_ROUTE_PARTS)
            if (
                len(parts) < PROCESS_ROUTE_PARTS
                or parts[1] != "processes"
                or not KEY.fullmatch(parts[2])
            ):
                self.reply(404, {"error": "unknown bridge route"})
                return
            key = parts[2]
            if len(parts) == PROCESS_ROUTE_PARTS and self.command == "POST":
                payload = json.loads(body)
                self.server.processes.launch(
                    key, payload["models"], lease=payload["lease"]
                )
                self.reply(200, {"started": True})
                return
            if len(parts) == PROCESS_ROUTE_PARTS and self.command == "DELETE":
                lease = json.loads(body)["lease"]
                if not isinstance(lease, str) or not KEY.fullmatch(lease):
                    raise ValueError("invalid group lease identity")
                stopped = self.server.processes.stop(key, lease=lease)
                self.reply(200, {"stopped": stopped})
                return
            with self.server.processes.lock:
                child = self.server.processes.children.get(key)
                if not child or child["process"].poll() is not None:
                    self.reply(503, {"error": "host runtime unavailable"})
                    return
                worker_socket = child["socket"]
            connection = http.client.HTTPConnection("localhost", timeout=1800)
            connection.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            connection.sock.settimeout(1800)
            try:
                connection.sock.connect(worker_socket)
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

    # BaseHTTPRequestHandler requires these exact HTTP-method hook names.
    do_GET = dispatch  # noqa: N815
    do_POST = dispatch  # noqa: N815
    do_DELETE = dispatch  # noqa: N815


class BridgeServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, config: dict):
        self.config = config
        self.processes = Processes(config)
        self._reaper_stop = threading.Event()
        self._reaper = None
        super().__init__(("127.0.0.1", config["port"]), Handler)

    def serve_forever(self, poll_interval=0.5) -> None:
        self._reaper = threading.Thread(target=self._reap_loop, daemon=True)
        self._reaper.start()
        try:
            super().serve_forever(poll_interval)
        finally:
            self._reaper_stop.set()

    def _reap_loop(self) -> None:
        while not self._reaper_stop.wait(1):
            try:
                self.processes.reap()
            except (OSError, subprocess.TimeoutExpired):
                log.exception("Host model retirement failed")

    def server_close(self) -> None:
        self._reaper_stop.set()
        if self._reaper is not None:
            self._reaper.join()
        try:
            self.processes.close()
        finally:
            super().server_close()


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
        server.server_close()


if __name__ == "__main__":
    main()

"""Unix-only instance controller, suitable for a host service supervisor.

The socket grants the narrow instance-deploy capability, never arbitrary Docker
access. Its private manifest is supplied by the canonical CLI, not HTTP callers.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import socketserver
import threading
from http.server import BaseHTTPRequestHandler
from pathlib import Path

from cli.commands.runtime_paths import cli_user_share_gid
from cli.instance_runtime import ContainerInstanceBackend
from cli.instance_state import InstanceConflictError, InstanceController

MAX_UNIX_SOCKET_PATH_BYTES = 103


class ControllerServer(socketserver.ThreadingMixIn, socketserver.UnixStreamServer):
    daemon_threads = True
    request_queue_size = 16

    def get_request(self):
        connection, address = super().get_request()
        connection.settimeout(35)
        return connection, address


class ControllerHandler(BaseHTTPRequestHandler):
    def log_message(self, *_args):
        # Request/model contents and credentials never enter controller logs.
        pass

    def send_json(self, status, document):
        self.send_bytes(status, json.dumps(document).encode())

    def send_bytes(self, status, body):
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        try:
            if self.path == "/status":
                self.send_json(200, self.server.controller.status())
            elif self.path == "/models":
                self.send_bytes(*self.server.controller.backend.native("/models"))
            else:
                self.send_json(404, {"error": "Unknown controller endpoint"})
        except (OSError, ValueError, RuntimeError):
            self.send_json(503, {"error": "Instance state is unavailable"})

    def do_POST(self):
        try:
            if self.headers.get("Transfer-Encoding"):
                raise ValueError("Transfer encoding is not supported")
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= 2 << 20:
                raise ValueError("Invalid body size")
            payload = json.loads(self.rfile.read(length))
            if not isinstance(payload, dict):
                raise ValueError("Request must be an object")
            if self.path == "/deploy":
                self.send_json(202, self.server.controller.submit(payload))
            elif self.path in {"/shutdown", "/configuration-applied"}:
                with self.server.controller.lock:
                    operation = self.server.controller.state.get("operation") or {}
                    if operation.get("phase") not in {None, "ready", "failed"}:
                        raise InstanceConflictError(
                            "An instance operation is in progress"
                        )
                    if self.path == "/configuration-applied":
                        self.server.controller.backend.checkpoint()
                    self.send_json(200, {"ok": True})
                    if self.path == "/shutdown":
                        threading.Thread(
                            target=self.server.shutdown, daemon=True
                        ).start()
            elif self.path == "/systemone":
                if (
                    set(payload) - {"deployment", "request", "expected_artifact"}
                    or not isinstance(payload.get("deployment"), str)
                    or not isinstance(payload.get("request"), dict)
                    or (
                        "expected_artifact" in payload
                        and not isinstance(payload["expected_artifact"], str)
                    )
                ):
                    raise ValueError("Invalid native inference envelope")
                self.send_bytes(
                    *self.server.controller.backend.native(self.path, payload)
                )
            else:
                self.send_json(404, {"error": "Unknown controller endpoint"})
        except InstanceConflictError as error:
            self.send_json(409, {"error": str(error)})
        except (ValueError, KeyError):
            self.send_json(400, {"error": "Invalid instance request"})
        except (OSError, RuntimeError):
            self.send_json(503, {"error": "Instance operation unavailable"})


def run(directory: Path):
    manifest = json.loads((directory / "manifest.json").read_text())
    with (directory / "controller.lock").open("a") as lock:
        os.chmod(lock.name, 0o600)
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        path = directory / "socket" / "control.sock"
        if len(os.fsencode(path)) > MAX_UNIX_SOCKET_PATH_BYTES:
            raise ValueError(
                "Instance socket path is too long; choose a shorter state root"
            )
        path.unlink(missing_ok=True)
        backend = ContainerInstanceBackend(manifest, directory)
        controller = InstanceController(directory, backend)
        with ControllerServer(str(path), ControllerHandler) as server:
            os.chmod(path, 0o660)
            os.chown(path, -1, cli_user_share_gid())
            server.controller = controller
            threading.Thread(target=controller.recover, daemon=True).start()
            server.serve_forever(poll_interval=0.5)
        path.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    arguments = parser.parse_args()
    run(arguments.directory)


if __name__ == "__main__":
    main()

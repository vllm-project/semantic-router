"""Headless local HTTP owner shared by the CLI and Dashboard."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import hmac
import json
import os
import signal
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from cli.runtime_env_names import runtime_env_name_is_allowed

from . import VERSION
from .contracts import catalog, plan
from .engine import Engine
from .offline import export_training, regrade, replay
from .report import compare, make_report
from .store import Store

PREFIX = "/api/sr-bench/v1"
DEFAULT_STORE = Path.home() / ".local" / "share" / "vllm-sr" / "sr-bench"
DEFAULT_URL = "http://127.0.0.1:8090"
MAX_ACTOR_ID_CHARS = 256
RUN_ROUTE_PARTS = 2
RUN_ACTION_ROUTE_PARTS = 3
CALL_ROUTE_PARTS = 4


def service_credentials():
    reference = os.environ.get("SR_BENCH_TOKEN_ENV", "SR_BENCH_TOKEN")
    if not runtime_env_name_is_allowed(reference) or reference in {
        "SR_BENCH_URL",
        "SR_BENCH_STORE",
        "SR_BENCH_TOKEN_ENV",
    }:
        raise ValueError("SR_BENCH_TOKEN_ENV must name a safe environment variable")
    token = os.environ.get(reference)
    if reference != "SR_BENCH_TOKEN" and not token:
        raise ValueError(
            f"The configured sr-bench service token is missing: {reference}"
        )
    return reference, token


def datasets(store):
    rows = []
    for path in sorted((store.root / "datasets").glob("*/manifest.json")):
        try:
            value = json.loads(path.read_text())
            rows.append({**value, "manifest_path": str(path)})
        except (OSError, ValueError):
            continue
    return rows


class Server(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address, store, token=None, store_identity=None):
        self.store = store
        self.engine = Engine(store)
        self.token = token
        self.store_identity = (
            store_identity or hashlib.sha256(str(store.root).encode()).hexdigest()
        )
        super().__init__(address, Handler)


class Handler(BaseHTTPRequestHandler):
    server_version = "sr-bench/1.0"

    def log_message(self, *args):
        pass

    def _send(self, status, value):
        payload = json.dumps(value, ensure_ascii=False, allow_nan=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(payload)

    def _actor(self):
        token = self.server.token
        if token and not hmac.compare_digest(
            self.headers.get("Authorization", ""), "Bearer " + token
        ):
            raise PermissionError("Service authorization required")
        actor = self.headers.get("X-SR-Bench-Actor-ID")
        role = self.headers.get("X-SR-Bench-Actor-Role")
        if actor or role:
            if not token:
                raise PermissionError(
                    "Trusted actor forwarding requires service authentication"
                )
            if not actor or role not in {"admin", "editor", "viewer"}:
                raise PermissionError("Invalid actor identity")
            if len(actor) > MAX_ACTOR_ID_CHARS:
                raise PermissionError("Invalid actor identity")
            return actor, role
        return "local", "local"

    def _body(self):
        length = int(self.headers.get("Content-Length", "0"))
        if length <= 0 or length > 8 * 1024 * 1024:
            raise ValueError("JSON body must be between 1 byte and 8 MiB")
        value = json.loads(self.rfile.read(length))
        if not isinstance(value, dict):
            raise ValueError("JSON body must be an object")
        return value

    def _registry_targets(self):
        file = self.server.store.root / "targets.json"
        data = json.loads(file.read_text()) if file.exists() else []
        if isinstance(data, dict):
            data = data.get("targets", [])
        if not isinstance(data, list):
            raise ValueError("Invalid server target registry")
        safe = {
            "id",
            "kind",
            "base_url",
            "model",
            "api_key_env",
            "config_hash",
            "prices",
            "preview_url",
            "expected_response_model",
            "cost_mode",
            "max_inference_calls",
            "preview_api_key_env",
            "request_params",
        }
        if any(set(t) - safe for t in data):
            raise ValueError("Server target registry contains unsupported fields")
        return data

    def _manifest(self, manifest, role):
        if role == "local":
            return manifest
        if not isinstance(manifest, dict):
            raise ValueError("manifest must be an object")
        registry = {t["id"]: t for t in self._registry_targets()}
        resolved = []
        for target in manifest.get("targets", []):
            registered = registry.get(target.get("id"))
            if registered is None:
                raise PermissionError(
                    "Target must be registered by the server operator"
                )
            if any(registered.get(k) != v for k, v in target.items()):
                raise PermissionError("Dashboard cannot override a registered target")
            resolved.append(registered)
        options_file = self.server.store.root / "benchmark-options.json"
        options = json.loads(options_file.read_text()) if options_file.exists() else {}
        for key, value in manifest.get("benchmark_options", {}).items():
            if options.get(key) != value:
                raise PermissionError(
                    "Benchmark harness options are configured by the server operator"
                )
        auxiliary = {}
        for key, target in manifest.get("auxiliary_targets", {}).items():
            registered = registry.get(key)
            if registered is None or any(
                registered.get(k) != v for k, v in target.items()
            ):
                raise PermissionError("Auxiliary target must match the server registry")
            auxiliary[key] = registered
        # Resolve judge/simulator references declared by the operator without exposing credentials.
        for config in options.values():
            for role_name in ("judge", "simulator"):
                ref = config.get(role_name)
                if ref and ref not in {t["id"] for t in resolved} and ref in registry:
                    auxiliary[ref] = registry[ref]
        return {
            **manifest,
            "targets": resolved,
            "auxiliary_targets": auxiliary,
            "benchmark_options": options,
        }

    def do_GET(self):
        self._handle("GET")

    def do_POST(self):
        self._handle("POST")

    def _handle(self, method):
        try:
            actor, role = self._actor()
            owner = None if role == "admin" else actor
            parsed = urlparse(self.path)
            path = parsed.path.rstrip("/")
            if method == "GET" and path in {"/health", PREFIX + "/health"}:
                return self._send(
                    200,
                    {
                        "version": VERSION,
                        "status": "ready",
                        "store_id": self.server.store_identity,
                    },
                )
            if not path.startswith(PREFIX + "/"):
                return self._send(404, {"error": "not found"})
            route = path[len(PREFIX) :].strip("/").split("/")
            if method == "POST" and role == "viewer":
                raise PermissionError("Editor access is required")
            if route == ["catalog"] and method == "GET":
                return self._send(200, catalog())
            if route == ["datasets"] and method == "GET":
                return self._send(200, {"datasets": datasets(self.server.store)})
            if route == ["targets"] and method == "GET":
                return self._send(200, {"targets": self._registry_targets()})
            if route == ["plans"] and method == "POST":
                body = self._body()
                frozen = plan(self._manifest(body.get("manifest", body), role))
                return self._send(
                    200,
                    {
                        "manifest": frozen,
                        "plan_sha256": frozen["plan_sha256"],
                        "total": len(frozen["cases"]) * len(frozen["targets"]),
                        "status": "validated",
                    },
                )
            if route == ["runs"]:
                if method == "GET":
                    return self._send(200, {"runs": self.server.store.list(owner)})
                body = self._body()
                manifest = body.get("manifest")
                if not manifest:
                    raise ValueError("manifest is required")
                return self._send(
                    201,
                    self.server.engine.start(
                        self._manifest(manifest, role),
                        actor,
                        body.get("idempotency_key"),
                    ),
                )
            if route == ["replays"] and method == "POST":
                body = self._body()
                for key in ("baseline_run_id", "preview_run_id"):
                    self.server.store.get(body[key], owner)
                return self._send(
                    201,
                    replay(
                        self.server.store,
                        body["baseline_run_id"],
                        body["preview_run_id"],
                        actor,
                        body.get("idempotency_key"),
                    ),
                )
            if route == ["comparisons"] and method == "POST":
                body = self._body()
                for key in ("baseline_run_id", "candidate_run_id"):
                    self.server.store.get(body[key], owner)
                return self._send(
                    200,
                    compare(
                        self.server.store,
                        body["baseline_run_id"],
                        body["candidate_run_id"],
                    ),
                )
            if len(route) >= RUN_ROUTE_PARTS and route[0] == "runs":
                run_id = route[1]
                run = self.server.store.get(run_id, owner)
                if len(route) == RUN_ROUTE_PARTS and method == "GET":
                    return self._send(200, run)
                if (
                    len(route) == CALL_ROUTE_PARTS
                    and route[2] == "calls"
                    and method == "GET"
                ):
                    return self._send(200, self.server.store.call(run_id, route[3]))
                if len(route) == RUN_ACTION_ROUTE_PARTS:
                    action = route[2]
                    if action in {"regrade", "export"} and method == "POST":
                        return self._send(
                            200,
                            (regrade if action == "regrade" else export_training)(
                                self.server.store, run_id
                            ),
                        )
                    if action in {"results", "calls"} and method == "GET":
                        query = parse_qs(parsed.query)
                        after = int(query.get("after", ["0"])[0])
                        limit = int(query.get("limit", ["100"])[0])
                        return self._send(
                            200, self.server.store.page(run_id, action, after, limit)
                        )
                    if action == "report" and method == "GET":
                        return self._send(200, make_report(self.server.store, run_id))
                    if action == "events" and method == "GET":
                        after = int(parse_qs(parsed.query).get("after", ["0"])[0])
                        return self._send(
                            200, {"events": self.server.store.events(run_id, after)}
                        )
                    if action == "cancel" and method == "POST":
                        return self._send(200, self.server.engine.cancel(run_id))
            self._send(404, {"error": "not found"})
        except PermissionError as exc:
            self._send(403, {"error": str(exc)})
        except KeyError:
            self._send(404, {"error": "resource not found"})
        except (ValueError, TypeError, json.JSONDecodeError) as exc:
            self._send(400, {"error": str(exc)})
        except Exception:
            self._send(
                500, {"error": "Internal service error; inspect the local journal"}
            )


def serve(store=DEFAULT_STORE, host="127.0.0.1", port=8090, store_identity=None):
    token_reference, token = service_credentials()
    if host not in {"127.0.0.1", "::1", "localhost"} and not token:
        raise ValueError(f"Non-loopback sr-bench service requires {token_reference}")
    if store_identity and (
        len(store_identity) != len(hashlib.sha256().hexdigest())
        or any(c not in "0123456789abcdef" for c in store_identity)
    ):
        raise ValueError("store-identity must be a SHA256 digest")
    root = Path(store).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    lock = (root / "service.lock").open("a+")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise ValueError("A service already owns this store") from None
    server = Server((host, port), Store(root), token, store_identity)
    (root / "service.json").write_text(
        json.dumps(
            {"pid": os.getpid(), "url": f"http://{host}:{port}", "version": VERSION}
        )
    )

    def stop(signum, frame):
        for event in list(server.engine.cancels.values()):
            event.set()
        threading.Thread(target=server.shutdown, daemon=True).start()

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    try:
        server.serve_forever(poll_interval=0.2)
    finally:
        server.server_close()
        for thread in list(server.engine.threads.values()):
            thread.join(timeout=5)
        lock.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--store", type=Path, default=DEFAULT_STORE)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8090)
    parser.add_argument("--store-identity")
    args = parser.parse_args()
    serve(args.store, args.host, args.port, args.store_identity)

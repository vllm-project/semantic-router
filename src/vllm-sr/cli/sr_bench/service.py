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
from .accounting import reconcile_usage
from .candidate_plans import candidate_manifest, validate_candidate_protocol
from .contracts import catalog, plan, planned_cells
from .datasets import DatasetReader
from .engine import Engine, ReviewedPlanChangedError
from .experiments import ActiveExperimentError, ExperimentDeletedError, Experiments
from .offline import export_training, regrade, replay
from .recovery import RecoveryPlanError, recover, recovery_plan
from .replay_validation import ReplayEligibilityError
from .report import compare, make_report
from .run_options import run_options
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
        self.experiments = Experiments(store)
        self.datasets = DatasetReader(store.root)
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

    def _plan(self, manifest, role, actor):
        frozen = plan(manifest, policy=lambda resolved: self._manifest(resolved, role))
        if membership := frozen.get("experiment"):
            self.server.experiments.get(
                membership["id"], None if role == "admin" else actor
            )
        return {
            "manifest": {
                k: v
                for k, v in frozen.items()
                if k != "cases" or "dataset" not in frozen
            },
            "plan_sha256": frozen["plan_sha256"],
            "total": len(planned_cells(frozen)),
            "status": "validated",
            "model_requests": 0,
        }

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
            if not actor or role not in {"admin", "write", "read"}:
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
            "capture_recipe",
            "native_limits",
        }
        if any(set(t) - safe for t in data):
            raise ValueError("Server target registry contains unsupported fields")
        return data

    def _manifest(self, manifest, role):
        if role == "local":
            return manifest
        if not isinstance(manifest, dict):
            raise ValueError("manifest must be an object")
        cases = manifest.get("cases")
        if (
            not isinstance(cases, list)
            or not cases
            or any(
                not isinstance(case, dict) or not isinstance(case.get("benchmark"), str)
                for case in cases
            )
        ):
            raise ValueError("at least one valid case is required")
        selected = {case.get("benchmark") for case in cases}
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
        # Only selected benchmarks contribute operator defaults. Explicit frozen
        # options remain verified above, including any intentionally retained scope.
        selected_options = {
            key: value for key, value in options.items() if key in selected
        }
        # Resolve only the auxiliary roles required by these selected benchmarks.
        for config in selected_options.values():
            for role_name in ("judge", "simulator"):
                ref = config.get(role_name)
                if ref and ref not in {t["id"] for t in resolved} and ref in registry:
                    auxiliary[ref] = registry[ref]
        result = {**manifest, "targets": resolved}
        if auxiliary or "auxiliary_targets" in manifest:
            result["auxiliary_targets"] = auxiliary
        effective_options = {
            **selected_options,
            **manifest.get("benchmark_options", {}),
        }
        if effective_options or "benchmark_options" in manifest:
            result["benchmark_options"] = effective_options
        return result

    def do_GET(self):
        self._handle("GET")

    def do_POST(self):
        self._handle("POST")

    def do_DELETE(self):
        self._handle("DELETE")

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
            if role == "read" and (
                method == "DELETE" or (method == "POST" and route != ["comparisons"])
            ):
                raise PermissionError("Write access is required")
            if (
                route in (["replay-options"], ["comparison-options"])
                and method == "GET"
            ):
                query = parse_qs(parsed.query, keep_blank_values=True, max_num_fields=3)
                if set(query) - {"baseline_run_id", "after", "limit"} or any(
                    len(value) != 1 or not value[0] for value in query.values()
                ):
                    raise ValueError("Invalid run options filters")
                return self._send(
                    200,
                    run_options(
                        self.server.store,
                        "replay" if route[0] == "replay-options" else "comparison",
                        query.get("baseline_run_id", [None])[0],
                        owner,
                        query.get("after", [None])[0],
                        int(query.get("limit", ["10"])[0]),
                    ),
                )
            if route[0] == "experiments":
                if method == "DELETE" and len(route) == RUN_ROUTE_PARTS:
                    if parsed.query or int(self.headers.get("Content-Length", "0")):
                        raise ValueError("Experiment deletion takes no body or filters")
                    return self._send(
                        200, self.server.experiments.delete(route[1], owner)
                    )
                if method == "GET":
                    query = parse_qs(
                        parsed.query, keep_blank_values=True, max_num_fields=2
                    )
                    if set(query) - {"after", "limit"} or any(
                        len(v) != 1 for v in query.values()
                    ):
                        raise ValueError("Invalid experiment page filters")
                    page = {
                        "after": int(query.get("after", ["0"])[0]),
                        "limit": int(query.get("limit", ["20"])[0]),
                    }
                    if len(route) == 1:
                        return self._send(
                            200, self.server.experiments.list(owner, **page)
                        )
                    if len(route) == RUN_ROUTE_PARTS:
                        return self._send(
                            200, self.server.experiments.get(route[1], owner)
                        )
                    if len(route) == RUN_ACTION_ROUTE_PARTS and route[2] == "runs":
                        return self._send(
                            200, self.server.experiments.runs(route[1], owner, **page)
                        )
                if method == "POST":
                    body = self._body()
                    if len(route) == 1:
                        return self._send(
                            201,
                            self.server.experiments.create(
                                body.get("name"), actor, body.get("idempotency_key")
                            ),
                        )
                    if len(route) == RUN_ACTION_ROUTE_PARTS and route[2] == "runs":
                        return self._send(
                            200,
                            self.server.experiments.attach(
                                route[1],
                                body.get("run_id"),
                                body.get("role"),
                                body.get("hypothesis", ""),
                                owner=owner,
                            ),
                        )
            if route == ["catalog"] and method == "GET":
                return self._send(200, catalog())
            if route == ["datasets"] and method == "GET":
                return self._send(200, {"datasets": datasets(self.server.store)})
            if route == ["datasets", "selection"] and method == "GET":
                query = parse_qs(parsed.query, keep_blank_values=True, max_num_fields=1)
                if set(query) != {"profile"} or len(query["profile"]) != 1:
                    raise ValueError("Dataset selection requires one profile")
                return self._send(
                    200, self.server.datasets.selection(query["profile"][0])
                )
            if route == ["datasets", "compose"] and method == "POST":
                body = self._body()
                return self._send(
                    200,
                    {
                        "dataset": self.server.datasets.compose(
                            body.get("dataset_ids"), body.get("benchmarks")
                        )
                    },
                )
            if route[0] == "datasets" and method == "GET":
                if len(route) == RUN_ROUTE_PARTS:
                    return self._send(200, self.server.datasets.detail(route[1]))
                if len(route) == RUN_ACTION_ROUTE_PARTS and route[2] == "cases":
                    query = parse_qs(parsed.query, max_num_fields=5)
                    if set(query) - {
                        "cursor",
                        "limit",
                        "benchmark",
                        "category",
                        "q",
                    } or any(len(values) != 1 for values in query.values()):
                        raise ValueError("Invalid dataset page filters")
                    return self._send(
                        200,
                        self.server.datasets.page(
                            route[1],
                            **{key: values[0] for key, values in query.items()},
                        ),
                    )
            if route == ["targets"] and method == "GET":
                return self._send(200, {"targets": self._registry_targets()})
            if route == ["plans"] and method == "POST":
                body = self._body()
                return self._send(
                    200, self._plan(body.get("manifest", body), role, actor)
                )
            if route == ["runs"]:
                if method == "GET":
                    return self._send(
                        200, {"runs": self.server.store.list(owner, summary=True)}
                    )
                body = self._body()
                manifest = body.get("manifest")
                if not manifest:
                    raise ValueError("manifest is required")
                return self._send(
                    201,
                    self.server.engine.start(
                        manifest,
                        actor,
                        body.get("idempotency_key"),
                        actor_role=role,
                        manifest_policy=lambda resolved: self._manifest(resolved, role),
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
                        actor_role=role,
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
                    if action == "candidate-plan" and method == "POST":
                        body = self._body()
                        if set(body) - {"target_ids", "mode", "name", "experiment"}:
                            raise ValueError("Unsupported candidate plan fields")
                        ids = body.get("target_ids")
                        if (
                            not isinstance(ids, list)
                            or not ids
                            or not all(isinstance(i, str) for i in ids)
                            or len(set(ids)) != len(ids)
                        ):
                            raise ValueError("Select distinct configured MoM targets")
                        registry = {t["id"]: t for t in self._registry_targets()}
                        if any(i not in registry for i in ids):
                            raise ValueError("Candidate target is not registered")
                        manifest = candidate_manifest(
                            run,
                            [registry[i] for i in ids],
                            body.get("mode", "live"),
                            body.get("name"),
                            body.get("experiment"),
                        )
                        result = self._plan(manifest, role, actor)
                        validate_candidate_protocol(run, result["manifest"])
                        return self._send(200, result)
                    if action == "reconcile-usage" and method == "POST":
                        return self._send(
                            200, reconcile_usage(self.server.store, run_id)
                        )
                    if action in {"recover-plan", "recover"} and method == "POST":
                        body = self._body()
                        result = (
                            recovery_plan(
                                self.server.store,
                                run_id,
                                body.get("mode", "undispatched"),
                            )
                            if action == "recover-plan"
                            else recover(
                                self.server.engine, run_id, body, actor, actor_role=role
                            )
                        )
                        return self._send(
                            200 if action == "recover-plan" else 201, result
                        )
                    if action in {"regrade", "export"} and method == "POST":
                        return self._send(
                            200,
                            (regrade if action == "regrade" else export_training)(
                                self.server.store, run_id
                            ),
                        )
                    if action in {"results", "calls"} and method == "GET":
                        query = parse_qs(
                            parsed.query, keep_blank_values=True, max_num_fields=3
                        )
                        allowed = (
                            {"after", "limit", "active"}
                            if action == "calls"
                            else {"after", "limit"}
                        )
                        if set(query) - allowed or any(
                            len(values) != 1 for values in query.values()
                        ):
                            raise ValueError("Invalid evidence page filters")
                        active = query.get("active", ["false"])[0]
                        if active not in {"true", "false"}:
                            raise ValueError("active must be true or false")
                        after = int(query.get("after", ["0"])[0])
                        limit = int(query.get("limit", ["100"])[0])
                        return self._send(
                            200,
                            self.server.store.page(
                                run_id, action, after, limit, active=active == "true"
                            ),
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
        except ReviewedPlanChangedError as exc:
            self._send(
                400,
                {
                    "error": str(exc),
                    "code": "reviewed_plan_changed",
                    "dispatch_started": False,
                    "model_requests": 0,
                },
            )
        except ReplayEligibilityError as exc:
            self._send(
                400,
                {
                    "error": str(exc),
                    "code": "replay_ineligible",
                    "reasons": exc.reasons,
                    "model_requests": 0,
                    "dispatch_started": False,
                },
            )
        except RecoveryPlanError as exc:
            self._send(
                400,
                {
                    "error": str(exc),
                    "code": "recovery_plan_required",
                    "dispatch_started": False,
                },
            )
        except ExperimentDeletedError as exc:
            self._send(
                409,
                {
                    "error": str(exc),
                    "code": "experiment_deleted",
                    "experiment_id": exc.identifier,
                },
            )
        except ActiveExperimentError as exc:
            self._send(
                409,
                {
                    "error": str(exc),
                    "code": "experiment_active_runs",
                    "active_run_count": exc.active_run_count,
                },
            )
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

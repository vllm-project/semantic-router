"""Canonical local-container data plane behind the instance control socket.

The manifest is written by ``serve``; requests may select only a mode and an
existing canonical deployment. No client-supplied image, command or URL enters
this owner. Dashboard, storage and external inference backends are never stopped.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import tempfile
import time
import urllib.error
import urllib.request
from http import HTTPStatus
from pathlib import Path

import yaml

from cli.container_management_listener import resolve_managed_management_listener
from cli.instance_state import write_private_json
from cli.management_credential import stack_management_credential
from cli.parser import parse_user_config
from cli.runtime_lifecycle_lock import acquire_runtime_lifecycle_lock
from cli.runtime_stack import RuntimeStackLayout


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def bounded_request(
    url, body=None, headers=None, timeout=30, method=None, response_headers=None
):
    request = urllib.request.Request(
        url, data=body, headers=headers or {}, method=method
    )
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())
    try:
        response = opener.open(request, timeout=timeout)
    except urllib.error.HTTPError as response_error:
        response = response_error
    with response:
        data = response.read((4 << 20) + 1)
        if len(data) > 4 << 20:
            raise ValueError("Runtime response exceeds the response limit")
        if response_headers is not None:
            response_headers.update(response.headers)
            if response.headers.get("ETag"):
                response_headers["ETag"] = response.headers["ETag"]
        return response.status, data


def copy_private(source, target):
    data = Path(source).read_bytes()
    fd, temporary = tempfile.mkstemp(dir=Path(target).parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
    finally:
        Path(temporary).unlink(missing_ok=True)


def restore_file(source, target):
    previous = Path(target).stat()
    fd, temporary = tempfile.mkstemp(dir=Path(target).parent)
    try:
        os.fchmod(fd, previous.st_mode & 0o777)
        os.fchown(fd, previous.st_uid, previous.st_gid)
        with os.fdopen(fd, "wb") as stream:
            stream.write(Path(source).read_bytes())
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
    finally:
        Path(temporary).unlink(missing_ok=True)


class ContainerInstanceBackend:
    """Apply capabilities to one persistent frontend using its canonical API.

    The host controller serializes CLI lifecycle operations. The frontend owns
    config mutation locking and compare-and-swap; never hold that lock across an
    HTTP mutation. No mode request can choose a process, image, path or URL.
    """

    def __init__(self, manifest: dict, directory: Path):
        self.manifest, self.directory = manifest, directory
        self.layout = RuntimeStackLayout(**manifest["layout"])
        self.runtime = manifest["runtime"]
        self.lock = None

    def endpoint(self):
        config = parse_user_config(self.manifest["config"], log_summary=False)
        listener = resolve_managed_management_listener(config, self.layout)
        token = stack_management_credential(
            self.manifest["state_root"], stack_layout=self.layout
        )
        return f"http://127.0.0.1:{listener['host_port']}", {
            "Authorization": f"Bearer {token}"
        }

    def api(self, path, payload=None, *, method=None, etag=None, response_headers=None):
        base, headers = self.endpoint()
        headers["Content-Type"] = "application/json"
        if etag:
            headers["If-Match"] = etag
        return bounded_request(
            base + path,
            json.dumps(payload).encode() if payload is not None else None,
            headers,
            method=method,
            response_headers=response_headers,
        )

    def observe(self):
        try:
            status, body = self.api("/api/v1/instance")
            state = json.loads(body)
            if status == HTTPStatus.OK and state.get("observed_mode") in {
                "router",
                "engine",
            }:
                return {
                    key: state.get(key)
                    for key in ("observed_mode", "active_deployment", "model")
                }
        except (OSError, ValueError, RuntimeError):
            pass
        return {"observed_mode": "unknown", "active_deployment": None, "model": None}

    def acquire(self):
        if self.lock is None:
            self.lock = acquire_runtime_lifecycle_lock(
                runtime=self.runtime, stack_name=self.layout.stack_name
            )

    def release(self):
        if self.lock is not None:
            self.lock.close()
            self.lock = None

    def read_document(self):
        headers = {}
        status, body = self.api("/api/v1/config", response_headers=headers)
        if status != HTTPStatus.OK or not headers.get("ETag"):
            raise RuntimeError("Unable to read canonical configuration")
        document = json.loads(body)
        if not isinstance(document, dict):
            raise ValueError("Canonical configuration must be an object")
        return document, headers["ETag"]

    def prepare(self, mode, deployment, operation):
        manifest_path = self.directory / "manifest.json"
        if manifest_path.exists():
            updated = json.loads(manifest_path.read_text())
            if (
                updated["layout"] != self.manifest["layout"]
                or updated["runtime"] != self.runtime
            ):
                raise ValueError("Instance manifest belongs to a different stack")
            self.manifest = updated
        self.acquire()
        try:
            observed = self.observe()
            if observed["observed_mode"] == "unknown":
                raise RuntimeError("The frontend has no active configuration")
            document, etag = self.read_document()
            candidate = copy.deepcopy(document)
            global_config = candidate.setdefault("global", {})
            global_config.setdefault("router", {})["enabled"] = mode == "router"
            catalog = global_config.setdefault("model_catalog", {})
            binding = catalog.setdefault("system", {}).get("decision_model") or {}
            selected = deployment or binding.get("deployment")
            resource = (catalog.get("deployments") or {}).get(selected)
            if mode == "engine" or deployment is not None:
                if not resource or resource.get("provider") != "model_runtime":
                    raise ValueError("Select a configured model_runtime deployment")
                catalog["system"]["decision_model"] = {"deployment": selected}
            suffix = hashlib.sha256(operation.encode()).hexdigest()[:16]
            previous = self.directory / f"previous-{suffix}.json"
            target = self.directory / f"candidate-{suffix}.json"
            # Payloads may contain credentials, so persist only in private state.
            write_private_json(previous, document)
            write_private_json(target, candidate)
            return {
                "mode": mode,
                "deployment": selected,
                "resource": resource,
                "previous_mode": observed["observed_mode"],
                "previous_deployment": observed.get("active_deployment"),
                "previous_model": observed.get("model"),
                "operation": operation,
                "etag": etag,
                "previous_document": str(previous),
                "candidate_document": str(target),
            }
        except BaseException:
            self.release()
            raise

    def publish(self, document, etag):
        status, body = self.api(
            "/api/v1/config",
            {"yaml": yaml.safe_dump(document, sort_keys=False)},
            method="PUT",
            etag=etag,
        )
        if status != HTTPStatus.OK:
            raise RuntimeError("Frontend rejected the configuration publication")
        result = json.loads(body)
        if result.get("activation_status") == "failed":
            raise RuntimeError("Frontend could not prepare the candidate generation")
        return result

    def activate(self, plan, progress):
        progress("applying")
        self.publish(
            json.loads(Path(plan["candidate_document"]).read_text()), plan["etag"]
        )
        progress("verifying")
        self.wait_ready(
            plan["mode"],
            plan["deployment"],
            (plan.get("resource") or {}).get("artifact"),
        )
        self.release()

    def wait_ready(self, mode, deployment=None, artifact=None):
        deadline = time.monotonic() + self.manifest.get("startup_timeout", 600)
        while time.monotonic() < deadline:
            state = self.observe()
            status, body = self.api("/api/v1/config/hash")
            activation = json.loads(body) if status == HTTPStatus.OK else {}
            if activation.get("activation_status") == "failed":
                raise RuntimeError("Frontend rejected the candidate generation")
            if (
                state["observed_mode"] == mode
                and (deployment is None or state["active_deployment"] == deployment)
                and activation.get("activation_status") == "active"
            ):
                if deployment is None:
                    return
                status, body = self.native("/models")
                cards = (
                    json.loads(body).get("deployments", [])
                    if status == HTTPStatus.OK
                    else []
                )
                if any(
                    card.get("id") == deployment
                    and card.get("ready")
                    and (
                        artifact is None
                        or (card.get("artifact") or card.get("repo")) == artifact
                    )
                    and "decisions" in card.get("surfaces", [])
                    for card in cards
                ):
                    return
            time.sleep(1)
        raise RuntimeError("Instance readiness deadline expired")

    def rollback(self, plan):
        self.acquire()
        try:
            previous = json.loads(Path(plan["previous_document"]).read_text())
            candidate = json.loads(Path(plan["candidate_document"]).read_text())
            current, etag = self.read_document()
            if current != previous:
                # This is the canonical writer's concurrency guard, not a
                # generation/hash invariant. Never overwrite another editor.
                if current != candidate:
                    raise RuntimeError(
                        "Configuration changed during the operation; recovery requires its owner"
                    )
                self.publish(previous, etag)
            self.wait_ready(
                plan["previous_mode"],
                plan.get("previous_deployment"),
                plan.get("previous_model"),
            )
        finally:
            self.release()

    def native(self, path, payload=None):
        if path == "/models":
            return self.api("/api/v1/diagnostics/models/systemone")
        return self.api("/api/v1/diagnostics/models/systemone", payload)

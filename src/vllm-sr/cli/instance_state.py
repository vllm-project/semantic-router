"""Durable operations for one CLI-owned instance, independent of its data plane."""

from __future__ import annotations

import copy
import json
import logging
import os
import re
import tempfile
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

MAX_DEPLOYMENT_LENGTH = 256
TERMINAL_PHASES = frozenset({"ready", "failed"})
REQUEST_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,95}\Z")


def timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_private_json(path: Path, value: Any) -> None:
    """Publish one complete journal record, including across a host restart."""
    fd, name = tempfile.mkstemp(prefix=".instance-", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(value, stream, sort_keys=True)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        Path(name).unlink(missing_ok=True)


class InstanceBackend(Protocol):
    def observe(self) -> dict: ...
    def prepare(self, mode: str, deployment: str | None, operation: str) -> dict: ...
    def activate(self, plan: dict, progress) -> None: ...
    def rollback(self, plan: dict) -> None: ...


class InstanceConflictError(ValueError):
    """An operation is in progress, or a request id names different work."""


class InstanceController:
    """Serializes lifecycle changes; the journal is written before any cutover.

    The backend applies canonical frontend generations with guarded publication.
    This owner never interprets a client path, command, image or endpoint.
    """

    def __init__(self, directory: Path, backend: InstanceBackend):
        self.directory, self.backend = directory, backend
        self.path = directory / "state.json"
        self.lock = threading.RLock()
        self.worker: threading.Thread | None = None
        if self.path.exists():
            self.state = json.loads(self.path.read_text())
        else:
            observed = backend.observe()
            self.state = {
                "desired_mode": observed.get("observed_mode", "unknown"),
                "deployment": observed.get("active_deployment"),
                "operation": None,
            }

    def status(self) -> dict:
        with self.lock:
            result = copy.deepcopy(self.state)
            result.pop("rollback", None)
            result.pop("previous_desired", None)
            result.pop("requests", None)
        result.update(self.backend.observe())
        result.update(
            ownership="managed",
            controller_available=True,
            can_switch="rollback" not in self.state,
        )
        return result

    def submit(self, request: dict) -> dict:
        if set(request) - {"mode", "deployment", "request_id"}:
            raise ValueError("Unknown instance operation field")
        mode, deployment, request_id = (
            request.get("mode"),
            request.get("deployment"),
            request.get("request_id"),
        )
        if mode not in {"router", "engine"}:
            raise ValueError("mode must be router or engine")
        if not isinstance(request_id, str) or not REQUEST_ID.fullmatch(request_id):
            raise ValueError("request_id must be a bounded unique operation identifier")
        if deployment is not None and (
            not isinstance(deployment, str)
            or not deployment
            or len(deployment) > MAX_DEPLOYMENT_LENGTH
        ):
            raise ValueError("deployment must identify a configured model deployment")
        with self.lock:
            completed = self.state.get("requests", {}).get(request_id)
            if completed:
                if (completed["target_mode"], completed.get("deployment")) != (
                    mode,
                    deployment,
                ):
                    raise InstanceConflictError(
                        "request_id already identifies a different operation"
                    )
                result = self.status()
                result["operation"] = copy.deepcopy(completed)
                return result
            previous = self.state.get("operation")
            if previous and previous["id"] == request_id:
                if (previous["target_mode"], previous.get("deployment")) != (
                    mode,
                    deployment,
                ):
                    raise InstanceConflictError(
                        "request_id already identifies a different operation"
                    )
                return self.status()
            if previous and previous["phase"] not in TERMINAL_PHASES:
                raise InstanceConflictError(
                    "An instance operation is already in progress"
                )
            if self.state.get("rollback"):
                raise InstanceConflictError(
                    "Previous deployment recovery requires controller restart"
                )
            self.state["previous_desired"] = {
                "desired_mode": self.state["desired_mode"],
                "deployment": self.state.get("deployment"),
            }
            self.state["operation"] = {
                "id": request_id,
                "target_mode": mode,
                "deployment": deployment,
                "phase": "preparing",
                "started_at": timestamp(),
            }
            self.state["desired_mode"] = mode
            self.state["deployment"] = deployment
            write_private_json(self.path, self.state)
            self.worker = threading.Thread(target=self._apply, daemon=True)
            self.worker.start()
            return self.status()

    def _progress(self, phase: str) -> None:
        with self.lock:
            self.state["operation"]["phase"] = phase
            write_private_json(self.path, self.state)

    def _apply(self) -> None:
        plan = None
        try:
            operation = self.state["operation"]
            plan = self.backend.prepare(
                operation["target_mode"], operation.get("deployment"), operation["id"]
            )
            with self.lock:
                self.state["rollback"] = plan
                write_private_json(self.path, self.state)
            self.backend.activate(plan, self._progress)
            with self.lock:
                self.state.pop("rollback", None)
                self.state.pop("previous_desired", None)
                self.state["operation"].update(phase="ready", finished_at=timestamp())
                self._remember()
                write_private_json(self.path, self.state)
        except (Exception, SystemExit) as error:
            # Do not expose subprocess arguments, model content or credentials.
            logging.error(
                "Instance deployment failed in phase %s (%s)",
                self.state["operation"]["phase"],
                type(error).__name__,
            )
            self._fail(plan, "Deployment failed; inspect the instance controller log")

    def _fail(self, plan: dict | None, error: str) -> None:
        restored = False
        if plan is not None:
            self._progress("rolling_back")
            try:
                self.backend.rollback(plan)
                restored = True
            except (Exception, SystemExit) as failure:
                logging.error("Instance rollback failed (%s)", type(failure).__name__)
                error = "Deployment failed and recovery needs operator attention"
        with self.lock:
            if plan is None:
                self.state.update(self.state.pop("previous_desired", {}))
            if restored:
                self.state["desired_mode"] = plan["previous_mode"]
                self.state["deployment"] = plan.get("previous_deployment")
                self.state.pop("rollback", None)
                self.state.pop("previous_desired", None)
            self.state["operation"].update(
                phase="failed",
                error=error,
                rolled_back=restored,
                finished_at=timestamp(),
            )
            self._remember()
            write_private_json(self.path, self.state)

    def _remember(self):
        operation = self.state["operation"]
        self.state.setdefault("requests", {})[operation["id"]] = copy.deepcopy(
            operation
        )

    def recover(self) -> None:
        """An interrupted cutover restores the retained previous service first."""
        operation = self.state.get("operation")
        if operation and (
            operation["phase"] not in TERMINAL_PHASES or self.state.get("rollback")
        ):
            self._fail(self.state.get("rollback"), "Interrupted deployment recovered")

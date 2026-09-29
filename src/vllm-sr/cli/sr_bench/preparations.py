"""Service-owned, durable dataset preparation jobs shared by every client."""

from __future__ import annotations

import copy
import json
import os
import re
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path

from .preparation_collections import CollectionError, execute_collection, pending_items
from .preparation_runtime import execute
from .sources import COUNTS, HF_SOURCES

JOB_ID = re.compile(r"prep-[0-9a-f]{32}\Z")
ACTIVE = {"queued", "running"}
PROFILES = ("smoke", "quick", "standard")


class PreparationBusyError(ValueError):
    """Only one dependency installer/source downloader owns a service at a time."""


class PreparationUnavailableError(RuntimeError):
    """The service cannot accept work; clients must retry explicitly."""


def timestamp():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        os.chmod(temporary, 0o600)
        json.dump(value, stream, ensure_ascii=False, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def preparation_options():
    from .contracts import BENCHMARKS  # noqa: PLC0415

    return {
        "benchmarks": [
            {
                "id": identifier,
                "name": name,
                "profiles": dict(zip(PROFILES, COUNTS[identifier], strict=True)),
                "source_url": (
                    "https://huggingface.co/datasets/" + HF_SOURCES[identifier][0]
                    if identifier in HF_SOURCES
                    else url
                ),
                "access_note": (
                    "Downloads a pinned source revision. Source access and license terms apply; "
                    "configure HF_TOKEN on the service for gated sources."
                ),
                "dependencies": (
                    ["pyarrow"] if identifier in {"mmlu-pro", "hle"} else []
                ),
            }
            for identifier, name, _, url in BENCHMARKS
            if identifier in COUNTS
        ]
    }


def validate_request(body):
    if not isinstance(body, dict) or set(body) - {
        "benchmark",
        "benchmarks",
        "profile",
        "seed",
        "limit",
    }:
        raise ValueError(
            "Preparation accepts only benchmark or benchmarks, profile, seed and single-benchmark limit"
        )
    collection = "benchmarks" in body
    benchmark, profile = body.get("benchmark"), body.get("profile", "quick")
    if collection:
        names = body["benchmarks"]
        if "benchmark" in body or "limit" in body:
            raise ValueError("Collection preparation cannot include benchmark or limit")
        if (
            not isinstance(names, list)
            or not 1 <= len(names) <= len(COUNTS)
            or any(not isinstance(name, str) or name not in COUNTS for name in names)
            or len(set(names)) != len(names)
        ):
            raise ValueError("Select distinct supported benchmarks")
    elif not isinstance(benchmark, str) or benchmark not in COUNTS:
        raise ValueError("Select a supported benchmark")
    if not isinstance(profile, str) or profile not in PROFILES:
        raise ValueError("Select smoke, quick or standard")
    seed = body.get("seed", 20260918)
    if type(seed) is not int or not 0 <= seed <= 2**53 - 1:
        raise ValueError(
            "seed must be a nonnegative JSON safe integer (at most 2^53 - 1)"
        )
    if collection:
        request = {"benchmarks": sorted(names), "profile": profile}
        if "seed" in body:
            request["seed"] = seed
        return request
    request = {"benchmark": benchmark, "profile": profile, "seed": seed}
    if body.get("limit") is not None:
        limit = body["limit"]
        if (
            type(limit) is not int
            or not 0 < limit <= COUNTS[benchmark][PROFILES.index(profile)]
        ):
            raise ValueError(
                "limit must be a positive integer within the profile budget"
            )
        request["limit"] = limit
    return request


class Preparations:
    def __init__(self, store, executor=None):
        self.store = Path(store).resolve()
        self.root = self.store / "dataset-preparations"
        self.root.mkdir(mode=0o700, exist_ok=True)
        if self.root.is_symlink():
            raise ValueError("Dataset preparation journal must not be a symlink")
        self.lock = threading.RLock()
        self.stopping = threading.Event()
        self.thread = None
        self.executor = executor or execute
        self.jobs = {}
        for path in self.root.glob("prep-*.json"):
            if not JOB_ID.fullmatch(path.stem) or path.is_symlink():
                continue
            job = json.loads(path.read_text())
            if job.get("id") != path.stem:
                raise ValueError("Invalid dataset preparation journal identity")
            if job["status"] in ACTIVE:
                job.update(
                    status="failed",
                    phase="failed",
                    updated_at=timestamp(),
                    error="Service restarted before preparation completed. Retry explicitly.",
                )
                for item in job.get("items", []):
                    if item["status"] != "completed":
                        item.update(
                            status="failed",
                            phase="failed",
                            error_code="service_restarted",
                            error="Service restarted before preparation completed. Retry explicitly.",
                        )
                write_json(path, job)
            self.jobs[job["id"]] = job

    def list(self):
        with self.lock:
            rows = sorted(
                self.jobs.values(), key=lambda row: row["created_at"], reverse=True
            )
            return {"preparations": copy.deepcopy(rows[:100])}

    def get(self, identifier):
        if not JOB_ID.fullmatch(identifier):
            raise KeyError(identifier)
        with self.lock:
            return copy.deepcopy(self.jobs[identifier])

    def submit(self, body):
        request = validate_request(body)
        with self.lock:
            if self.stopping.is_set():
                raise PreparationUnavailableError(
                    "Dataset preparation service is stopping. Retry explicitly."
                )
            for job in reversed(list(self.jobs.values())):
                if job["request"] != request:
                    continue
                if job["status"] in ACTIVE or self._complete_dataset_exists(job):
                    return copy.deepcopy(job)
            if any(job["status"] in ACTIVE for job in self.jobs.values()):
                raise PreparationBusyError(
                    "Another dataset is being prepared; wait for it to finish"
                )
            identifier = "prep-" + uuid.uuid4().hex
            job = {
                "id": identifier,
                **request,
                "request": request,
                "status": "queued",
                "phase": "queued",
                "created_at": timestamp(),
                "updated_at": timestamp(),
            }
            if "benchmarks" in request:
                job.update(items=pending_items(request), model_requests=0)
            write_json(self.root / (identifier + ".json"), job)
            self.jobs[identifier] = job
            self.thread = threading.Thread(
                target=self._run, args=(identifier,), daemon=True
            )
            try:
                self.thread.start()
            except RuntimeError as exc:
                self.thread = None
                self._update(
                    identifier,
                    status="failed",
                    phase="failed",
                    error_code="preparation_unavailable",
                    error="Preparation worker could not start. Retry explicitly.",
                )
                raise PreparationUnavailableError(
                    "Preparation worker could not start. Retry explicitly."
                ) from exc
            return copy.deepcopy(job)

    def _complete_dataset_exists(self, job):
        # Completed collections must re-check today's selection conflicts and
        # content integrity. The next job reuses verified data, not stale proof.
        if "benchmarks" in job["request"]:
            return False
        if job["status"] != "completed":
            return False
        dataset = job.get("dataset", {})
        identifier = dataset.get("id", "")
        if not re.fullmatch(r"[0-9a-f]{64}", identifier):
            return False
        path = self.store / "datasets" / identifier / "manifest.json"
        return path.is_file() and json.loads(path.read_text()) == dataset

    def _update(self, identifier, **changes):
        with self.lock:
            job = self.jobs[identifier]
            job.update(copy.deepcopy(changes), updated_at=timestamp())
            write_json(self.root / (identifier + ".json"), job)

    def _run(self, identifier):
        try:
            self._update(identifier, status="running", phase="checking_dependencies")
            request = self.jobs[identifier]["request"]
            if "benchmarks" in request:
                dataset = execute_collection(
                    request,
                    self.store,
                    self.executor,
                    lambda **changes: self._update(identifier, **changes),
                    self.stopping,
                )
            else:
                dataset = self.executor(
                    request,
                    self.store,
                    lambda phase: self._update(identifier, phase=phase),
                    self.stopping,
                )
            self._update(
                identifier, status="completed", phase="completed", dataset=dataset
            )
        except Exception as exc:
            # Runtime errors are controlled messages; unexpected errors never expose
            # source rows, credentials, filesystem paths, or subprocess output.
            from .preparation_runtime import PreparationError  # noqa: PLC0415

            message = (
                str(exc)
                if isinstance(exc, PreparationError)
                else "Dataset preparation failed. Check the worker and retry explicitly."
            )
            details = (
                {"error_code": exc.code} if isinstance(exc, CollectionError) else {}
            )
            self._update(
                identifier, status="failed", phase="failed", error=message, **details
            )

    def close(self):
        with self.lock:
            self.stopping.set()
            thread = self.thread
        if thread is not None:
            thread.join(timeout=10)

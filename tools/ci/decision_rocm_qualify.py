#!/usr/bin/env python3
"""Produce a six-model ROCm qualification receipt from live Decision runtimes.

This command consumes an already published staging image by immutable digest.
It launches each model through ``drun``, checks real HTTP responses and runtime
telemetry, then writes evidence accepted by ``decision_rocm_promotion.py``.
It does not build, publish, or promote an image.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import socket
import subprocess
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from decision_rocm_promotion import (
    IMAGE,
    MODEL_IDS,
    REVISION,
    SHA256,
    validate_receipt,
    validate_registry_candidate,
)
from decision_runtime.contracts import (
    SystemOneBatchRequest,
    SystemOneBatchResponse,
    SystemOneRequest,
    SystemOneResponse,
    validate_batch_response_for_request,
    validate_response_for_request,
)
from image_artifacts import DECISION_RUNTIME_BASES, source_sha

MODEL_SUFFIXES = (
    "Kai-0.6B",
    "Lex-0.6B",
    "Eos-0.8B",
    "Sol-2B",
    "Nox-4B",
    "Lux-9B",
)
MODEL_NAMES = tuple(
    f"llm-semantic-router/Decision-1.0-{suffix}" for suffix in MODEL_SUFFIXES
)
PHYSICAL_BATCH = 8
MAX_CONCURRENCY = 8
MAX_QUEUE = 32
MAX_ACTIVE_ROWS = 4096
CONCURRENT_REQUESTS = 8
CONCURRENT_QUESTIONS = 8
WIDE_QUESTIONS = 32
WIDE_STATES = 8
_MAX_HTTP_BODY = 4 * 1024 * 1024
_HTTP_OK = 200
_MAX_TCP_PORT = 65535
_MAX_GPU_DEVICE_INDEX = 9999
_MIN_STARTUP_TIMEOUT = 30
_MAX_STARTUP_TIMEOUT = 1800
_MAX_REQUEST_TIMEOUT = 300
_OWNER = re.compile(r"[A-Za-z0-9][A-Za-z0-9-]*\Z")
_METRIC_VALUE = re.compile(r"(?:0|[1-9][0-9]*)(?:\.[0-9]+)?\Z")


class QualificationError(RuntimeError):
    """An actual runtime check failed; no promotable receipt may be written."""


@dataclass(frozen=True, slots=True)
class Options:
    owner: str
    candidate_ref: str
    output_dir: Path
    port: int
    gpu_device: int
    startup_timeout: int = 900
    request_timeout: int = 120


@dataclass(frozen=True, slots=True)
class HTTPObservation:
    status: int
    body: bytes
    elapsed_ms: float


class RuntimeIO(Protocol):
    """The only live process and HTTP seam; tests substitute fake services."""

    def command(self, arguments: list[str], *, timeout: int) -> str: ...

    def http(
        self, method: str, url: str, body: bytes | None, *, timeout: int
    ) -> HTTPObservation: ...


class LiveRuntimeIO:
    """Use the installed CLI and loopback HTTP, without a shell."""

    def command(self, arguments: list[str], *, timeout: int) -> str:
        try:
            result = subprocess.run(
                arguments,
                check=False,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except (OSError, subprocess.TimeoutExpired) as error:
            raise QualificationError(
                "Decision runtime command did not complete"
            ) from error
        if result.returncode != 0:
            raise QualificationError(
                f"Decision runtime command failed with exit code {result.returncode}"
            )
        return result.stdout

    def http(
        self, method: str, url: str, body: bytes | None, *, timeout: int
    ) -> HTTPObservation:
        request = Request(
            url,
            data=body,
            method=method,
            headers={"Content-Type": "application/json"} if body is not None else {},
        )
        started = time.perf_counter()
        try:
            with urlopen(request, timeout=timeout) as response:
                payload = response.read(_MAX_HTTP_BODY + 1)
                status = response.status
        except HTTPError as error:
            raise QualificationError(
                f"Decision HTTP {method} request returned {error.code}"
            ) from error
        except (OSError, URLError) as error:
            raise QualificationError(
                "Decision HTTP request did not complete"
            ) from error
        if len(payload) > _MAX_HTTP_BODY:
            raise QualificationError("Decision HTTP response exceeds evidence limit")
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        if status != _HTTP_OK or not payload or not math.isfinite(elapsed_ms):
            raise QualificationError("Decision HTTP response is unavailable")
        return HTTPObservation(status, payload, elapsed_ms)


class EvidenceWriter:
    """Write only below one fresh output directory and hash exact raw bytes."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.hashes: dict[str, str] = {}
        self._lock = threading.Lock()

    def raw(self, relative: str, payload: bytes) -> None:
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        with self._lock:
            self.hashes[relative] = _digest(path.read_bytes())

    def document(self, relative: str, value: object) -> str:
        encoded = _json_bytes(value)
        self.raw(relative, encoded)
        return self.hashes[relative]


def _digest(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def _validate_options(options: Options) -> None:
    if not _OWNER.fullmatch(options.owner):
        raise QualificationError("qualification owner is invalid")
    expected = (
        f"ghcr.io/{options.owner.lower()}/semantic-router/"
        "decision-runtime-rocm-staging@"
    )
    if not options.candidate_ref.startswith(expected) or not SHA256.fullmatch(
        options.candidate_ref.removeprefix(expected)
    ):
        raise QualificationError("ROCm candidate must be the owner's staging digest")
    if not 1 <= options.port <= _MAX_TCP_PORT:
        raise QualificationError("qualification port is invalid")
    if not 0 <= options.gpu_device <= _MAX_GPU_DEVICE_INDEX:
        raise QualificationError("qualification GPU device is invalid")
    if not _MIN_STARTUP_TIMEOUT <= options.startup_timeout <= _MAX_STARTUP_TIMEOUT:
        raise QualificationError("qualification startup timeout is invalid")
    if not 1 <= options.request_timeout <= _MAX_REQUEST_TIMEOUT:
        raise QualificationError("qualification request timeout is invalid")
    if options.output_dir.exists() or options.output_dir.is_symlink():
        raise QualificationError("qualification output directory already exists")


def _require_free_port(port: int) -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.settimeout(1.0)
        if probe.connect_ex(("127.0.0.1", port)) == 0:
            raise QualificationError("qualification port is already in use")


def _require_clean_source() -> None:
    try:
        dirty = subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=all"], text=True
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise QualificationError(
            "qualification source status is unavailable"
        ) from error
    if dirty:
        raise QualificationError("qualification requires a clean source checkout")


def _parse_launch(
    output: str, *, model: str, port: int, instance: str
) -> tuple[str, str]:
    fields: dict[str, str] = {}
    for line in output.splitlines():
        match = re.fullmatch(
            r"  (Instance|Model|Endpoint|Backend|Artifact|Mode): (.+)", line
        )
        if match:
            fields[match.group(1)] = match.group(2)
    identity = fields.get("Model", "")
    if not identity.startswith(model + "@"):
        raise QualificationError("drun launch reported a different model")
    revision = identity.removeprefix(model + "@")
    artifact = fields.get("Artifact", "")
    if (
        fields.get("Instance") != instance
        or REVISION.fullmatch(revision) is None
        or SHA256.fullmatch(artifact) is None
        or fields.get("Backend", "").split(" ", 1)[0] != "rocm"
        or fields.get("Mode") != "detached"
        or fields.get("Endpoint") != f"http://127.0.0.1:{port}/v1/systemone"
    ):
        raise QualificationError("drun launch identity is incomplete or inconsistent")
    return revision, artifact


def _read_json(observation: HTTPObservation, *, label: str) -> dict:
    if observation.status != _HTTP_OK:
        raise QualificationError(f"{label} response was not successful")
    try:
        value = json.loads(observation.body)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise QualificationError(f"{label} response is not JSON") from error
    if not isinstance(value, dict):
        raise QualificationError(f"{label} response is not an object")
    return value


def _check_status(status: dict, *, model: str, revision: str, artifact: str) -> None:
    provenance = status.get("artifact")
    scheduler = status.get("scheduler")
    if (
        status.get("status") != "ready"
        or status.get("models") != [model]
        or not isinstance(provenance, dict)
        or provenance.get("model") != model
        or provenance.get("revision") != revision
        or provenance.get("content_sha256") != artifact.removeprefix("sha256:")
        or not isinstance(provenance.get("manifest_sha256"), str)
        or SHA256.fullmatch("sha256:" + provenance["manifest_sha256"]) is None
        or not isinstance(scheduler, list)
        or len(scheduler) != 1
        or not isinstance(scheduler[0], dict)
        or scheduler[0].get("model") != model
        or scheduler[0].get("max_concurrency") != MAX_CONCURRENCY
        or scheduler[0].get("max_queue") != MAX_QUEUE
        or scheduler[0].get("max_active_rows") != MAX_ACTIVE_ROWS
    ):
        raise QualificationError("live status does not attest the launched artifact")


def _questions(count: int) -> dict[str, dict]:
    examples = (
        {
            "type": "noul",
            "instructions": "Does the state request a refund?",
        },
        {
            "type": "choice",
            "instructions": "Classify the customer request.",
            "criteria": {
                "billing": "Payment, invoice, or refund",
                "technical": "Product error or malfunction",
            },
        },
        {
            "type": "score",
            "instructions": "Rate urgency using this ordered rubric.",
            "criteria": ["routine", "important", "urgent"],
        },
    )
    return {
        f"q{index:02d}": {**examples[index % len(examples)]} for index in range(count)
    }


def _single(model: str, count: int, *, sequence: int = 0) -> dict:
    return {
        "model": model,
        "state": (
            f"Synthetic request {sequence}: I was charged twice. "
            "Please refund the duplicate payment."
        ),
        "questions": _questions(count),
    }


def _batch(model: str, *, states: int, questions: int) -> dict:
    return {
        "model": model,
        "states": [
            {
                "id": f"state-{index}",
                "state": (
                    f"Synthetic state {index}: payment duplicated; "
                    "please refund the extra charge."
                ),
            }
            for index in range(states)
        ],
        "questions": _questions(questions),
    }


def _post_validated(
    io: RuntimeIO,
    writer: EvidenceWriter,
    *,
    base_url: str,
    relative: str,
    payload: dict,
    batch: bool,
    timeout: int,
) -> float:
    request = (
        SystemOneBatchRequest.model_validate(payload)
        if batch
        else SystemOneRequest.model_validate(payload)
    )
    path = "/v1/decision/batches" if batch else "/v1/systemone"
    raw_request = _json_bytes(payload)
    writer.raw(f"{relative}-request.json", raw_request)
    observation = io.http("POST", base_url + path, raw_request, timeout=timeout)
    writer.raw(f"{relative}-response.json", observation.body)
    if observation.status != _HTTP_OK:
        raise QualificationError("Decision HTTP answer was not successful")
    try:
        if batch:
            response = SystemOneBatchResponse.model_validate_json(observation.body)
            validate_batch_response_for_request(request, response)
        else:
            response = SystemOneResponse.model_validate_json(observation.body)
            validate_response_for_request(request, response)
    except ValueError as error:
        raise QualificationError(
            "Decision HTTP answer violated its request contract"
        ) from error
    if not math.isfinite(observation.elapsed_ms) or observation.elapsed_ms <= 0:
        raise QualificationError("Decision HTTP latency is invalid")
    return observation.elapsed_ms


def _metric(body: bytes, name: str, model: str) -> float:
    prefix = f'{name}{{model="{model}"}} '
    for line in body.decode("utf-8").splitlines():
        if line.startswith(prefix):
            value = line.removeprefix(prefix).strip()
            if _METRIC_VALUE.fullmatch(value) is None:
                break
            parsed = float(value)
            if math.isfinite(parsed):
                return parsed
            break
    raise QualificationError(f"runtime metric {name} is missing or invalid")


def _percentile(values: list[float], percent: float) -> float:
    ordered = sorted(values)
    return ordered[max(0, math.ceil(len(ordered) * percent) - 1)]


def _concurrency_probe(
    io: RuntimeIO,
    writer: EvidenceWriter,
    *,
    base_url: str,
    model: str,
    relative: str,
    timeout: int,
) -> dict[str, float | int]:
    barrier = threading.Barrier(CONCURRENT_REQUESTS)
    started = time.perf_counter()

    def one(index: int) -> float:
        try:
            barrier.wait(timeout=10)
        except threading.BrokenBarrierError as error:
            raise QualificationError(
                "concurrent Decision workload did not start"
            ) from error
        return _post_validated(
            io,
            writer,
            base_url=base_url,
            relative=f"{relative}/concurrent-{index:02d}",
            payload=_single(model, CONCURRENT_QUESTIONS, sequence=index + 1),
            batch=False,
            timeout=timeout,
        )

    with ThreadPoolExecutor(max_workers=CONCURRENT_REQUESTS) as pool:
        latencies = list(pool.map(one, range(CONCURRENT_REQUESTS)))
    wall_seconds = time.perf_counter() - started
    if not math.isfinite(wall_seconds) or wall_seconds <= 0:
        raise QualificationError("concurrent Decision window is invalid")
    decisions = CONCURRENT_REQUESTS * CONCURRENT_QUESTIONS
    return {
        "requests": CONCURRENT_REQUESTS,
        "questions_per_request": CONCURRENT_QUESTIONS,
        "decisions": decisions,
        "wall_ms": wall_seconds * 1000.0,
        "p50_ms": _percentile(latencies, 0.50),
        "p95_ms": _percentile(latencies, 0.95),
        "decisions_per_second": decisions / wall_seconds,
    }


def qualify_model(
    io: RuntimeIO,
    writer: EvidenceWriter,
    *,
    options: Options,
    model: str,
    source_revision: str,
    revision: str,
    artifact: str,
    slug: str,
) -> dict:
    """Observe one live model; only completed checks become success evidence."""

    base_url = f"http://127.0.0.1:{options.port}"
    relative = f"raw/{slug}"
    ready = io.http("GET", base_url + "/ready", None, timeout=options.request_timeout)
    writer.raw(f"{relative}/ready.json", ready.body)
    if _read_json(ready, label="ready") != {"ready": True}:
        raise QualificationError("Decision runtime did not report ready")
    status_response = io.http(
        "GET", base_url + "/api/status", None, timeout=options.request_timeout
    )
    writer.raw(f"{relative}/status.json", status_response.body)
    status = _read_json(status_response, label="status")
    _check_status(status, model=model, revision=revision, artifact=artifact)

    single_ms = _post_validated(
        io,
        writer,
        base_url=base_url,
        relative=f"{relative}/mixed-single",
        payload=_single(model, 3),
        batch=False,
        timeout=options.request_timeout,
    )
    batch_ms = _post_validated(
        io,
        writer,
        base_url=base_url,
        relative=f"{relative}/two-state-batch",
        payload=_batch(model, states=2, questions=3),
        batch=True,
        timeout=options.request_timeout,
    )
    wide_ms = _post_validated(
        io,
        writer,
        base_url=base_url,
        relative=f"{relative}/wide-single",
        payload=_single(model, WIDE_QUESTIONS),
        batch=False,
        timeout=options.request_timeout,
    )
    wide_batch_ms = _post_validated(
        io,
        writer,
        base_url=base_url,
        relative=f"{relative}/wide-batch",
        payload=_batch(model, states=WIDE_STATES, questions=CONCURRENT_QUESTIONS),
        batch=True,
        timeout=options.request_timeout,
    )
    before = io.http(
        "GET", base_url + "/metrics", None, timeout=options.request_timeout
    )
    writer.raw(f"{relative}/metrics-before.txt", before.body)
    performance = _concurrency_probe(
        io,
        writer,
        base_url=base_url,
        model=model,
        relative=relative,
        timeout=options.request_timeout,
    )
    after = io.http("GET", base_url + "/metrics", None, timeout=options.request_timeout)
    writer.raw(f"{relative}/metrics-after.txt", after.body)
    batches = _metric(
        after.body, "decision_runtime_physical_batches_total", model
    ) - _metric(before.body, "decision_runtime_physical_batches_total", model)
    rows = _metric(
        after.body, "decision_runtime_physical_batch_rows_total", model
    ) - _metric(before.body, "decision_runtime_physical_batch_rows_total", model)
    seconds = _metric(
        after.body, "decision_runtime_physical_batch_duration_seconds_total", model
    ) - _metric(
        before.body, "decision_runtime_physical_batch_duration_seconds_total", model
    )
    if (
        not batches.is_integer()
        or not rows.is_integer()
        or batches < 1
        or batches >= rows
        or rows != performance["decisions"]
        or seconds <= 0
    ):
        raise QualificationError(
            "ROCm physical forward telemetry did not cover the workload"
        )
    performance.update(
        {
            "physical_batches": int(batches),
            "physical_rows": int(rows),
            "physical_forward_ms": seconds * 1000.0,
            "rows_per_physical_batch": rows / batches,
        }
    )
    return {
        "image_ref": options.candidate_ref,
        "source_sha": source_revision,
        "model_id": model,
        "revision": revision,
        "artifact_content_id": artifact,
        "backend": "rocm",
        "device": "rocm",
        "result": "passed",
        "checks": {
            "artifact.identity": True,
            "health": True,
            "single.noul": True,
            "single.choice": True,
            "single.score": True,
            "batch.mixed_two_states": True,
            "rocm.execution": True,
            "single.mixed_32_questions": True,
            "batch.mixed_eight_states": True,
            "concurrency.eight_requests": True,
        },
        "latency_ms": {
            "mixed_single": single_ms,
            "two_state_batch": batch_ms,
            "wide_single": wide_ms,
            "wide_batch": wide_batch_ms,
        },
        "performance": performance,
        "raw_sha256": {
            path: digest
            for path, digest in sorted(writer.hashes.items())
            if path.startswith(relative + "/")
        },
    }


def _qualify_managed_model(
    io: RuntimeIO,
    writer: EvidenceWriter,
    options: Options,
    model: str,
    source_revision: str,
) -> dict:
    slug = model.rsplit("/", 1)[-1].removeprefix("Decision-1.0-").lower()
    _require_free_port(options.port)
    instance = f"qualify-{slug}-{uuid.uuid4().hex[:12]}"
    arguments = [
        "vllm-sr",
        "drun",
        "run",
        model,
        "--backend",
        "rocm",
        "--image",
        options.candidate_ref,
        "--host",
        "127.0.0.1",
        "--port",
        str(options.port),
        "--gpu-device",
        str(options.gpu_device),
        "--max-batch",
        str(PHYSICAL_BATCH),
        "--max-concurrency",
        str(MAX_CONCURRENCY),
        "--max-queue",
        str(MAX_QUEUE),
        "--instance-name",
        instance,
        "--startup-timeout",
        str(options.startup_timeout),
        "--detach",
    ]
    launch_attempted = False
    failure: BaseException | None = None
    try:
        launch_attempted = True
        output = io.command(arguments, timeout=options.startup_timeout + 60)
        writer.raw(f"raw/{slug}/drun-launch.txt", output.encode())
        revision, artifact = _parse_launch(
            output, model=model, port=options.port, instance=instance
        )
        evidence = qualify_model(
            io,
            writer,
            options=options,
            model=model,
            source_revision=source_revision,
            revision=revision,
            artifact=artifact,
            slug=slug,
        )
    except BaseException as error:
        failure = error
        raise
    finally:
        if launch_attempted:
            try:
                # The CLI can time out after creating an owned container.
                io.command(["vllm-sr", "drun", "stop", instance], timeout=60)
            except Exception:
                if failure is None:
                    raise
    return evidence


def qualify_all(
    options: Options,
    *,
    io: RuntimeIO | None = None,
    inspect_candidate=validate_registry_candidate,
) -> Path:
    """Qualify six live models and atomically finish one promotion receipt."""

    _validate_options(options)
    _require_clean_source()
    revision = source_sha()
    base_image, _, _ = DECISION_RUNTIME_BASES[IMAGE]
    receipt: dict = {
        "schema": 1,
        "image": IMAGE,
        "source_sha": revision,
        "base_image": base_image,
        "candidate_ref": options.candidate_ref,
        "backend": "rocm",
        "platform": "linux/amd64",
        "models": [],
    }
    if set(MODEL_NAMES) != MODEL_IDS:
        raise QualificationError("qualification model catalog differs from promotion")
    inspect_candidate(receipt)
    _require_free_port(options.port)
    options.output_dir.mkdir(parents=True)
    writer = EvidenceWriter(options.output_dir)
    runtime_io = io or LiveRuntimeIO()
    for model in MODEL_NAMES:
        evidence = _qualify_managed_model(runtime_io, writer, options, model, revision)
        slug = model.rsplit("/", 1)[-1].removeprefix("Decision-1.0-").lower()
        relative = f"models/{slug}.json"
        evidence_digest = writer.document(relative, evidence)
        receipt["models"].append(
            {
                "id": model,
                "revision": evidence["revision"],
                "artifact_content_id": evidence["artifact_content_id"],
                "evidence_file": relative,
                "evidence_sha256": evidence_digest,
            }
        )
    for relative, digest in writer.hashes.items():
        if _digest((options.output_dir / relative).read_bytes()) != digest:
            raise QualificationError("raw qualification evidence changed on disk")
    pending_path = options.output_dir / "qualification.pending.json"
    pending_path.write_bytes(_json_bytes(receipt))
    validate_receipt(pending_path, owner=options.owner, revision=revision)
    receipt_path = options.output_dir / "qualification.json"
    pending_path.replace(receipt_path)
    return receipt_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--owner", required=True)
    parser.add_argument("--candidate-ref", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--port", required=True, type=int)
    parser.add_argument("--gpu-device", required=True, type=int)
    parser.add_argument("--startup-timeout", type=int, default=900)
    parser.add_argument("--request-timeout", type=int, default=120)
    args = parser.parse_args()
    try:
        path = qualify_all(Options(**vars(args)))
    except (QualificationError, ValueError, OSError) as error:
        parser.exit(1, f"ROCm qualification failed: {error}\n")
    print(path)


if __name__ == "__main__":
    main()

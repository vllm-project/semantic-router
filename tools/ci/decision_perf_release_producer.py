#!/usr/bin/env python3
"""Measure protected, live old/new Decision services and produce gate evidence.

The protected baseline document identifies already running, isolated loopback
services. This command verifies their immutable container images, performs an
independent semantic preflight, then runs the tracked benchmark for every model
and workload shape. It never starts an unreviewed old runtime or invents a
performance result when a protected service is unavailable.
"""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import stat
import subprocess
import sys
from pathlib import Path
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, ProxyHandler, build_opener

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from cli.decision_runtime.catalog_adapter import default_artifact_cache_root
from decision_perf_release_gate import (
    ARRIVAL_POLICY,
    CONCURRENCIES,
    MIN_WORKFLOWS_PER_ROUND,
    PHYSICAL_BATCH,
    ROUNDS,
    SCHEMA,
    SHAPES,
    _read_json,
    _reject_nonfinite,
    _unique_pairs,
    validate_report,
)
from decision_rocm_promotion import MODEL_IDS, validate_receipt

from bench.decision_runtime import semantic_audit
from bench.decision_runtime.semantic_audit import audit_cohorts
from bench.decision_runtime.semantic_cases import generate_cases
from bench.decision_runtime.semantic_metrics import validate_metrics_url
from bench.decision_runtime.semantic_runner import _harness_digest
from bench.decision_runtime.semantic_transport import batch_url
from bench.decision_runtime.transport import Endpoint, validate_endpoint_url
from decision_runtime.artifacts import ArtifactError, open_verified_artifact
from decision_runtime.catalog_adapter import resolve_decision_runtime_model

SHA = re.compile(r"[0-9a-f]{40}\Z")
HASH = re.compile(r"[0-9a-f]{64}\Z")
IMAGE_REF = re.compile(r"[^\s@]+@sha256:[0-9a-f]{64}\Z")
IMAGE_ID = re.compile(r"sha256:[0-9a-f]{64}\Z")
CONTAINER_ID = re.compile(r"[0-9a-f]{64}\Z")
ENV_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")
GPU_INDEX = re.compile(r"(?:0|[1-9][0-9]{0,3})\Z")
MODEL_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._/-]{0,127}\Z")
ARGUMENT_FLAG = re.compile(r"--[A-Za-z][A-Za-z0-9-]*\Z")
MAX_STATUS_BYTES = 1024 * 1024
NEW_MAX_CONCURRENCY = 4
NEW_MAX_QUEUE = 32
RAW_FILES = (
    "receipt.json",
    "audit.jsonl",
    "samples.jsonl",
    "workflows.jsonl",
    "metrics.jsonl",
)


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, request, fp, code, msg, headers, newurl):
        return None


LOOPBACK_OPENER = build_opener(ProxyHandler({}), _NoRedirect())
semantic_audit.OPENER = LOOPBACK_OPENER


class ProducerError(ValueError):
    """A protected identity or live benchmark prerequisite is not satisfied."""


def _require(value: object, pattern: re.Pattern[str], label: str) -> str:
    if not isinstance(value, str) or not pattern.fullmatch(value):
        raise ProducerError(f"{label} is invalid")
    return value


def _json(path: Path) -> dict:
    return _read_json(path)


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, value: object) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return _digest(path)


def _loopback(value: object, label: str, *, path: str) -> str:
    if not isinstance(value, str):
        raise ProducerError(f"{label} must be a loopback URL")
    validated = (
        validate_endpoint_url(value)
        if path == "/v1/systemone"
        else validate_metrics_url(value)
    )
    parsed = urlsplit(validated)
    if (
        parsed.scheme != "http"
        or parsed.hostname != "127.0.0.1"
        or parsed.port is None
        or parsed.path != path
        or parsed.query
        or parsed.fragment
    ):
        raise ProducerError(f"{label} must be an exact loopback {path} URL")
    return validated


def _docker_inspect(kind: str, identity: str) -> dict:
    result = subprocess.run(
        ["docker", "inspect", "--type", kind, identity],
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    if result.returncode or len(result.stdout) > 16 * 1024 * 1024:
        raise ProducerError(f"protected {kind} identity is unavailable")
    rows = json.loads(
        result.stdout, object_pairs_hook=_unique_pairs, parse_constant=_reject_nonfinite
    )
    if not isinstance(rows, list) or len(rows) != 1 or not isinstance(rows[0], dict):
        raise ProducerError(f"protected {kind} inspection is invalid")
    return rows[0]


def _mount_digest(source: Path) -> str:
    """Hash an exact regular file or an ordered, symlink-free directory tree."""

    if source.is_symlink():
        raise ProducerError("protected source mount uses a symlink")
    if source.is_file():
        return _digest(source)
    if not source.is_dir():
        raise ProducerError("protected source mount is not a regular tree")
    digest = hashlib.sha256(b"decision-mounted-tree-v1\0")
    for path in sorted(source.rglob("*")):
        relative = path.relative_to(source).as_posix().encode("utf-8")
        if path.is_symlink():
            raise ProducerError("protected source tree contains a symlink")
        mode = path.stat(follow_symlinks=False).st_mode
        if stat.S_ISDIR(mode):
            digest.update(b"D\0" + relative + b"\0")
        elif stat.S_ISREG(mode):
            digest.update(b"F\0" + relative + b"\0")
            with path.open("rb") as handle:
                while chunk := handle.read(1024 * 1024):
                    digest.update(chunk)
            digest.update(b"\0")
        else:
            raise ProducerError("protected source tree contains a special file")
    return digest.hexdigest()


def _old_launch_locator(
    container: dict, locator: dict, destination: str, label: str
) -> None:
    """Require the declared old Docker launch to name a verified mount."""

    config = container.get("Config") or {}
    kind = locator["kind"]
    if kind in ("argument", "command_path"):
        entrypoint = config.get("Entrypoint") or []
        command = config.get("Cmd") or []
        if (
            not isinstance(entrypoint, list)
            or not isinstance(command, list)
            or any(not isinstance(item, str) for item in entrypoint + command)
        ):
            raise ProducerError("old service launch arguments are invalid")
        words = entrypoint + command
        if kind == "argument":
            flag = locator["flag"]
            found = any(
                word == flag
                and index + 1 < len(words)
                and words[index + 1] == destination
                or word == f"{flag}={destination}"
                for index, word in enumerate(words)
            )
        else:
            found = locator["path"] in words
    elif kind == "environment":
        environment = config.get("Env") or []
        name = locator["name"]
        found = (
            isinstance(environment, list)
            and sum(
                item == f"{name}={destination}"
                for item in environment
                if isinstance(item, str)
            )
            == 1
            and not any(
                isinstance(item, str)
                and item.startswith(f"{name}=")
                and item != f"{name}={destination}"
                for item in environment
            )
        )
    else:
        found = config.get("WorkingDir") == destination
    if not found:
        raise ProducerError(f"old service launch does not name its {label} mount")


def _validate_locator(locator: object, *, label: str, destination: str) -> dict:
    if not isinstance(locator, dict):
        raise ProducerError(f"old {label} launch locator is missing")
    kind = locator.get("kind")
    if label == "core" and kind != "command_path":
        raise ProducerError("old core must be the running process script or executable")
    if kind == "argument":
        _require(locator.get("flag"), ARGUMENT_FLAG, f"{label} argument")
    elif kind == "environment":
        _require(locator.get("name"), ENV_NAME, f"{label} environment")
    elif kind == "command_path" and label == "core":
        path = locator.get("path")
        if (
            not isinstance(path, str)
            or not path.startswith(destination.rstrip("/") + "/")
            and path != destination
            or path != os.path.normpath(path)
        ):
            raise ProducerError("old core command path is outside its mount")
    elif kind != "working_dir":
        raise ProducerError(f"old {label} launch locator is invalid")
    return locator


def _running_command(container: dict) -> list[str]:
    """Read only the live container-init command; never expose its arguments."""

    pid = container.get("State", {}).get("Pid")
    if type(pid) is not int or pid < 1:
        raise ProducerError("old container process identity is unavailable")
    try:
        with (Path("/proc") / str(pid) / "cmdline").open("rb") as handle:
            payload = handle.read(65537)
        if len(payload) > 65536 or not payload.endswith(b"\0"):
            raise ProducerError("old container process command is unavailable")
        words = [word.decode("utf-8") for word in payload.split(b"\0")[:-1]]
    except (OSError, UnicodeError) as error:
        raise ProducerError("old container process command is unavailable") from error
    if not words or any(not word for word in words):
        raise ProducerError("old container process command is invalid")
    return words


def _old_core_process(container: dict, core_path: str) -> None:
    """Require PID 1 to execute the mounted source, not merely mention it."""

    config = container.get("Config") or {}
    entrypoint = config.get("Entrypoint") or []
    command = config.get("Cmd") or []
    if (
        not isinstance(entrypoint, list)
        or not isinstance(command, list)
        or any(not isinstance(word, str) for word in entrypoint + command)
    ):
        raise ProducerError("old core launch command is invalid")
    declared = entrypoint + command
    running = _running_command(container)
    if declared != running:
        raise ProducerError("old running process differs from its declared launch")
    if running[0] == core_path:
        return
    interpreter = Path(running[0]).name
    if (
        interpreter in {"python", "python3", "python3.11", "python3.12", "python3.13"}
        and len(running) > 1
        and running[1] == core_path
    ):
        return
    raise ProducerError("old mounted core is not the executed process source")


def _declared_launch_batch(container: dict) -> int:
    config = container.get("Config") or {}
    entrypoint = config.get("Entrypoint") or []
    command = config.get("Cmd") or []
    if (
        not isinstance(entrypoint, list)
        or not isinstance(command, list)
        or any(not isinstance(word, str) for word in entrypoint + command)
    ):
        raise ProducerError("candidate launch command is invalid")
    words = entrypoint + command
    values = []
    for index, word in enumerate(words):
        if word == "--max-batch" and index + 1 < len(words):
            values.append(words[index + 1])
        elif word.startswith("--max-batch="):
            values.append(word.removeprefix("--max-batch="))
    if len(values) != 1 or not values[0].isdecimal():
        raise ProducerError("candidate launch batch size is ambiguous or missing")
    return int(values[0])


def _candidate_process(
    container: dict,
    image: dict,
    *,
    model_id: str,
    revision: str,
    artifact_content_id: str,
    physical_batch_size: int,
    gpu_device: str,
) -> None:
    """Bind the timed candidate to drun's exact, live, image-owned command."""

    model = resolve_decision_runtime_model(model_id, revision=revision, backend="rocm")
    python = {
        "vela": "/opt/vllm-sr/venvs/vela/bin/python",
        "qwen3.5": "/opt/vllm-sr/venvs/qwen35/bin/python",
    }.get(model.profile.family)
    if python is None:
        raise ProducerError("candidate model family has no approved entrypoint")
    expected = [
        python,
        "-m",
        "decision_runtime.entrypoint",
        "--model",
        model_id,
        "--revision",
        revision,
        "--backend",
        "rocm",
        "--artifact-root",
        "/opt/vllm-sr/decision-artifact",
        "--artifact-content-id",
        artifact_content_id,
        "--host",
        "0.0.0.0",
        "--port",
        "8000",
        "--max-batch",
        str(physical_batch_size),
        "--max-concurrency",
        str(NEW_MAX_CONCURRENCY),
        "--max-queue",
        str(NEW_MAX_QUEUE),
    ]
    image_config = image.get("Config") or {}
    config = container.get("Config") or {}
    if (
        image_config.get("Entrypoint") not in (None, [])
        or config.get("Entrypoint") not in (None, [])
        or config.get("Cmd") != expected
        or config.get("WorkingDir") != image_config.get("WorkingDir")
        or config.get("User") != image_config.get("User")
        or _running_command(container) != expected
    ):
        raise ProducerError(
            "candidate live process is not the approved drun entrypoint"
        )

    def environment(value: object) -> dict[str, str]:
        if not isinstance(value, list):
            raise ProducerError("candidate environment is invalid")
        result: dict[str, str] = {}
        for item in value:
            if not isinstance(item, str) or "=" not in item:
                raise ProducerError("candidate environment is invalid")
            name, content = item.split("=", 1)
            if not name or name in result:
                raise ProducerError("candidate environment is ambiguous")
            result[name] = content
        return result

    expected_env = environment(image_config.get("Env") or [])
    expected_env.update(
        {"TOKENIZERS_PARALLELISM": "false", "ROCR_VISIBLE_DEVICES": gpu_device}
    )
    if environment(config.get("Env") or []) != expected_env:
        raise ProducerError("candidate environment differs from the approved image")


def _verified_old_artifact(source: Path, row: dict) -> None:
    try:
        model = resolve_decision_runtime_model(
            row["model_id"], revision=row["revision"], backend="rocm"
        )
        artifact = open_verified_artifact(
            source, model, expected_content_id=row["artifact_content_id"]
        )
    except (ArtifactError, ValueError, OSError) as error:
        raise ProducerError(
            "old artifact mount failed full content verification"
        ) from error
    if (
        artifact.repository_id != row["model_id"]
        or artifact.revision != row["revision"]
        or artifact.content_id != row["artifact_content_id"]
        or artifact.manifest is None
        or artifact.manifest.sha256 != row["artifact_manifest_sha256"]
    ):
        raise ProducerError("old artifact mount differs from qualified model snapshot")


def _container_image(
    arm: dict,
    *,
    source_sha: str | None,
    gpu_device: str,
    old_core_sha256: str | None = None,
    old_overlay: str | None = None,
    old_artifact: dict | None = None,
    new_artifact_content_id: str | None = None,
    new_physical_batch_size: int | None = None,
    new_model_id: str | None = None,
    new_model_revision: str | None = None,
) -> str:
    ref = arm.get("image_ref")
    if source_sha is None and isinstance(ref, str) and IMAGE_ID.fullmatch(ref):
        pass  # A protected old image may be immutable and local-only.
    else:
        ref = _require(ref, IMAGE_REF, "immutable image reference")
    container_id = _require(arm.get("container_id"), CONTAINER_ID, "container ID")
    image = _docker_inspect("image", ref)
    image_id = _require(image.get("Id"), IMAGE_ID, "local image ID")
    labels = image.get("Config", {}).get("Labels") or {}
    if source_sha is not None and (
        not isinstance(labels, dict)
        or labels.get("ai.vllm-sr.decision.backend") != "rocm"
        or labels.get("ai.vllm-sr.decision.source-state") != "clean"
        or labels.get("org.opencontainers.image.revision") != source_sha
    ):
        raise ProducerError("protected image source or backend labels disagree")
    container = _docker_inspect("container", container_id)
    if (
        container.get("Id") != container_id
        or container.get("Image") != image_id
        or container.get("State", {}).get("Running") is not True
    ):
        raise ProducerError("protected service is not running the declared image")
    environment = container.get("Config", {}).get("Env") or []
    if not isinstance(environment, list):
        raise ProducerError("protected GPU environment is invalid")
    visible = [
        item.removeprefix("ROCR_VISIBLE_DEVICES=")
        for item in environment
        if isinstance(item, str) and item.startswith("ROCR_VISIBLE_DEVICES=")
    ]
    if visible != [gpu_device]:
        raise ProducerError("protected service does not select the same ROCm device")
    for selector in ("HIP_VISIBLE_DEVICES=", "GPU_DEVICE_ORDINAL="):
        conflicting = [
            item.removeprefix(selector)
            for item in environment
            if isinstance(item, str) and item.startswith(selector)
        ]
        if conflicting and conflicting != [gpu_device]:
            raise ProducerError("protected service has conflicting GPU visibility")
    device_bindings = container.get("HostConfig", {}).get("Devices") or []
    if not isinstance(device_bindings, list) or {
        (device.get("PathOnHost"), device.get("PathInContainer"))
        for device in device_bindings
        if isinstance(device, dict)
    } != {("/dev/kfd", "/dev/kfd"), ("/dev/dri", "/dev/dri")}:
        raise ProducerError("protected service has a different ROCm device mapping")
    if old_core_sha256 is not None:
        if old_artifact is None:
            raise ProducerError("old artifact attestation is required")
        mounts = container.get("Mounts") or []
        declared = arm.get("mounts", [])
        if (
            not isinstance(mounts, list)
            or not isinstance(declared, list)
            or len(mounts) != len(declared)
        ):
            raise ProducerError(
                "old service mount inventory differs from protected baseline"
            )
        expected = {
            item.get("destination"): item.get("sha256")
            for item in declared
            if isinstance(item, dict)
        }
        if len(expected) != len(declared):
            raise ProducerError("old service mount declarations are invalid")
        observed = set()
        artifact_source = None
        core_destination = arm.get("core_mount_destination")
        for mount in mounts:
            if (
                not isinstance(mount, dict)
                or mount.get("Type") != "bind"
                or mount.get("RW") is not False
                or not isinstance(mount.get("Destination"), str)
                or mount["Destination"] not in expected
            ):
                raise ProducerError("old service has an unattested or writable mount")
            actual = _mount_digest(Path(mount["Source"]))
            if actual != expected[mount["Destination"]]:
                raise ProducerError("old mounted source changed from protected digest")
            observed.add(actual)
            if (
                old_artifact is not None
                and mount["Destination"] == arm["artifact_mount_destination"]
            ):
                artifact_source = Path(mount["Source"])
        if arm.get("core_source_kind") == "mounted":
            if (
                core_destination == arm["artifact_mount_destination"]
                or expected.get(core_destination) != old_core_sha256
                or old_core_sha256 not in observed
            ):
                raise ProducerError("old core mount is not the declared source")
            _old_core_process(container, arm["core_locator"]["path"])
        elif arm.get("core_source_kind") == "baked":
            raise ProducerError(
                "old baked core requires independent protected process/source attestation"
            )
        else:
            raise ProducerError("old core source kind is invalid")
        if (
            old_overlay not in (None, "none")
            and old_overlay.removeprefix("sha256:") not in observed
        ):
            raise ProducerError(
                "old source overlay is not attested by a mounted digest"
            )
        if old_artifact is not None:
            if artifact_source is None:
                raise ProducerError("old artifact mount is missing")
            _verified_old_artifact(artifact_source, old_artifact)
            _old_launch_locator(
                container,
                arm["artifact_locator"],
                arm["artifact_mount_destination"],
                "artifact",
            )
    if new_artifact_content_id is not None:
        if _declared_launch_batch(container) != new_physical_batch_size:
            raise ProducerError(
                "candidate running batch size differs from protected config"
            )
        if (
            source_sha is None
            or new_model_id is None
            or new_model_revision is None
            or new_physical_batch_size is None
        ):
            raise ProducerError("candidate model identity is incomplete")
        _candidate_process(
            container,
            image,
            model_id=new_model_id,
            revision=new_model_revision,
            artifact_content_id=new_artifact_content_id,
            physical_batch_size=new_physical_batch_size,
            gpu_device=gpu_device,
        )
        mounts = container.get("Mounts") or []
        expected_source = (
            default_artifact_cache_root() / "sha256" / new_artifact_content_id
        )
        if (
            not isinstance(mounts, list)
            or len(mounts) != 1
            or mounts[0].get("Type") != "bind"
            or mounts[0].get("RW") is not False
            or mounts[0].get("Destination") != "/opt/vllm-sr/decision-artifact"
            or mounts[0].get("Source") != str(expected_source)
        ):
            raise ProducerError("candidate has an unattested source mount")
    url = urlsplit(arm["url"])
    ports = container.get("NetworkSettings", {}).get("Ports") or {}
    bound_ports = {
        binding.get("HostPort")
        for bindings in ports.values()
        if isinstance(bindings, list)
        for binding in bindings
        if isinstance(binding, dict) and binding.get("HostIp") == "127.0.0.1"
    }
    if str(url.port) not in bound_ports:
        raise ProducerError("protected service URL is not a loopback container port")
    if source_sha is not None:
        runtime_bindings = ports.get("8000/tcp")
        if runtime_bindings != [{"HostIp": "127.0.0.1", "HostPort": str(url.port)}]:
            raise ProducerError(
                "candidate API URL is not bound to its approved runtime listener"
            )
    metrics_url = arm.get("metrics_url")
    if metrics_url is not None:
        metrics = urlsplit(metrics_url)
        if source_sha is not None and metrics.port != url.port:
            raise ProducerError("candidate metrics must use the API listener")
        if str(metrics.port) not in bound_ports:
            raise ProducerError(
                "protected metrics URL is not this container's loopback port"
            )
    return image_id


def _status(url: str) -> dict:
    base = url.rsplit("/v1/systemone", 1)[0]
    with LOOPBACK_OPENER.open(base + "/api/status", timeout=30) as response:
        payload = response.read(MAX_STATUS_BYTES + 1)
    if len(payload) > MAX_STATUS_BYTES:
        raise ProducerError("new runtime status exceeds evidence limit")
    result = json.loads(
        payload, object_pairs_hook=_unique_pairs, parse_constant=_reject_nonfinite
    )
    if not isinstance(result, dict):
        raise ProducerError("new runtime status is invalid")
    return result


def _validate_config(
    config: dict, *, candidate_ref: str, qualification: dict
) -> list[dict]:
    if config.get("schema_version") != "decision-paired-baseline-v1":
        raise ProducerError("protected baseline schema is invalid")
    hardware = config.get("hardware")
    if (
        not isinstance(hardware, str)
        or re.fullmatch(r"[A-Za-z0-9 ._+()-]{1,128}", hardware) is None
        or "AMD Instinct" not in hardware
    ):
        raise ProducerError("protected hardware label is invalid")
    _require(config.get("gpu_device"), GPU_INDEX, "ROCm GPU index")
    if config.get("gpu_exclusivity") != "dedicated_gpu_no_unrelated_compute":
        raise ProducerError("protected runner GPU exclusivity is not attested")
    if config.get("gpu_clock_policy", "unobserved") not in (
        "unobserved",
        "protected_fixed",
        "default_dynamic",
    ):
        raise ProducerError("protected GPU clock policy is invalid")
    records = config.get("models")
    if not isinstance(records, list) or len(records) != len(MODEL_IDS):
        raise ProducerError("protected baseline requires six model services")
    indexed = {row.get("model_id"): row for row in records if isinstance(row, dict)}
    if set(indexed) != MODEL_IDS or len(indexed) != len(records):
        raise ProducerError("protected baseline model inventory is invalid")
    qualified = {row["id"]: row for row in qualification["models"]}
    for model_id, row in indexed.items():
        if row.get("revision") != qualified[model_id]["revision"]:
            raise ProducerError("protected old/new model revisions differ")
        if row.get("artifact_content_id") != qualified[model_id][
            "artifact_content_id"
        ].removeprefix("sha256:"):
            raise ProducerError("protected old/new artifact identities differ")
        for field in (
            "artifact_content_id",
            "artifact_metadata_sha256",
            "artifact_manifest_sha256",
            "old_core_source_sha256",
        ):
            _require(row.get(field), HASH, field)
        batch_size = row.get("new_physical_batch_size")
        if type(batch_size) is not int or not 1 <= batch_size <= PHYSICAL_BATCH:
            raise ProducerError(
                "protected batch size requires independent tuning qualification above eight"
            )
        old_batch = row.get("old_physical_batch_size")
        if type(old_batch) is not int or not 1 <= old_batch <= 4096:
            raise ProducerError("protected old physical batch declaration is invalid")
        metadata_path = (
            default_artifact_cache_root()
            / "sha256"
            / row["artifact_content_id"]
            / ".vllm-sr-artifact.json"
        )
        if _digest(metadata_path) != row["artifact_metadata_sha256"]:
            raise ProducerError(
                "protected artifact metadata differs from the verified local cache"
            )
        metadata = _json(metadata_path)
        if (
            metadata.get("repository_id") != model_id
            or metadata.get("revision") != row["revision"]
            or metadata.get("manifest", {}).get("sha256")
            != row["artifact_manifest_sha256"]
        ):
            raise ProducerError(
                "protected artifact metadata names a different model snapshot"
            )
        if row.get("old_arm_overlay") != "none" and not (
            isinstance(row.get("old_arm_overlay"), str)
            and IMAGE_ID.fullmatch(row["old_arm_overlay"])
        ):
            raise ProducerError("old arm overlay description is missing")
        old_model_id = row.get("old_model_id", model_id)
        if (
            not isinstance(old_model_id, str)
            or MODEL_ID.fullmatch(old_model_id) is None
        ):
            raise ProducerError("old model adapter ID is invalid")
        if row.get("old_response_mode", "decision_v1") not in (
            "decision_v1",
            "legacy_preview",
        ):
            raise ProducerError("old response adapter is invalid")
        for arm in ("old", "new"):
            endpoint = row.get(arm)
            if not isinstance(endpoint, dict):
                raise ProducerError(f"{arm} protected service is missing")
            endpoint["url"] = _loopback(
                endpoint.get("url"), f"{arm} URL", path="/v1/systemone"
            )
            if arm == "new":
                endpoint["metrics_url"] = _loopback(
                    endpoint.get("metrics_url"), "new metrics URL", path="/metrics"
                )
                if endpoint["image_ref"] != candidate_ref:
                    raise ProducerError(
                        "new service does not declare the current candidate digest"
                    )
            elif endpoint.get("metrics_url") is not None:
                endpoint["metrics_url"] = _loopback(
                    endpoint["metrics_url"], "old metrics URL", path="/metrics"
                )
            image_ref = endpoint.get("image_ref")
            if (
                arm == "old"
                and isinstance(image_ref, str)
                and IMAGE_ID.fullmatch(image_ref)
            ):
                pass
            else:
                _require(image_ref, IMAGE_REF, f"{arm} image reference")
            _require(endpoint.get("container_id"), CONTAINER_ID, f"{arm} container ID")
            if arm == "old":
                mounts = endpoint.get("mounts", [])
                if not isinstance(mounts, list) or any(
                    not isinstance(mount, dict)
                    or not isinstance(mount.get("destination"), str)
                    or not mount["destination"].startswith("/")
                    or HASH.fullmatch(str(mount.get("sha256", ""))) is None
                    for mount in mounts
                ):
                    raise ProducerError(
                        "protected old mount digest inventory is invalid"
                    )
                destinations = [mount["destination"] for mount in mounts]
                if len(destinations) != len(set(destinations)) or any(
                    destination != os.path.normpath(destination)
                    or any(
                        other != destination
                        and other.startswith(destination.rstrip("/") + "/")
                        for other in destinations
                    )
                    for destination in destinations
                ):
                    raise ProducerError("old mount destinations overlap or are invalid")
                artifact_destination = endpoint.get("artifact_mount_destination")
                if artifact_destination not in destinations:
                    raise ProducerError(
                        "old artifact must be a declared read-only mount"
                    )
                _validate_locator(
                    endpoint.get("artifact_locator"),
                    label="artifact",
                    destination=artifact_destination,
                )
                kind = endpoint.get("core_source_kind")
                if kind == "mounted":
                    core_destination = endpoint.get("core_mount_destination")
                    if (
                        core_destination == artifact_destination
                        or core_destination not in destinations
                        or next(
                            mount["sha256"]
                            for mount in mounts
                            if mount["destination"] == core_destination
                        )
                        != row["old_core_source_sha256"]
                    ):
                        raise ProducerError("old mounted core declaration is invalid")
                    _validate_locator(
                        endpoint.get("core_locator"),
                        label="core",
                        destination=core_destination,
                    )
                elif kind == "baked":
                    raise ProducerError(
                        "old baked core requires independent protected process/source attestation"
                    )
                else:
                    raise ProducerError("old core source kind is missing")
        token_env = row.get("old_token_env")
        if token_env is not None:
            _require(token_env, ENV_NAME, "old token environment name")
            if not os.environ.get(token_env):
                raise ProducerError("old token environment is empty")
    return [indexed[model_id] for model_id in sorted(MODEL_IDS)]


def _preflight(row: dict, q: int, s: int, directory: Path) -> dict:
    old_model_id = row.get("old_model_id", row["model_id"])
    cases = generate_cases(
        row["model_id"],
        old_model_id,
        question_count=q,
        state_count=s,
        variants=4,
        seed=17,
    )
    old = Endpoint(
        "old",
        row["old"]["url"],
        os.environ.get(row.get("old_token_env", "")),
        row.get("old_response_mode", "decision_v1"),
    )
    new = Endpoint("new", row["new"]["url"], None)
    audit_path = directory / "preflight-audit.jsonl"
    with audit_path.open("x", encoding="utf-8") as handle:
        audit = audit_cohorts(
            {(q, s): cases},
            old,
            new,
            Endpoint("new", batch_url(new.url), None),
            timeout=120.0,
            probability_tolerance=0.01,
            output_handle=handle,
        )
    _write(directory / "preflight-summary.json", audit)
    if audit["status"] != "passed" or audit["mismatch_counts"]:
        raise ProducerError("untimed protected semantic preflight failed")
    return audit


def _measure(
    row: dict, q: int, s: int, directory: Path, source_sha: str, hardware: str
) -> None:
    command = [
        sys.executable,
        "-m",
        "bench.decision_runtime",
        "semantic",
        "--model",
        row["model_id"],
        "--old-url",
        row["old"]["url"],
        "--new-url",
        row["new"]["url"],
        "--old-source-ref",
        "sha256:" + row["old_core_source_sha256"],
        "--new-source-ref",
        source_sha,
        "--old-model-revision",
        row["revision"],
        "--new-model-revision",
        row["revision"],
        "--old-hardware",
        hardware,
        "--new-hardware",
        hardware,
        "--old-network-scope",
        "loopback",
        "--new-network-scope",
        "loopback",
        "--old-physical-batch-size",
        str(row["old_physical_batch_size"]),
        "--new-physical-batch-size",
        str(row["new_physical_batch_size"]),
        "--new-metrics-url",
        row["new"]["metrics_url"],
        "--question-counts",
        str(q),
        "--state-counts",
        str(s),
        "--concurrencies",
        ",".join(map(str, CONCURRENCIES)),
        "--variants",
        "4",
        "--seed",
        "17",
        "--warmup",
        "2",
        "--latency-workflows",
        "16",
        "--throughput-workflows",
        str(MIN_WORKFLOWS_PER_ROUND),
        "--rounds",
        str(ROUNDS),
        "--timeout",
        "120",
        "--parity-policy",
        "require",
        "--probability-tolerance",
        "0.01",
        "--output-dir",
        str(directory / "measured"),
    ]
    if row.get("old_model_id", row["model_id"]) != row["model_id"]:
        command.extend(("--old-model-id", row["old_model_id"]))
    if row.get("old_response_mode", "decision_v1") != "decision_v1":
        command.extend(("--old-response-mode", row["old_response_mode"]))
    if row["old"].get("metrics_url"):
        command.extend(("--old-metrics-url", row["old"]["metrics_url"]))
    if row.get("old_token_env"):
        command.extend(("--old-token-env", row["old_token_env"]))
    environment = {
        key: value
        for key, value in os.environ.items()
        if key.lower() not in {"http_proxy", "https_proxy", "all_proxy"}
    }
    environment["NO_PROXY"] = "127.0.0.1,localhost"
    environment["no_proxy"] = environment["NO_PROXY"]
    result = subprocess.run(
        command,
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode:
        raise ProducerError(
            f"tracked semantic benchmark failed for {row['model_id']} q{q}/s{s}"
        )


def _compact_arm(raw: dict) -> dict:
    throughput = raw["throughput"]
    return {
        "throughput_decisions_per_second": throughput[
            "successful_decisions_per_second"
        ],
        "throughput_window_workflow_ms": throughput["workflow_latency"],
        "low_load_probe_workflow_ms": raw["latency"]["workflow_latency"],
        **{
            field: throughput[field]
            for field in (
                "attempted_workflows",
                "successful_workflows",
                "failed_workflows",
                "attempted_decisions",
                "successful_decisions",
            )
        },
        "round_windows_seconds": [
            round_["window_seconds"] for round_ in throughput["rounds"]
        ],
    }


def _shape(
    row: dict, q: int, s: int, directory: Path, output_dir: Path, source_sha: str
) -> dict:
    raw = _json(directory / "measured" / "receipt.json")
    preflight = _json(directory / "preflight-summary.json")
    if (
        raw.get("source_commit") != source_sha
        or raw.get("harness_sha256") != _harness_digest()
    ):
        raise ProducerError("benchmark receipt is not from the checked-out harness")
    cells = []
    for concurrency, cell in zip(CONCURRENCIES, raw["shapes"], strict=True):
        if cell["concurrency"] != concurrency:
            raise ProducerError("benchmark concurrency inventory differs")
        summary = cell["summary"]
        old, new = summary["arms"]["old"], summary["arms"]["new"]
        telemetry = summary["telemetry"]["arms"]["new"]
        counters = telemetry["counter_deltas"]
        cells.append(
            {
                "concurrency": concurrency,
                "old": _compact_arm(old),
                "new": _compact_arm(new),
                "new_over_old_decisions_per_second": summary["comparison"][
                    "new_over_old_successful_decisions_per_second"
                ],
                "old_over_new_low_load_p50_ms": summary["comparison"][
                    "old_over_new_p50_workflow_latency"
                ],
                "new_physical_batches": counters["physical_batches"],
                "new_physical_batch_rows": counters["physical_batch_rows"],
                "new_observed_rows_per_physical_batch": telemetry[
                    "observed_rows_per_physical_batch"
                ],
            }
        )
    relative = Path("raw") / (row["model_id"].rsplit("/", 1)[-1].lower() + f"-q{q}s{s}")
    target = output_dir / relative
    target.mkdir(parents=True)
    sources = {
        "receipt": directory / "measured" / "receipt.json",
        "preflight": directory / "preflight-summary.json",
        "preflight_audit": directory / "preflight-audit.jsonl",
    }
    sources.update(
        {name.split(".", 1)[0]: directory / "measured" / name for name in RAW_FILES[1:]}
    )
    evidence = {}
    for label, source in sources.items():
        destination = target / source.name
        shutil.copyfile(source, destination)
        evidence[f"raw_{label}_path"] = str(destination.relative_to(output_dir))
        evidence[f"raw_{label}_sha256"] = _digest(destination)
    return {
        "question_count": q,
        "state_count": s,
        "preflight_status": preflight["status"],
        "preflight_mismatch_counts": preflight["mismatch_counts"],
        "preflight_max_absolute_probability_delta": preflight[
            "absolute_probability_delta"
        ]["max"],
        "formal_audit_status": raw["audit"]["status"],
        "formal_audit_mismatch_counts": raw["audit"]["mismatch_counts"],
        "harness_sha256": raw["harness_sha256"],
        "benchmark_source_sha": raw["source_commit"],
        "new_runtime_source_sha": source_sha,
        "arrival_policy": ARRIVAL_POLICY,
        **evidence,
        "cells": cells,
    }


def produce(args: argparse.Namespace) -> Path:
    source_sha = _require(args.source_sha, SHA, "source SHA")
    if not args.run_id.isdecimal() or not args.run_attempt.isdecimal():
        raise ProducerError("protected run identity is invalid")
    actual_sha = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    if source_sha != actual_sha or subprocess.check_output(
        ["git", "status", "--porcelain=v1", "-uall"], cwd=ROOT, text=True
    ):
        raise ProducerError("producer requires the exact clean source checkout")
    candidate_ref = _require(args.candidate_ref, IMAGE_REF, "candidate digest")
    qualification = validate_receipt(
        args.qualification_receipt, owner=args.owner, revision=source_sha
    )
    if qualification["candidate_ref"] != candidate_ref:
        raise ProducerError("qualification receipt names another candidate image")
    config = _json(args.baseline_config)
    rows = _validate_config(
        config, candidate_ref=candidate_ref, qualification=qualification
    )
    if args.output_dir.exists() or args.output_dir.is_symlink():
        raise ProducerError("performance output directory already exists")
    args.output_dir.mkdir(parents=True)
    report = {
        "schema_version": SCHEMA,
        "scope": "synthetic same-revision Decision HTTP performance, not task-quality evaluation",
        "source_sha": source_sha,
        "candidate_ref": candidate_ref,
        "run_id": args.run_id,
        "run_attempt": args.run_attempt,
        "new_runtime_image_source_sha": source_sha,
        "source_image_match": True,
        "harness_sha256": _harness_digest(),
        "environment": {
            "hardware": "declared isolated "
            + config["hardware"]
            + " per paired comparison",
            "network": "both arms loopback HTTP on same validation host",
            "gpu_exclusivity": "dedicated_gpu_no_unrelated_compute; externally attested by protected runner",
            "gpu_clock_policy": config.get("gpu_clock_policy", "unobserved"),
            "service_residency": "both service containers running during alternating waves; HBM residency unmeasured",
            "physical_batch_policy": "per-model launch size; occupancy from per-round metrics",
            "throughput_rounds_per_cell": ROUNDS,
            "concurrencies": list(CONCURRENCIES),
            "workload_shapes": [{"questions": q, "states": s} for q, s in SHAPES],
            "arrival_policy": ARRIVAL_POLICY,
            "new_max_concurrency": NEW_MAX_CONCURRENCY,
            "new_max_queue": NEW_MAX_QUEUE,
            "scheduler_policy": "new runtime admits at most four requests and queues at most 32; HTTP c1/c8/c32 is client-side",
        },
        "protected_baseline_config_sha256": _digest(args.baseline_config),
        "models": [],
    }
    for row in rows:
        old_id = _container_image(
            row["old"],
            source_sha=None,
            gpu_device=config["gpu_device"],
            old_core_sha256=row["old_core_source_sha256"],
            old_overlay=row["old_arm_overlay"],
            old_artifact=row,
        )
        new_id = _container_image(
            row["new"],
            source_sha=source_sha,
            gpu_device=config["gpu_device"],
            new_artifact_content_id=row["artifact_content_id"],
            new_physical_batch_size=row["new_physical_batch_size"],
            new_model_id=row["model_id"],
            new_model_revision=row["revision"],
        )
        if old_id == new_id:
            raise ProducerError("old and new runtime images are identical")
        status = _status(row["new"]["url"])
        artifact = status.get("artifact", {})
        scheduler = status.get("scheduler", [])
        if (
            status.get("status") != "ready"
            or status.get("models") != [row["model_id"]]
            or artifact.get("model") != row["model_id"]
            or artifact.get("revision") != row["revision"]
            or artifact.get("content_sha256") != row["artifact_content_id"]
            or artifact.get("manifest_sha256") != row["artifact_manifest_sha256"]
            or not isinstance(scheduler, list)
            or len(scheduler) != 1
            or scheduler[0].get("model") != row["model_id"]
            or scheduler[0].get("max_concurrency") != NEW_MAX_CONCURRENCY
            or scheduler[0].get("max_queue") != NEW_MAX_QUEUE
        ):
            raise ProducerError("live candidate artifact or scheduler identity differs")
        model = {
            "model_id": row["model_id"],
            "same_old_new_revision": row["revision"],
            "same_old_new_artifact_content_id": row["artifact_content_id"],
            "artifact_metadata_sha256": row["artifact_metadata_sha256"],
            "artifact_manifest_sha256": row["artifact_manifest_sha256"],
            "old_core_source_sha256": row["old_core_source_sha256"],
            "old_arm_overlay": row["old_arm_overlay"],
            "old_image_id": old_id,
            "new_image_id": new_id,
            "old_physical_batch_size": row["old_physical_batch_size"],
            "new_physical_batch_size": row["new_physical_batch_size"],
            "new_scheduler": {
                field: scheduler[0][field] for field in ("max_concurrency", "max_queue")
            },
            "shapes": [],
        }
        for q, s in SHAPES:
            work = (
                args.output_dir
                / "work"
                / (row["model_id"].rsplit("/", 1)[-1].lower() + f"-q{q}s{s}")
            )
            work.mkdir(parents=True)
            _preflight(row, q, s, work)
            _measure(row, q, s, work, source_sha, config["hardware"])
            model["shapes"].append(_shape(row, q, s, work, args.output_dir, source_sha))
        if (
            _container_image(
                row["old"],
                source_sha=None,
                gpu_device=config["gpu_device"],
                old_core_sha256=row["old_core_source_sha256"],
                old_overlay=row["old_arm_overlay"],
                old_artifact=row,
            )
            != old_id
            or _container_image(
                row["new"],
                source_sha=source_sha,
                gpu_device=config["gpu_device"],
                new_artifact_content_id=row["artifact_content_id"],
                new_physical_batch_size=row["new_physical_batch_size"],
                new_model_id=row["model_id"],
                new_model_revision=row["revision"],
            )
            != new_id
        ):
            raise ProducerError("protected service identity changed during measurement")
        report["models"].append(model)
    report_path = args.output_dir / "report.json"
    _write(report_path, report)
    validate_report(
        report_path,
        source_sha=source_sha,
        qualification=qualification,
        candidate_ref=candidate_ref,
        run_id=args.run_id,
        run_attempt=args.run_attempt,
    )
    return report_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-config", required=True, type=Path)
    parser.add_argument("--qualification-receipt", required=True, type=Path)
    parser.add_argument("--candidate-ref", required=True)
    parser.add_argument("--owner", required=True)
    parser.add_argument("--source-sha", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--run-attempt", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    try:
        path = produce(args)
    except (
        ProducerError,
        OSError,
        KeyError,
        TypeError,
        ValueError,
        subprocess.CalledProcessError,
        json.JSONDecodeError,
    ) as error:
        parser.exit(1, f"Decision paired measurement failed: {error}\n")
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

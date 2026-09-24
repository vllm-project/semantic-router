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
import secrets
import shutil
import stat
import subprocess
import sys
from pathlib import Path
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener

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
from decision_runtime.artifacts import (
    ArtifactError,
    open_verified_artifact,
    parse_artifact_manifest,
)
from decision_runtime.catalog_adapter import resolve_decision_runtime_model
from decision_runtime.runtime_profile import (
    RuntimeProfileError,
    validate_relative_artifact_path,
)

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
MAX_ATTESTATION_BYTES = 16 * 1024
OLD_ATTESTATION_SCHEMA = "decision-old-baseline-attestation-v1"
OLD_ATTESTATION_PATH = "/api/decision-baseline-attestation"
OLD_ARTIFACT_LAYOUT = "full_snapshot_selected_data_v1"
NEW_MAX_CONCURRENCY = 4
NEW_MAX_QUEUE = 32
RAW_FILES = (
    "receipt.json",
    "audit.jsonl",
    "samples.jsonl",
    "workflows.jsonl",
    "metrics.jsonl",
    "timed-semantic.jsonl.gz",
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
    """Hash a regular file or an unambiguously framed, symlink-free tree."""

    try:
        if (
            not source.is_absolute()
            or source == Path("/")
            or source.resolve(strict=True) != source
        ):
            raise ProducerError("protected source mount path is not canonical")
        return _unchecked_mount_digest(source)
    except OSError:
        raise ProducerError("protected source mount is unreadable") from None


def _verify_bind_inode(
    container: dict, source: Path, destination: str, *, proc_root: Path = Path("/proc")
) -> None:
    """Require host Source to name the inode actually bound into live PID 1."""

    pid = (container.get("State") or {}).get("Pid")
    if type(pid) is not int or pid < 1:
        raise ProducerError("old container process identity is unavailable")
    if (
        not destination.startswith("/")
        or destination == "/"
        or destination != os.path.normpath(destination)
    ):
        raise ProducerError("old bind destination is invalid")
    mounted = proc_root / str(pid) / "root" / destination.lstrip("/")
    try:
        source_stat = source.stat(follow_symlinks=False)
        mounted_stat = mounted.stat(follow_symlinks=False)
    except OSError:
        raise ProducerError("old bind inode is unavailable") from None
    if (
        source_stat.st_dev,
        source_stat.st_ino,
        stat.S_IFMT(source_stat.st_mode),
    ) != (
        mounted_stat.st_dev,
        mounted_stat.st_ino,
        stat.S_IFMT(mounted_stat.st_mode),
    ):
        raise ProducerError("old bind source differs from the live mounted inode")


def _unchecked_mount_digest(source: Path) -> str:

    if source.is_symlink():
        raise ProducerError("protected source mount uses a symlink")
    if source.is_file():
        return _digest(source)
    if not source.is_dir():
        raise ProducerError("protected source mount is not a regular tree")
    digest = hashlib.sha256(b"decision-mounted-tree-v2\0")

    def frame(value: bytes) -> None:
        digest.update(len(value).to_bytes(8, "big"))
        digest.update(value)

    for path in sorted(source.rglob("*")):
        relative = path.relative_to(source).as_posix().encode("utf-8")
        if path.is_symlink():
            raise ProducerError("protected source tree contains a symlink")
        mode = path.stat(follow_symlinks=False).st_mode
        if stat.S_ISDIR(mode):
            frame(b"D")
            frame(relative)
            digest.update((0).to_bytes(8, "big"))
        elif stat.S_ISREG(mode):
            frame(b"F")
            frame(relative)
            size = path.stat(follow_symlinks=False).st_size
            digest.update(size.to_bytes(8, "big"))
            observed_size = 0
            with path.open("rb") as handle:
                while chunk := handle.read(1024 * 1024):
                    digest.update(chunk)
                    observed_size += len(chunk)
            if (
                observed_size != size
                or path.stat(follow_symlinks=False).st_size != size
            ):
                raise ProducerError("protected source changed while hashing")
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
    if label in ("core", "adapter") and kind != "command_path":
        raise ProducerError("old core must be the running process script or executable")
    if kind == "argument":
        _require(locator.get("flag"), ARGUMENT_FLAG, f"{label} argument")
    elif kind == "environment":
        _require(locator.get("name"), ENV_NAME, f"{label} environment")
    elif kind == "command_path" and label in ("core", "adapter"):
        path = locator.get("path")
        if (
            not isinstance(path, str)
            or not path.startswith(destination.rstrip("/") + "/")
            and path != destination
            or path != os.path.normpath(path)
        ):
            raise ProducerError(f"old {label} command path is outside its mount")
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


def _process_start_ticks(container: dict) -> int:
    """Bind a live attestation to Docker's container-init process lifetime."""

    pid = container.get("State", {}).get("Pid")
    if type(pid) is not int or pid < 1:
        raise ProducerError("old container process identity is unavailable")
    try:
        stat_line = (Path("/proc") / str(pid) / "stat").read_text(encoding="ascii")
        closing = stat_line.rfind(")")
        if closing < 0:
            raise ProducerError("old container process lifetime is unavailable")
        fields = stat_line[closing + 2 :].split()
        start_ticks = int(fields[19])
        status = (Path("/proc") / str(pid) / "status").read_text(encoding="ascii")
        namespaces = [
            line.split()[1:]
            for line in status.splitlines()
            if line.startswith("NSpid:")
        ]
    except (OSError, UnicodeError, IndexError, ValueError) as error:
        raise ProducerError("old container process lifetime is unavailable") from error
    if (
        not namespaces
        or not namespaces[0]
        or namespaces[0][-1] != "1"
        or start_ticks < 1
    ):
        raise ProducerError("old container init is not PID 1 in its namespace")
    return start_ticks


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


def _import_locator(locator: object, destination: str) -> dict:
    if not isinstance(locator, dict) or locator.get("kind") != "python_import":
        raise ProducerError("old imported core locator is invalid")
    path = locator.get("path")
    module = locator.get("module")
    if (
        not isinstance(path, str)
        or (path != destination and not path.startswith(destination.rstrip("/") + "/"))
        or path != os.path.normpath(path)
        or not isinstance(module, str)
        or re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*", module)
        is None
    ):
        raise ProducerError("old imported core path or module is invalid")
    return locator


def _old_live_attestation(container: dict, arm: dict, row: dict) -> dict:
    """Validate live imports and return proof with private paths redacted."""

    start_ticks = _process_start_ticks(container)
    declared_command = _running_command(container)
    challenge = secrets.token_hex(32)
    base = arm["url"].rsplit("/v1/systemone", 1)[0]
    request = Request(
        base + OLD_ATTESTATION_PATH + "?challenge=" + challenge,
        headers=(
            {"Authorization": "Bearer " + os.environ[row["old_token_env"]]}
            if row.get("old_token_env")
            else {}
        ),
    )
    try:
        with LOOPBACK_OPENER.open(request, timeout=30) as response:
            media_type = (
                response.headers.get("Content-Type", "")
                .split(";", 1)[0]
                .strip()
                .lower()
            )
            if response.status != 200 or media_type != "application/json":
                raise ProducerError("old live attestation is not a JSON 200 response")
            payload = response.read(MAX_ATTESTATION_BYTES + 1)
    except OSError as error:
        raise ProducerError("old live attestation endpoint is unavailable") from error
    if len(payload) > MAX_ATTESTATION_BYTES:
        raise ProducerError("old live attestation exceeds evidence limit")
    try:
        evidence = json.loads(
            payload, object_pairs_hook=_unique_pairs, parse_constant=_reject_nonfinite
        )
    except (UnicodeError, ValueError) as error:
        raise ProducerError("old live attestation is invalid JSON") from error
    if not isinstance(evidence, dict) or set(evidence) != {
        "schema_version",
        "challenge",
        "pid",
        "process_start_ticks",
        "adapter_path",
        "adapter_sha256",
        "imported_module",
        "imported_core_path",
        "core_mount_sha256",
        "loaded_artifact_root",
        "loaded_artifact_content_id",
        "model_id",
        "revision",
    }:
        raise ProducerError("old live attestation has an invalid contract")
    if (
        type(evidence["pid"]) is not int
        or type(evidence["process_start_ticks"]) is not int
    ):
        raise ProducerError("old live process identity has an invalid type")
    expected = {
        "schema_version": OLD_ATTESTATION_SCHEMA,
        "challenge": challenge,
        "pid": 1,
        "process_start_ticks": start_ticks,
        "adapter_path": arm["adapter_locator"]["path"],
        "adapter_sha256": row["old_adapter_source_sha256"],
        "imported_module": arm["core_locator"]["module"],
        "imported_core_path": arm["core_locator"]["path"],
        "core_mount_sha256": row["old_core_source_sha256"],
        "loaded_artifact_root": arm["artifact_mount_destination"],
        "loaded_artifact_content_id": row["artifact_content_id"],
        "model_id": row["model_id"],
        "revision": row["revision"],
    }
    if evidence != expected:
        raise ProducerError("old live import or loaded artifact identity differs")
    if (
        _process_start_ticks(container) != start_ticks
        or _running_command(container) != declared_command
    ):
        raise ProducerError("old container init changed during live attestation")
    for field in (
        "adapter_path",
        "imported_module",
        "imported_core_path",
        "loaded_artifact_root",
    ):
        evidence[field + "_sha256"] = hashlib.sha256(
            evidence.pop(field).encode()
        ).hexdigest()
    return evidence


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


def _verified_old_snapshot(source: Path, core_source: Path, row: dict) -> str:
    """Bind a complete old code/weight snapshot to qualified selected data."""

    try:
        model = resolve_decision_runtime_model(
            row["model_id"], revision=row["revision"], backend="rocm"
        )
        artifact = open_verified_artifact(
            default_artifact_cache_root() / "sha256" / row["artifact_content_id"],
            model,
            expected_content_id=row["artifact_content_id"],
        )
    except (ArtifactError, ValueError, OSError) as error:
        raise ProducerError(
            "qualified selected-data artifact failed full content verification"
        ) from error
    if (
        artifact.repository_id != row["model_id"]
        or artifact.revision != row["revision"]
        or artifact.content_id != row["artifact_content_id"]
        or artifact.manifest is None
        or artifact.manifest.sha256 != row["artifact_manifest_sha256"]
    ):
        raise ProducerError("qualified selected-data artifact identity differs")
    if (
        source.is_symlink()
        or not source.is_dir()
        or core_source.is_symlink()
        or not core_source.is_dir()
    ):
        raise ProducerError("old full snapshot and core must be regular directories")

    def regular_file(root: Path, relative: str) -> Path:
        try:
            validate_relative_artifact_path(relative, field="old snapshot file")
        except RuntimeProfileError as error:
            raise ProducerError("old snapshot path is invalid") from error
        path = root / relative
        if path.is_symlink() or not path.is_file():
            raise ProducerError("old snapshot file is missing or is a symlink")
        return path

    try:
        bindings_path = regular_file(core_source, "BINDINGS.json")
        if bindings_path.stat().st_size > 1024 * 1024:
            raise ProducerError("old binding inventory is oversized")
        bindings = json.loads(
            bindings_path.read_bytes(),
            object_pairs_hook=_unique_pairs,
            parse_constant=_reject_nonfinite,
        )
        binding = bindings[row.get("old_model_id", row["model_id"])]
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise ProducerError("old binding inventory is invalid") from error
    manifest_name = artifact.manifest.path
    if (
        not isinstance(binding, dict)
        or binding.get("repo") != row["model_id"]
        or binding.get("revision") != row["revision"]
        or binding.get("manifest_file") != manifest_name
        or binding.get("manifest_sha256") != artifact.manifest.sha256
        or not isinstance(binding.get("files"), dict)
        or not binding["files"]
    ):
        raise ProducerError("old binding differs from qualified model revision")
    manifest_path = regular_file(source, manifest_name)
    manifest_bytes = manifest_path.read_bytes()
    if (
        len(manifest_bytes) != artifact.manifest.size_bytes
        or hashlib.sha256(manifest_bytes).hexdigest() != artifact.manifest.sha256
    ):
        raise ProducerError("old full snapshot manifest differs from qualification")
    try:
        json.loads(
            manifest_bytes,
            object_pairs_hook=_unique_pairs,
            parse_constant=_reject_nonfinite,
        )
        inventory = parse_artifact_manifest(manifest_bytes, manifest_path=manifest_name)
    except (UnicodeError, ValueError, ArtifactError) as error:
        raise ProducerError("old full snapshot manifest is invalid") from error

    declared = binding["files"]
    actual = set()
    for path in source.rglob("*"):
        if path.is_symlink():
            raise ProducerError("old full snapshot contains a symlink")
        mode = path.stat(follow_symlinks=False).st_mode
        if stat.S_ISREG(mode):
            actual.add(path.relative_to(source).as_posix())
        elif not stat.S_ISDIR(mode):
            raise ProducerError("old full snapshot contains a special file")
    if set(declared) != actual or manifest_name not in declared:
        raise ProducerError("old full snapshot file roster differs from old binding")
    for relative, reference in declared.items():
        path = regular_file(source, relative)
        if (
            not isinstance(reference, dict)
            or set(reference) != {"bytes", "sha256"}
            or type(reference["bytes"]) is not int
            or reference["bytes"] < 0
            or not isinstance(reference["sha256"], str)
            or HASH.fullmatch(reference["sha256"]) is None
            or path.stat().st_size != reference["bytes"]
            or _mount_digest(path) != reference["sha256"]
        ):
            raise ProducerError("old full snapshot file differs from old binding")
    if declared[manifest_name] != {
        "bytes": artifact.manifest.size_bytes,
        "sha256": artifact.manifest.sha256,
    }:
        raise ProducerError("old binding manifest identity differs")
    inventory_by_path = {item.repository_path: item for item in inventory.values()}
    if any(
        declared.get(item.repository_path)
        != {"bytes": item.size_bytes, "sha256": item.sha256}
        for item in inventory_by_path.values()
    ):
        raise ProducerError("old binding differs from a model-manifest file")
    selected_files = []
    for item in artifact.files:
        old_item = inventory_by_path.get(item.repository_path)
        if (
            old_item is None
            or old_item.sha256 != item.sha256
            or old_item.size_bytes != item.size_bytes
            or declared.get(item.repository_path)
            != {"bytes": item.size_bytes, "sha256": item.sha256}
        ):
            raise ProducerError("old selected data differs from qualified artifact")
        selected_files.append(
            {
                "path": item.repository_path,
                "sha256": old_item.sha256,
                "size_bytes": old_item.size_bytes,
            }
        )
    observed_receipt = {
        "schema_version": 2,
        "repository_id": binding["repo"],
        "revision": binding["revision"],
        "manifest": {
            "path": manifest_name,
            "sha256": hashlib.sha256(manifest_bytes).hexdigest(),
            "size_bytes": len(manifest_bytes),
        },
        "files": selected_files,
    }
    observed_content_id = hashlib.sha256(
        json.dumps(observed_receipt, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if observed_content_id != artifact.content_id:
        raise ProducerError("old selected-data content identity differs")
    return observed_content_id


def _container_image(
    arm: dict,
    *,
    source_sha: str | None,
    gpu_device: str,
    old_core_sha256: str | None = None,
    old_adapter_sha256: str | None = None,
    old_overlay: str | None = None,
    old_artifact: dict | None = None,
    old_attestations: list[dict] | None = None,
    old_snapshot_proofs: list[dict] | None = None,
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
        # A directly mounted old core cannot provide the required imported-core
        # and loaded-artifact proof. Reject it before examining live bind inodes
        # so this independent policy violation is not masked by mount failures.
        if arm.get("core_source_kind") == "mounted":
            raise ProducerError(
                "direct old core lacks mandatory loaded-artifact process proof"
            )
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
            item.get("destination"): (item.get("sha256"), item.get("kind"))
            for item in declared
            if isinstance(item, dict)
        }
        if len(expected) != len(declared):
            raise ProducerError("old service mount declarations are invalid")
        observed = set()
        seen_destinations = set()
        artifact_source = None
        core_source = None
        adapter_source = None
        snapshot_sha256 = None
        core_destination = arm.get("core_mount_destination")
        bind_sources: list[tuple[Path, str]] = []
        for mount in mounts:
            if (
                not isinstance(mount, dict)
                or mount.get("Type") != "bind"
                or mount.get("RW") is not False
                or not isinstance(mount.get("Destination"), str)
                or mount["Destination"] not in expected
                or mount["Destination"] in seen_destinations
                or not isinstance(mount.get("Source"), str)
            ):
                raise ProducerError("old service has an unattested or writable mount")
            seen_destinations.add(mount["Destination"])
            source_text = mount["Source"]
            if (
                not source_text.startswith("/")
                or source_text == "/"
                or source_text != os.path.normpath(source_text)
            ):
                raise ProducerError("old bind source path is not canonical")
            source = Path(source_text)
            declared_digest, declared_kind = expected[mount["Destination"]]
            if (declared_kind == "file" and not source.is_file()) or (
                declared_kind == "tree" and not source.is_dir()
            ):
                raise ProducerError("old mounted source kind differs")
            _verify_bind_inode(container, source, mount["Destination"])
            actual = _mount_digest(source)
            _verify_bind_inode(container, source, mount["Destination"])
            if actual != declared_digest:
                raise ProducerError("old mounted source changed from protected digest")
            bind_sources.append((source, mount["Destination"]))
            observed.add(actual)
            if (
                old_artifact is not None
                and mount["Destination"] == arm["artifact_mount_destination"]
            ):
                artifact_source = Path(mount["Source"])
                snapshot_sha256 = actual
            if mount["Destination"] == core_destination:
                core_source = Path(mount["Source"])
            if mount["Destination"] == arm.get("adapter_mount_destination"):
                adapter_source = Path(mount["Source"])
        if seen_destinations != set(expected):
            raise ProducerError(
                "old service mount inventory differs from protected baseline"
            )
        if arm.get("core_source_kind") == "mounted_adapter":
            adapter_destination = arm.get("adapter_mount_destination")
            if (
                old_adapter_sha256 is None
                or len(declared) != 3
                or adapter_destination
                in (None, core_destination, arm["artifact_mount_destination"])
                or expected.get(adapter_destination) != (old_adapter_sha256, "file")
                or core_destination == arm["artifact_mount_destination"]
                or expected.get(core_destination) != (old_core_sha256, "tree")
                or expected.get(arm["artifact_mount_destination"], (None, None))[1]
                != "tree"
                or arm.get("artifact_layout") != OLD_ARTIFACT_LAYOUT
                or adapter_source is None
                or not adapter_source.is_file()
                or core_source is None
                or not core_source.is_dir()
            ):
                raise ProducerError(
                    "old adapter, core, or full snapshot declaration differs"
                )
            _old_core_process(container, arm["adapter_locator"]["path"])
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
            if (
                artifact_source is None
                or core_source is None
                or snapshot_sha256 is None
            ):
                raise ProducerError("old full snapshot or core mount is missing")
            try:
                selected_content_id = _verified_old_snapshot(
                    artifact_source, core_source, old_artifact
                )
            except OSError:
                raise ProducerError("old full snapshot is unreadable") from None
            for source, destination in bind_sources:
                _verify_bind_inode(container, source, destination)
            if old_snapshot_proofs is not None:
                old_snapshot_proofs.append(
                    {
                        "full_snapshot_sha256": snapshot_sha256,
                        "selected_data_content_id": selected_content_id,
                    }
                )
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
    elif (
        old_core_sha256 is not None and arm.get("core_source_kind") == "mounted_adapter"
    ):
        if old_artifact is None:
            raise ProducerError("old artifact attestation is required")
        api_port = arm.get("api_container_port")
        if type(api_port) is not int or not 1 <= api_port <= 65535:
            raise ProducerError("old API container port is invalid")
        expected_binding = [{"HostIp": "127.0.0.1", "HostPort": str(url.port)}]
        if ports.get(f"{api_port}/tcp") != expected_binding:
            raise ProducerError("old API listener is not the declared container port")
        evidence = _old_live_attestation(container, arm, old_artifact)
        if old_attestations is not None:
            old_attestations.append(evidence)
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
    if config.get("schema_version") != "decision-paired-baseline-v2":
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
                    or mount.get("kind") not in ("file", "tree")
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
                    raise ProducerError(
                        "direct old core lacks mandatory loaded-artifact process proof"
                    )
                elif kind == "mounted_adapter":
                    if endpoint.get("artifact_layout") != OLD_ARTIFACT_LAYOUT:
                        raise ProducerError("old full snapshot layout is required")
                    if endpoint.get("artifact_locator") != {
                        "kind": "argument",
                        "flag": "--artifact-root",
                    }:
                        raise ProducerError(
                            "old full snapshot launch argument is required"
                        )
                    port = endpoint.get("api_container_port")
                    if type(port) is not int or not 1 <= port <= 65535:
                        raise ProducerError("old API container port is invalid")
                    core_destination = endpoint.get("core_mount_destination")
                    adapter_destination = endpoint.get("adapter_mount_destination")
                    adapter_sha = _require(
                        row.get("old_adapter_source_sha256"),
                        HASH,
                        "old adapter source digest",
                    )
                    if (
                        len(mounts) != 3
                        or core_destination not in destinations
                        or adapter_destination not in destinations
                        or len(
                            {
                                core_destination,
                                adapter_destination,
                                artifact_destination,
                            }
                        )
                        != 3
                        or next(
                            (mount["sha256"], mount["kind"])
                            for mount in mounts
                            if mount["destination"] == core_destination
                        )
                        != (row["old_core_source_sha256"], "tree")
                        or next(
                            (mount["sha256"], mount["kind"])
                            for mount in mounts
                            if mount["destination"] == adapter_destination
                        )
                        != (adapter_sha, "file")
                        or next(
                            mount["kind"]
                            for mount in mounts
                            if mount["destination"] == artifact_destination
                        )
                        != "tree"
                    ):
                        raise ProducerError(
                            "old adapter or imported core declaration is invalid"
                        )
                    _validate_locator(
                        endpoint.get("adapter_locator"),
                        label="adapter",
                        destination=adapter_destination,
                    )
                    if endpoint["adapter_locator"]["path"] != adapter_destination:
                        raise ProducerError(
                            "old adapter must execute its exact file mount"
                        )
                    _import_locator(endpoint.get("core_locator"), core_destination)
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
        "--timed-semantic-evidence",
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
        {
            name.split(".", 1)[0].replace("timed-semantic", "timed_semantic"): directory
            / "measured"
            / name
            for name in RAW_FILES[1:]
        }
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
        old_attestations: list[dict] = []
        old_snapshot_proofs: list[dict] = []
        old_id = _container_image(
            row["old"],
            source_sha=None,
            gpu_device=config["gpu_device"],
            old_core_sha256=row["old_core_source_sha256"],
            old_adapter_sha256=row.get("old_adapter_source_sha256"),
            old_overlay=row["old_arm_overlay"],
            old_artifact=row,
            old_attestations=old_attestations,
            old_snapshot_proofs=old_snapshot_proofs,
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
            "old_core_source_kind": row["old"]["core_source_kind"],
            "old_artifact_layout": row["old"]["artifact_layout"],
            "old_full_snapshot_sha256": next(
                mount["sha256"]
                for mount in row["old"]["mounts"]
                if mount["destination"] == row["old"]["artifact_mount_destination"]
            ),
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
        if row["old"]["core_source_kind"] == "mounted_adapter":
            model["old_adapter_source_sha256"] = row["old_adapter_source_sha256"]
            model["old_source_declaration"] = {
                "adapter_path_sha256": hashlib.sha256(
                    row["old"]["adapter_locator"]["path"].encode()
                ).hexdigest(),
                "imported_core_path_sha256": hashlib.sha256(
                    row["old"]["core_locator"]["path"].encode()
                ).hexdigest(),
                "imported_module_sha256": hashlib.sha256(
                    row["old"]["core_locator"]["module"].encode()
                ).hexdigest(),
                "artifact_mount_path_sha256": hashlib.sha256(
                    row["old"]["artifact_mount_destination"].encode()
                ).hexdigest(),
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
                old_adapter_sha256=row.get("old_adapter_source_sha256"),
                old_overlay=row["old_arm_overlay"],
                old_artifact=row,
                old_attestations=old_attestations,
                old_snapshot_proofs=old_snapshot_proofs,
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
        if (
            len(old_snapshot_proofs) != 2
            or old_snapshot_proofs[0] != old_snapshot_proofs[1]
            or old_snapshot_proofs[0]["selected_data_content_id"]
            != row["artifact_content_id"]
        ):
            raise ProducerError("old full snapshot changed during measurement")
        model["old_snapshot_verifications"] = old_snapshot_proofs
        if len(old_attestations) == 2:
            path = (
                args.output_dir
                / "raw"
                / (row["model_id"].rsplit("/", 1)[-1].lower() + "-old-attestation.json")
            )
            model["old_attestation_path"] = path.relative_to(args.output_dir).as_posix()
            model["old_attestation_sha256"] = _write(
                path,
                {
                    "schema_version": OLD_ATTESTATION_SCHEMA,
                    "observations": old_attestations,
                },
            )
        else:
            raise ProducerError("old live process proof is incomplete")
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

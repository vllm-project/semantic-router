#!/usr/bin/env python3
"""Qualify the exact published CPU image with three live Decision models.

The protected runner downloads model artifacts on the host. The container gets
only a verified read-only artifact mount, never a Hugging Face credential.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from decision_cpu_receipt import (
    IMAGE,
    MAX_RAW_BYTES,
    MODEL_IDS,
    OWNER,
    RAW_FILES,
    REQUIRED_CHECKS,
    REVISION,
    SHA256,
    _digest,
    published_ref,
    validate_receipt,
)
from decision_rocm_qualify import (
    EvidenceWriter,
    LiveRuntimeIO,
    RuntimeIO,
    _batch,
    _check_status,
    _post_validated,
    _read_json,
    _require_clean_source,
    _require_free_port,
    _single,
)
from image_artifacts import DECISION_RUNTIME_BASES, source_sha


class QualificationError(RuntimeError):
    """A live CPU qualification check failed."""


@dataclass(frozen=True, slots=True)
class Options:
    owner: str
    published_ref: str
    output_dir: Path
    port: int
    startup_timeout: int = 900
    request_timeout: int = 120


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


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
        raise QualificationError("CPU drun launch reported a different model")
    revision = identity.removeprefix(model + "@")
    artifact = fields.get("Artifact", "")
    if (
        fields.get("Instance") != instance
        or REVISION.fullmatch(revision) is None
        or SHA256.fullmatch(artifact) is None
        or fields.get("Backend", "").split(" ", 1)[0] != "cpu"
        or fields.get("Mode") != "detached"
        or fields.get("Endpoint") != f"http://127.0.0.1:{port}/v1/systemone"
    ):
        raise QualificationError("CPU drun launch identity is incomplete")
    return revision, artifact


def _attest_live_image(io: RuntimeIO, *, image: str, instance: str) -> bytes:
    """Bind the running container's image ID to the pulled digest, not a tag."""

    image_rows = json.loads(
        io.command(["docker", "image", "inspect", image], timeout=30)
    )
    container_rows = json.loads(
        io.command(["docker", "inspect", f"vllm-sr-drun-{instance}"], timeout=30)
    )
    if (
        not isinstance(image_rows, list)
        or len(image_rows) != 1
        or not isinstance(image_rows[0], dict)
        or not isinstance(container_rows, list)
        or len(container_rows) != 1
        or not isinstance(container_rows[0], dict)
    ):
        raise QualificationError("CPU image or live container inspection is incomplete")
    local = image_rows[0]
    running = container_rows[0]
    image_id = local.get("Id")
    config = running.get("Config")
    labels = config.get("Labels") if isinstance(config, dict) else None
    command = config.get("Cmd") if isinstance(config, dict) else None
    state = running.get("State")
    host_config = running.get("HostConfig")
    repo_digests = local.get("RepoDigests")
    if (
        not isinstance(image_id, str)
        or SHA256.fullmatch(image_id) is None
        or not isinstance(repo_digests, list)
        or image not in repo_digests
        or running.get("Image") != image_id
        or not isinstance(config, dict)
        or config.get("Image") != image
        or not isinstance(labels, dict)
        or labels.get("ai.vllm-sr.drun.managed") != "true"
        or labels.get("ai.vllm-sr.drun.instance") != instance
        or labels.get("ai.vllm-sr.drun.image") != image
        or not isinstance(command, list)
        or command.count("--backend") != 1
        or command[command.index("--backend") + 1 : command.index("--backend") + 2]
        != ["cpu"]
        or not isinstance(host_config, dict)
        or host_config.get("Devices") not in (None, [])
        or host_config.get("DeviceRequests") not in (None, [])
        or not isinstance(state, dict)
        or state.get("Status") != "running"
    ):
        raise QualificationError("CPU live container is not bound to published digest")
    # Never archive Docker's raw inspection: it can contain host paths or env.
    return _json_bytes({"published_ref": image, "local_image_id": image_id})


def _qualify_model(
    io: RuntimeIO, writer: EvidenceWriter, options: Options, model: str, revision: str
) -> dict:
    slug = model.rsplit("/", 1)[-1].removeprefix("Decision-1.0-").lower()
    _require_free_port(options.port)
    instance = f"qualify-cpu-{slug}-{uuid.uuid4().hex[:12]}"
    command = [
        "vllm-sr",
        "drun",
        "run",
        model,
        "--backend",
        "cpu",
        "--image",
        options.published_ref,
        "--host",
        "127.0.0.1",
        "--port",
        str(options.port),
        "--max-batch",
        "8",
        "--max-concurrency",
        "8",
        "--max-queue",
        "32",
        "--runtime",
        "docker",
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
        output = io.command(command, timeout=options.startup_timeout + 60)
        writer.raw(f"raw/{slug}/drun-launch.txt", output.encode())
        model_revision, artifact = _parse_launch(
            output, model=model, port=options.port, instance=instance
        )
        writer.raw(
            f"raw/{slug}/image-attestation.json",
            _attest_live_image(io, image=options.published_ref, instance=instance),
        )
        base_url = f"http://127.0.0.1:{options.port}"
        ready = io.http(
            "GET", base_url + "/ready", None, timeout=options.request_timeout
        )
        writer.raw(f"raw/{slug}/ready.json", ready.body)
        if _read_json(ready, label="CPU ready") != {"ready": True}:
            raise QualificationError("CPU runtime did not report ready")
        status_reply = io.http(
            "GET", base_url + "/api/status", None, timeout=options.request_timeout
        )
        writer.raw(f"raw/{slug}/status.json", status_reply.body)
        _check_status(
            _read_json(status_reply, label="CPU status"),
            model=model,
            revision=model_revision,
            artifact=artifact,
        )
        single_ms = _post_validated(
            io,
            writer,
            base_url=base_url,
            relative=f"raw/{slug}/mixed-single",
            payload=_single(model, 3),
            batch=False,
            timeout=options.request_timeout,
        )
        batch_ms = _post_validated(
            io,
            writer,
            base_url=base_url,
            relative=f"raw/{slug}/two-state-batch",
            payload=_batch(model, states=2, questions=3),
            batch=True,
            timeout=options.request_timeout,
        )
        if not math.isfinite(single_ms) or not math.isfinite(batch_ms):
            raise QualificationError("CPU HTTP latency is invalid")
        return {
            "image_ref": options.published_ref,
            "source_sha": revision,
            "model_id": model,
            "revision": model_revision,
            "artifact_content_id": artifact,
            "backend": "cpu",
            "device": "cpu",
            "result": "passed",
            "checks": {check: True for check in sorted(REQUIRED_CHECKS)},
            "latency_ms": {"mixed_single": single_ms, "two_state_batch": batch_ms},
            "raw_sha256": {
                name: digest
                for name, digest in sorted(writer.hashes.items())
                if name.startswith(f"raw/{slug}/")
            },
        }
    except BaseException as error:
        failure = error
        raise
    finally:
        if launch_attempted:
            try:
                # drun stop checks the registry and immutable ownership labels.
                # Try it even if the launch command timed out after container creation.
                io.command(["vllm-sr", "drun", "stop", instance], timeout=60)
            except Exception:
                if failure is None:
                    raise


def qualify_all(
    options: Options,
    *,
    io: RuntimeIO | None = None,
    inspect_image: Callable[[str, str], None] | None = None,
) -> Path:
    """Write a promotable receipt only after all three CPU models pass live probes."""

    if OWNER.fullmatch(options.owner) is None:
        raise QualificationError("CPU qualification owner is invalid")
    if (
        not isinstance(options.published_ref, str)
        or "@" not in options.published_ref
        or published_ref(options.owner, options.published_ref.rsplit("@", 1)[1])
        != options.published_ref
        or not 1 <= options.port <= 65535
        or not 30 <= options.startup_timeout <= 1800
        or not 1 <= options.request_timeout <= 300
        or options.output_dir.exists()
        or options.output_dir.is_symlink()
    ):
        raise QualificationError("CPU qualification options are invalid")
    _require_clean_source()
    revision = source_sha()
    if inspect_image is None:
        from decision_image_lock_release import inspect_published_image

        inspect_image = lambda reference, source: inspect_published_image(
            reference, revision=source, backend="cpu"
        )
    inspect_image(options.published_ref, revision)
    _require_free_port(options.port)
    options.output_dir.mkdir(parents=True)
    writer = EvidenceWriter(options.output_dir)
    runtime_io = io or LiveRuntimeIO()
    base, _, _ = DECISION_RUNTIME_BASES[IMAGE]
    record = {
        "schema": 1,
        "image": IMAGE,
        "source_sha": revision,
        "base_image": base,
        "published_ref": options.published_ref,
        "backend": "cpu",
        "platform": "linux/amd64",
        "models": [],
    }
    for model in sorted(MODEL_IDS):
        evidence = _qualify_model(runtime_io, writer, options, model, revision)
        slug = model.rsplit("/", 1)[-1].removeprefix("Decision-1.0-").lower()
        relative = f"models/{slug}.json"
        evidence_digest = writer.document(relative, evidence)
        record["models"].append(
            {
                "id": model,
                "revision": evidence["revision"],
                "artifact_content_id": evidence["artifact_content_id"],
                "evidence_file": relative,
                "evidence_sha256": evidence_digest,
            }
        )
    pending = options.output_dir / "qualification.pending.json"
    pending.write_bytes(_json_bytes(record))
    validate_receipt(
        pending,
        owner=options.owner,
        revision=revision,
        published_image=options.published_ref,
    )
    final = options.output_dir / "qualification.json"
    pending.replace(final)
    return final


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--owner", required=True)
    parser.add_argument("--published-ref", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--port", required=True, type=int)
    parser.add_argument("--startup-timeout", type=int, default=900)
    parser.add_argument("--request-timeout", type=int, default=120)
    args = parser.parse_args()
    try:
        receipt = qualify_all(Options(**vars(args)))
    except (OSError, ValueError, QualificationError) as error:
        parser.exit(1, f"CPU qualification failed: {error}\n")
    print(receipt)


if __name__ == "__main__":
    main()

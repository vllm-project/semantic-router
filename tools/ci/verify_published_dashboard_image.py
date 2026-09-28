#!/usr/bin/env python3
"""Verify the published Dashboard manifest and runtime image variants."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
import urllib.request
import uuid
from collections.abc import Sequence
from http import HTTPStatus

EXPECTED_PLATFORMS = {"linux/amd64", "linux/arm64"}
EXPECTED_HEALTH = {
    "status": "healthy",
    "service": "semantic-router-dashboard",
}
IMAGE_BY_DIGEST = re.compile(r"^\S+@sha256:[0-9a-f]{64}$")
MANIFEST_DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")


class VerificationError(RuntimeError):
    """Raised when the published Dashboard image violates its contract."""


def run_command(
    command: Sequence[str], *, check: bool = True
) -> subprocess.CompletedProcess[str]:
    """Run one Docker command and retain output for diagnostics."""
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if check and result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip() or "no command output"
        raise VerificationError(f"{' '.join(command)} failed: {detail}")
    return result


def manifest_platforms(raw_manifest: str) -> dict[str, str]:
    """Map supported platforms to child digests in the published OCI index."""
    try:
        document = json.loads(raw_manifest)
    except json.JSONDecodeError as error:
        raise VerificationError(
            f"published manifest is not valid JSON: {error}"
        ) from error

    if not isinstance(document, dict):
        raise VerificationError("published manifest JSON root is not an object")
    manifests = document.get("manifests")
    if not isinstance(manifests, list):
        raise VerificationError("published image is not a multi-platform image index")

    platforms: dict[str, str] = {}
    for manifest in manifests:
        if not isinstance(manifest, dict):
            raise VerificationError("published image index has an invalid descriptor")
        platform = manifest.get("platform", {})
        if not isinstance(platform, dict):
            raise VerificationError("published image index has an invalid platform")
        operating_system = platform.get("os")
        architecture = platform.get("architecture")
        if isinstance(operating_system, str) and isinstance(architecture, str):
            name = f"{operating_system}/{architecture}"
            if name not in EXPECTED_PLATFORMS:
                continue
            child_digest = manifest.get("digest")
            if (
                not isinstance(child_digest, str)
                or MANIFEST_DIGEST.fullmatch(child_digest) is None
            ):
                raise VerificationError(
                    f"published {name} manifest has no valid digest"
                )
            if name in platforms:
                raise VerificationError(
                    f"published image has duplicate {name} manifests"
                )
            if child_digest in platforms.values():
                raise VerificationError(
                    "published platforms cannot share one child manifest digest"
                )
            platforms[name] = child_digest
    return platforms


def verify_manifest(image: str) -> dict[str, str]:
    """Bind each supported platform to a child of the published index digest."""
    result = run_command(["docker", "buildx", "imagetools", "inspect", "--raw", image])
    platforms = manifest_platforms(result.stdout)
    missing = EXPECTED_PLATFORMS - platforms.keys()
    if missing:
        raise VerificationError(
            "published manifest is missing platforms: " + ", ".join(sorted(missing))
        )
    print("Published Dashboard manifest: " + ", ".join(sorted(platforms)))
    return platforms


def published_port(container: str) -> int:
    """Return the ephemeral host port mapped to the Dashboard HTTP port."""
    output = run_command(["docker", "port", container, "8700/tcp"]).stdout.strip()
    first_binding = output.splitlines()[0] if output else ""
    raw_port = first_binding.rsplit(":", 1)[-1]
    if not raw_port.isdigit():
        raise VerificationError(f"could not resolve Dashboard port from {output!r}")
    return int(raw_port)


def container_is_running(container: str) -> bool:
    """Return whether Docker still reports the test container as running."""
    result = run_command(
        ["docker", "inspect", "--format", "{{.State.Running}}", container],
        check=False,
    )
    return result.returncode == 0 and result.stdout.strip() == "true"


def validate_health_response(status: int, body: bytes) -> None:
    """Assert the Dashboard's public health response contract."""
    if status != HTTPStatus.OK:
        raise VerificationError(f"Dashboard health endpoint returned HTTP {status}")
    try:
        response = json.loads(body)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise VerificationError(
            f"Dashboard health response is not valid JSON: {error}"
        ) from error
    if not isinstance(response, dict):
        raise VerificationError("Dashboard health response JSON root is not an object")
    for key, expected in EXPECTED_HEALTH.items():
        if response.get(key) != expected:
            raise VerificationError(
                f"Dashboard health response has {key}={response.get(key)!r}, "
                f"expected {expected!r}"
            )


def wait_for_health(container: str, timeout_seconds: float = 90) -> None:
    """Wait until the Dashboard container serves its healthy response."""
    url = f"http://127.0.0.1:{published_port(container)}/healthz"
    deadline = time.monotonic() + timeout_seconds
    last_error = "health endpoint was not ready"

    while time.monotonic() < deadline:
        if not container_is_running(container):
            raise VerificationError(
                "Dashboard container exited before becoming healthy"
            )
        try:
            with urllib.request.urlopen(url, timeout=5) as response:
                validate_health_response(response.status, response.read())
            return
        except OSError as error:
            last_error = str(error)
            time.sleep(2)

    raise VerificationError(
        f"Dashboard did not become healthy within {timeout_seconds:g}s: {last_error}"
    )


def verify_runtime(image: str, platform: str) -> None:
    """Start one published platform variant and verify its health endpoint."""
    container = (
        f"published-dashboard-{platform.rsplit('/', 1)[-1]}-{uuid.uuid4().hex[:8]}"
    )
    command = [
        "docker",
        "run",
        "--detach",
        "--platform",
        platform,
        "--name",
        container,
        "--publish",
        "127.0.0.1::8700",
        "--env",
        "DASHBOARD_READONLY=true",
        "--env",
        "DASHBOARD_RUNTIME_CONFIG_WRITABLE=false",
        "--env",
        "DASHBOARD_RECIPE_STORE_WRITABLE=false",
        "--env",
        "EVALUATION_ENABLED=false",
        "--env",
        "MCP_ENABLED=false",
        "--env",
        "ML_PIPELINE_ENABLED=false",
        "--env",
        "OPENCLAW_ENABLED=false",
        image,
    ]

    try:
        run_command(command)
        wait_for_health(container)
    except VerificationError as error:
        logs = run_command(["docker", "logs", container], check=False)
        detail = (logs.stderr + logs.stdout).strip()
        if detail:
            detail = f"\nContainer logs:\n{detail[-4000:]}"
        raise VerificationError(
            f"{platform} runtime check failed: {error}{detail}"
        ) from error
    finally:
        run_command(["docker", "rm", "--force", container], check=False)

    print(f"Published Dashboard runtime: {platform} /healthz is healthy")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "image",
        help="published Dashboard image pinned to its multi-platform manifest digest",
    )
    args = parser.parse_args()

    if IMAGE_BY_DIGEST.fullmatch(args.image) is None:
        print(
            "published Dashboard verification requires an image@sha256:<digest> reference",
            file=sys.stderr,
        )
        return 2

    try:
        platforms = verify_manifest(args.image)
        repository = args.image.rsplit("@", 1)[0]
        for platform in sorted(EXPECTED_PLATFORMS):
            # Docker's classic image store cannot pull a second architecture
            # through the same multi-platform digest after the first one.
            # Each child digest is immutable and was verified in that index.
            verify_runtime(f"{repository}@{platforms[platform]}", platform)
    except VerificationError as error:
        print(f"Published Dashboard verification failed: {error}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

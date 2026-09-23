"""Low-level container seams for one standalone Decision runtime."""

from __future__ import annotations

import json
import os
import re
import subprocess
import time
import urllib.error
import urllib.request
from collections.abc import Mapping
from contextlib import suppress
from dataclasses import dataclass
from http import HTTPStatus
from pathlib import Path, PurePosixPath
from typing import Protocol

from cli.consts import (
    CONTAINER_RUNTIME_ENV,
    DEFAULT_NOFILE_LIMIT,
    IMAGE_PULL_POLICY_NEVER,
    SUPPORTED_CONTAINER_RUNTIMES,
)
from cli.container_images import _ensure_image_available
from cli.container_run_command import (
    append_amd_gpu_passthrough,
    append_env_vars,
    append_nvidia_gpu_passthrough,
    append_port_mappings,
    build_base_run_command,
)
from cli.container_runtime import get_container_runtime, reset_container_runtime_cache
from cli.container_services import (
    CONTAINER_STOP_COMMAND_TIMEOUT_SECONDS,
    CONTAINER_STOP_GRACE_SECONDS,
    container_logs,
)
from cli.container_start_runner import run_container_specs
from cli.decision_runtime.catalog import DecisionRuntimeMount, ResolvedDecisionRuntime
from cli.decision_runtime.gpu_device import (
    GPU_DEVICE_INDEX,
    ROCM_VISIBLE_DEVICES_ENV,
)
from cli.decision_runtime.image_reference import (
    ImmutableImageReferenceError,
    is_local_docker_image_id,
    validate_decision_image_reference,
)

MANAGED_LABEL = "ai.vllm-sr.drun.managed"
INSTANCE_LABEL = "ai.vllm-sr.drun.instance"
IDENTITY_LABEL = "ai.vllm-sr.drun.identity"
IMAGE_LABEL = "ai.vllm-sr.drun.image"
_PUBLIC_TUNING_ENVIRONMENT = {
    "DECISION_RUNTIME_LOG_LEVEL": re.compile(r"(?:critical|error|warning|info|debug)"),
    "DECISION_CPU_THREADS": re.compile(
        r"(?:[1-9]|[1-9][0-9]|1[0-9][0-9]|2[0-4][0-9]|25[0-6])"
    ),
    "OMP_NUM_THREADS": re.compile(r"[1-9][0-9]{0,3}"),
    "TOKENIZERS_PARALLELISM": re.compile(r"(?:true|false)"),
    ROCM_VISIBLE_DEVICES_ENV: GPU_DEVICE_INDEX,
}
_CONTAINER_ID = re.compile(r"[0-9a-f]{64}")
_CONTAINER_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}")
_INSTANCE_NAME = re.compile(r"[a-z0-9](?:[a-z0-9_.-]{0,62}[a-z0-9])?")
_SHA256_DIGEST = re.compile(r"sha256:[0-9a-f]{64}")
_INSPECT_TIMEOUT_SECONDS = 10
_READINESS_BODY_LIMIT = 4096
_NO_SUCH_CONTAINER_MARKERS = (
    "no such container",
    "no such object",
    "no container with name or id",
)


class DecisionContainerError(RuntimeError):
    """A standalone Decision runtime container operation failed."""


class DecisionContainerOwnershipError(DecisionContainerError):
    """A container exists but cannot be proven to belong to this launch."""


@dataclass(frozen=True)
class DecisionContainerLaunch:
    """Fully resolved, secret-free container launch inputs."""

    runtime: str
    container_name: str
    instance_name: str
    identity_digest: str
    host: str
    port: int
    pull_policy: str
    runtime_spec: ResolvedDecisionRuntime

    def identity(self, *, container_id: str | None = None) -> DecisionContainerIdentity:
        """Return the immutable ownership contract encoded in container labels."""

        return DecisionContainerIdentity(
            runtime=self.runtime,
            container_name=self.container_name,
            instance_name=self.instance_name,
            identity_digest=self.identity_digest,
            image=self.runtime_spec.image,
            container_id=container_id,
        )


@dataclass(frozen=True)
class DecisionContainerIdentity:
    """Expected immutable identity for one managed container."""

    runtime: str
    container_name: str
    instance_name: str
    identity_digest: str
    image: str
    container_id: str | None = None


@dataclass(frozen=True)
class DecisionContainerObservation:
    """An ownership-verified runtime observation."""

    identity: DecisionContainerIdentity
    state: str


@dataclass(frozen=True)
class DecisionContainerCandidate:
    """Non-authoritative summary of a container carrying the managed label."""

    runtime: str
    container_id: str
    container_name: str
    state: str
    instance_name: str | None
    identity_digest: str | None
    image: str
    label_contract_valid: bool


class DecisionContainerDriver(Protocol):
    """Minimal lifecycle seam used by the control-plane."""

    def ensure_image(self, launch: DecisionContainerLaunch) -> None:
        """Make the selected immutable runtime image available."""

    def start(self, launch: DecisionContainerLaunch) -> DecisionContainerObservation:
        """Create the managed container and return verified ownership."""

    def inspect(
        self, identity: DecisionContainerIdentity
    ) -> DecisionContainerObservation | None:
        """Return an ownership-verified observation, or ``None`` if absent."""

    def wait_ready(
        self,
        launch: DecisionContainerLaunch,
        ownership: DecisionContainerIdentity,
        *,
        startup_timeout: int,
    ) -> None:
        """Wait for the runtime's declared health endpoint."""

    def probe_ready(
        self,
        ownership: DecisionContainerIdentity,
        *,
        host: str,
        port: int,
    ) -> bool:
        """Probe the strict current readiness contract without changing state."""

    def follow_logs(self, ownership: DecisionContainerIdentity) -> None:
        """Attach the foreground command to container logs."""

    def stop_and_remove(self, ownership: DecisionContainerIdentity) -> None:
        """Stop and remove only a container whose ownership still matches."""

    def list_managed(self, runtime: str) -> tuple[DecisionContainerCandidate, ...]:
        """List labeled candidates without adopting or mutating them."""


class LowLevelDecisionContainerDriver:
    """Container implementation built only from low-level CLI seams."""

    def ensure_image(self, launch: DecisionContainerLaunch) -> None:
        if _validated_launch_image(launch):
            _ensure_local_docker_image_id(launch.runtime_spec.image)
            return
        try:
            _ensure_image_available(launch.runtime_spec.image, launch.pull_policy)
        except Exception as error:
            raise DecisionContainerError(
                f"Unable to prepare immutable Decision runtime image "
                f"{launch.runtime_spec.image!r}: {_safe_detail(error)}"
            ) from error

    def start(self, launch: DecisionContainerLaunch) -> DecisionContainerObservation:
        command = build_decision_container_command(launch)
        try:
            return_code, stdout, stderr = run_container_specs(
                [("decision-runtime", launch.container_name, [command])],
                storage_secret_values={},
            )
        except Exception as error:
            raise DecisionContainerError(
                "Decision runtime container command could not be executed: "
                f"{_safe_detail(error)}"
            ) from error
        if return_code != 0:
            detail = _safe_detail(
                stderr or stdout or "container runtime returned an error"
            )
            raise DecisionContainerError(
                f"Failed to start Decision runtime container "
                f"{launch.container_name!r}: {detail}"
            )

        output_ids = [
            line.strip().lower()
            for line in stdout.splitlines()
            if _CONTAINER_ID.fullmatch(line.strip().lower())
        ]
        expected = launch.identity(container_id=output_ids[-1] if output_ids else None)
        try:
            observation = self.inspect(expected)
        except DecisionContainerOwnershipError:
            raise
        except DecisionContainerError as error:
            raise DecisionContainerOwnershipError(
                "Decision runtime container was created, but its ownership could not "
                f"be verified: {error}"
            ) from error
        if observation is None:
            raise DecisionContainerOwnershipError(
                "Decision runtime container was created, but it disappeared before "
                "ownership could be verified."
            )
        return observation

    def inspect(
        self, identity: DecisionContainerIdentity
    ) -> DecisionContainerObservation | None:
        reference = identity.container_id or identity.container_name
        inspected = _inspect_container(identity.runtime, reference)
        if inspected is None:
            return None
        container_id, state, labels, image = inspected
        if identity.container_id is not None and container_id != identity.container_id:
            raise DecisionContainerOwnershipError(
                "Refusing to manage a container whose immutable ID changed."
            )
        expected_labels = {
            MANAGED_LABEL: "true",
            INSTANCE_LABEL: identity.instance_name,
            IDENTITY_LABEL: identity.identity_digest,
            IMAGE_LABEL: identity.image,
        }
        if any(labels.get(key) != value for key, value in expected_labels.items()):
            raise DecisionContainerOwnershipError(
                f"Refusing to manage container {identity.container_name!r}: "
                "ownership labels do not match the registry."
            )
        if image != identity.image:
            raise DecisionContainerOwnershipError(
                f"Refusing to manage container {identity.container_name!r}: "
                "its immutable image does not match the registry."
            )
        return DecisionContainerObservation(
            identity=DecisionContainerIdentity(
                runtime=identity.runtime,
                container_name=identity.container_name,
                instance_name=identity.instance_name,
                identity_digest=identity.identity_digest,
                image=identity.image,
                container_id=container_id,
            ),
            state=state,
        )

    def list_managed(self, runtime: str) -> tuple[DecisionContainerCandidate, ...]:
        """Surface managed-label candidates; registry evidence remains authoritative."""

        try:
            result = subprocess.run(
                [
                    runtime,
                    "ps",
                    "--all",
                    "--no-trunc",
                    "--filter",
                    f"label={MANAGED_LABEL}=true",
                    "--format",
                    "{{.ID}}",
                ],
                capture_output=True,
                text=True,
                check=False,
                timeout=_INSPECT_TIMEOUT_SECONDS,
            )
        except (OSError, subprocess.SubprocessError) as error:
            raise DecisionContainerError(
                "Decision runtime managed-container discovery failed: "
                f"{_safe_detail(error)}"
            ) from error
        if result.returncode != 0:
            raise DecisionContainerError(
                "Decision runtime managed-container discovery failed: "
                f"{_safe_detail(result.stderr or result.stdout)}"
            )
        references = tuple(
            line.strip().lower() for line in result.stdout.splitlines() if line.strip()
        )
        if any(_CONTAINER_ID.fullmatch(reference) is None for reference in references):
            raise DecisionContainerError(
                "Decision runtime managed-container discovery returned an invalid ID."
            )
        candidates = []
        for reference in dict.fromkeys(references):
            snapshot = _inspect_container_snapshot(runtime, reference)
            if snapshot is None:
                continue
            container_id, state, labels, image, container_name = snapshot
            instance_name = labels.get(INSTANCE_LABEL)
            identity_digest = labels.get(IDENTITY_LABEL)
            labeled_image = labels.get(IMAGE_LABEL)
            label_contract_valid = (
                labels.get(MANAGED_LABEL) == "true"
                and isinstance(instance_name, str)
                and _INSTANCE_NAME.fullmatch(instance_name) is not None
                and container_name == f"vllm-sr-drun-{instance_name}"
                and isinstance(identity_digest, str)
                and _SHA256_DIGEST.fullmatch(identity_digest) is not None
                and labeled_image == image
            )
            if label_contract_valid:
                try:
                    validate_decision_image_reference(image)
                except ImmutableImageReferenceError:
                    label_contract_valid = False
                if is_local_docker_image_id(image) and runtime != "docker":
                    label_contract_valid = False
            candidates.append(
                DecisionContainerCandidate(
                    runtime=runtime,
                    container_id=container_id,
                    container_name=container_name,
                    state=state,
                    instance_name=instance_name if label_contract_valid else None,
                    identity_digest=(identity_digest if label_contract_valid else None),
                    image=image,
                    label_contract_valid=label_contract_valid,
                )
            )
        return tuple(candidates)

    def wait_ready(
        self,
        launch: DecisionContainerLaunch,
        ownership: DecisionContainerIdentity,
        *,
        startup_timeout: int,
    ) -> None:
        deadline = time.monotonic() + startup_timeout
        probe_url = _host_url(
            _probe_host(launch.host),
            launch.port,
            launch.runtime_spec.health_path,
        )
        while time.monotonic() < deadline:
            observation = self.inspect(ownership)
            if observation is None:
                raise DecisionContainerError(
                    "Decision runtime container disappeared during startup."
                )
            if observation.state != "running":
                self._show_startup_logs(ownership)
                raise DecisionContainerError(
                    "Decision runtime container exited during startup "
                    f"({observation.state})."
                )
            if _http_ready(
                probe_url, timeout=min(2.0, max(0.1, deadline - time.monotonic()))
            ):
                final_observation = self.inspect(ownership)
                if final_observation is None or final_observation.state != "running":
                    raise DecisionContainerError(
                        "Decision runtime container changed state after its readiness "
                        "probe succeeded."
                    )
                return
            time.sleep(min(0.25, max(0.0, deadline - time.monotonic())))
        self._show_startup_logs(ownership)
        raise DecisionContainerError(
            f"Decision runtime did not become ready within {startup_timeout}s."
        )

    def probe_ready(
        self,
        ownership: DecisionContainerIdentity,
        *,
        host: str,
        port: int,
    ) -> bool:
        """Return current strict ``/ready`` health for the same owned container."""

        observation = self.inspect(ownership)
        if observation is None or observation.state != "running":
            return False
        if not _http_ready(
            _host_url(_probe_host(host), port, "/ready"),
            timeout=2.0,
        ):
            return False
        final_observation = self.inspect(ownership)
        return final_observation is not None and final_observation.state == "running"

    def follow_logs(self, ownership: DecisionContainerIdentity) -> None:
        reference = ownership.container_id or ownership.container_name
        try:
            succeeded = container_logs(reference, follow=True, merge_output=True)
        except Exception as error:
            raise DecisionContainerError(
                f"Failed to follow Decision runtime logs: {_safe_detail(error)}"
            ) from error
        if not succeeded:
            raise DecisionContainerError(
                f"Failed to follow Decision runtime logs for "
                f"{ownership.container_name!r}."
            )

    def stop_and_remove(self, ownership: DecisionContainerIdentity) -> None:
        observation = self.inspect(ownership)
        if observation is None:
            return
        container_id = observation.identity.container_id
        if container_id is None:  # inspect always supplies one; fail closed.
            raise DecisionContainerOwnershipError(
                "Refusing cleanup without an immutable container ID."
            )
        if observation.state in {
            "running",
            "restarting",
            "paused",
        }:
            _run_destructive_container_command(
                ownership.runtime,
                [
                    "stop",
                    "--time",
                    str(CONTAINER_STOP_GRACE_SECONDS),
                    container_id,
                ],
                timeout=CONTAINER_STOP_COMMAND_TIMEOUT_SECONDS,
                action="stop",
            )
        elif observation.state == "removing":
            raise DecisionContainerError(
                "Decision runtime container removal is already in progress."
            )

        # Re-inspect after the stop. This prevents a stale name or ID from ever
        # being used for the destructive remove step.
        observation = self.inspect(observation.identity)
        if observation is None:
            return
        container_id = observation.identity.container_id
        if container_id is None:
            raise DecisionContainerOwnershipError(
                "Refusing cleanup without an immutable container ID."
            )
        _run_destructive_container_command(
            ownership.runtime,
            ["rm", container_id],
            timeout=_INSPECT_TIMEOUT_SECONDS,
            action="remove",
        )

    @staticmethod
    def _show_startup_logs(ownership: DecisionContainerIdentity) -> None:
        if ownership.container_id is None:
            return
        with suppress(Exception):
            container_logs(ownership.container_id, follow=False, tail=100, timeout=5)


def select_container_runtime(requested: str | None) -> str:
    """Resolve one Docker-compatible runtime for this CLI process."""

    if requested is not None:
        normalized = requested.strip().lower()
        if normalized not in SUPPORTED_CONTAINER_RUNTIMES:
            raise DecisionContainerError(
                f"Unsupported container runtime {requested!r}; choose "
                f"{', '.join(SUPPORTED_CONTAINER_RUNTIMES)}."
            )
        os.environ[CONTAINER_RUNTIME_ENV] = normalized
        reset_container_runtime_cache()
    try:
        return get_container_runtime()
    except SystemExit as error:
        raise DecisionContainerError(
            "No usable Docker-compatible container runtime is available."
        ) from error
    except Exception as error:
        raise DecisionContainerError(
            f"Container runtime selection failed: {_safe_detail(error)}"
        ) from error


def build_decision_container_command(
    launch: DecisionContainerLaunch,
) -> list[str]:
    """Build one stable, inspectable container command from resolved inputs."""

    spec = launch.runtime_spec
    local_image_id = _validated_launch_image(launch)
    validate_decision_environment(spec.environment)
    if ROCM_VISIBLE_DEVICES_ENV in spec.environment and spec.backend != "rocm":
        raise DecisionContainerError(
            "ROCR_VISIBLE_DEVICES is supported only for the ROCm Decision backend."
        )
    command = build_base_run_command(
        launch.runtime,
        DEFAULT_NOFILE_LIMIT,
        None,
        launch.container_name,
        start_immediately=True,
    )
    command.extend(["--label", f"{MANAGED_LABEL}=true"])
    command.extend(["--label", f"{INSTANCE_LABEL}={launch.instance_name}"])
    command.extend(["--label", f"{IDENTITY_LABEL}={launch.identity_digest}"])
    command.extend(["--label", f"{IMAGE_LABEL}={spec.image}"])
    append_port_mappings(
        command,
        [(launch.host, launch.port, spec.container_port)],
    )
    append_env_vars(
        command,
        {key: spec.environment[key] for key in sorted(spec.environment)},
    )
    for mount in spec.mounts:
        source, target = validate_decision_mount(mount)
        command.extend(
            [
                "--mount",
                f"type=bind,src={source},dst={target},readonly",
            ]
        )
    if spec.backend == "rocm":
        append_amd_gpu_passthrough(command, "amd")
    elif spec.backend == "cuda":
        append_nvidia_gpu_passthrough(command, launch.runtime)
    if local_image_id:
        command.append("--pull=never")
    command.append(spec.image)
    command.extend(spec.command)
    return command


def _validated_launch_image(launch: DecisionContainerLaunch) -> bool:
    """Keep local IDs on Docker's no-pull path at every launch seam."""

    try:
        validate_decision_image_reference(launch.runtime_spec.image)
    except ImmutableImageReferenceError as error:
        raise DecisionContainerError(str(error)) from error
    local_image_id = is_local_docker_image_id(launch.runtime_spec.image)
    if local_image_id and (
        launch.runtime != "docker" or launch.pull_policy != IMAGE_PULL_POLICY_NEVER
    ):
        raise DecisionContainerError(
            "A local Docker image ID requires Docker and image pull policy 'never'."
        )
    return local_image_id


def _ensure_local_docker_image_id(image_id: str) -> None:
    """Inspect an exact local image ID without invoking any pull mechanism."""

    try:
        result = subprocess.run(
            ["docker", "image", "inspect", "--format", "{{.Id}}", image_id],
            capture_output=True,
            text=True,
            check=False,
            timeout=_INSPECT_TIMEOUT_SECONDS,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise DecisionContainerError(
            f"Local Docker image ID could not be inspected: {_safe_detail(error)}"
        ) from error
    if result.returncode != 0 or result.stdout.strip() != image_id:
        raise DecisionContainerError(
            "The exact local Docker image ID is not present; no pull was attempted."
        )


def validate_decision_environment(environment: object) -> None:
    """Allow only documented, value-constrained public runtime tunings."""

    if not isinstance(environment, Mapping):
        raise DecisionContainerError("Decision runtime environment is invalid.")
    for key, value in environment.items():
        validator = (
            _PUBLIC_TUNING_ENVIRONMENT.get(key) if isinstance(key, str) else None
        )
        if validator is None:
            raise DecisionContainerError(
                "Decision runtime launch profiles may contain only documented public "
                "tuning environment variables."
            )
        if not isinstance(value, str) or validator.fullmatch(value) is None:
            raise DecisionContainerError(
                "Decision runtime public tuning environment value is invalid."
            )


def validate_decision_mount(mount: object) -> tuple[str, str]:
    """Validate one immutable artifact bind without shell or mount-option ambiguity."""

    if not isinstance(mount, DecisionRuntimeMount):
        raise DecisionContainerError("Decision runtime mount is invalid.")
    if not isinstance(mount.source, str) or not isinstance(mount.target, str):
        raise DecisionContainerError("Decision runtime mount paths are invalid.")
    if any(
        not value
        or value != value.strip()
        or "\0" in value
        or "," in value
        or any(ord(character) < 32 for character in value)
        for value in (mount.source, mount.target)
    ):
        raise DecisionContainerError("Decision runtime mount paths are invalid.")

    source = Path(mount.source)
    if not source.is_absolute() or source.is_symlink() or not source.is_dir():
        raise DecisionContainerError(
            "Decision runtime mount source must be an existing absolute directory."
        )
    try:
        resolved_source = source.resolve(strict=True)
    except OSError as error:
        raise DecisionContainerError(
            "Decision runtime mount source is unavailable."
        ) from error
    if str(resolved_source) != mount.source:
        raise DecisionContainerError(
            "Decision runtime mount source must be a canonical absolute directory."
        )

    target = PurePosixPath(mount.target)
    if (
        not target.is_absolute()
        or target == PurePosixPath("/")
        or target.as_posix() != mount.target
        or any(part in {"", ".", ".."} for part in target.parts[1:])
        or "\\" in mount.target
        or ":" in mount.target
    ):
        raise DecisionContainerError(
            "Decision runtime mount target must be a canonical absolute POSIX path."
        )
    return str(resolved_source), target.as_posix()


def _probe_host(host: str) -> str:
    if host == "0.0.0.0":
        return "127.0.0.1"
    if host == "::":
        return "::1"
    return host


def _host_url(host: str, port: int, path: str) -> str:
    rendered_host = f"[{host}]" if ":" in host and not host.startswith("[") else host
    return f"http://{rendered_host}:{port}{path}"


def _http_ready(url: str, *, timeout: float) -> bool:
    request = urllib.request.Request(url, headers={"Accept": "application/json"})
    try:
        opener = urllib.request.build_opener(_RejectRedirects)
        with opener.open(request, timeout=timeout) as response:
            if response.status != HTTPStatus.OK:
                return False
            payload = response.read(_READINESS_BODY_LIMIT + 1)
    except (OSError, urllib.error.URLError):
        return False
    if len(payload) > _READINESS_BODY_LIMIT:
        return False
    try:
        decoded = json.loads(payload)
    except (UnicodeError, json.JSONDecodeError):
        return False
    return (
        isinstance(decoded, dict)
        and set(decoded) == {"ready"}
        and decoded["ready"] is True
    )


class _RejectRedirects(urllib.request.HTTPRedirectHandler):
    """Keep readiness bound to the configured endpoint and origin."""

    def redirect_request(self, request, file_pointer, code, message, headers, url):
        del request, file_pointer, code, message, headers, url


def _inspect_container(
    runtime: str, reference: str
) -> tuple[str, str, dict[str, str], str] | None:
    snapshot = _inspect_container_snapshot(runtime, reference)
    if snapshot is None:
        return None
    container_id, state, labels, image, _container_name = snapshot
    return container_id, state, labels, image


def _inspect_container_snapshot(
    runtime: str, reference: str
) -> tuple[str, str, dict[str, str], str, str] | None:
    try:
        result = subprocess.run(
            [runtime, "inspect", reference],
            capture_output=True,
            text=True,
            check=False,
            timeout=_INSPECT_TIMEOUT_SECONDS,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise DecisionContainerError(
            f"Decision runtime ownership inspection failed: {_safe_detail(error)}"
        ) from error
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "").lower()
        if any(marker in detail for marker in _NO_SUCH_CONTAINER_MARKERS):
            return None
        raise DecisionContainerError(
            "Decision runtime ownership inspection failed: "
            f"{_safe_detail(result.stderr or result.stdout)}"
        )
    try:
        payload = json.loads(result.stdout)
    except (TypeError, json.JSONDecodeError) as error:
        raise DecisionContainerError(
            "Decision runtime ownership inspection returned invalid JSON."
        ) from error
    if (
        not isinstance(payload, list)
        or len(payload) != 1
        or not isinstance(payload[0], dict)
    ):
        raise DecisionContainerError(
            "Decision runtime ownership inspection returned an invalid record."
        )
    snapshot = payload[0]
    config = snapshot.get("Config")
    state_record = snapshot.get("State")
    if not isinstance(config, dict) or not isinstance(state_record, dict):
        raise DecisionContainerError(
            "Decision runtime ownership inspection omitted required fields."
        )
    container_id = snapshot.get("Id")
    state = state_record.get("Status")
    labels = config.get("Labels")
    image = config.get("Image")
    container_name = snapshot.get("Name")
    if isinstance(container_name, str):
        container_name = container_name.removeprefix("/")
    if (
        not isinstance(container_id, str)
        or not _CONTAINER_ID.fullmatch(container_id.lower())
        or not isinstance(state, str)
        or state.lower()
        not in {
            "created",
            "restarting",
            "running",
            "removing",
            "paused",
            "exited",
            "dead",
        }
        or not isinstance(labels, dict)
        or any(
            not isinstance(key, str) or not isinstance(value, str)
            for key, value in labels.items()
        )
        or not isinstance(image, str)
        or not isinstance(container_name, str)
        or _CONTAINER_NAME.fullmatch(container_name) is None
    ):
        raise DecisionContainerError(
            "Decision runtime ownership inspection returned invalid fields."
        )
    return container_id.lower(), state.lower(), labels, image, container_name


def _run_destructive_container_command(
    runtime: str,
    arguments: list[str],
    *,
    timeout: float,
    action: str,
) -> None:
    try:
        result = subprocess.run(
            [runtime, *arguments],
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise DecisionContainerError(
            f"Failed to {action} Decision runtime container: {_safe_detail(error)}"
        ) from error
    if result.returncode != 0:
        raise DecisionContainerError(
            f"Failed to {action} Decision runtime container: "
            f"{_safe_detail(result.stderr or result.stdout)}"
        )


def _safe_detail(value: object, *, limit: int = 1000) -> str:
    rendered = " ".join(str(value).split()) or "no diagnostic was provided"
    if len(rendered) > limit:
        return rendered[: limit - 3] + "..."
    return rendered

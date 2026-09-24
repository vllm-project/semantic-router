from __future__ import annotations

import importlib
import json
import os
import stat
import subprocess
import sys
import threading
from dataclasses import replace
from pathlib import Path

import pytest
from click.testing import CliRunner

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

decision_command = importlib.import_module("cli.commands.decision")
catalog_module = importlib.import_module("cli.decision_runtime.catalog")
container_module = importlib.import_module("cli.decision_runtime.container")
registry_module = importlib.import_module("cli.decision_runtime.registry")
from cli.decision_runtime import lifecycle  # noqa: E402
from cli.decision_runtime.catalog import (  # noqa: E402
    DecisionRuntimeMount,
    DecisionRuntimeRequest,
    ResolvedDecisionRuntime,
)
from cli.decision_runtime.container import (  # noqa: E402
    DecisionContainerCandidate,
    DecisionContainerError,
    DecisionContainerIdentity,
    DecisionContainerLaunch,
    DecisionContainerObservation,
    DecisionContainerOwnershipError,
    LowLevelDecisionContainerDriver,
    build_decision_container_command,
)
from cli.decision_runtime.image_reference import (  # noqa: E402
    ImmutableImageReferenceError,
    validate_decision_image_reference,
    validate_immutable_image_reference,
)
from cli.decision_runtime.lifecycle import (  # noqa: E402
    DecisionServeOptions,
    run_decision_runtime,
)
from cli.decision_runtime.management import (  # noqa: E402
    DecisionDiscoveryStatus,
    DecisionManagedStatus,
    DecisionManagementError,
    DecisionOrphanStatus,
    forget_decision_instance,
    list_decision_instances,
    status_decision_instance,
    stop_decision_instance,
)
from cli.decision_runtime.registry import (  # noqa: E402
    DecisionInstanceRecord,
    DecisionInstanceRegistry,
    DecisionRegistryError,
    DecisionRegistryStaleRecordError,
)
from cli.main import main  # noqa: E402

REVISION = "a" * 40
ARTIFACT_DIGEST = f"sha256:{'b' * 64}"
IMAGE = f"example.test/decision-runtime@sha256:{'c' * 64}"
LOCAL_IMAGE_ID = f"sha256:{'9' * 64}"
MODEL = "llm-semantic-router/Decision-1.0-Kai-0.6B"
SOL_MODEL = "llm-semantic-router/Decision-1.0-Sol-2B"
CONTAINER_ID = "d" * 64


class FixtureResolver:
    def __init__(self) -> None:
        self.requests: list[DecisionRuntimeRequest] = []

    def resolve(self, request: DecisionRuntimeRequest) -> ResolvedDecisionRuntime:
        self.requests.append(request)
        return ResolvedDecisionRuntime(
            canonical_model=MODEL,
            revision=request.revision or REVISION,
            backend="rocm" if request.backend == "auto" else request.backend,
            dtype="bfloat16",
            image=request.image or IMAGE,
            artifact_digest=ARTIFACT_DIGEST,
            command=("python", "-m", "decision_runtime.service"),
            environment={"DECISION_RUNTIME_LOG_LEVEL": "info"},
            max_batch=request.max_batch or 8,
            max_concurrency=request.max_concurrency or 8,
            max_queue=32 if request.max_queue is None else request.max_queue,
        )


class FakeDriver:
    def __init__(self, *, fail_at: str | None = None) -> None:
        self.fail_at = fail_at
        self.events: list[object] = []

    def ensure_image(self, launch: DecisionContainerLaunch) -> None:
        self.events.append(("ensure", launch))
        self._maybe_fail("ensure")

    def start(self, launch: DecisionContainerLaunch) -> DecisionContainerObservation:
        self.events.append(("start", launch))
        self._maybe_fail("start")
        return DecisionContainerObservation(
            identity=launch.identity(container_id=CONTAINER_ID),
            state="running",
            restart_policy=launch.restart_policy,
        )

    def inspect(
        self, identity: DecisionContainerIdentity
    ) -> DecisionContainerObservation | None:
        self.events.append(("inspect", identity))
        self._maybe_fail("inspect")
        return DecisionContainerObservation(identity=identity, state="running")

    def wait_ready(
        self,
        launch: DecisionContainerLaunch,
        ownership: DecisionContainerIdentity,
        *,
        startup_timeout: int,
    ) -> None:
        self.events.append(("ready", launch, ownership, startup_timeout))
        self._maybe_fail("ready")

    def probe_ready(
        self,
        ownership: DecisionContainerIdentity,
        *,
        host: str,
        port: int,
    ) -> bool:
        self.events.append(("probe", ownership, host, port))
        self._maybe_fail("probe")
        return True

    def follow_logs(self, ownership: DecisionContainerIdentity) -> None:
        self.events.append(("logs", ownership))
        self._maybe_fail("logs")

    def stop_and_remove(self, ownership: DecisionContainerIdentity) -> None:
        self.events.append(("stop", ownership))
        self._maybe_fail("stop")

    def _maybe_fail(self, stage: str) -> None:
        if self.fail_at == stage:
            raise DecisionContainerError(f"fixture {stage} failure")


def _launch(
    spec: ResolvedDecisionRuntime | None = None,
    *,
    identity_digest: str | None = None,
) -> DecisionContainerLaunch:
    runtime_spec = spec or FixtureResolver().resolve(
        DecisionRuntimeRequest(MODEL, None, "rocm", None, None, None, None)
    )
    return DecisionContainerLaunch(
        runtime="docker",
        container_name="vllm-sr-decision-fixture",
        instance_name="fixture",
        identity_digest=identity_digest or f"sha256:{'e' * 64}",
        host="127.0.0.1",
        port=8000,
        pull_policy="never",
        runtime_spec=runtime_spec,
    )


def _inspect_completed_process(
    identity: DecisionContainerIdentity,
    *,
    state: str = "running",
    labels: dict[str, str] | None = None,
    image: str | None = None,
    container_id: str | None = None,
    restart_policy: str = "no",
) -> subprocess.CompletedProcess[str]:
    effective_labels = labels or {
        "ai.vllm-sr.decision.managed": "true",
        "ai.vllm-sr.decision.instance": identity.instance_name,
        "ai.vllm-sr.decision.identity": identity.identity_digest,
        "ai.vllm-sr.decision.image": identity.image,
    }
    payload = [
        {
            "Id": container_id or identity.container_id or CONTAINER_ID,
            "Name": f"/{identity.container_name}",
            "State": {"Status": state},
            "HostConfig": {"RestartPolicy": {"Name": restart_policy}},
            "Config": {
                "Labels": effective_labels,
                "Image": image or identity.image,
            },
        }
    ]
    return subprocess.CompletedProcess(
        [identity.runtime, "inspect"],
        0,
        stdout=json.dumps(payload),
        stderr="",
    )


def test_main_always_registers_decision_recovery_commands():
    result = CliRunner().invoke(main, ["--help"])

    assert result.exit_code == 0, result.output
    assert "decision" in main.commands
    assert " decision " in result.output


def test_decision_management_commands_do_not_load_catalog(monkeypatch):
    monkeypatch.setattr(
        decision_command,
        "default_catalog_resolver",
        lambda: (_ for _ in ()).throw(AssertionError("catalog must not be loaded")),
    )
    monkeypatch.setattr(decision_command, "list_decision_instances", lambda: ())

    result = CliRunner().invoke(main, ["decision", "list"])

    assert result.exit_code == 0, result.output
    assert "No managed Decision runtime instances." in result.output


def test_decision_list_renders_registry_runtime_and_readiness_separately(monkeypatch):
    statuses = (
        DecisionManagedStatus(
            record=_record("active-start", "id-active-start", 8000),
            runtime_state="running",
            readiness_state="not-applicable",
            ownership_verified=True,
        ),
        DecisionManagedStatus(
            record=replace(
                _record("healthy", "id-healthy", 8001),
                state="running",
                container_id=CONTAINER_ID,
            ),
            runtime_state="running",
            readiness_state="not-probed",
            ownership_verified=True,
        ),
        DecisionManagedStatus(
            record=replace(
                _record("needs-cleanup", "id-needs-cleanup", 8002),
                state="cleanup-required",
                container_id=CONTAINER_ID,
            ),
            runtime_state="running",
            readiness_state="not-applicable",
            ownership_verified=True,
        ),
    )
    monkeypatch.setattr(decision_command, "list_decision_instances", lambda: statuses)

    result = CliRunner().invoke(decision_command.decision, ["list"])

    assert result.exit_code == 0, result.output
    assert (
        "active-start\tregistry=starting\truntime=running\t"
        "readiness=not-applicable\townership=verified"
    ) in result.output
    assert (
        "healthy\tregistry=running\truntime=running\t"
        "readiness=not-probed\townership=verified"
    ) in result.output
    assert (
        "needs-cleanup\tregistry=cleanup-required\truntime=running\t"
        "readiness=not-applicable\townership=verified"
    ) in result.output
    assert "\trunning\tverified\t" not in result.output


def test_decision_status_renders_current_strict_readiness(monkeypatch):
    managed = DecisionManagedStatus(
        record=replace(
            _record("healthy", "id-healthy", 8000),
            state="running",
            container_id=CONTAINER_ID,
        ),
        runtime_state="running",
        readiness_state="ready",
        ownership_verified=True,
    )
    monkeypatch.setattr(
        decision_command,
        "status_decision_instance",
        lambda _instance_name: managed,
    )

    result = CliRunner().invoke(decision_command.decision, ["status", "healthy"])

    assert result.exit_code == 0, result.output
    assert (
        "healthy\tregistry=running\truntime=running\t"
        "readiness=ready\townership=verified"
    ) in result.output


def test_decision_list_surfaces_each_runtime_discovery_warning(monkeypatch):
    monkeypatch.setattr(
        decision_command,
        "list_decision_instances",
        lambda: (
            DecisionDiscoveryStatus(runtime="docker"),
            DecisionDiscoveryStatus(runtime="podman"),
        ),
    )

    result = CliRunner().invoke(decision_command.decision, ["list"])

    assert result.exit_code == 0, result.output
    assert "Warning: docker managed-label discovery is unavailable" in result.output
    assert "Warning: podman managed-label discovery is unavailable" in result.output


def test_decision_list_identifies_orphan_runtime_without_adopting_it(monkeypatch):
    orphan = DecisionOrphanStatus(
        candidate=DecisionContainerCandidate(
            runtime="podman",
            container_id="f" * 64,
            container_name="vllm-sr-decision-orphan",
            state="running",
            instance_name="orphan",
            identity_digest=f"sha256:{'f' * 64}",
            image=IMAGE,
            label_contract_valid=True,
        )
    )
    monkeypatch.setattr(
        decision_command,
        "list_decision_instances",
        lambda: (orphan,),
    )

    result = CliRunner().invoke(decision_command.decision, ["list"])

    assert result.exit_code == 0, result.output
    assert "orphan\tregistry=unregistered\truntime=running" in result.output
    assert "readiness=not-probed\tengine=podman" in result.output
    assert "it was not adopted or changed" in result.output


def test_default_catalog_resolver_uses_integrated_adapter():
    from cli.decision_runtime.catalog_adapter import (  # noqa: PLC0415
        IntegratedDecisionCatalogResolver,
    )

    assert isinstance(
        catalog_module.default_catalog_resolver(), IntegratedDecisionCatalogResolver
    )


def test_decision_group_exposes_lifecycle_commands():
    result = CliRunner().invoke(decision_command.decision, ["--help"])

    assert result.exit_code == 0, result.output
    for command in ("serve", "list", "status", "stop", "forget"):
        assert command in result.output
    assert "run" not in decision_command.decision.commands

    forget_help = CliRunner().invoke(decision_command.decision, ["forget", "--help"])
    assert forget_help.exit_code == 0, forget_help.output
    assert "--force" in forget_help.output
    assert "no container is changed" in forget_help.output


def test_decision_serve_help_exposes_required_model_and_lifecycle_options():
    result = CliRunner().invoke(decision_command.decision, ["serve", "--help"])

    assert result.exit_code == 0, result.output
    assert "Usage: decision serve [OPTIONS] MODEL" in result.output
    for option in (
        "--revision",
        "--host",
        "--port",
        "--backend",
        "--max-batch",
        "--max-concurrency",
        "--max-queue",
        "--cpu-threads",
        "--instance-name",
        "--image",
        "--image-pull-policy",
        "--runtime",
        "--startup-timeout",
        "--detach",
    ):
        assert option in result.output
    assert "[auto|rocm|cuda|cpu]" in result.output
    assert "--dtype" not in result.output
    assert "mlx" not in result.output.lower()


def test_decision_serve_rejects_removed_dtype_override():
    result = CliRunner().invoke(
        decision_command.decision, ["serve", MODEL, "--dtype", "float32"]
    )

    assert result.exit_code == 2
    assert "No such option '--dtype'" in result.output


def test_decision_serve_requires_exact_positional_model():
    result = CliRunner().invoke(decision_command.decision, ["serve"])

    assert result.exit_code == 2
    assert "Missing argument 'MODEL'" in result.output


def test_decision_does_not_accept_model_as_implicit_serve_command():
    result = CliRunner().invoke(decision_command.decision, [MODEL])

    assert result.exit_code == 2
    assert "No such command" in result.output


def test_decision_rejects_invalid_cpu_thread_count_before_catalog_resolution():
    result = CliRunner().invoke(
        decision_command.decision, ["serve", MODEL, "--cpu-threads", "0"]
    )

    assert result.exit_code == 2
    assert "--cpu-threads" in result.output


def test_decision_cli_forwards_cpu_thread_override(monkeypatch):
    options_seen: list[DecisionServeOptions] = []

    def record_options(options: DecisionServeOptions, **_kwargs: object) -> None:
        options_seen.append(options)

    monkeypatch.setattr(decision_command, "default_catalog_resolver", object)
    monkeypatch.setattr(decision_command, "run_decision_runtime", record_options)

    result = CliRunner().invoke(
        decision_command.decision,
        ["serve", MODEL, "--backend", "cpu", "--cpu-threads", "6"],
    )

    assert result.exit_code == 0, result.output
    assert len(options_seen) == 1
    assert options_seen[0].backend == "cpu"
    assert options_seen[0].cpu_threads == 6


def test_decision_cli_forwards_one_rocm_gpu_device(monkeypatch):
    options_seen: list[DecisionServeOptions] = []

    def record_options(options: DecisionServeOptions, **_kwargs: object) -> None:
        options_seen.append(options)

    monkeypatch.setattr(decision_command, "default_catalog_resolver", object)
    monkeypatch.setattr(decision_command, "run_decision_runtime", record_options)

    result = CliRunner().invoke(
        decision_command.decision,
        ["serve", MODEL, "--backend", "rocm", "--gpu-device", "2"],
    )

    assert result.exit_code == 0, result.output
    assert len(options_seen) == 1
    assert options_seen[0].backend == "rocm"
    assert options_seen[0].gpu_device == "2"


def test_decision_cli_no_longer_exposes_experimental_graph_flag():
    result = CliRunner().invoke(
        decision_command.decision,
        ["serve", SOL_MODEL, "--experimental-qwen-rocm-graph-b8"],
    )

    assert result.exit_code == 2
    assert "No such option" in result.output


def test_decision_cli_forwards_exact_local_docker_image_id(monkeypatch):
    options_seen: list[DecisionServeOptions] = []

    def record_options(options: DecisionServeOptions, **_kwargs: object) -> None:
        options_seen.append(options)

    monkeypatch.setattr(decision_command, "default_catalog_resolver", object)
    monkeypatch.setattr(decision_command, "run_decision_runtime", record_options)

    result = CliRunner().invoke(
        decision_command.decision,
        [
            "serve",
            MODEL,
            "--image",
            LOCAL_IMAGE_ID,
            "--image-pull-policy",
            "never",
            "--runtime",
            "docker",
        ],
    )

    assert result.exit_code == 0, result.output
    assert len(options_seen) == 1
    assert options_seen[0].image == LOCAL_IMAGE_ID
    assert options_seen[0].image_pull_policy == "never"
    assert options_seen[0].runtime == "docker"


def test_decision_cli_runs_fixture_lifecycle_end_to_end(monkeypatch, tmp_path: Path):
    resolver = FixtureResolver()
    driver = FakeDriver()
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path))
    monkeypatch.setattr(decision_command, "default_catalog_resolver", lambda: resolver)
    monkeypatch.setattr(lifecycle, "LowLevelDecisionContainerDriver", lambda: driver)
    monkeypatch.setattr(
        lifecycle,
        "select_container_runtime",
        lambda requested: requested or "docker",
    )

    result = CliRunner().invoke(
        decision_command.decision,
        [
            "serve",
            MODEL,
            "--revision",
            REVISION,
            "--port",
            "8123",
            "--backend",
            "rocm",
            "--max-batch",
            "16",
            "--max-concurrency",
            "24",
            "--max-queue",
            "96",
            "--instance-name",
            "kai-fixture",
            "--image",
            IMAGE,
            "--image-pull-policy",
            "never",
            "--runtime",
            "podman",
            "--startup-timeout",
            "45",
            "--detach",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "Decision runtime ready" in result.output
    assert f"Model: {MODEL}@{REVISION}" in result.output
    assert "Endpoint: http://127.0.0.1:8123/v1/systemone" in result.output
    assert "Mode: detached" in result.output
    assert [event[0] for event in driver.events] == ["ensure", "start", "ready"]
    assert driver.events[-1][3] == 45
    assert resolver.requests == [
        DecisionRuntimeRequest(
            model=MODEL,
            revision=REVISION,
            backend="rocm",
            image=IMAGE,
            max_batch=16,
            max_concurrency=24,
            max_queue=96,
        )
    ]
    records = DecisionInstanceRegistry().records()
    assert len(records) == 1
    assert records[0].instance_name == "kai-fixture"
    assert records[0].runtime == "podman"
    assert records[0].state == "running"
    assert records[0].artifact_digest == ARTIFACT_DIGEST
    assert records[0].container_id == CONTAINER_ID
    assert records[0].image == IMAGE


def test_decision_cli_opt_in_restart_policy_is_detached_and_docker_only(
    monkeypatch, tmp_path: Path
):
    driver = FakeDriver()
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path))
    monkeypatch.setattr(decision_command, "default_catalog_resolver", FixtureResolver)
    monkeypatch.setattr(lifecycle, "LowLevelDecisionContainerDriver", lambda: driver)
    monkeypatch.setattr(
        lifecycle,
        "select_container_runtime",
        lambda requested: requested or "docker",
    )

    result = CliRunner().invoke(
        decision_command.decision,
        [
            "serve",
            MODEL,
            "--instance-name",
            "durable-fixture",
            "--runtime",
            "docker",
            "--detach",
            "--restart-policy",
            "unless-stopped",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "Mode: detached" in result.output
    assert "Restart policy: unless-stopped" in result.output
    launch = next(event[1] for event in driver.events if event[0] == "start")
    assert launch.restart_policy == "unless-stopped"
    assert DecisionInstanceRegistry().get("durable-fixture").state == "running"


@pytest.mark.parametrize(
    ("options", "runtime", "message"),
    (
        (
            DecisionServeOptions(model=MODEL, restart_policy="unless-stopped"),
            "docker",
            "requires --detach",
        ),
        (
            DecisionServeOptions(
                model=MODEL, detach=True, restart_policy="unless-stopped"
            ),
            "podman",
            "requires the Docker container runtime",
        ),
    ),
)
def test_durable_restart_policy_rejects_foreground_and_podman_before_reservation(
    options: DecisionServeOptions, runtime: str, message: str, tmp_path: Path
):
    registry = DecisionInstanceRegistry(tmp_path / "instances.json")
    driver = FakeDriver()

    with pytest.raises(lifecycle.DecisionLifecycleError, match=message):
        run_decision_runtime(
            options,
            resolver=FixtureResolver(),
            registry=registry,
            driver=driver,
            runtime_selector=lambda _requested: runtime,
        )

    assert driver.events == []
    assert not registry.path.exists()


def test_durable_restart_policy_must_be_observed_before_ready(tmp_path: Path):
    class IgnoredRestartDriver(FakeDriver):
        def start(
            self, launch: DecisionContainerLaunch
        ) -> DecisionContainerObservation:
            self.events.append(("start", launch))
            return DecisionContainerObservation(
                identity=launch.identity(container_id=CONTAINER_ID),
                state="running",
                restart_policy="no",
            )

    registry = DecisionInstanceRegistry(tmp_path / "instances.json")
    driver = IgnoredRestartDriver()

    with pytest.raises(lifecycle.DecisionLifecycleError, match="did not apply"):
        run_decision_runtime(
            DecisionServeOptions(
                model=MODEL,
                instance_name="ignored-restart",
                detach=True,
                restart_policy="unless-stopped",
            ),
            resolver=FixtureResolver(),
            registry=registry,
            driver=driver,
            runtime_selector=lambda _requested: "docker",
        )

    assert [event[0] for event in driver.events] == ["ensure", "start", "stop"]
    assert registry.records() == ()


def test_durable_instance_remains_owned_and_status_ready_after_process_restart(
    monkeypatch, tmp_path: Path
):
    class ReconnectedDriver(FakeDriver):
        def __init__(self) -> None:
            super().__init__()
            self.state = "restarting"

        def inspect(self, identity: DecisionContainerIdentity):
            self.events.append(("inspect", identity))
            return DecisionContainerObservation(
                identity=identity,
                state=self.state,
                restart_policy="unless-stopped",
            )

    registry = DecisionInstanceRegistry(tmp_path / "instances.json")
    run_decision_runtime(
        DecisionServeOptions(
            model=MODEL,
            instance_name="recovered",
            detach=True,
            restart_policy="unless-stopped",
        ),
        resolver=FixtureResolver(),
        registry=registry,
        driver=FakeDriver(),
        runtime_selector=lambda _requested: "docker",
    )
    reconnected = ReconnectedDriver()
    restarting = status_decision_instance(
        "recovered", registry=registry, driver=reconnected
    )
    assert restarting.record.state == "running"
    assert restarting.runtime_state == "restarting"
    assert restarting.readiness_state == "not-ready"
    assert restarting.ownership_verified is True
    assert restarting.restart_policy == "unless-stopped"
    assert registry.get("recovered").state == "running"

    reconnected.state = "running"
    status = status_decision_instance(
        "recovered", registry=registry, driver=reconnected
    )

    assert status.record.state == "running"
    assert status.runtime_state == "running"
    assert status.readiness_state == "ready"
    assert status.ownership_verified is True
    assert status.restart_policy == "unless-stopped"
    monkeypatch.setattr(
        decision_command, "status_decision_instance", lambda _name: status
    )
    output = CliRunner().invoke(decision_command.decision, ["status", "recovered"])
    assert output.exit_code == 0, output.output
    assert "ownership=verified\trestart=unless-stopped" in output.output

    stopped = stop_decision_instance("recovered", registry=registry, driver=reconnected)
    assert stopped.action == "stopped"
    assert registry.records() == ()
    assert any(event[0] == "stop" for event in reconnected.events)


def test_foreground_launch_follows_logs_then_cleans_registry(tmp_path: Path):
    registry = DecisionInstanceRegistry(tmp_path / "instances.json")
    driver = FakeDriver()
    receipts = []

    receipt = run_decision_runtime(
        DecisionServeOptions(model=MODEL, instance_name="foreground"),
        resolver=FixtureResolver(),
        registry=registry,
        driver=driver,
        runtime_selector=lambda _requested: "docker",
        on_ready=receipts.append,
    )

    assert receipt.detached is False
    assert receipts == [receipt]
    assert [event[0] for event in driver.events] == [
        "ensure",
        "start",
        "ready",
        "logs",
        "stop",
    ]
    assert registry.records() == ()


def test_identical_launches_receive_distinct_ownership_identities(tmp_path: Path):
    registry = DecisionInstanceRegistry(tmp_path / "instances.json")
    driver = FakeDriver()

    for _attempt in range(2):
        run_decision_runtime(
            DecisionServeOptions(model=MODEL, instance_name="same-launch"),
            resolver=FixtureResolver(),
            registry=registry,
            driver=driver,
            runtime_selector=lambda _requested: "docker",
        )

    launches = [event[1] for event in driver.events if event[0] == "start"]
    assert len(launches) == 2
    assert launches[0].identity_digest != launches[1].identity_digest


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("runtime", "podman"),
        ("container_name", "vllm-sr-decision-somebody-else"),
        ("instance_name", "somebody-else"),
        ("identity_digest", f"sha256:{'f' * 64}"),
        ("image", f"example.test/other@sha256:{'f' * 64}"),
        ("container_id", "d" * 12),
    ),
)
def test_untrusted_start_receipt_is_never_cleaned_by_guess(
    field: str,
    value: str,
    tmp_path: Path,
):
    class BadReceiptDriver(FakeDriver):
        def start(
            self, launch: DecisionContainerLaunch
        ) -> DecisionContainerObservation:
            self.events.append(("start", launch))
            identity = replace(
                launch.identity(container_id=CONTAINER_ID),
                **{field: value},
            )
            return DecisionContainerObservation(identity=identity, state="running")

    registry = DecisionInstanceRegistry(tmp_path / "instances.json")
    driver = BadReceiptDriver()
    with pytest.raises(
        DecisionContainerOwnershipError,
        match="full, matching ownership receipt",
    ):
        run_decision_runtime(
            DecisionServeOptions(model=MODEL, instance_name="bad-receipt"),
            resolver=FixtureResolver(),
            registry=registry,
            driver=driver,
            runtime_selector=lambda _requested: "docker",
        )

    assert [event[0] for event in driver.events] == ["ensure", "start"]
    assert registry.records()[0].state == "cleanup-required"


def test_untrusted_start_state_is_never_accepted(tmp_path: Path):
    class BadStateDriver(FakeDriver):
        def start(
            self, launch: DecisionContainerLaunch
        ) -> DecisionContainerObservation:
            self.events.append(("start", launch))
            return DecisionContainerObservation(
                identity=launch.identity(container_id=CONTAINER_ID),
                state="unknown",
            )

    registry = DecisionInstanceRegistry(tmp_path / "instances.json")
    driver = BadStateDriver()
    with pytest.raises(
        DecisionContainerOwnershipError,
        match="full, matching ownership receipt",
    ):
        run_decision_runtime(
            DecisionServeOptions(model=MODEL, instance_name="bad-state"),
            resolver=FixtureResolver(),
            registry=registry,
            driver=driver,
            runtime_selector=lambda _requested: "docker",
        )

    assert [event[0] for event in driver.events] == ["ensure", "start"]
    assert registry.records()[0].state == "cleanup-required"


def test_foreground_log_failure_cleans_container_only_once(tmp_path: Path):
    registry = DecisionInstanceRegistry(tmp_path / "instances.json")
    driver = FakeDriver(fail_at="logs")

    with pytest.raises(DecisionContainerError, match="fixture logs failure"):
        run_decision_runtime(
            DecisionServeOptions(model=MODEL, instance_name="foreground-failure"),
            resolver=FixtureResolver(),
            registry=registry,
            driver=driver,
            runtime_selector=lambda _requested: "docker",
        )

    assert [event[0] for event in driver.events] == [
        "ensure",
        "start",
        "ready",
        "logs",
        "stop",
    ]
    assert registry.records() == ()


def test_zero_length_queue_is_a_valid_explicit_overload_policy(tmp_path: Path):
    registry = DecisionInstanceRegistry(tmp_path / "instances.json")

    receipt = run_decision_runtime(
        DecisionServeOptions(
            model=MODEL,
            instance_name="no-queue",
            max_queue=0,
            detach=True,
        ),
        resolver=FixtureResolver(),
        registry=registry,
        driver=FakeDriver(),
        runtime_selector=lambda _requested: "docker",
    )

    assert receipt.instance_name == "no-queue"
    assert registry.records()[0].max_queue == 0


def test_startup_failure_rolls_back_owned_container_and_registry(tmp_path: Path):
    registry = DecisionInstanceRegistry(tmp_path / "instances.json")
    driver = FakeDriver(fail_at="ready")

    with pytest.raises(DecisionContainerError, match="fixture ready failure"):
        run_decision_runtime(
            DecisionServeOptions(model=MODEL, instance_name="rollback"),
            resolver=FixtureResolver(),
            registry=registry,
            driver=driver,
            runtime_selector=lambda _requested: "docker",
        )

    assert [event[0] for event in driver.events] == [
        "ensure",
        "start",
        "ready",
        "stop",
    ]
    assert registry.records() == ()


def test_start_command_failure_retains_evidence_without_guessing_cleanup(
    tmp_path: Path,
):
    registry = DecisionInstanceRegistry(tmp_path / "instances.json")
    driver = FakeDriver(fail_at="start")

    with pytest.raises(DecisionContainerError, match="fixture start failure"):
        run_decision_runtime(
            DecisionServeOptions(model=MODEL, instance_name="start-rollback"),
            resolver=FixtureResolver(),
            registry=registry,
            driver=driver,
            runtime_selector=lambda _requested: "docker",
        )

    assert [event[0] for event in driver.events] == ["ensure", "start"]
    assert registry.records()[0].state == "cleanup-required"


def test_failed_container_cleanup_is_retained_for_reconciliation(tmp_path: Path):
    class CleanupFailDriver(FakeDriver):
        def wait_ready(
            self,
            launch: DecisionContainerLaunch,
            ownership: DecisionContainerIdentity,
            *,
            startup_timeout: int,
        ) -> None:
            self.events.append(("ready", launch, ownership, startup_timeout))
            raise DecisionContainerError("fixture readiness failure")

        def stop_and_remove(self, ownership: DecisionContainerIdentity) -> None:
            self.events.append(("stop", ownership))
            raise DecisionContainerError("fixture cleanup failure")

    registry = DecisionInstanceRegistry(tmp_path / "instances.json")
    driver = CleanupFailDriver()

    with pytest.raises(DecisionContainerError, match="fixture readiness failure"):
        run_decision_runtime(
            DecisionServeOptions(model=MODEL, instance_name="orphan"),
            resolver=FixtureResolver(),
            registry=registry,
            driver=driver,
            runtime_selector=lambda _requested: "docker",
        )

    records = registry.records()
    assert len(records) == 1
    assert records[0].instance_name == "orphan"
    assert records[0].state == "cleanup-required"


def test_resolver_cannot_silently_change_explicit_limits(tmp_path: Path):
    class BadResolver(FixtureResolver):
        def resolve(self, request: DecisionRuntimeRequest) -> ResolvedDecisionRuntime:
            return replace(super().resolve(request), max_batch=7)

    with pytest.raises(
        lifecycle.DecisionLifecycleError,
        match="changed the explicitly requested max batch",
    ):
        run_decision_runtime(
            DecisionServeOptions(model=MODEL, max_batch=8),
            resolver=BadResolver(),
            registry=DecisionInstanceRegistry(tmp_path / "instances.json"),
            driver=FakeDriver(),
            runtime_selector=lambda _requested: "docker",
        )


def test_cpu_thread_override_reaches_owned_cpu_container(tmp_path: Path):
    registry = DecisionInstanceRegistry(tmp_path / "instances.json")
    driver = FakeDriver()

    run_decision_runtime(
        DecisionServeOptions(model=MODEL, backend="cpu", cpu_threads=6, detach=True),
        resolver=FixtureResolver(),
        registry=registry,
        driver=driver,
        runtime_selector=lambda _requested: "docker",
    )

    launch = next(event[1] for event in driver.events if event[0] == "start")
    assert launch.runtime_spec.environment["DECISION_CPU_THREADS"] == "6"
    assert "DECISION_CPU_THREADS=6" in build_decision_container_command(launch)
    assert registry.records()[0].state == "running"


@pytest.mark.parametrize("backend", ("rocm", "cuda"))
def test_cpu_thread_override_rejects_explicit_gpu_before_catalog_resolution(
    backend: str,
) -> None:
    class MustNotResolve:
        def resolve(self, _request: DecisionRuntimeRequest) -> ResolvedDecisionRuntime:
            raise AssertionError("invalid CPU tuning reached the catalog")

    with pytest.raises(lifecycle.DecisionLifecycleError, match="only for the CPU"):
        run_decision_runtime(
            DecisionServeOptions(model=MODEL, backend=backend, cpu_threads=6),
            resolver=MustNotResolve(),
        )


def test_cpu_thread_override_rejects_auto_gpu_before_registry_mutation(
    tmp_path: Path,
) -> None:
    registry = DecisionInstanceRegistry(tmp_path / "instances.json")
    driver = FakeDriver()

    with pytest.raises(lifecycle.DecisionLifecycleError, match="only for the CPU"):
        run_decision_runtime(
            DecisionServeOptions(model=MODEL, cpu_threads=6),
            resolver=FixtureResolver(),
            registry=registry,
            driver=driver,
            runtime_selector=lambda _requested: "docker",
        )

    assert registry.records() == ()
    assert driver.events == []


def test_two_rocm_gpu_devices_launch_independently_on_distinct_ports(tmp_path: Path):
    class PerDeviceDriver(FakeDriver):
        def start(
            self, launch: DecisionContainerLaunch
        ) -> DecisionContainerObservation:
            self.events.append(("start", launch))
            device = int(launch.runtime_spec.environment["ROCR_VISIBLE_DEVICES"])
            return DecisionContainerObservation(
                identity=launch.identity(container_id=f"{device + 1:064x}"),
                state="running",
            )

    registry = DecisionInstanceRegistry(tmp_path / "instances.json")
    driver = PerDeviceDriver()
    resolver = FixtureResolver()

    for device, port in (("0", 8100), ("2", 8102)):
        run_decision_runtime(
            DecisionServeOptions(
                model=MODEL,
                backend="rocm",
                gpu_device=device,
                port=port,
                instance_name=f"gpu-{device}",
                detach=True,
            ),
            resolver=resolver,
            registry=registry,
            driver=driver,
            runtime_selector=lambda _requested: "docker",
        )

    launches = {
        event[1].instance_name: event[1]
        for event in driver.events
        if event[0] == "start"
    }
    records = {record.instance_name: record for record in registry.records()}
    assert set(launches) == set(records) == {"gpu-0", "gpu-2"}
    assert launches["gpu-0"].runtime_spec.environment["ROCR_VISIBLE_DEVICES"] == "0"
    assert launches["gpu-2"].runtime_spec.environment["ROCR_VISIBLE_DEVICES"] == "2"
    assert "ROCR_VISIBLE_DEVICES=0" in build_decision_container_command(
        launches["gpu-0"]
    )
    assert "ROCR_VISIBLE_DEVICES=2" in build_decision_container_command(
        launches["gpu-2"]
    )
    assert launches["gpu-0"].identity_digest != launches["gpu-2"].identity_digest
    assert records["gpu-0"].identity_digest == launches["gpu-0"].identity_digest
    assert records["gpu-2"].identity_digest == launches["gpu-2"].identity_digest
    assert records["gpu-0"].endpoint.startswith("http://127.0.0.1:8100/")
    assert records["gpu-2"].endpoint.startswith("http://127.0.0.1:8102/")
    assert records["gpu-0"].container_id != records["gpu-2"].container_id
    assert all(record.state == "running" for record in records.values())


def test_rocm_gpu_device_combines_with_exact_local_image_id(tmp_path: Path):
    driver = FakeDriver()
    run_decision_runtime(
        DecisionServeOptions(
            model=MODEL,
            backend="rocm",
            gpu_device="2",
            image=LOCAL_IMAGE_ID,
            image_pull_policy="never",
            detach=True,
        ),
        resolver=FixtureResolver(),
        registry=DecisionInstanceRegistry(tmp_path / "instances.json"),
        driver=driver,
        runtime_selector=lambda _requested: "docker",
    )

    launch = next(event[1] for event in driver.events if event[0] == "start")
    command = build_decision_container_command(launch)
    assert "ROCR_VISIBLE_DEVICES=2" in command
    assert "--pull=never" in command
    assert command[-4] == LOCAL_IMAGE_ID


@pytest.mark.parametrize(
    "device", ("-1", " 2", "2 ", "0,1", "2-3", "02", "10000", "GPU-deadbeef")
)
def test_invalid_gpu_device_rejected_before_catalog_resolution(device: str) -> None:
    class MustNotResolve:
        def resolve(self, _request: DecisionRuntimeRequest) -> ResolvedDecisionRuntime:
            raise AssertionError("invalid GPU selector reached the catalog")

    with pytest.raises(lifecycle.DecisionLifecycleError, match="one GPU index"):
        run_decision_runtime(
            DecisionServeOptions(model=MODEL, gpu_device=device),
            resolver=MustNotResolve(),
        )


@pytest.mark.parametrize("backend", ("cpu", "cuda"))
def test_gpu_device_rejects_unsupported_explicit_backend_before_catalog(
    backend: str,
) -> None:
    class MustNotResolve:
        def resolve(self, _request: DecisionRuntimeRequest) -> ResolvedDecisionRuntime:
            raise AssertionError("unsupported GPU selector reached the catalog")

    with pytest.raises(lifecycle.DecisionLifecycleError, match="only for the ROCm"):
        run_decision_runtime(
            DecisionServeOptions(model=MODEL, backend=backend, gpu_device="2"),
            resolver=MustNotResolve(),
        )


@pytest.mark.parametrize("backend", ("cpu", "cuda"))
def test_gpu_device_rejects_auto_resolved_unsupported_backend_before_registry(
    tmp_path: Path, backend: str
) -> None:
    class UnsupportedResolver(FixtureResolver):
        def resolve(self, request: DecisionRuntimeRequest) -> ResolvedDecisionRuntime:
            return replace(super().resolve(request), backend=backend)

    registry = DecisionInstanceRegistry(tmp_path / "instances.json")
    driver = FakeDriver()
    with pytest.raises(lifecycle.DecisionLifecycleError, match="only for the ROCm"):
        run_decision_runtime(
            DecisionServeOptions(model=MODEL, gpu_device="2"),
            resolver=UnsupportedResolver(),
            registry=registry,
            driver=driver,
            runtime_selector=lambda _requested: "docker",
        )

    assert registry.records() == ()
    assert driver.events == []


def test_catalog_cannot_inject_rocm_gpu_visibility_without_selector(
    tmp_path: Path,
) -> None:
    class MaskingResolver(FixtureResolver):
        def resolve(self, request: DecisionRuntimeRequest) -> ResolvedDecisionRuntime:
            spec = super().resolve(request)
            return replace(
                spec,
                environment={**spec.environment, "ROCR_VISIBLE_DEVICES": "2"},
            )

    registry = DecisionInstanceRegistry(tmp_path / "instances.json")
    driver = FakeDriver()
    with pytest.raises(lifecycle.DecisionLifecycleError, match="through --gpu-device"):
        run_decision_runtime(
            DecisionServeOptions(model=MODEL),
            resolver=MaskingResolver(),
            registry=registry,
            driver=driver,
            runtime_selector=lambda _requested: "docker",
        )

    assert registry.records() == ()
    assert driver.events == []


def test_resolved_runtime_detaches_from_retained_environment_mapping():
    source = {"DECISION_RUNTIME_LOG_LEVEL": "info"}
    spec = replace(
        FixtureResolver().resolve(
            DecisionRuntimeRequest(MODEL, None, "rocm", None, None, None, None)
        ),
        environment=source,
    )

    source["DECISION_RUNTIME_LOG_LEVEL"] = "debug"

    assert dict(spec.environment) == {"DECISION_RUNTIME_LOG_LEVEL": "info"}
    with pytest.raises(TypeError):
        spec.environment["DECISION_RUNTIME_LOG_LEVEL"] = "debug"  # type: ignore[index]
    command = build_decision_container_command(_launch(spec))
    assert "DECISION_RUNTIME_LOG_LEVEL=info" in command
    assert "DECISION_RUNTIME_LOG_LEVEL=debug" not in command


def test_container_command_mounts_verified_artifact_read_only(tmp_path: Path):
    artifact_root = tmp_path / "artifact"
    artifact_root.mkdir()
    spec = replace(
        FixtureResolver().resolve(
            DecisionRuntimeRequest(MODEL, None, "rocm", None, None, None, None)
        ),
        mounts=(
            DecisionRuntimeMount(
                source=str(artifact_root),
                target="/opt/vllm-sr/decision-artifact",
            ),
        ),
    )

    command = build_decision_container_command(_launch(spec))

    mount_index = command.index("--mount")
    assert command[mount_index + 1] == (
        f"type=bind,src={artifact_root},dst=/opt/vllm-sr/decision-artifact,readonly"
    )
    assert mount_index < command.index(IMAGE)


@pytest.mark.parametrize(
    "target",
    ("relative", "/", "/opt/../artifact", "/opt/artifact,ro"),
)
def test_invalid_artifact_mount_target_is_rejected_before_start(
    tmp_path: Path, target: str
):
    artifact_root = tmp_path / "artifact"
    artifact_root.mkdir()
    spec = replace(
        FixtureResolver().resolve(
            DecisionRuntimeRequest(MODEL, None, "rocm", None, None, None, None)
        ),
        mounts=(DecisionRuntimeMount(str(artifact_root), target),),
    )
    driver = FakeDriver()

    class InvalidMountResolver:
        def resolve(self, _request: DecisionRuntimeRequest) -> ResolvedDecisionRuntime:
            return spec

    with pytest.raises(DecisionContainerError, match=r"mount target|mount paths"):
        run_decision_runtime(
            DecisionServeOptions(model=MODEL, detach=True),
            resolver=InvalidMountResolver(),
            registry=DecisionInstanceRegistry(tmp_path / "instances.json"),
            driver=driver,
            runtime_selector=lambda _requested: "docker",
        )
    assert driver.events == []


def test_noncanonical_or_symlink_artifact_mount_is_rejected(tmp_path: Path):
    artifact_root = tmp_path / "artifact"
    artifact_root.mkdir()
    artifact_link = tmp_path / "artifact-link"
    artifact_link.symlink_to(artifact_root, target_is_directory=True)

    for source in (f"{artifact_root}/.", str(artifact_link)):
        spec = replace(
            FixtureResolver().resolve(
                DecisionRuntimeRequest(MODEL, None, "rocm", None, None, None, None)
            ),
            mounts=(DecisionRuntimeMount(source, "/opt/decision-artifact"),),
        )
        with pytest.raises(DecisionContainerError, match="mount source"):
            build_decision_container_command(_launch(spec))


def test_duplicate_artifact_mount_targets_are_rejected_before_start(tmp_path: Path):
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()

    class DuplicateMountResolver(FixtureResolver):
        def resolve(self, request: DecisionRuntimeRequest) -> ResolvedDecisionRuntime:
            return replace(
                super().resolve(request),
                mounts=(
                    DecisionRuntimeMount(str(first), "/opt/decision-artifact"),
                    DecisionRuntimeMount(str(second), "/opt/decision-artifact"),
                ),
            )

    driver = FakeDriver()
    with pytest.raises(
        lifecycle.DecisionLifecycleError, match="mount targets must be unique"
    ):
        run_decision_runtime(
            DecisionServeOptions(model=MODEL, detach=True),
            resolver=DuplicateMountResolver(),
            registry=DecisionInstanceRegistry(tmp_path / "instances.json"),
            driver=driver,
            runtime_selector=lambda _requested: "docker",
        )
    assert driver.events == []


def test_artifact_mount_is_part_of_managed_container_identity(tmp_path: Path):
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    base = FixtureResolver().resolve(
        DecisionRuntimeRequest(MODEL, None, "rocm", None, None, None, None)
    )

    def digest(source: Path) -> str:
        spec = replace(
            base,
            mounts=(DecisionRuntimeMount(str(source), "/opt/decision-artifact"),),
        )
        return lifecycle._identity_digest(
            instance_id="00000000-0000-0000-0000-000000000001",
            instance_name="fixture",
            endpoint="http://127.0.0.1:8000/v1/systemone",
            runtime="docker",
            spec=spec,
        )

    assert digest(first) != digest(second)


def test_rocm_gpu_visibility_is_part_of_managed_container_identity():
    base = FixtureResolver().resolve(
        DecisionRuntimeRequest(MODEL, None, "rocm", None, None, None, None)
    )

    def digest(device: str) -> str:
        spec = replace(
            base,
            environment={**base.environment, "ROCR_VISIBLE_DEVICES": device},
        )
        return lifecycle._identity_digest(
            instance_id="00000000-0000-0000-0000-000000000001",
            instance_name="fixture",
            endpoint="http://127.0.0.1:8000/v1/systemone",
            runtime="docker",
            spec=spec,
        )

    assert digest("0") != digest("2")


def test_resolved_runtime_defaults_to_strict_readiness_endpoint():
    spec = FixtureResolver().resolve(
        DecisionRuntimeRequest(MODEL, None, "rocm", None, None, None, None)
    )

    assert spec.health_path == "/ready"
    assert _launch(spec).runtime_spec.health_path == "/ready"


def test_host_binding_is_canonicalized_before_identity_and_argv(tmp_path: Path):
    registry = DecisionInstanceRegistry(tmp_path / "instances.json")
    driver = FakeDriver()

    receipt = run_decision_runtime(
        DecisionServeOptions(
            model=MODEL,
            instance_name="canonical-host",
            host="0:0:0:0:0:0:0:1",
            detach=True,
        ),
        resolver=FixtureResolver(),
        registry=registry,
        driver=driver,
        runtime_selector=lambda _requested: "docker",
    )

    launch = next(event[1] for event in driver.events if event[0] == "start")
    assert launch.host == "::1"
    assert receipt.endpoint == "http://[::1]:8000/v1/systemone"
    assert registry.records()[0].endpoint == receipt.endpoint


def test_explicit_image_is_validated_before_catalog_resolution(tmp_path: Path):
    class MustNotResolve:
        def resolve(self, _request):
            raise AssertionError("unsafe image reached the catalog adapter")

    with pytest.raises(
        lifecycle.DecisionLifecycleError,
        match="image override is invalid",
    ):
        run_decision_runtime(
            DecisionServeOptions(model=MODEL, image="--runtime@sha256:" + "c" * 64),
            resolver=MustNotResolve(),
            registry=DecisionInstanceRegistry(tmp_path / "instances.json"),
            driver=FakeDriver(),
            runtime_selector=lambda _requested: "docker",
        )


@pytest.mark.parametrize(
    ("pull_policy", "runtime", "message"),
    (
        ("ifnotpresent", "docker", "image-pull-policy never"),
        ("always", "docker", "image-pull-policy never"),
        ("never", "podman", "Docker container runtime"),
    ),
)
def test_local_image_id_requires_no_pull_docker_before_catalog_resolution(
    monkeypatch,
    pull_policy: str,
    runtime: str,
    message: str,
) -> None:
    class MustNotResolve:
        def resolve(self, _request):
            raise AssertionError("invalid local image request reached catalog")

    monkeypatch.setattr(decision_command, "default_catalog_resolver", MustNotResolve)
    result = CliRunner().invoke(
        decision_command.decision,
        [
            "serve",
            MODEL,
            "--image",
            LOCAL_IMAGE_ID,
            "--image-pull-policy",
            pull_policy,
            "--runtime",
            runtime,
        ],
    )

    assert result.exit_code == 1
    assert message in result.output
    assert not isinstance(result.exception, AssertionError)


def test_explicit_local_image_id_is_retained_in_launch_and_registry(tmp_path: Path):
    registry = DecisionInstanceRegistry(tmp_path / "instances.json")
    driver = FakeDriver()

    run_decision_runtime(
        DecisionServeOptions(
            model=MODEL,
            image=LOCAL_IMAGE_ID,
            image_pull_policy="never",
            runtime="docker",
            instance_name="local-image",
            detach=True,
        ),
        resolver=FixtureResolver(),
        registry=registry,
        driver=driver,
        runtime_selector=lambda _requested: "docker",
    )

    launch = next(event[1] for event in driver.events if event[0] == "start")
    record = registry.get("local-image")
    assert launch.runtime_spec.image == LOCAL_IMAGE_ID
    assert record.image == LOCAL_IMAGE_ID
    assert record.identity_digest == launch.identity_digest
    assert record.state == "running"


def test_local_image_id_refuses_selected_non_docker_runtime(tmp_path: Path):
    registry = DecisionInstanceRegistry(tmp_path / "instances.json")
    driver = FakeDriver()

    with pytest.raises(
        lifecycle.DecisionLifecycleError, match="Docker container runtime"
    ):
        run_decision_runtime(
            DecisionServeOptions(
                model=MODEL, image=LOCAL_IMAGE_ID, image_pull_policy="never"
            ),
            resolver=FixtureResolver(),
            registry=registry,
            driver=driver,
            runtime_selector=lambda _requested: "podman",
        )

    assert registry.records() == ()
    assert driver.events == []


def test_catalog_cannot_inject_local_image_id_without_explicit_override(
    tmp_path: Path,
) -> None:
    class UnexpectedImageResolver(FixtureResolver):
        def resolve(self, request: DecisionRuntimeRequest) -> ResolvedDecisionRuntime:
            return replace(super().resolve(request), image=LOCAL_IMAGE_ID)

    registry = DecisionInstanceRegistry(tmp_path / "instances.json")
    driver = FakeDriver()
    with pytest.raises(lifecycle.DecisionLifecycleError, match="explicit --image"):
        run_decision_runtime(
            DecisionServeOptions(model=MODEL, image_pull_policy="never"),
            resolver=UnexpectedImageResolver(),
            registry=registry,
            driver=driver,
            runtime_selector=lambda _requested: "docker",
        )

    assert registry.records() == ()
    assert driver.events == []


@pytest.mark.parametrize(
    "reference",
    (
        f"decision-runtime@sha256:{'a' * 64}",
        f"ghcr.io/vllm-project/decision-runtime:1.0@sha256:{'b' * 64}",
        f"localhost:5000/team/decision_runtime@sha256:{'c' * 64}",
    ),
)
def test_immutable_image_reference_accepts_conservative_valid_forms(reference: str):
    assert validate_immutable_image_reference(reference) == reference


def test_exact_local_image_id_is_a_separate_override_form():
    assert validate_decision_image_reference(LOCAL_IMAGE_ID) == LOCAL_IMAGE_ID
    with pytest.raises(ImmutableImageReferenceError):
        validate_immutable_image_reference(LOCAL_IMAGE_ID)


@pytest.mark.parametrize(
    "reference",
    (
        f"sha256:{'9' * 63}",
        f"sha256:{'A' * 64}",
        f"sha256:{'9' * 64}extra",
        f" sha256:{'9' * 64}",
    ),
)
def test_local_image_id_rejects_abbreviated_or_ambiguous_forms(reference: str):
    with pytest.raises(ImmutableImageReferenceError):
        validate_decision_image_reference(reference)


@pytest.mark.parametrize(
    "reference",
    (
        "decision-runtime:latest",
        f"https://example.test/runtime@sha256:{'a' * 64}",
        f"--runtime@sha256:{'a' * 64}",
        f" example.test/runtime@sha256:{'a' * 64}",
        f"example.test/runtime@@sha256:{'a' * 64}",
        f"example.test/runtime@sha256:{'A' * 64}",
        f"example.test/runtime@sha256:{'a' * 63}",
        f"example.test:0/runtime@sha256:{'a' * 64}",
        f"example.test:65536/runtime@sha256:{'a' * 64}",
        f"example.test:port/runtime@sha256:{'a' * 64}",
        f"bad..example/runtime@sha256:{'a' * 64}",
        f"bad.example./runtime@sha256:{'a' * 64}",
        f"example.test//runtime@sha256:{'a' * 64}",
        f"example.test/Runtime@sha256:{'a' * 64}",
    ),
)
def test_immutable_image_reference_rejects_unsafe_or_ambiguous_forms(reference: str):
    with pytest.raises(ImmutableImageReferenceError):
        validate_immutable_image_reference(reference)


def test_container_command_is_stable_and_contains_no_registry_secrets(monkeypatch):
    container_module = importlib.import_module("cli.decision_runtime.container")
    monkeypatch.setattr(
        container_module,
        "append_nvidia_gpu_passthrough",
        lambda _command, _runtime: None,
    )
    spec = replace(
        FixtureResolver().resolve(
            DecisionRuntimeRequest(MODEL, None, "cuda", None, None, None, None)
        ),
        backend="cuda",
        environment={
            "TOKENIZERS_PARALLELISM": "false",
            "DECISION_RUNTIME_LOG_LEVEL": "info",
        },
    )
    launch = DecisionContainerLaunch(
        runtime="docker",
        container_name="vllm-sr-decision-fixture",
        instance_name="fixture",
        identity_digest=f"sha256:{'d' * 64}",
        host="127.0.0.1",
        port=8000,
        pull_policy="never",
        runtime_spec=spec,
    )

    assert build_decision_container_command(launch) == [
        "docker",
        "run",
        "-d",
        "--name",
        "vllm-sr-decision-fixture",
        "--ulimit",
        "nofile=65536:65536",
        "--label",
        "ai.vllm-sr.decision.managed=true",
        "--label",
        "ai.vllm-sr.decision.instance=fixture",
        "--label",
        f"ai.vllm-sr.decision.identity=sha256:{'d' * 64}",
        "--label",
        f"ai.vllm-sr.decision.image={IMAGE}",
        "-p",
        "127.0.0.1:8000:8000",
        "-e",
        "DECISION_RUNTIME_LOG_LEVEL=info",
        "-e",
        "TOKENIZERS_PARALLELISM=false",
        IMAGE,
        "python",
        "-m",
        "decision_runtime.service",
    ]


def test_durable_docker_command_sets_only_opted_in_restart_policy():
    default_command = build_decision_container_command(_launch())
    durable_command = build_decision_container_command(
        replace(_launch(), restart_policy="unless-stopped")
    )

    assert "--restart" not in default_command
    assert durable_command[durable_command.index("--restart") + 1] == "unless-stopped"
    assert durable_command.index("--restart") < durable_command.index(IMAGE)
    with pytest.raises(DecisionContainerError, match="requires Docker"):
        build_decision_container_command(
            replace(_launch(), runtime="podman", restart_policy="unless-stopped")
        )


def test_inspection_reports_actual_restart_policy_without_changing_ownership(
    monkeypatch,
):
    ownership = _launch().identity(container_id=CONTAINER_ID)
    monkeypatch.setattr(
        container_module.subprocess,
        "run",
        lambda _command, **_kwargs: _inspect_completed_process(
            ownership, restart_policy="unless-stopped"
        ),
    )

    observation = LowLevelDecisionContainerDriver().inspect(ownership)

    assert observation is not None
    assert observation.identity == ownership
    assert observation.restart_policy == "unless-stopped"


def test_default_rocm_launch_keeps_all_gpu_visibility_unset():
    assert "ROCR_VISIBLE_DEVICES" not in _launch().runtime_spec.environment
    assert not any(
        part.startswith("ROCR_VISIBLE_DEVICES=")
        for part in build_decision_container_command(_launch())
    )


def test_decision_rocm_passthrough_uses_devices_without_debug_privileges(
    tmp_path: Path,
):
    container_module = importlib.import_module("cli.decision_runtime.container")
    kfd = tmp_path / "kfd"
    kfd.touch()
    dri = tmp_path / "dri"
    dri.mkdir()
    (dri / "renderD128").touch()
    command: list[str] = []

    container_module._append_decision_rocm_devices(command, kfd=kfd, dri=dri)

    assert command[:4] == ["--device", str(kfd), "--device", str(dri)]
    assert "SYS_PTRACE" not in command
    assert "seccomp=unconfined" not in command


def test_local_image_id_is_inspected_exactly_and_run_with_pull_disabled(monkeypatch):
    launch = _launch(replace(_launch().runtime_spec, image=LOCAL_IMAGE_ID))
    commands: list[list[str]] = []
    monkeypatch.setattr(
        container_module,
        "_ensure_image_available",
        lambda *_args: (_ for _ in ()).throw(AssertionError("pull path reached")),
    )

    def fake_run(command, **_kwargs):
        commands.append(command)
        return subprocess.CompletedProcess(command, 0, stdout=f"{LOCAL_IMAGE_ID}\n")

    monkeypatch.setattr(container_module.subprocess, "run", fake_run)

    LowLevelDecisionContainerDriver().ensure_image(launch)
    command = build_decision_container_command(launch)

    assert commands == [
        ["docker", "image", "inspect", "--format", "{{.Id}}", LOCAL_IMAGE_ID]
    ]
    assert command.index("--pull=never") < command.index(LOCAL_IMAGE_ID)
    assert command[-4:] == [LOCAL_IMAGE_ID, "python", "-m", "decision_runtime.service"]


@pytest.mark.parametrize(
    ("returncode", "inspected_id"),
    ((1, ""), (0, f"sha256:{'8' * 64}")),
)
def test_local_image_id_must_exist_with_exact_inspected_id(
    monkeypatch, returncode: int, inspected_id: str
) -> None:
    launch = _launch(replace(_launch().runtime_spec, image=LOCAL_IMAGE_ID))
    monkeypatch.setattr(
        container_module,
        "_ensure_image_available",
        lambda *_args: (_ for _ in ()).throw(AssertionError("pull path reached")),
    )
    monkeypatch.setattr(
        container_module.subprocess,
        "run",
        lambda command, **_kwargs: subprocess.CompletedProcess(
            command, returncode, stdout=inspected_id
        ),
    )

    with pytest.raises(DecisionContainerError, match="no pull was attempted"):
        LowLevelDecisionContainerDriver().ensure_image(launch)


@pytest.mark.parametrize(
    ("runtime", "pull_policy"),
    (("docker", "always"), ("docker", "ifnotpresent"), ("podman", "never")),
)
def test_local_image_id_rejects_pull_or_non_docker_at_container_seams(
    monkeypatch, runtime: str, pull_policy: str
) -> None:
    launch = replace(
        _launch(replace(_launch().runtime_spec, image=LOCAL_IMAGE_ID)),
        runtime=runtime,
        pull_policy=pull_policy,
    )
    monkeypatch.setattr(
        container_module.subprocess,
        "run",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("container command reached")
        ),
    )
    monkeypatch.setattr(
        container_module,
        "_ensure_image_available",
        lambda *_args: (_ for _ in ()).throw(AssertionError("pull path reached")),
    )

    with pytest.raises(DecisionContainerError, match="policy 'never'"):
        LowLevelDecisionContainerDriver().ensure_image(launch)
    with pytest.raises(DecisionContainerError, match="policy 'never'"):
        build_decision_container_command(launch)


def test_repository_digest_keeps_existing_image_preparation_path(monkeypatch):
    seen: list[tuple[str, str]] = []
    monkeypatch.setattr(
        container_module,
        "_ensure_image_available",
        lambda image, policy: seen.append((image, policy)),
    )

    LowLevelDecisionContainerDriver().ensure_image(_launch())

    assert seen == [(IMAGE, "never")]


@pytest.mark.parametrize(
    "environment",
    (
        {"HF_TOKEN": "must-not-reach-argv"},
        {"A_PROFILE": "apparently-innocent-but-not-public"},
        {"DECISION_RUNTIME_LOG_LEVEL": "invalid-value"},
        {"OMP_NUM_THREADS": "12345"},
        {"DECISION_CPU_THREADS": "257"},
        {"ROCR_VISIBLE_DEVICES": "0,1"},
        {"ROCR_VISIBLE_DEVICES": "-1"},
    ),
)
def test_container_command_rejects_nonpublic_or_unbounded_environment_values(
    environment: dict[str, str],
):
    spec = replace(
        FixtureResolver().resolve(
            DecisionRuntimeRequest(MODEL, None, "cuda", None, None, None, None)
        ),
        backend="cuda",
        environment=environment,
    )
    launch = DecisionContainerLaunch(
        runtime="docker",
        container_name="vllm-sr-decision-fixture",
        instance_name="fixture",
        identity_digest=f"sha256:{'d' * 64}",
        host="127.0.0.1",
        port=8000,
        pull_policy="never",
        runtime_spec=spec,
    )

    with pytest.raises(DecisionContainerError, match="public tuning") as error:
        build_decision_container_command(launch)

    assert all(value not in str(error.value) for value in environment.values())


@pytest.mark.parametrize("backend", ("cpu", "cuda"))
def test_container_command_rejects_rocm_visibility_on_other_backends(backend: str):
    spec = replace(
        _launch().runtime_spec,
        backend=backend,
        environment={"ROCR_VISIBLE_DEVICES": "2"},
    )

    with pytest.raises(DecisionContainerError, match="only for the ROCm"):
        build_decision_container_command(_launch(spec))


def test_low_level_start_preserves_actionable_runtime_stderr(monkeypatch):
    monkeypatch.setattr(
        container_module,
        "run_container_specs",
        lambda *_args, **_kwargs: (
            125,
            "",
            "container name is already in use by an unrelated workload",
        ),
    )

    with pytest.raises(
        DecisionContainerError,
        match="already in use by an unrelated workload",
    ):
        LowLevelDecisionContainerDriver().start(_launch())


def test_label_mismatch_never_issues_a_destructive_container_command(monkeypatch):
    ownership = _launch().identity(container_id=CONTAINER_ID)
    commands: list[list[str]] = []
    mismatched_labels = {
        "ai.vllm-sr.decision.managed": "true",
        "ai.vllm-sr.decision.instance": ownership.instance_name,
        "ai.vllm-sr.decision.identity": f"sha256:{'f' * 64}",
        "ai.vllm-sr.decision.image": ownership.image,
    }

    def fake_run(command, **_kwargs):
        commands.append(command)
        return _inspect_completed_process(ownership, labels=mismatched_labels)

    monkeypatch.setattr(container_module.subprocess, "run", fake_run)

    with pytest.raises(DecisionContainerOwnershipError, match="labels do not match"):
        LowLevelDecisionContainerDriver().stop_and_remove(ownership)

    assert [command[1] for command in commands] == ["inspect"]


def test_image_mismatch_never_issues_a_destructive_container_command(monkeypatch):
    ownership = _launch().identity(container_id=CONTAINER_ID)
    commands: list[list[str]] = []

    def fake_run(command, **_kwargs):
        commands.append(command)
        return _inspect_completed_process(
            ownership,
            image=f"example.test/other@sha256:{'f' * 64}",
        )

    monkeypatch.setattr(container_module.subprocess, "run", fake_run)

    with pytest.raises(DecisionContainerOwnershipError, match="image does not match"):
        LowLevelDecisionContainerDriver().stop_and_remove(ownership)

    assert [command[1] for command in commands] == ["inspect"]


def test_local_image_id_ownership_requires_exact_inspected_image(monkeypatch):
    launch = _launch(replace(_launch().runtime_spec, image=LOCAL_IMAGE_ID))
    ownership = launch.identity(container_id=CONTAINER_ID)
    monkeypatch.setattr(
        container_module.subprocess,
        "run",
        lambda _command, **_kwargs: _inspect_completed_process(ownership),
    )

    observation = LowLevelDecisionContainerDriver().inspect(ownership)
    assert observation is not None
    assert observation.identity.image == LOCAL_IMAGE_ID

    expected_other_image = replace(ownership, image=IMAGE)
    monkeypatch.setattr(
        container_module.subprocess,
        "run",
        lambda _command, **_kwargs: _inspect_completed_process(
            expected_other_image, image=LOCAL_IMAGE_ID
        ),
    )
    with pytest.raises(DecisionContainerOwnershipError, match="image does not match"):
        LowLevelDecisionContainerDriver().inspect(expected_other_image)


@pytest.mark.parametrize(
    "inspect_result",
    (
        subprocess.CompletedProcess(
            ["docker", "inspect"],
            125,
            stdout="",
            stderr="permission denied",
        ),
        subprocess.CompletedProcess(
            ["docker", "inspect"],
            0,
            stdout="not-json",
            stderr="",
        ),
        subprocess.CompletedProcess(
            ["docker", "inspect"],
            0,
            stdout=json.dumps(
                [
                    {
                        "Id": "d" * 12,
                        "State": {"Status": "running"},
                        "Config": {"Labels": {}, "Image": IMAGE},
                    }
                ]
            ),
            stderr="",
        ),
    ),
)
def test_unverifiable_inspect_never_issues_stop_or_remove(
    monkeypatch,
    inspect_result: subprocess.CompletedProcess[str],
):
    ownership = _launch().identity(container_id=CONTAINER_ID)
    commands: list[list[str]] = []

    def fake_run(command, **_kwargs):
        commands.append(command)
        return inspect_result

    monkeypatch.setattr(container_module.subprocess, "run", fake_run)

    with pytest.raises(DecisionContainerError):
        LowLevelDecisionContainerDriver().stop_and_remove(ownership)

    assert [command[1] for command in commands] == ["inspect"]


def test_verified_cleanup_uses_only_full_container_id(monkeypatch):
    ownership = _launch().identity(container_id=CONTAINER_ID)
    commands: list[list[str]] = []
    inspect_states = iter(("running", "exited"))

    def fake_run(command, **_kwargs):
        commands.append(command)
        if command[1] == "inspect":
            return _inspect_completed_process(
                ownership,
                state=next(inspect_states),
            )
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(container_module.subprocess, "run", fake_run)

    LowLevelDecisionContainerDriver().stop_and_remove(ownership)

    assert [command[1] for command in commands] == [
        "inspect",
        "stop",
        "inspect",
        "rm",
    ]
    assert commands[1][-1] == CONTAINER_ID
    assert commands[3][-1] == CONTAINER_ID


@pytest.mark.parametrize("image", (IMAGE, LOCAL_IMAGE_ID))
def test_managed_label_discovery_is_read_only_and_structurally_validated(
    monkeypatch, image: str
):
    identity = DecisionContainerIdentity(
        runtime="docker",
        container_name="vllm-sr-decision-orphan",
        instance_name="orphan",
        identity_digest=f"sha256:{'f' * 64}",
        image=image,
        container_id=CONTAINER_ID,
    )
    commands: list[list[str]] = []

    def fake_run(command, **_kwargs):
        commands.append(command)
        if command[1] == "ps":
            return subprocess.CompletedProcess(
                command,
                0,
                stdout=f"{CONTAINER_ID}\n",
                stderr="",
            )
        return _inspect_completed_process(identity)

    monkeypatch.setattr(container_module.subprocess, "run", fake_run)

    candidates = LowLevelDecisionContainerDriver().list_managed("docker")

    assert [command[1] for command in commands] == ["ps", "inspect"]
    assert candidates == (
        DecisionContainerCandidate(
            runtime="docker",
            container_id=CONTAINER_ID,
            container_name="vllm-sr-decision-orphan",
            state="running",
            instance_name="orphan",
            identity_digest=f"sha256:{'f' * 64}",
            image=image,
            label_contract_valid=True,
        ),
    )


def test_readiness_success_is_rechecked_against_the_same_container(monkeypatch):
    launch = _launch()
    ownership = launch.identity(container_id=CONTAINER_ID)
    driver = LowLevelDecisionContainerDriver()
    observations = iter(
        (
            DecisionContainerObservation(identity=ownership, state="running"),
            None,
        )
    )
    monkeypatch.setattr(driver, "inspect", lambda _identity: next(observations))
    monkeypatch.setattr(container_module, "_http_ready", lambda *_args, **_kwargs: True)

    with pytest.raises(DecisionContainerError, match="changed state"):
        driver.wait_ready(launch, ownership, startup_timeout=1)


def test_readiness_probe_uses_strict_ready_endpoint(monkeypatch):
    launch = _launch()
    ownership = launch.identity(container_id=CONTAINER_ID)
    driver = LowLevelDecisionContainerDriver()
    urls: list[str] = []
    monkeypatch.setattr(
        driver,
        "inspect",
        lambda _identity: DecisionContainerObservation(
            identity=ownership,
            state="running",
        ),
    )

    def ready(url: str, **_kwargs) -> bool:
        urls.append(url)
        return True

    monkeypatch.setattr(container_module, "_http_ready", ready)

    driver.wait_ready(launch, ownership, startup_timeout=1)

    assert urls == ["http://127.0.0.1:8000/ready"]


def test_current_readiness_probe_uses_strict_ready_and_rechecks_identity(monkeypatch):
    launch = _launch()
    ownership = launch.identity(container_id=CONTAINER_ID)
    driver = LowLevelDecisionContainerDriver()
    observations = iter(
        (
            DecisionContainerObservation(identity=ownership, state="running"),
            DecisionContainerObservation(identity=ownership, state="running"),
        )
    )
    inspected = []
    urls = []

    def inspect(identity: DecisionContainerIdentity):
        inspected.append(identity)
        return next(observations)

    def ready(url: str, **_kwargs) -> bool:
        urls.append(url)
        return True

    monkeypatch.setattr(driver, "inspect", inspect)
    monkeypatch.setattr(container_module, "_http_ready", ready)

    assert driver.probe_ready(ownership, host="0.0.0.0", port=8000) is True
    assert inspected == [ownership, ownership]
    assert urls == ["http://127.0.0.1:8000/ready"]


@pytest.mark.parametrize(
    ("status", "payload", "expected"),
    (
        (200, b'{"ready":true}', True),
        (201, b'{"ready":true}', False),
        (200, b"", False),
        (200, b'{"ready":false}', False),
        (200, b'{"ready":1}', False),
        (200, b'{"ready":true,"status":"ok"}', False),
        (200, b"not-json", False),
        (200, b"x" * 4097, False),
    ),
)
def test_http_readiness_requires_exact_200_json_contract(
    monkeypatch,
    status: int,
    payload: bytes,
    expected: bool,
):
    handlers = []

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self, limit: int) -> bytes:
            assert limit == 4097
            return payload

    response = Response()
    response.status = status

    class Opener:
        def open(self, _request, *, timeout: float):
            assert timeout == 0.5
            return response

    def build_opener(*configured_handlers):
        handlers.extend(configured_handlers)
        return Opener()

    monkeypatch.setattr(container_module.urllib.request, "build_opener", build_opener)

    assert (
        container_module._http_ready("http://127.0.0.1/ready", timeout=0.5) is expected
    )
    assert handlers == [container_module._RejectRedirects]


def test_readiness_redirect_handler_never_follows_redirects():
    handler = container_module._RejectRedirects()

    assert (
        handler.redirect_request(None, None, 302, "Found", {}, "http://elsewhere")
        is None
    )


def test_startup_failure_logs_use_verified_container_id(monkeypatch):
    launch = _launch()
    ownership = launch.identity(container_id=CONTAINER_ID)
    driver = LowLevelDecisionContainerDriver()
    logged: list[str] = []
    monkeypatch.setattr(
        driver,
        "inspect",
        lambda _identity: DecisionContainerObservation(
            identity=ownership,
            state="exited",
        ),
    )
    monkeypatch.setattr(
        container_module,
        "container_logs",
        lambda reference, **_kwargs: logged.append(reference) or True,
    )

    with pytest.raises(DecisionContainerError, match="exited during startup"):
        driver.wait_ready(launch, ownership, startup_timeout=1)

    assert logged == [CONTAINER_ID]


def test_mlx_requires_native_driver_before_container_access(tmp_path: Path):
    class MlxResolver(FixtureResolver):
        def resolve(self, request: DecisionRuntimeRequest) -> ResolvedDecisionRuntime:
            return replace(super().resolve(request), backend="mlx")

    resolver = MlxResolver()
    driver = FakeDriver()

    with pytest.raises(
        lifecycle.DecisionLifecycleError,
        match="MLX serving requires a native Decision runtime driver",
    ):
        run_decision_runtime(
            DecisionServeOptions(model=MODEL),
            resolver=resolver,
            registry=DecisionInstanceRegistry(tmp_path / "instances.json"),
            driver=driver,
            runtime_selector=lambda _requested: (_ for _ in ()).throw(
                AssertionError("container runtime must not be selected for MLX")
            ),
        )

    assert driver.events == []
    assert not (tmp_path / "instances.json").exists()


def test_mutable_resolved_image_is_rejected_before_container_access(tmp_path: Path):
    class MutableImageResolver(FixtureResolver):
        def resolve(self, request: DecisionRuntimeRequest) -> ResolvedDecisionRuntime:
            return replace(
                super().resolve(request), image="example.test/runtime:latest"
            )

    driver = FakeDriver()
    with pytest.raises(
        lifecycle.DecisionLifecycleError,
        match="safe digest-qualified OCI reference",
    ):
        run_decision_runtime(
            DecisionServeOptions(model=MODEL),
            resolver=MutableImageResolver(),
            registry=DecisionInstanceRegistry(tmp_path / "instances.json"),
            driver=driver,
            runtime_selector=lambda _requested: (_ for _ in ()).throw(
                AssertionError("container runtime must not be selected")
            ),
        )

    assert driver.events == []
    assert not (tmp_path / "instances.json").exists()


def test_detached_instance_can_be_safely_stopped_from_registry(tmp_path: Path):
    registry = DecisionInstanceRegistry(tmp_path / "instances.json")
    record = replace(
        _record("detached", "id-detached", 8000),
        state="running",
        container_id=CONTAINER_ID,
    )
    registry.reserve(record)
    driver = FakeDriver()

    receipt = stop_decision_instance(
        "detached",
        registry=registry,
        driver=driver,
    )

    assert receipt.action == "stopped"
    assert receipt.registry_durable is True
    assert [event[0] for event in driver.events] == ["inspect", "stop"]
    assert registry.records() == ()


def test_starting_missing_container_is_preserved_against_stop_race(tmp_path: Path):
    class MissingDriver(FakeDriver):
        def inspect(self, identity: DecisionContainerIdentity):
            self.events.append(("inspect", identity))

    registry = DecisionInstanceRegistry(tmp_path / "instances.json")
    record = _record("active-start", "id-active-start", 8000)
    registry.reserve(record)
    driver = MissingDriver()

    with pytest.raises(DecisionManagementError, match="launch process may still own"):
        stop_decision_instance(
            "active-start",
            registry=registry,
            driver=driver,
        )

    assert registry.get("active-start") == record
    assert [event[0] for event in driver.events] == ["inspect"]


def test_forget_refuses_starting_reservation_without_explicit_force(tmp_path: Path):
    registry = DecisionInstanceRegistry(tmp_path / "instances.json")
    record = _record("active-start", "id-active-start", 8000)
    registry.reserve(record)

    with pytest.raises(DecisionManagementError, match="pass --force"):
        forget_decision_instance("active-start", registry=registry)

    assert registry.get("active-start") == record


def test_force_forget_starting_reservation_only_removes_registry_evidence(
    tmp_path: Path,
):
    registry = DecisionInstanceRegistry(tmp_path / "instances.json")
    registry.reserve(_record("abandoned-start", "id-abandoned-start", 8000))

    receipt = forget_decision_instance(
        "abandoned-start",
        force=True,
        registry=registry,
    )

    assert receipt.action == "registry-record-forgotten"
    assert receipt.unknown_container_preserved is True
    assert registry.records() == ()


def test_blocked_image_prepare_cannot_be_orphaned_by_concurrent_stop(tmp_path: Path):
    entered = threading.Event()
    resume = threading.Event()
    registry = DecisionInstanceRegistry(tmp_path / "instances.json")
    receipts = []
    failures: list[BaseException] = []

    class BlockingDriver(FakeDriver):
        def ensure_image(self, launch: DecisionContainerLaunch) -> None:
            self.events.append(("ensure", launch))
            entered.set()
            assert resume.wait(timeout=5)

    class MissingDriver(FakeDriver):
        def inspect(self, identity: DecisionContainerIdentity):
            self.events.append(("inspect", identity))

    launch_driver = BlockingDriver()

    def launch() -> None:
        try:
            receipts.append(
                run_decision_runtime(
                    DecisionServeOptions(
                        model=MODEL,
                        instance_name="concurrent-start",
                        detach=True,
                    ),
                    resolver=FixtureResolver(),
                    registry=registry,
                    driver=launch_driver,
                    runtime_selector=lambda _requested: "docker",
                )
            )
        except BaseException as error:  # surfaced by the assertions below.
            failures.append(error)

    worker = threading.Thread(target=launch)
    worker.start()
    assert entered.wait(timeout=5)
    try:
        with pytest.raises(
            DecisionManagementError, match="launch process may still own"
        ):
            stop_decision_instance(
                "concurrent-start",
                registry=registry,
                driver=MissingDriver(),
            )
    finally:
        resume.set()
        worker.join(timeout=5)

    assert not worker.is_alive()
    assert failures == []
    assert len(receipts) == 1
    assert registry.get("concurrent-start").state == "running"


def test_concurrent_list_cannot_advance_or_rollback_a_healthy_launch(
    tmp_path: Path,
):
    container_visible = threading.Event()
    return_start_receipt = threading.Event()
    registry = DecisionInstanceRegistry(tmp_path / "instances.json")
    receipts = []
    failures: list[BaseException] = []

    class ConcurrentListDriver(FakeDriver):
        def start(
            self,
            launch: DecisionContainerLaunch,
        ) -> DecisionContainerObservation:
            self.events.append(("start", launch))
            container_visible.set()
            assert return_start_receipt.wait(timeout=5)
            return DecisionContainerObservation(
                identity=launch.identity(container_id=CONTAINER_ID),
                state="running",
            )

        def inspect(
            self,
            identity: DecisionContainerIdentity,
        ) -> DecisionContainerObservation:
            self.events.append(("inspect", identity))
            return DecisionContainerObservation(
                identity=replace(identity, container_id=CONTAINER_ID),
                state="running",
            )

    driver = ConcurrentListDriver()

    def launch() -> None:
        try:
            receipts.append(
                run_decision_runtime(
                    DecisionServeOptions(
                        model=MODEL,
                        instance_name="concurrent-list",
                        detach=True,
                    ),
                    resolver=FixtureResolver(),
                    registry=registry,
                    driver=driver,
                    runtime_selector=lambda _requested: "docker",
                )
            )
        except BaseException as error:  # surfaced by the assertions below.
            failures.append(error)

    worker = threading.Thread(target=launch)
    worker.start()
    assert container_visible.wait(timeout=5)
    try:
        statuses = list_decision_instances(registry=registry, driver=driver)
        managed = next(
            status for status in statuses if isinstance(status, DecisionManagedStatus)
        )
        assert managed.record.state == "starting"
        assert managed.record.container_id is None
        assert managed.runtime_state == "running"
        assert managed.readiness_state == "not-applicable"
        reserved = registry.get("concurrent-list")
        assert reserved.generation == 0
        assert reserved.container_id is None
    finally:
        return_start_receipt.set()
        worker.join(timeout=5)

    assert not worker.is_alive()
    assert failures == []
    assert len(receipts) == 1
    retained = registry.get("concurrent-list")
    assert retained.state == "running"
    assert retained.container_id == CONTAINER_ID
    assert retained.generation == 2
    assert all(event[0] != "stop" for event in driver.events)


@pytest.mark.parametrize("state", ("running", "cleanup-required"))
def test_missing_nonstarting_container_record_can_be_explicitly_stopped(
    state: str,
    tmp_path: Path,
):
    class MissingDriver(FakeDriver):
        def inspect(self, identity: DecisionContainerIdentity):
            self.events.append(("inspect", identity))

    registry = DecisionInstanceRegistry(tmp_path / "instances.json")
    record = replace(
        _record(f"missing-{state}", f"id-{state}", 8000),
        state=state,
        container_id=CONTAINER_ID,
    )
    registry.reserve(record)

    receipt = stop_decision_instance(
        record.instance_name,
        registry=registry,
        driver=MissingDriver(),
    )

    assert receipt.action == "stale-record-removed"
    assert registry.records() == ()


def test_ownership_mismatch_preserves_container_and_evidence_until_forget(
    tmp_path: Path,
):
    class MismatchDriver(FakeDriver):
        def inspect(self, identity: DecisionContainerIdentity):
            self.events.append(("inspect", identity))
            raise DecisionContainerOwnershipError("fixture identity mismatch")

        def stop_and_remove(self, ownership: DecisionContainerIdentity) -> None:
            raise AssertionError("unknown container must not be stopped")

    registry = DecisionInstanceRegistry(tmp_path / "instances.json")
    record = replace(
        _record("mismatch", "id-mismatch", 8000),
        state="running",
        container_id=CONTAINER_ID,
    )
    registry.reserve(record)
    driver = MismatchDriver()

    with pytest.raises(DecisionManagementError, match="unknown container"):
        stop_decision_instance("mismatch", registry=registry, driver=driver)

    retained = registry.get("mismatch")
    assert retained.state == "cleanup-required"
    assert [event[0] for event in driver.events] == ["inspect"]

    receipt = forget_decision_instance("mismatch", registry=registry)
    assert receipt.unknown_container_preserved is True
    assert registry.records() == ()


def test_cleanup_required_is_never_resurrected_from_container_state(tmp_path: Path):
    registry = DecisionInstanceRegistry(tmp_path / "instances.json")
    record = replace(
        _record("needs-cleanup", "id-needs-cleanup", 8000),
        state="cleanup-required",
        container_id=CONTAINER_ID,
    )
    registry.reserve(record)

    managed = status_decision_instance(
        "needs-cleanup",
        registry=registry,
        driver=FakeDriver(),
    )

    assert managed.runtime_state == "running"
    assert managed.readiness_state == "not-applicable"
    assert managed.ownership_verified is True
    assert managed.record.state == "cleanup-required"
    assert registry.get("needs-cleanup").state == "cleanup-required"


def test_status_probes_readiness_only_for_authoritative_running_state(
    tmp_path: Path,
):
    registry = DecisionInstanceRegistry(tmp_path / "instances.json")
    running = replace(
        _record("healthy", "id-healthy", 8000),
        state="running",
        container_id=CONTAINER_ID,
    )
    registry.reserve(running)
    driver = FakeDriver()

    managed = status_decision_instance(
        "healthy",
        registry=registry,
        driver=driver,
    )

    assert managed.record == running
    assert managed.runtime_state == "running"
    assert managed.readiness_state == "ready"
    assert [event[0] for event in driver.events] == ["inspect", "probe"]
    assert driver.events[-1][2:] == ("127.0.0.1", 8000)


def test_status_reports_running_instance_not_ready_when_strict_probe_fails(
    tmp_path: Path,
):
    class NotReadyDriver(FakeDriver):
        def probe_ready(
            self,
            ownership: DecisionContainerIdentity,
            *,
            host: str,
            port: int,
        ) -> bool:
            super().probe_ready(ownership, host=host, port=port)
            return False

    registry = DecisionInstanceRegistry(tmp_path / "instances.json")
    registry.reserve(
        replace(
            _record("not-ready", "id-not-ready", 8000),
            state="running",
            container_id=CONTAINER_ID,
        )
    )

    managed = status_decision_instance(
        "not-ready",
        registry=registry,
        driver=NotReadyDriver(),
    )

    assert managed.runtime_state == "running"
    assert managed.readiness_state == "not-ready"


def test_registry_generation_and_transition_graph_make_cleanup_terminal(
    tmp_path: Path,
):
    registry = DecisionInstanceRegistry(tmp_path / "instances.json")
    registry.reserve(_record("terminal", "id-terminal", 8000))
    identity = registry.transition(
        "terminal",
        "id-terminal",
        expected_generation=0,
        expected_state="starting",
        state="starting",
        container_id=CONTAINER_ID,
    ).record
    assert identity is not None
    cleanup = registry.transition(
        "terminal",
        "id-terminal",
        expected_generation=identity.generation,
        expected_state="starting",
        state="cleanup-required",
        container_id=CONTAINER_ID,
    ).record
    assert cleanup is not None

    with pytest.raises(DecisionRegistryStaleRecordError, match="lifecycle state"):
        registry.transition(
            "terminal",
            "id-terminal",
            expected_generation=identity.generation,
            expected_state="starting",
            state="running",
            container_id=CONTAINER_ID,
        )
    with pytest.raises(DecisionRegistryError, match="transition is invalid"):
        registry.transition(
            "terminal",
            "id-terminal",
            expected_generation=cleanup.generation,
            expected_state="cleanup-required",
            state="running",
            container_id=CONTAINER_ID,
        )

    retained = registry.get("terminal")
    assert retained.state == "cleanup-required"
    assert retained.generation == 2


def test_registry_remove_cannot_delete_a_newer_lifecycle_generation(tmp_path: Path):
    registry = DecisionInstanceRegistry(tmp_path / "instances.json")
    registry.reserve(_record("newer", "id-newer", 8000))
    current = registry.transition(
        "newer",
        "id-newer",
        expected_generation=0,
        expected_state="starting",
        state="starting",
        container_id=CONTAINER_ID,
    ).record
    assert current is not None

    with pytest.raises(DecisionRegistryStaleRecordError, match="lifecycle state"):
        registry.remove(
            "newer",
            "id-newer",
            expected_generation=0,
            expected_state="starting",
        )

    assert registry.get("newer") == current


def test_launch_cannot_overwrite_concurrent_cleanup_required_transition(
    tmp_path: Path,
):
    registry = DecisionInstanceRegistry(tmp_path / "instances.json")

    class ConcurrentCleanupDriver(FakeDriver):
        def wait_ready(
            self,
            launch: DecisionContainerLaunch,
            ownership: DecisionContainerIdentity,
            *,
            startup_timeout: int,
        ) -> None:
            super().wait_ready(
                launch,
                ownership,
                startup_timeout=startup_timeout,
            )
            current = registry.get(launch.instance_name)
            registry.transition(
                current.instance_name,
                current.instance_id,
                expected_generation=current.generation,
                expected_state=current.state,
                state="cleanup-required",
                container_id=ownership.container_id,
            )

    driver = ConcurrentCleanupDriver()
    with pytest.raises(DecisionRegistryStaleRecordError, match="lifecycle state"):
        run_decision_runtime(
            DecisionServeOptions(
                model=MODEL, instance_name="cleanup-race", detach=True
            ),
            resolver=FixtureResolver(),
            registry=registry,
            driver=driver,
            runtime_selector=lambda _requested: "docker",
        )

    retained = registry.get("cleanup-race")
    assert retained.state == "cleanup-required"
    assert retained.generation == 2
    assert [event[0] for event in driver.events] == [
        "ensure",
        "start",
        "ready",
        "stop",
    ]


def test_list_does_not_swallow_registry_integrity_failures(tmp_path: Path):
    class FailingRegistry(DecisionInstanceRegistry):
        def transition(self, *args, **kwargs):
            raise DecisionRegistryError("fixture registry storage failure")

    class ExitedDriver(FakeDriver):
        def inspect(self, identity: DecisionContainerIdentity):
            return DecisionContainerObservation(
                identity=replace(identity, container_id=CONTAINER_ID),
                state="exited",
            )

    registry = FailingRegistry(tmp_path / "instances.json")
    registry.reserve(_record("registered", "id-registered", 8000))

    with pytest.raises(DecisionRegistryError, match="storage failure"):
        list_decision_instances(registry=registry, driver=ExitedDriver())


def test_list_surfaces_label_owned_orphan_without_adopting_it(tmp_path: Path):
    class DiscoveryDriver(FakeDriver):
        def __init__(self) -> None:
            super().__init__()
            self.discovered_runtimes = []

        def list_managed(self, runtime: str):
            self.discovered_runtimes.append(runtime)
            if runtime == "podman":
                return ()
            return (
                DecisionContainerCandidate(
                    runtime="docker",
                    container_id="f" * 64,
                    container_name="vllm-sr-decision-orphan",
                    state="running",
                    instance_name="orphan",
                    identity_digest=f"sha256:{'f' * 64}",
                    image=IMAGE,
                    label_contract_valid=True,
                ),
            )

    registry = DecisionInstanceRegistry(tmp_path / "instances.json")
    registered = replace(
        _record("registered", "id-registered", 8000),
        state="running",
        container_id=CONTAINER_ID,
    )
    registry.reserve(registered)

    driver = DiscoveryDriver()
    statuses = list_decision_instances(registry=registry, driver=driver)

    assert len(statuses) == 2
    orphan = next(
        status for status in statuses if isinstance(status, DecisionOrphanStatus)
    )
    assert orphan.candidate.instance_name == "orphan"
    assert driver.discovered_runtimes == ["docker", "podman"]
    assert registry.records() == (registered,)


def test_empty_registry_still_discovers_docker_and_podman_without_adoption(
    tmp_path: Path,
):
    class EmptyDiscoveryDriver(FakeDriver):
        def __init__(self) -> None:
            super().__init__()
            self.discovered_runtimes = []

        def list_managed(self, runtime: str):
            self.discovered_runtimes.append(runtime)
            return ()

    registry = DecisionInstanceRegistry(tmp_path / "instances.json")
    driver = EmptyDiscoveryDriver()

    statuses = list_decision_instances(registry=registry, driver=driver)

    assert statuses == ()
    assert driver.discovered_runtimes == ["docker", "podman"]
    assert registry.records() == ()


def test_list_keeps_registry_status_when_orphan_discovery_is_unavailable(
    tmp_path: Path,
):
    class UnavailableDiscoveryDriver(FakeDriver):
        def list_managed(self, runtime: str):
            raise DecisionContainerError(f"fixture {runtime} unavailable")

    registry = DecisionInstanceRegistry(tmp_path / "instances.json")
    registered = replace(
        _record("registered", "id-registered", 8000),
        state="running",
        container_id=CONTAINER_ID,
    )
    registry.reserve(registered)

    statuses = list_decision_instances(
        registry=registry,
        driver=UnavailableDiscoveryDriver(),
    )

    assert len(statuses) == 3
    assert any(isinstance(status, DecisionManagedStatus) for status in statuses)
    discoveries = {
        status.runtime
        for status in statuses
        if isinstance(status, DecisionDiscoveryStatus)
    }
    assert discoveries == {"docker", "podman"}


def test_list_and_status_never_mutate_starting_to_record_observed_identity(
    tmp_path: Path,
):
    class RunningDriver(FakeDriver):
        def inspect(self, identity: DecisionContainerIdentity):
            self.events.append(("inspect", identity))
            return DecisionContainerObservation(
                identity=replace(identity, container_id=CONTAINER_ID),
                state="running",
            )

    registry = DecisionInstanceRegistry(tmp_path / "instances.json")
    record = _record("still-starting", "id-still-starting", 8000)
    registry.reserve(record)
    driver = RunningDriver()

    listed = list_decision_instances(registry=registry, driver=driver)[0]
    inspected = status_decision_instance(
        "still-starting",
        registry=registry,
        driver=driver,
    )

    assert listed.record == record
    assert inspected.record == record
    assert listed.runtime_state == "running"
    assert inspected.runtime_state == "running"
    assert listed.readiness_state == "not-applicable"
    assert inspected.readiness_state == "not-applicable"
    assert registry.get("still-starting") == record


def test_registry_is_private_secret_free_and_rejects_duplicate_binding(
    tmp_path: Path,
):
    registry = DecisionInstanceRegistry(tmp_path / "instances.json")
    first = _record("first", "id-first", 8000)
    registry.reserve(first)

    mode = stat.S_IMODE(registry.path.stat().st_mode)
    document = json.loads(registry.path.read_text(encoding="utf-8"))
    assert mode == 0o600
    assert set(document) == {"version", "instances"}
    assert set(document["instances"]["first"]) == set(first.__dataclass_fields__)
    assert all(
        marker not in registry.path.read_text(encoding="utf-8").lower()
        for marker in ("token", "password", "secret", "api_key")
    )

    with pytest.raises(DecisionRegistryError, match="host port conflicts"):
        registry.reserve(_record("second", "id-second", 8000))


def test_registry_rejects_wildcard_and_specific_host_on_same_port(tmp_path: Path):
    registry = DecisionInstanceRegistry(tmp_path / "instances.json")
    wildcard = replace(
        _record("wildcard", "id-wildcard", 8000),
        endpoint="http://0.0.0.0:8000/v1/systemone",
    )
    registry.reserve(wildcard)

    with pytest.raises(DecisionRegistryError, match="host port conflicts"):
        registry.reserve(_record("loopback", "id-loopback", 8000))


def test_registry_validates_record_before_creating_state(tmp_path: Path):
    registry = DecisionInstanceRegistry(tmp_path / "instances.json")
    invalid = replace(_record("invalid", "id-invalid", 8000), backend="auto")

    with pytest.raises(DecisionRegistryError, match="record is invalid"):
        registry.reserve(invalid)

    assert not registry.path.exists()
    assert not registry.lock_path.exists()


def test_registry_rejects_local_docker_image_id_for_podman(tmp_path: Path):
    registry = DecisionInstanceRegistry(tmp_path / "instances.json")
    invalid = replace(
        _record("invalid", "id-invalid", 8000),
        image=LOCAL_IMAGE_ID,
        runtime="podman",
    )

    with pytest.raises(DecisionRegistryError, match="record is invalid"):
        registry.reserve(invalid)

    assert not registry.path.exists()


def test_registry_atomic_replace_failure_preserves_previous_document(
    monkeypatch, tmp_path: Path
):
    registry = DecisionInstanceRegistry(tmp_path / "instances.json")
    first = _record("first", "id-first", 8000)
    registry.reserve(first)
    previous = registry.path.read_bytes()

    def fail_replace(_source, _destination):
        raise OSError("fixture replace failure")

    monkeypatch.setattr(registry_module.os, "replace", fail_replace)
    with pytest.raises(DecisionRegistryError, match="not committed atomically"):
        registry.transition(
            "first",
            "id-first",
            expected_generation=0,
            expected_state="starting",
            state="running",
            container_id=CONTAINER_ID,
        )

    assert registry.path.read_bytes() == previous
    assert list(tmp_path.glob(".instances.json.*.tmp")) == []


def test_registry_file_fsync_failure_preserves_previous_document(
    monkeypatch,
    tmp_path: Path,
):
    registry = DecisionInstanceRegistry(tmp_path / "instances.json")
    registry.reserve(_record("first", "id-first", 8000))
    previous = registry.path.read_bytes()
    real_fsync = registry_module.os.fsync

    def fail_regular_file_fsync(descriptor: int):
        if stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise OSError("fixture file fsync failure")
        return real_fsync(descriptor)

    monkeypatch.setattr(registry_module.os, "fsync", fail_regular_file_fsync)

    with pytest.raises(DecisionRegistryError, match="not committed atomically"):
        registry.transition(
            "first",
            "id-first",
            expected_generation=0,
            expected_state="starting",
            state="running",
            container_id=CONTAINER_ID,
        )

    assert registry.path.read_bytes() == previous


def test_registry_wraps_temporary_file_creation_failure(monkeypatch, tmp_path: Path):
    registry = DecisionInstanceRegistry(tmp_path / "instances.json")
    assert registry.records() == ()

    def fail_mkstemp(**_kwargs):
        raise OSError("fixture mkstemp failure")

    monkeypatch.setattr(registry_module.tempfile, "mkstemp", fail_mkstemp)

    with pytest.raises(DecisionRegistryError, match="temporary file cannot be created"):
        registry.reserve(_record("first", "id-first", 8000))


def test_registry_first_use_fsyncs_each_new_parent_entry(monkeypatch, tmp_path: Path):
    path = tmp_path / "nested" / "private" / "instances.json"
    registry = DecisionInstanceRegistry(path)
    synced: list[Path] = []
    real_fsync_directory = registry_module._fsync_directory

    def track_directory(directory: Path):
        synced.append(Path(directory))
        return real_fsync_directory(directory)

    monkeypatch.setattr(registry_module, "_fsync_directory", track_directory)

    commit = registry.reserve(_record("first", "id-first", 8000))

    assert commit.durable is True
    assert synced[:2] == [tmp_path, tmp_path / "nested"]
    assert synced[-1] == tmp_path / "nested" / "private"


def test_registry_commit_reports_post_replace_directory_fsync_failure(
    monkeypatch,
    tmp_path: Path,
):
    registry = DecisionInstanceRegistry(tmp_path / "instances.json")
    registry.reserve(_record("first", "id-first", 8000))

    def fail_directory_fsync(_path: Path):
        raise OSError("fixture directory fsync failure")

    monkeypatch.setattr(registry_module, "_fsync_directory", fail_directory_fsync)

    commit = registry.transition(
        "first",
        "id-first",
        expected_generation=0,
        expected_state="starting",
        state="running",
        container_id=CONTAINER_ID,
    )

    assert commit.durable is False
    assert registry.get("first").state == "running"
    assert registry.get("first").container_id == CONTAINER_ID


def test_registry_write_order_is_file_fsync_replace_directory_fsync(
    monkeypatch,
    tmp_path: Path,
):
    registry = DecisionInstanceRegistry(tmp_path / "instances.json")
    registry.reserve(_record("first", "id-first", 8000))
    events: list[str] = []
    real_fsync = registry_module.os.fsync
    real_replace = registry_module.os.replace
    real_fsync_directory = registry_module._fsync_directory

    def track_fsync(descriptor: int):
        if stat.S_ISREG(os.fstat(descriptor).st_mode):
            events.append("file-fsync")
        return real_fsync(descriptor)

    def track_replace(source, destination):
        events.append("replace")
        return real_replace(source, destination)

    def track_directory(directory: Path):
        events.append("directory-fsync")
        return real_fsync_directory(directory)

    monkeypatch.setattr(registry_module.os, "fsync", track_fsync)
    monkeypatch.setattr(registry_module.os, "replace", track_replace)
    monkeypatch.setattr(registry_module, "_fsync_directory", track_directory)

    registry.transition(
        "first",
        "id-first",
        expected_generation=0,
        expected_state="starting",
        state="running",
        container_id=CONTAINER_ID,
    )

    assert events == ["file-fsync", "replace", "directory-fsync"]


def test_nondurable_reservation_never_starts_or_reports_a_runtime(
    monkeypatch,
    tmp_path: Path,
):
    registry = DecisionInstanceRegistry(tmp_path / "instances.json")
    assert registry.records() == ()
    driver = FakeDriver()

    def fail_directory_fsync(_path: Path):
        raise OSError("fixture directory fsync failure")

    monkeypatch.setattr(registry_module, "_fsync_directory", fail_directory_fsync)

    with pytest.raises(
        lifecycle.DecisionLifecycleError,
        match="reservation was committed, but crash durability",
    ):
        run_decision_runtime(
            DecisionServeOptions(model=MODEL, instance_name="nondurable"),
            resolver=FixtureResolver(),
            registry=registry,
            driver=driver,
            runtime_selector=lambda _requested: "docker",
        )

    assert driver.events == []
    assert registry.get("nondurable").state == "starting"


def _record(name: str, instance_id: str, port: int) -> DecisionInstanceRecord:
    return DecisionInstanceRecord(
        instance_id=instance_id,
        instance_name=name,
        container_name=f"vllm-sr-decision-{name}",
        model=MODEL,
        revision=REVISION,
        endpoint=f"http://127.0.0.1:{port}/v1/systemone",
        backend="rocm",
        dtype="bfloat16",
        artifact_digest=ARTIFACT_DIGEST,
        identity_digest=f"sha256:{'e' * 64}",
        image=IMAGE,
        container_id=None,
        runtime="docker",
        max_batch=8,
        max_concurrency=8,
        max_queue=32,
        state="starting",
        generation=0,
        created_at="2026-09-23T00:00:00+00:00",
    )

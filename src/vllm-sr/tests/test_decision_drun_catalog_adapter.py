"""Integration tests for the host-side ``drun`` catalog bridge."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest
from cli.decision_runtime.catalog import DecisionCatalogError, DecisionRuntimeRequest
from cli.decision_runtime.catalog_adapter import (
    IntegratedDecisionCatalogResolver,
    default_artifact_cache_root,
)
from decision_runtime.artifacts import VerifiedArtifact
from decision_runtime.catalog_adapter import ResolvedRuntimeModel

MODEL = "llm-semantic-router/Decision-1.0-Kai-0.6B"
REVISION = "7185f514f54b8f93c55998b1e8f9c5cc67f0d029"
IMAGE = f"example.test/decision-runtime-rocm@sha256:{'a' * 64}"
LOCAL_IMAGE_ID = f"sha256:{'9' * 64}"


@dataclass
class RecordingArtifacts:
    root: Path

    def __post_init__(self) -> None:
        self.calls: list[ResolvedRuntimeModel] = []

    def materialize(self, model: ResolvedRuntimeModel) -> VerifiedArtifact:
        self.calls.append(model)
        return VerifiedArtifact(
            root=self.root,
            data_root=self.root,
            content_id="b" * 64,
            repository_id=model.repository_id,
            revision=model.catalog.revision,
            files=(),
        )


def request(**overrides: object) -> DecisionRuntimeRequest:
    values: dict[str, object] = {
        "model": MODEL,
        "revision": None,
        "backend": "rocm",
        "dtype": None,
        "image": None,
        "max_batch": None,
        "max_concurrency": None,
        "max_queue": None,
    }
    values.update(overrides)
    return DecisionRuntimeRequest(**values)  # type: ignore[arg-type]


def resolver(tmp_path: Path, **overrides: object) -> IntegratedDecisionCatalogResolver:
    artifact_root = tmp_path / "artifact"
    artifact_root.mkdir()
    values: dict[str, object] = {
        "artifacts": RecordingArtifacts(artifact_root),
        "images": {"rocm": IMAGE},
        "detect_backend": lambda: "rocm",
    }
    values.update(overrides)
    return IntegratedDecisionCatalogResolver(**values)  # type: ignore[arg-type]


def test_bridge_materializes_exact_model_and_builds_read_only_launch(
    tmp_path: Path,
) -> None:
    bridge = resolver(tmp_path)

    spec = bridge.resolve(request())

    assert spec.canonical_model == MODEL
    assert spec.revision == REVISION
    assert spec.backend == "rocm"
    assert spec.dtype == "float32"
    assert spec.image == IMAGE
    assert spec.artifact_digest == f"sha256:{'b' * 64}"
    assert (spec.max_batch, spec.max_concurrency, spec.max_queue) == (8, 8, 8)
    assert spec.environment == {"TOKENIZERS_PARALLELISM": "false"}
    assert len(spec.mounts) == 1
    assert spec.mounts[0].source == str(tmp_path / "artifact")
    assert spec.mounts[0].target == "/opt/vllm-sr/decision-artifact"
    assert spec.command == (
        "/opt/vllm-sr/venvs/vela/bin/python",
        "-m",
        "decision_runtime.entrypoint",
        "--model",
        MODEL,
        "--revision",
        REVISION,
        "--backend",
        "rocm",
        "--artifact-root",
        "/opt/vllm-sr/decision-artifact",
        "--artifact-content-id",
        "b" * 64,
        "--host",
        "0.0.0.0",
        "--port",
        "8000",
        "--max-batch",
        "8",
        "--max-concurrency",
        "8",
        "--max-queue",
        "8",
    )
    assert [item.catalog.model_id for item in bridge.artifacts.calls] == [MODEL]  # type: ignore[attr-defined]


def test_explicit_tuning_is_forwarded_without_changing_model_identity(
    tmp_path: Path,
) -> None:
    spec = resolver(tmp_path).resolve(
        request(max_batch=16, max_concurrency=32, max_queue=64, image=IMAGE)
    )

    assert (spec.max_batch, spec.max_concurrency, spec.max_queue) == (16, 32, 64)
    assert spec.command[-6:] == (
        "--max-batch",
        "16",
        "--max-concurrency",
        "32",
        "--max-queue",
        "64",
    )


def test_explicit_local_image_id_is_forwarded_without_catalog_image_fallback(
    tmp_path: Path,
) -> None:
    spec = resolver(tmp_path, images={}).resolve(request(image=LOCAL_IMAGE_ID))

    assert spec.image == LOCAL_IMAGE_ID


def test_packaged_image_inventory_still_requires_repository_digest(
    tmp_path: Path,
) -> None:
    bridge = resolver(tmp_path, images={"rocm": LOCAL_IMAGE_ID})

    with pytest.raises(
        DecisionCatalogError, match="safe digest-qualified OCI reference"
    ):
        bridge.resolve(request())

    assert bridge.artifacts.calls == []  # type: ignore[attr-defined]


def test_auto_backend_uses_injected_detector(tmp_path: Path) -> None:
    detected: list[bool] = []
    bridge = resolver(
        tmp_path,
        detect_backend=lambda: detected.append(True) or "rocm",
    )

    assert bridge.resolve(request(backend="auto")).backend == "rocm"
    assert detected == [True]


def test_missing_released_image_fails_before_artifact_download(tmp_path: Path) -> None:
    bridge = resolver(tmp_path, images={})

    with pytest.raises(DecisionCatalogError, match=r"no released.*image"):
        bridge.resolve(request())

    assert bridge.artifacts.calls == []  # type: ignore[attr-defined]


@pytest.mark.parametrize(
    ("overrides", "message"),
    (
        ({"revision": "main"}, "full lowercase Git SHA"),
        ({"dtype": "bfloat16"}, "requires backbone dtype"),
        ({"backend": "mlx"}, "rocm, cuda, or cpu"),
    ),
)
def test_incompatible_launch_request_fails_before_artifact_download(
    tmp_path: Path, overrides: dict[str, object], message: str
) -> None:
    bridge = resolver(tmp_path)

    with pytest.raises(DecisionCatalogError, match=message):
        bridge.resolve(request(**overrides))

    assert bridge.artifacts.calls == []  # type: ignore[attr-defined]


def test_explicit_immutable_revision_reaches_artifact_and_container(
    tmp_path: Path,
) -> None:
    bridge = resolver(tmp_path)
    revision = "c" * 40
    spec = bridge.resolve(request(revision=revision))

    assert spec.revision == revision
    assert spec.command[spec.command.index("--revision") + 1] == revision
    assert bridge.artifacts.calls[0].catalog.revision == revision  # type: ignore[attr-defined]


def test_noncanonical_artifact_root_is_rejected(tmp_path: Path) -> None:
    artifact_root = tmp_path / "artifact"
    artifact_root.mkdir()
    link = tmp_path / "link"
    link.symlink_to(artifact_root, target_is_directory=True)
    bridge = IntegratedDecisionCatalogResolver(
        artifacts=RecordingArtifacts(link),
        images={"rocm": IMAGE},
        detect_backend=lambda: "rocm",
    )

    with pytest.raises(DecisionCatalogError, match="symlink root"):
        bridge.resolve(request())


def test_default_cache_root_requires_absolute_xdg_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("XDG_CACHE_HOME", "relative/cache")

    with pytest.raises(DecisionCatalogError, match="must be an absolute"):
        default_artifact_cache_root()

    monkeypatch.setenv("XDG_CACHE_HOME", "/var/tmp/decision-cache")
    assert default_artifact_cache_root() == Path(
        "/var/tmp/decision-cache/vllm-sr/decision-runtime/artifacts"
    )

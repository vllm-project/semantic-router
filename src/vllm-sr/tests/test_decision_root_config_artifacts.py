"""Root-config publication layout and immutable artifact receipt tests."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from importlib import resources
from pathlib import Path

import pytest
from decision_runtime.artifacts import (
    ArtifactError,
    ArtifactIntegrityError,
    ArtifactManifestError,
    ArtifactNotFoundError,
    ArtifactResolver,
    HfHubArtifactFetcher,
    open_verified_artifact,
    parse_decision_config,
)
from decision_runtime.catalog_adapter import resolve_decision_runtime_model
from decision_runtime.decision_config import (
    DecisionConfigError,
    validate_decision_weight_index,
)
from decision_runtime.runtime_factory import _artifact_provenance
from decision_runtime.runtime_profile import ArtifactSelection, parse_runtime_profile
from huggingface_hub.utils import EntryNotFoundError, LocalEntryNotFoundError
from requests import Response

_REVISION = "c" * 40
_QWEN = "llm-semantic-router/Decision-1.0-Eos-0.8B"
_VELA = "llm-semantic-router/Decision-1.0-Kai-0.6B"


@dataclass
class SnapshotFetcher:
    files: dict[str, Path]

    def __post_init__(self) -> None:
        self.calls: list[tuple[str, str, str]] = []

    def fetch(self, *, repository_id: str, revision: str, filename: str) -> Path:
        self.calls.append((repository_id, revision, filename))
        try:
            return self.files[filename]
        except KeyError:
            raise ArtifactNotFoundError("file absent at immutable revision") from None


def _descriptor(family: str, model_name: str) -> dict:
    if family == "qwen3.5":
        return {
            "decision_format": "vllm-sr-decision",
            "format_version": 1,
            "model_name": model_name,
            "runtime_family": "qwen3.5-decision",
            "model_config": "decision_config.json",
            "backbone": {
                "config": "backbone/config.json",
                "weights": ["backbone/model.safetensors"],
            },
            "tokenizer": {
                "json": "tokenizer.json",
                "config": "tokenizer_config.json",
            },
            "decision_weights": {"decision_head": "decision_head.safetensors"},
            "calibration": {"temperature": 1.25},
        }
    return {
        "decision_format": "vllm-sr-decision",
        "format_version": 1,
        "model_name": model_name,
        "runtime_family": "vela-encoder",
        "model_config": "native/decision_config.json",
        "backbone": {
            "config": "native/encoder/config.json",
            "weights": ["native/encoder/model.safetensors"],
        },
        "tokenizer": {
            "json": "native/tokenizer/tokenizer.json",
            "config": "native/tokenizer/tokenizer_config.json",
            "special_tokens_map": "native/tokenizer/special_tokens_map.json",
        },
        "decision_weights": {
            "choice_encoder": "native/choice_encoder.safetensors",
            "score_encoder": "native/score_encoder.safetensors",
            "decision_heads": "native/decision_heads.safetensors",
        },
    }


def _snapshot(
    tmp_path: Path, model_id: str, family: str, *, descriptor: dict | None = None
):
    model = resolve_decision_runtime_model(model_id, revision=_REVISION)
    model = replace(
        model,
        profile=replace(
            model.profile, artifact=ArtifactSelection(config_path="config.json")
        ),
    )
    descriptor = descriptor or _descriptor(family, model.template_id)
    config = parse_decision_config(
        json.dumps(descriptor).encode(), model_name=model.template_id, family=family
    )
    payloads = {
        "config.json": json.dumps(descriptor).encode(),
        **{name: f"snapshot:{name}".encode() for name in config.files},
    }
    files = {}
    for name, payload in payloads.items():
        path = tmp_path / "source" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        files[name] = path
    return model, SnapshotFetcher(files), config


def test_packaged_profile_without_artifact_defaults_to_root_config() -> None:
    resource = resources.files("decision_runtime.profiles").joinpath(
        "qwen35", "Decision-1.0-Eos-0.8B.json"
    )
    document = json.loads(resource.read_bytes())
    document.pop("artifact", None)
    profile = parse_runtime_profile(json.dumps(document).encode(), revision=_REVISION)
    assert profile.artifact.config_path == "config.json"
    assert profile.artifact.manifest_path is None


@pytest.mark.parametrize("model_id,family", [(_QWEN, "qwen3.5"), (_VELA, "vela")])
def test_root_config_materializes_declared_data_only_at_pinned_revision(
    tmp_path: Path, model_id: str, family: str
) -> None:
    model, fetcher, config = _snapshot(tmp_path, model_id, family)
    artifact = ArtifactResolver(fetcher, tmp_path / "cache").materialize(model)

    assert artifact.config is not None
    assert (
        artifact.config.sha256
        == hashlib.sha256(fetcher.files["config.json"].read_bytes()).hexdigest()
    )
    assert artifact.manifest is None
    assert artifact.repository_config == config
    assert _artifact_provenance(model, artifact) == {
        "model": model_id,
        "revision": _REVISION,
        "config_sha256": artifact.config.sha256,
        "content_sha256": artifact.content_id,
    }
    assert artifact.data_root == (
        artifact.root / "native" if family == "vela" else artifact.root
    )
    assert {item.repository_path for item in artifact.files} == set(config.files)
    assert not (artifact.root / "MODEL_MANIFEST.json").exists()
    assert {call[:2] for call in fetcher.calls} == {(model_id, _REVISION)}
    assert (
        open_verified_artifact(
            artifact.root, model, expected_content_id=artifact.content_id
        )
        == artifact
    )

    selected = artifact.root / config.backbone_weights[0]
    selected.chmod(0o644)
    selected.write_bytes(b"corrupted")
    with pytest.raises(ArtifactIntegrityError, match=r"size|digest"):
        open_verified_artifact(
            artifact.root, model, expected_content_id=artifact.content_id
        )


def test_root_config_rejects_wrong_identity_code_and_unsafe_paths() -> None:
    descriptor = _descriptor("qwen3.5", "Decision-1.0-Eos-0.8B")

    def payload():
        return json.dumps(descriptor).encode()

    with pytest.raises(DecisionConfigError, match="model name"):
        parse_decision_config(
            payload(), model_name="Decision-1.0-Sol-2B", family="qwen3.5"
        )
    with pytest.raises(DecisionConfigError, match="family"):
        parse_decision_config(
            payload(), model_name=descriptor["model_name"], family="vela"
        )
    descriptor["backbone"]["weights"] = ["../runtime.py"]
    with pytest.raises(DecisionConfigError, match="unsafe path"):
        parse_decision_config(
            payload(), model_name=descriptor["model_name"], family="qwen3.5"
        )
    descriptor["backbone"]["weights"] = ["backbone/model.safetensors"]
    descriptor["execution_code"] = "runtime.py"
    with pytest.raises(DecisionConfigError, match="fields"):
        parse_decision_config(
            payload(), model_name=descriptor["model_name"], family="qwen3.5"
        )
    duplicated = payload().replace(
        b'"model_name":', b'"model_name": "Injected", "model_name":', 1
    )
    with pytest.raises(DecisionConfigError, match="valid JSON"):
        parse_decision_config(
            duplicated, model_name=descriptor["model_name"], family="qwen3.5"
        )


def test_root_config_accepts_dynamic_complete_shards_and_checks_index() -> None:
    descriptor = _descriptor("qwen3.5", "Decision-1.0-Nox-4B")
    weights = [
        "backbone/part-one.safetensors",
        "backbone/part-two.safetensors",
    ]
    descriptor["backbone"] = {
        "config": "backbone/config.json",
        "weights": weights,
        "index": "backbone/model.safetensors.index.json",
    }
    config = parse_decision_config(
        json.dumps(descriptor).encode(),
        model_name=descriptor["model_name"],
        family="qwen3.5",
    )
    assert set(config.backbone_weights) == set(weights)
    assert config.backbone_index == "backbone/model.safetensors.index.json"
    validate_decision_weight_index(
        json.dumps(
            {
                "weight_map": {
                    "layer.0": "part-one.safetensors",
                    "layer.1": "part-two.safetensors",
                }
            }
        ).encode(),
        config,
    )
    with pytest.raises(DecisionConfigError, match="differs from config shards"):
        validate_decision_weight_index(
            b'{"weight_map":{"layer.0":"other.safetensors"}}', config
        )
    descriptor["backbone"].pop("index")
    with pytest.raises(DecisionConfigError, match="indexed weight layout"):
        parse_decision_config(
            json.dumps(descriptor).encode(),
            model_name=descriptor["model_name"],
            family="qwen3.5",
        )


@pytest.mark.parametrize(
    "model_id,family,data_root", [(_QWEN, "qwen3.5", "v2"), (_VELA, "vela", "release")]
)
def test_root_config_drives_renamed_model_files_and_data_root(
    tmp_path: Path, model_id: str, family: str, data_root: str
) -> None:
    descriptor = _descriptor(family, model_id.rsplit("/", 1)[-1])
    descriptor["model_config"] = f"{data_root}/metadata.json"
    descriptor["backbone"] = {
        "config": f"{data_root}/body/config.json",
        "weights": [
            (
                f"{data_root}/body/model.safetensors"
                if family == "qwen3.5"
                else f"{data_root}/body/encoder-v2.safetensors"
            )
        ],
    }
    descriptor["tokenizer"] = {
        "json": f"{data_root}/text/tokenizer.json",
        "config": f"{data_root}/text/tokenizer_config.json",
    }
    descriptor["decision_weights"] = (
        {"decision_head": f"{data_root}/heads/scorer-v2.safetensors"}
        if family == "qwen3.5"
        else {
            "choice_encoder": f"{data_root}/heads/choice-v2.safetensors",
            "score_encoder": f"{data_root}/heads/score-v2.safetensors",
            "decision_heads": f"{data_root}/heads/decision-v2.safetensors",
        }
    )
    model, fetcher, config = _snapshot(
        tmp_path, model_id, family, descriptor=descriptor
    )
    artifact = ArtifactResolver(fetcher, tmp_path / "cache").materialize(model)

    assert config.data_root_relative == data_root
    assert artifact.data_root == artifact.root / data_root
    assert {item.repository_path for item in artifact.files} == set(config.files)
    assert (
        open_verified_artifact(
            artifact.root, model, expected_content_id=artifact.content_id
        )
        == artifact
    )


def test_old_manifest_fallback_requires_config_absence_not_invalidity(
    tmp_path: Path,
) -> None:
    model, fetcher, _ = _snapshot(tmp_path, _VELA, "vela")
    fetcher.files.pop("config.json")
    names = (
        "INVENTORY.json",
        "STATE_LAYOUT.json",
        "choice_encoder.safetensors",
        "decision_config.json",
        "decision_heads.safetensors",
        "encoder/config.json",
        "encoder/model.safetensors",
        "score_encoder.safetensors",
        "tokenizer/tokenizer.json",
        "tokenizer/tokenizer_config.json",
    )
    inventory = {}
    for name in names:
        repository_path = f"native/{name}"
        if repository_path not in fetcher.files:
            path = tmp_path / "source" / repository_path
            path.write_bytes(f"legacy:{name}".encode())
            fetcher.files[repository_path] = path
        inventory[name] = {
            "bytes": fetcher.files[repository_path].stat().st_size,
            "sha256": hashlib.sha256(
                fetcher.files[repository_path].read_bytes()
            ).hexdigest(),
        }
    manifest_path = tmp_path / "source" / "native" / "MANIFEST.json"
    manifest_path.write_text(json.dumps({"files": inventory}), encoding="utf-8")
    fetcher.files["native/MANIFEST.json"] = manifest_path

    artifact = ArtifactResolver(fetcher, tmp_path / "cache").materialize(model)
    assert artifact.config is None
    assert artifact.manifest is not None
    assert artifact.manifest.path == "native/MANIFEST.json"
    assert _artifact_provenance(model, artifact) == {
        "model": _VELA,
        "revision": _REVISION,
        "manifest_sha256": artifact.manifest.sha256,
        "content_sha256": artifact.content_id,
    }
    assert fetcher.calls[:2] == [
        (_VELA, _REVISION, "config.json"),
        (_VELA, _REVISION, "native/MANIFEST.json"),
    ]

    fetcher.calls.clear()
    fetcher.files["config.json"] = tmp_path / "bad-config.json"
    fetcher.files["config.json"].write_bytes(b"invalid JSON")
    with pytest.raises(ArtifactManifestError, match="root config"):
        ArtifactResolver(fetcher, tmp_path / "other-cache").materialize(model)
    assert [call[2] for call in fetcher.calls] == ["config.json"]


def test_fetch_error_does_not_downgrade_to_legacy_manifest(tmp_path: Path) -> None:
    model, fetcher, _ = _snapshot(tmp_path, _QWEN, "qwen3.5")

    def unavailable(*, repository_id: str, revision: str, filename: str):
        raise ArtifactError("Hub unavailable")

    fetcher.fetch = unavailable  # type: ignore[method-assign]
    with pytest.raises(ArtifactError, match="Hub unavailable"):
        ArtifactResolver(fetcher, tmp_path / "cache").materialize(model)


def test_hub_remote_404_is_the_only_config_fallback_signal(monkeypatch) -> None:
    response = Response()
    response.status_code = 404

    def absent(**_kwargs):
        raise EntryNotFoundError("private URL", response=response)

    monkeypatch.setattr("huggingface_hub.hf_hub_download", absent)
    with pytest.raises(ArtifactNotFoundError, match="absent") as raised:
        HfHubArtifactFetcher().fetch(
            repository_id=_QWEN, revision=_REVISION, filename="config.json"
        )
    assert "private URL" not in str(raised.value)

    def cache_miss(**_kwargs):
        raise LocalEntryNotFoundError("private cache path")

    monkeypatch.setattr("huggingface_hub.hf_hub_download", cache_miss)
    with pytest.raises(ArtifactError, match="Unable to fetch") as raised:
        HfHubArtifactFetcher(local_files_only=True).fetch(
            repository_id=_QWEN, revision=_REVISION, filename="config.json"
        )
    assert type(raised.value) is ArtifactError
    assert "private cache path" not in str(raised.value)

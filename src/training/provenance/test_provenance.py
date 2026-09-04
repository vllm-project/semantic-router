import json
import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from provenance import (
    ArtifactManifest,
    DatasetManifest,
    EvaluationManifest,
    ProvenanceError,
    RunManifest,
    dump_manifest,
    load_bundle,
    load_manifest,
    manifest_id,
    tree_digest,
    validate_bundle,
    validate_manifest,
)
from provenance.__main__ import main
from provenance.digest import tree_files
from provenance.validate import Bundle


def _dataset() -> DatasetManifest:
    return DatasetManifest(
        name="jailbreak-v1",
        sources=[
            {
                "name": "lmsys/toxic-chat",
                "revision": "abc123",
                "license": "cc-by-nc-4.0",
                "config": "toxicchat0124",
            }
        ],
        splits={"train": 800, "validation": 100, "test": 100},
        preprocessing=["dedupe", "balance"],
        label_mapping={"benign": 0, "jailbreak": 1},
    )


def _bundle(tmp_path: pathlib.Path) -> tuple[Bundle, pathlib.Path]:
    artifact_dir = tmp_path / "model"
    artifact_dir.mkdir()
    (artifact_dir / "config.json").write_text("{}")
    (artifact_dir / "weights.bin").write_bytes(b"\x00\x01")
    dataset = _dataset()
    run = RunManifest(
        name="jailbreak-run-1",
        task="prompt_guard",
        dataset_id=manifest_id(dataset),
        base_model={
            "name": "llm-semantic-router/mmbert-32k-yarn",
            "revision": "def456",
        },
        code_revision="0123abcd",
        dependencies={"torch": "2.4.0"},
        seed=42,
        hyperparameters={"epochs": 3, "learning_rate": 3e-5},
    )
    evaluation = EvaluationManifest(
        run_id=manifest_id(run),
        dataset_id=manifest_id(dataset),
        split="test",
        sample_count=100,
        command="python eval.py",
        metrics={"accuracy": 0.97},
    )
    files = tree_files(artifact_dir)
    artifact = ArtifactManifest(
        run_id=manifest_id(run),
        evaluation_id=manifest_id(evaluation),
        format="safetensors-merged",
        files=files,
        digest=tree_digest(files),
        label_mapping=dataset.label_mapping,
    )
    return Bundle(dataset, run, evaluation, artifact), artifact_dir


def _write_bundle(bundle: Bundle, directory: pathlib.Path) -> None:
    dump_manifest(bundle.dataset, directory / "dataset.json")
    dump_manifest(bundle.run, directory / "run.json")
    dump_manifest(bundle.evaluation, directory / "evaluation.json")
    dump_manifest(bundle.artifact, directory / "artifact.json")


def test_manifest_round_trip_preserves_identity(tmp_path):
    path = tmp_path / "dataset.json"
    dump_manifest(_dataset(), path)
    loaded = load_manifest(path)
    assert isinstance(loaded, DatasetManifest)
    assert manifest_id(loaded) == manifest_id(_dataset())
    assert json.loads(path.read_text())["kind"] == "dataset"


def test_manifest_id_changes_with_content():
    changed = _dataset()
    changed.splits["train"] = 801
    assert manifest_id(changed) != manifest_id(_dataset())


def test_valid_bundle_passes(tmp_path):
    bundle, artifact_dir = _bundle(tmp_path)
    validate_bundle(bundle, artifact_dir)


def test_missing_license_fails():
    dataset = _dataset()
    del dataset.sources[0]["license"]
    with pytest.raises(ProvenanceError, match="sources\\[0\\] missing 'license'"):
        validate_manifest(dataset)


def test_missing_seed_fails():
    run = RunManifest(
        name="r",
        task="t",
        dataset_id="sha256:x",
        base_model={"name": "m", "revision": "r"},
        code_revision="c",
    )
    with pytest.raises(ProvenanceError, match="missing required field 'seed'"):
        validate_manifest(run)


def test_mismatched_dataset_id_fails(tmp_path):
    bundle, artifact_dir = _bundle(tmp_path)
    bundle.run.dataset_id = "sha256:stale"
    with pytest.raises(ProvenanceError) as err:
        validate_bundle(bundle, artifact_dir)
    messages = "\n".join(err.value.errors)
    assert "run.dataset_id" in messages
    assert "evaluation.run_id" in messages
    assert "artifact.run_id" in messages


def test_artifact_digest_mismatch_fails(tmp_path):
    bundle, artifact_dir = _bundle(tmp_path)
    (artifact_dir / "weights.bin").write_bytes(b"\x00\x02")
    with pytest.raises(ProvenanceError, match=r"files differ from .*weights\.bin"):
        validate_bundle(bundle, artifact_dir)


def test_artifact_digest_field_must_match_files(tmp_path):
    bundle, _ = _bundle(tmp_path)
    bundle.artifact.digest = "sha256:wrong"
    with pytest.raises(ProvenanceError, match="digest does not match files"):
        validate_manifest(bundle.artifact)


def test_artifact_file_paths_must_be_relative(tmp_path):
    bundle, _ = _bundle(tmp_path)
    digest = "sha256:" + "0" * 64
    bundle.artifact.files = {"/abs/weights.bin": digest, "../up.bin": digest}
    bundle.artifact.digest = tree_digest(bundle.artifact.files)
    with pytest.raises(ProvenanceError) as err:
        validate_manifest(bundle.artifact)
    assert [e for e in err.value.errors if "must be relative" in e] == err.value.errors
    assert "/abs/weights.bin" in err.value.errors[0]
    assert "../up.bin" in err.value.errors[1]


def test_secret_like_key_and_value_fail():
    dataset = _dataset()
    dataset.extra = {"hf_token": "x", "note": "hf_" + "a" * 30}
    with pytest.raises(ProvenanceError) as err:
        validate_manifest(dataset)
    assert any("secret-like key" in e for e in err.value.errors)
    assert any("secret-like value" in e for e in err.value.errors)


@pytest.mark.parametrize(
    ("kind", "field", "value", "message"),
    [
        ("evaluation", "sample_count", 0, "sample_count must be a positive int"),
        ("evaluation", "command", "", "missing required field 'command'"),
        ("evaluation", "sample_count", -3, "sample_count must be a positive int"),
        ("evaluation", "sample_count", "100", "sample_count must be a positive int"),
        ("evaluation", "metrics", {"accuracy": "0.9"}, "metrics must be a map"),
        ("evaluation", "metrics", {"accuracy": float("nan")}, "metrics must be a map"),
        ("evaluation", "run_id", "sha256:short", "run_id must be a sha256"),
        ("dataset", "splits", {"train": -1}, "splits must be a map"),
        ("dataset", "splits", {"train": "800"}, "splits must be a map"),
        ("run", "seed", True, "seed must be a non-negative int"),
        ("run", "dataset_id", "abc", "dataset_id must be a sha256"),
        ("artifact", "digest", "md5:abc", "digest must be a sha256"),
        ("artifact", "files", {"weights.bin": "sha256:xyz"}, "files must be a map"),
        ("artifact", "files", ["weights.bin"], "files must be a map"),
        ("dataset", "sources", [1], "sources must be a list of objects"),
        ("dataset", "sources", "lmsys/toxic-chat", "sources must be a list of objects"),
        ("run", "base_model", "mmbert", "base_model must be a map"),
        ("run", "base_model", {"name": 1}, "base_model must be a map"),
        ("dataset", "schema_version", "1", "schema_version must be an int"),
        ("dataset", "schema_version", 2, "unsupported schema_version 2"),
    ],
)
def test_type_and_format_violations_fail(tmp_path, kind, field, value, message):
    bundle, _ = _bundle(tmp_path)
    manifest = getattr(bundle, kind)
    setattr(manifest, field, value)
    with pytest.raises(ProvenanceError, match=message):
        validate_manifest(manifest)


def test_source_fields_must_be_strings():
    dataset = _dataset()
    dataset.sources[0]["license"] = 1
    with pytest.raises(
        ProvenanceError, match=r"sources\[0\]\.license must be a string"
    ):
        validate_manifest(dataset)


def test_serialized_schema_version_is_validated(tmp_path):
    path = tmp_path / "dataset.json"
    data = _dataset().to_dict()
    data["schema_version"] = "1"
    path.write_text(json.dumps(data))
    with pytest.raises(ProvenanceError, match="schema_version must be an int"):
        validate_manifest(load_manifest(path))


def test_unknown_kind_rejected(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text('{"kind": "weights"}')
    with pytest.raises(ValueError, match="unknown manifest kind"):
        load_manifest(path)


def test_unknown_field_rejected(tmp_path):
    path = tmp_path / "run.json"
    path.write_text('{"kind": "run", "seeds": 1}')
    with pytest.raises(ValueError, match="run: unknown fields \\['seeds'\\]"):
        load_manifest(path)


def test_cli_validate_bundle(tmp_path, capsys):
    bundle, artifact_dir = _bundle(tmp_path)
    _write_bundle(bundle, tmp_path)
    assert (
        main(["validate-bundle", str(tmp_path), "--artifact-dir", str(artifact_dir)])
        == 0
    )
    assert capsys.readouterr().out.strip() == "ok"
    assert main(["id", str(tmp_path / "run.json")]) == 0
    assert capsys.readouterr().out.strip() == manifest_id(bundle.run)
    assert load_bundle(tmp_path).run == bundle.run


def test_cli_reports_errors(tmp_path, capsys):
    bundle, _ = _bundle(tmp_path)
    bundle.evaluation.metrics = {}
    _write_bundle(bundle, tmp_path)
    assert main(["validate-bundle", str(tmp_path)]) == 1
    assert "missing required field 'metrics'" in capsys.readouterr().err
    assert main(["validate", str(tmp_path / "missing.json")]) == 1

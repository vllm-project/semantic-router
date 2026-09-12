"""Tests for the bytes a quality baseline run measures (#3197)."""

import argparse
import fnmatch
import pathlib
import sys

import pytest

TEST_DIR = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(TEST_DIR))

import baseline_artifact  # noqa: E402
from artifact_inventory import LoadSite, ServedArtifact  # noqa: E402
from baseline_artifact import BaselineError, resolve_measured_artifact  # noqa: E402
from provenance.crossref import artifact_identity_digest, file_digest  # noqa: E402
from provenance.emit import write_manifest  # noqa: E402

REPO = "llm-semantic-router/mmbert32k-jailbreak-detector-merged"
REVISION = "b" * 40
CONFIG_JSON = b'{"architectures": ["ModernBertForSequenceClassification"]}'


def served_artifact():
    site = LoadSite(
        location="prompt_guard",
        model_path="models/mmbert32k-jailbreak-detector-merged",
        model_ref="prompt_guard",
        threshold=0.7,
        mapping_path=None,
        positive_labels=("jailbreak",),
    )
    return ServedArtifact(task="jailbreak", model_path=site.model_path, sites=(site,))


def artifact_dir(tmp_path, name, body=CONFIG_JSON):
    directory = tmp_path / name
    directory.mkdir()
    (directory / "config.json").write_bytes(body)
    return directory


def artifact_manifest(tmp_path, directory, repo=REPO):
    """Publish a manifest for the bytes in ``directory`` and return its path."""
    files = [
        {
            "path": path.name,
            "size_bytes": path.stat().st_size,
            "digest": file_digest(path),
        }
        for path in sorted(directory.iterdir())
    ]
    manifest = {
        "schema_version": "v1",
        "kind": "artifact",
        "id": "mmbert32k-jailbreak-detector-merged",
        "task": "jailbreak",
        "identity": {
            "repo": repo,
            "revision": REVISION,
            "digest": artifact_identity_digest(files),
        },
        "files": files,
        "label_mapping": {"benign": 0, "jailbreak": 1},
        "runtime": {
            "architecture": "ModernBertForSequenceClassification",
            "max_position_embeddings": 32768,
            "num_labels": 2,
        },
    }
    return write_manifest(manifest, tmp_path / "artifact.manifest.yaml")


def resolve(manifest, artifact_dir=None, artifact_repo=None):
    args = argparse.Namespace(
        config=pathlib.Path("config/config.yaml"),
        artifact_dir=artifact_dir,
        artifact_manifest=manifest,
        artifact_repo=artifact_repo,
    )
    return resolve_measured_artifact(args, served_artifact())


def test_a_local_artifact_matching_its_manifest_is_measured(tmp_path):
    directory = artifact_dir(tmp_path, "local")
    manifest = artifact_manifest(tmp_path, directory)

    measured = resolve(manifest, artifact_dir=directory)

    assert measured.model_dir == directory
    assert (measured.repo, measured.revision) == (REPO, REVISION)


def test_local_bytes_that_are_not_the_referenced_artifact_are_refused(tmp_path):
    directory = artifact_dir(tmp_path, "local")
    manifest = artifact_manifest(tmp_path, directory)
    (directory / "config.json").write_bytes(b'{"architectures": ["Other"]}')

    with pytest.raises(BaselineError, match="cannot be measured under that identity"):
        resolve(manifest, artifact_dir=directory)


def test_a_missing_file_is_refused_rather_than_scored(tmp_path):
    directory = artifact_dir(tmp_path, "local")
    manifest = artifact_manifest(tmp_path, directory)
    (directory / "config.json").unlink()

    with pytest.raises(BaselineError, match=r"config\.json is missing"):
        resolve(manifest, artifact_dir=directory)


def test_a_local_artifact_without_a_manifest_is_refused(tmp_path):
    with pytest.raises(BaselineError, match="requires --artifact-manifest"):
        resolve(None, artifact_dir=artifact_dir(tmp_path, "local"))


def test_a_download_is_fetched_at_the_revision_the_manifest_names(
    tmp_path, monkeypatch
):
    directory = artifact_dir(tmp_path, "snapshot")
    manifest = artifact_manifest(tmp_path, directory)
    calls = {}

    def fake_download(repo, revision, patterns):
        calls.update(repo=repo, revision=revision, patterns=patterns)
        return directory

    monkeypatch.setattr(baseline_artifact, "download_artifact", fake_download)

    measured = resolve(manifest)

    assert calls == {"repo": REPO, "revision": REVISION, "patterns": ["config.json"]}
    assert (measured.repo, measured.revision) == (REPO, REVISION)


def test_downloaded_bytes_that_are_not_the_referenced_artifact_are_refused(
    tmp_path, monkeypatch
):
    directory = artifact_dir(tmp_path, "snapshot")
    manifest = artifact_manifest(tmp_path, directory)
    other = artifact_dir(tmp_path, "other", body=b'{"architectures": ["Other"]}')
    monkeypatch.setattr(
        baseline_artifact, "download_artifact", lambda repo, revision, patterns: other
    )

    with pytest.raises(BaselineError, match="cannot be measured under that identity"):
        resolve(manifest)


def snapshot_matching(source, destination, patterns):
    """Copy the files an allow_patterns download would fetch.

    The Hub filters a repository by fnmatch before anything is transferred, so
    a fake that filters the same way answers the question the real download
    answers: is the closure the run asks for complete.
    """
    destination.mkdir()
    for path in sorted(source.iterdir()):
        if any(fnmatch.fnmatch(path.name, pattern) for pattern in patterns):
            (destination / path.name).write_bytes(path.read_bytes())
    return destination


def test_a_bin_only_artifact_is_downloaded_with_its_weight_shards(
    tmp_path, monkeypatch
):
    """A checkpoint without safetensors still has to arrive whole.

    Transformers resolves sharded pytorch_model.bin weights through their index,
    so fetching the index without the shards leaves the run with a directory it
    cannot load.
    """
    repository = artifact_dir(tmp_path, "repository")
    (repository / "pytorch_model.bin.index.json").write_bytes(b"{}")
    (repository / "pytorch_model-00001-of-00002.bin").write_bytes(b"first shard")
    (repository / "pytorch_model-00002-of-00002.bin").write_bytes(b"second shard")

    monkeypatch.setattr(
        baseline_artifact,
        "download_artifact",
        lambda repo, revision, patterns: snapshot_matching(
            repository, tmp_path / "snapshot", patterns
        ),
    )
    monkeypatch.setattr(baseline_artifact, "resolve_hf_revision", lambda repo: REVISION)

    measured = resolve(None, artifact_repo=REPO)

    assert sorted(path.name for path in measured.model_dir.iterdir()) == [
        "config.json",
        "pytorch_model-00001-of-00002.bin",
        "pytorch_model-00002-of-00002.bin",
        "pytorch_model.bin.index.json",
    ]


def test_a_candidate_repo_the_manifest_does_not_describe_is_refused(tmp_path):
    directory = artifact_dir(tmp_path, "snapshot")
    manifest = artifact_manifest(tmp_path, directory)

    with pytest.raises(BaselineError, match="--artifact-manifest describes"):
        resolve(manifest, artifact_repo="llm-semantic-router/some-candidate")

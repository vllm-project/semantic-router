"""Tests for the Router Model provenance manifest contract."""

import pathlib
import sys

import pytest
import yaml

TEST_DIR = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(TEST_DIR))

from provenance.crossref import artifact_identity_digest, validate_bundle  # noqa: E402
from provenance.manifest import ManifestError, load_manifest  # noqa: E402
from provenance.metrics import (  # noqa: E402
    abstention_curve,
    calibration_metrics,
    classification_metrics,
    latency_percentiles,
)
from provenance.redaction import RedactionError  # noqa: E402

DATASET_REVISION = "a" * 40
ARTIFACT_REVISION = "b" * 40
CODE_REVISION = "c" * 40
BASE_REVISION = "d" * 40
FILE_DIGEST = "sha256:" + "1" * 64


def dataset_manifest(**overrides):
    manifest = {
        "schema_version": "v1",
        "kind": "dataset",
        "id": "jailbreak-detection-dataset",
        "task": "jailbreak",
        "source": {
            "type": "huggingface",
            "locator": "llm-semantic-router/jailbreak-detection-dataset",
            "revision": DATASET_REVISION,
        },
        "license": "unknown-upstream",
        "splits": [{"name": "test", "rows": 827, "digest": "sha256:" + "2" * 64}],
        "preprocessing": {
            "text_field": "text",
            "label_field": "label",
            "steps": ["load the published split without shuffling"],
        },
        "label_mapping": {"benign": 0, "jailbreak": 1},
    }
    manifest.update(overrides)
    return manifest


def run_manifest(**overrides):
    manifest = {
        "schema_version": "v1",
        "kind": "run",
        "id": "prompt-guard-r8",
        "task": "jailbreak",
        "base_model": {"repo": "jhu-clsp/mmBERT-base", "revision": BASE_REVISION},
        "code": {
            "repo": "vllm-project/semantic-router",
            "revision": CODE_REVISION,
            "entrypoint": "src/training/model_classifier/x.py",
            "dirty": False,
        },
        "dataset_refs": [
            {
                "id": "jailbreak-detection-dataset",
                "revision": DATASET_REVISION,
                "splits": ["test"],
            }
        ],
        "seed": 42,
        "hyperparameters": {"lora_rank": 8},
        "dependencies": {"python": "3.12.3", "packages": {"torch": "2.13.0"}},
        "label_mapping": {"benign": 0, "jailbreak": 1},
    }
    manifest.update(overrides)
    return manifest


def artifact_manifest(**overrides):
    files = [{"path": "config.json", "size_bytes": 12, "digest": FILE_DIGEST}]
    manifest = {
        "schema_version": "v1",
        "kind": "artifact",
        "id": "mmbert32k-jailbreak-detector-merged",
        "task": "jailbreak",
        "identity": {
            "repo": "llm-semantic-router/mmbert32k-jailbreak-detector-merged",
            "revision": ARTIFACT_REVISION,
            "digest": artifact_identity_digest(files),
        },
        "run_ref": {"id": "prompt-guard-r8"},
        "files": files,
        "label_mapping": {"benign": 0, "jailbreak": 1},
        "runtime": {
            "architecture": "ModernBertForSequenceClassification",
            "max_position_embeddings": 32768,
            "num_labels": 2,
        },
    }
    manifest.update(overrides)
    return manifest


def evaluation_manifest(**overrides):
    manifest = {
        "schema_version": "v1",
        "kind": "evaluation",
        "id": "jailbreak-test-baseline",
        "task": "jailbreak",
        "artifact_ref": {
            "id": "mmbert32k-jailbreak-detector-merged",
            "revision": ARTIFACT_REVISION,
            "digest": artifact_manifest()["identity"]["digest"],
        },
        "dataset_ref": {
            "id": "jailbreak-detection-dataset",
            "revision": DATASET_REVISION,
            "splits": ["test"],
        },
        "split_rule": "by_source",
        "harness": {
            "code": {
                "repo": "vllm-project/semantic-router",
                "revision": CODE_REVISION,
                "entrypoint": "src/training/model_eval/quality_baseline.py",
                "dirty": False,
            },
            "device": "cuda",
            "batch_size": 32,
            "max_length": 512,
            "sample_limit": None,
            "seed": 42,
            "dependencies": {"python": "3.12.3", "packages": {"torch": "2.13.0"}},
        },
        "label_mapping": {"benign": 0, "jailbreak": 1},
        "metrics": {
            "rows": 827,
            "accuracy": 0.9,
            "macro_f1": 0.9,
            "weighted_f1": 0.9,
            "per_label": {
                "benign": {
                    "precision": 0.9,
                    "recall": 0.9,
                    "f1": 0.9,
                    "support": 400,
                },
                "jailbreak": {
                    "precision": 0.9,
                    "recall": 0.9,
                    "f1": 0.9,
                    "support": 427,
                },
            },
        },
        "calibration": {
            "bin_count": 2,
            "ece": 0.05,
            "mce": 0.1,
            "brier": 0.08,
            "bins": [
                {
                    "lower": 0.0,
                    "upper": 0.5,
                    "count": 0,
                    "confidence": None,
                    "accuracy": None,
                },
                {
                    "lower": 0.5,
                    "upper": 1.0,
                    "count": 827,
                    "confidence": 0.95,
                    "accuracy": 0.9,
                },
            ],
        },
        "abstention": {
            "curve": [
                {
                    "threshold": 0.7,
                    "coverage": 0.8,
                    "selective_accuracy": 0.95,
                    "abstained": 165,
                }
            ]
        },
        "performance": {
            "latency_ms": {"mean": 2.0, "p50": 1.8, "p95": 3.0, "p99": 4.0},
            "peak_memory_mb": 1500.0,
        },
    }
    manifest.update(overrides)
    return manifest


def write_bundle(directory, **replacements):
    manifests = {
        "dataset": dataset_manifest(),
        "run": run_manifest(),
        "artifact": artifact_manifest(),
        "evaluation": evaluation_manifest(),
    }
    manifests.update(replacements)
    for kind, manifest in manifests.items():
        if manifest is None:
            continue
        path = directory / f"{kind}.manifest.yaml"
        path.write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")
    return directory


def write_one(tmp_path, manifest, name="one.manifest.yaml"):
    path = tmp_path / name
    path.write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")
    return path


def test_valid_bundle_passes(tmp_path):
    summary = validate_bundle(write_bundle(tmp_path))
    assert summary["artifacts"] == ["mmbert32k-jailbreak-detector-merged"]
    assert summary["evaluations"] == ["jailbreak-test-baseline"]


def test_unknown_field_is_rejected(tmp_path):
    path = write_one(tmp_path, dataset_manifest(extra_field="nope"))
    with pytest.raises(ManifestError, match="extra_field"):
        load_manifest(path)


def test_branch_name_is_not_an_acceptable_revision(tmp_path):
    manifest = dataset_manifest()
    manifest["source"]["revision"] = "main"
    with pytest.raises(ManifestError, match=r"source\.revision"):
        load_manifest(write_one(tmp_path, manifest))


def test_schema_version_must_match_exactly(tmp_path):
    path = write_one(tmp_path, dataset_manifest(schema_version="v2"))
    with pytest.raises(ManifestError, match="schema_version"):
        load_manifest(path)


def test_non_contiguous_label_mapping_is_rejected(tmp_path):
    manifest = dataset_manifest(label_mapping={"benign": 0, "jailbreak": 2})
    with pytest.raises(ManifestError, match=r"indices 0\.\.1"):
        load_manifest(write_one(tmp_path, manifest))


def test_yaml_anchors_are_rejected(tmp_path):
    path = tmp_path / "anchor.manifest.yaml"
    path.write_text(
        "schema_version: &v v1\nkind: dataset\nid: x\nalias: *v\n", encoding="utf-8"
    )
    with pytest.raises(ManifestError, match="anchors"):
        load_manifest(path)


def test_embedded_token_is_rejected(tmp_path):
    manifest = dataset_manifest(description="pulled with hf_abcdefghijklmnopqrstuv")
    with pytest.raises(RedactionError, match="Hugging Face token"):
        load_manifest(write_one(tmp_path, manifest))


def test_inline_url_credentials_are_rejected(tmp_path):
    manifest = dataset_manifest()
    manifest["source"]["locator"] = "https://user:swordfish@example.com/dataset"
    with pytest.raises(RedactionError, match="inline URL credentials"):
        load_manifest(write_one(tmp_path, manifest))


def test_machine_specific_path_is_rejected(tmp_path):
    manifest = dataset_manifest(description="cached under /home/runner/datasets")
    with pytest.raises(RedactionError, match="absolute POSIX path"):
        load_manifest(write_one(tmp_path, manifest))


def test_permuted_label_order_fails_cross_reference(tmp_path):
    """The failure mode this contract exists to catch."""
    permuted = evaluation_manifest()
    permuted["label_mapping"] = {"benign": 1, "jailbreak": 0}
    with pytest.raises(ManifestError, match="label_mapping differs"):
        validate_bundle(write_bundle(tmp_path, evaluation=permuted))


def test_artifact_digest_must_match_the_file_list(tmp_path):
    tampered = artifact_manifest()
    tampered["files"] = [
        {"path": "config.json", "size_bytes": 12, "digest": "sha256:" + "9" * 64}
    ]
    with pytest.raises(ManifestError, match="file list hashes to"):
        validate_bundle(write_bundle(tmp_path, artifact=tampered))


def test_evaluation_referencing_a_different_artifact_revision_fails(tmp_path):
    stale = evaluation_manifest()
    stale["artifact_ref"]["revision"] = "e" * 40
    with pytest.raises(ManifestError, match=r"artifact_ref\.revision"):
        validate_bundle(write_bundle(tmp_path, evaluation=stale))


def test_unresolvable_dataset_reference_fails(tmp_path):
    orphan = evaluation_manifest()
    orphan["dataset_ref"]["id"] = "some-other-dataset"
    with pytest.raises(ManifestError, match="resolves to no dataset manifest"):
        validate_bundle(write_bundle(tmp_path, evaluation=orphan))


def test_silent_row_subset_fails(tmp_path):
    partial = evaluation_manifest()
    partial["metrics"]["rows"] = 100
    with pytest.raises(ManifestError, match="unexplained subset"):
        validate_bundle(write_bundle(tmp_path, evaluation=partial))


def test_more_rows_than_the_split_holds_fails(tmp_path):
    inflated = evaluation_manifest()
    inflated["metrics"]["rows"] = 5000
    with pytest.raises(ManifestError, match="holds only"):
        validate_bundle(write_bundle(tmp_path, evaluation=inflated))


def test_artifact_without_a_run_reference_fails(tmp_path):
    orphan = artifact_manifest()
    orphan.pop("run_ref")
    with pytest.raises(ManifestError, match="no run_ref"):
        validate_bundle(write_bundle(tmp_path, artifact=orphan))


def test_per_label_metrics_must_cover_every_declared_label(tmp_path):
    incomplete = evaluation_manifest()
    incomplete["metrics"]["per_label"].pop("benign")
    with pytest.raises(ManifestError, match="omits declared labels"):
        validate_bundle(write_bundle(tmp_path, evaluation=incomplete))


def test_bundle_without_an_evaluation_fails(tmp_path):
    with pytest.raises(ManifestError, match="no evaluation manifest"):
        validate_bundle(write_bundle(tmp_path, evaluation=None))


def test_evaluation_without_a_split_rule_fails(tmp_path):
    unstated = evaluation_manifest()
    unstated.pop("split_rule")
    with pytest.raises(ManifestError, match="split_rule"):
        load_manifest(write_one(tmp_path, unstated))


def test_row_level_split_is_recorded_rather_than_rejected(tmp_path):
    leaky = evaluation_manifest(split_rule="by_row")
    assert load_manifest(write_one(tmp_path, leaky))["split_rule"] == "by_row"


def test_an_invented_split_rule_is_rejected(tmp_path):
    manifest = evaluation_manifest(split_rule="by_prompt")
    with pytest.raises(ManifestError, match="split_rule"):
        load_manifest(write_one(tmp_path, manifest))


def test_composite_dataset_must_pin_its_upstreams(tmp_path):
    manifest = dataset_manifest()
    manifest["source"] = {
        "type": "composite",
        "locator": "src/training/model_classifier/x.py::build",
        "revision": CODE_REVISION,
    }
    with pytest.raises(ManifestError, match="components"):
        load_manifest(write_one(tmp_path, manifest))


EXPECTED_ROWS = 4
EXPECTED_ACCURACY = 0.75


def test_classification_metrics_count_each_label():
    """Per label support and F1 come from the rows, not from a rounded average."""
    metrics = classification_metrics(
        [1, 1, 0, 0], [1, 0, 0, 0], {"benign": 0, "jailbreak": 1}
    )
    assert metrics["rows"] == EXPECTED_ROWS
    assert metrics["accuracy"] == EXPECTED_ACCURACY
    assert metrics["per_label"]["jailbreak"] == {
        "precision": 1.0,
        "recall": 0.5,
        "f1": pytest.approx(2 / 3),
        "support": 2,
    }
    assert metrics["macro_f1"] == pytest.approx((2 / 3 + 0.8) / 2)


CALIBRATION_ROWS = (1, 1, 0)
CALIBRATION_BINS = 4
LATENCY_SAMPLES_MS = (5.0, 1.0, 3.0, 2.0, 4.0)
EXPECTED_MEAN_MS = 3.0
EXPECTED_P50_MS = 3.0
EXPECTED_P95_MS = 5.0


def test_calibration_bins_cover_the_unit_interval():
    """Every row lands in exactly one bin, and empty bins report no numbers."""
    calibration = calibration_metrics(
        [1, 1, 0], [1, 0, 0], [0.95, 0.55, 0.85], bin_count=CALIBRATION_BINS
    )
    assert calibration["bin_count"] == CALIBRATION_BINS
    assert sum(entry["count"] for entry in calibration["bins"]) == len(CALIBRATION_ROWS)
    empty = [entry for entry in calibration["bins"] if entry["count"] == 0]
    assert all(entry["confidence"] is None for entry in empty)
    assert 0.0 <= calibration["ece"] <= 1.0
    assert calibration["mce"] >= calibration["ece"]


def test_abstention_curve_reports_coverage_and_selective_accuracy():
    """A threshold that drops the wrong answer raises accuracy on what is left."""
    curve = abstention_curve([1, 0], [1, 1], [0.9, 0.6], thresholds=(0.5, 0.8))["curve"]
    assert curve[0] == {
        "threshold": 0.5,
        "coverage": 1.0,
        "selective_accuracy": 0.5,
        "abstained": 0,
    }
    assert curve[1] == {
        "threshold": 0.8,
        "coverage": 0.5,
        "selective_accuracy": 1.0,
        "abstained": 1,
    }


def test_abstention_curve_reports_no_accuracy_when_everything_abstains():
    """Selective accuracy over an empty selection is unknown, not perfect."""
    curve = abstention_curve([1], [1], [0.4], thresholds=(0.9,))["curve"]
    assert curve[0]["coverage"] == 0.0
    assert curve[0]["selective_accuracy"] is None


def test_latency_percentiles_use_nearest_rank():
    percentiles = latency_percentiles(LATENCY_SAMPLES_MS)
    assert percentiles["mean"] == EXPECTED_MEAN_MS
    assert percentiles["p50"] == EXPECTED_P50_MS
    assert percentiles["p95"] == EXPECTED_P95_MS


def test_metrics_reject_misaligned_inputs():
    with pytest.raises(ValueError):
        classification_metrics([1, 0], [1], {"benign": 0, "jailbreak": 1})
    with pytest.raises(ValueError):
        calibration_metrics([1], [1], [0.9], bin_count=1)
    with pytest.raises(ValueError):
        latency_percentiles([])

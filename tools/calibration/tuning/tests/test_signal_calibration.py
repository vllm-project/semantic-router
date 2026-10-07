"""Tests for calibrating a signal family onto label_correctness/v1."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tuning.confidence_calibration import ConfidenceCalibrationError
from tuning.signal_calibration import apply, build_signal_artifact, fit_isotonic, main

LABELS = ["biology", "law", "math"]


def _rows(prefix: str, spec: list[tuple[str, str, float]]) -> list[dict]:
    return [
        {
            "id": f"{prefix}-{index}",
            "category": category,
            "label": label,
            "score": score,
        }
        for index, (category, label, score) in enumerate(spec)
    ]


def _manifest(
    root: Path, calibration: list[dict], held_out: list[dict], **overrides
) -> Path:
    for split, rows in (("calibration", calibration), ("held_out", held_out)):
        (root / f"{split}.json").write_text(json.dumps(rows), encoding="utf-8")
    manifest = {
        "schema_version": "signal-calibration/v1",
        "name": "test-domain",
        "family": "domain",
        "scale": "label_correctness/v1",
        "method": "isotonic",
        "dataset": {"name": "fixture", "version": "v1", "digest": "sha256:fixture"},
        "population": "fixture rows",
        "outcome": "top label equals the gold category",
        "model": {
            "id": "fixture/model",
            "revision": "r1",
            "labels": LABELS,
            "model_sha256": "a" * 64,
        },
        "operating_threshold": 0.5,
        "splits": {"calibration": "calibration.json", "held_out": "held_out.json"},
        "policy": {"rollback_identity": "uncalibrated"},
        **overrides,
    }
    path = root / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path


# An overconfident classifier: right about two times in three at 0.9 and 0.95.
OVERCONFIDENT = [
    ("math", "math", 0.95),
    ("math", "math", 0.95),
    ("law", "math", 0.95),
    ("biology", "biology", 0.9),
    ("biology", "biology", 0.9),
    ("law", "biology", 0.9),
    ("law", "law", 0.55),
    ("math", "law", 0.55),
]


def test_isotonic_pools_violators_into_rising_knots():
    pairs = [(0.5, False), (0.6, True), (0.7, False), (0.9, True), (0.95, True)]
    knots = fit_isotonic(pairs)
    flat = [value for knot in knots for value in knot]
    assert flat == pytest.approx([0.5, 0.0, 0.65, 0.5, 0.925, 1.0])
    assert apply(knots, 0.1) == 0.0
    assert apply(knots, 0.575) == pytest.approx(0.25)
    assert apply(knots, 1.0) == 1.0


def test_artifact_reports_held_out_scores_and_gaps(tmp_path):
    manifest = _manifest(
        tmp_path, _rows("c", OVERCONFIDENT), _rows("h", OVERCONFIDENT[:6])
    )
    artifact = build_signal_artifact(manifest)

    assert artifact["status"] == "calibrated"
    held_out = artifact["metrics"]["held_out"]
    assert held_out["calibrated"]["brier"] < held_out["raw"]["brier"]
    assert held_out["calibrated"]["ece_10"] < held_out["raw"]["ece_10"]
    assert artifact["operating_point"]["coverage"] == 1.0
    assert artifact["operating_point"]["accuracy_when_matched"] == pytest.approx(4 / 6)
    assert set(artifact["failure_slices"]["by_category"]) == {"biology", "law", "math"}
    assert artifact["split_counts"] == {"calibration": 8, "held_out": 6}
    assert artifact["artifact_id"] == build_signal_artifact(manifest)["artifact_id"]


def test_mapping_that_does_not_transfer_is_not_calibrated(tmp_path):
    held_out = [("law", "law", 0.95)] * 4 + [("law", "math", 0.55)] * 4
    manifest = _manifest(tmp_path, _rows("c", OVERCONFIDENT), _rows("h", held_out))
    assert build_signal_artifact(manifest)["status"] == "no_improvement"


def test_held_out_label_without_rows_is_an_unsupported_region(tmp_path):
    rows = [("math", "math", 0.9), ("math", "math", 0.6), ("law", "law", 0.7)]
    manifest = _manifest(tmp_path, _rows("c", OVERCONFIDENT), _rows("h", rows))
    assert build_signal_artifact(manifest)["unsupported_regions"] == [
        "biology: no held-out rows"
    ]


@pytest.mark.parametrize(
    ("held_out", "overrides", "message"),
    [
        (_rows("c", OVERCONFIDENT[:1]), {}, "empty or repeated"),
        (
            [{"id": "x", "category": "law", "label": "art", "score": 0.9}],
            {},
            "known label",
        ),
        ([{"id": "x", "category": "law", "label": "law", "score": 1.5}], {}, "[0, 1]"),
        (_rows("h", OVERCONFIDENT), {"family": "embedding"}, "domain family"),
        (_rows("h", OVERCONFIDENT), {"operating_threshold": 0.3}, "[0.5, 1]"),
        (
            _rows("h", OVERCONFIDENT),
            {"model": {"id": "fixture/model", "revision": "r1", "labels": LABELS}},
            "model_sha256",
        ),
    ],
)
def test_unsafe_inputs_are_refused(tmp_path, held_out, overrides, message):
    manifest = _manifest(tmp_path, _rows("c", OVERCONFIDENT), held_out, **overrides)
    with pytest.raises(ConfidenceCalibrationError, match=message):
        build_signal_artifact(manifest)


def test_cli_writes_the_artifact(tmp_path, capsys):
    manifest = _manifest(tmp_path, _rows("c", OVERCONFIDENT), _rows("h", OVERCONFIDENT))
    output = tmp_path / "artifact.json"
    assert main(["--manifest", str(manifest), "--output", str(output)]) == 0
    assert json.loads(output.read_text())["scale"] == "label_correctness/v1"
    assert capsys.readouterr().out.startswith("calibrated:")

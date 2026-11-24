"""Unit tests for the distribution-calibrated proposer (sub-issue 7, item D).

The golden case is the BUG-3 recalibration: `difficulty gte 2.5` -> the
observed p75 (1.75) across all four sites, matching PR #5568.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml
from recipe_lab import diagnose
from recipe_lab.cli import main
from recipe_lab.contract import dataset_content_hash
from recipe_lab.insights import resolve_insight_refs
from recipe_lab.proposers import distribution_calibration, rule_threshold
from tests.fixtures import pilot_rows_jsonl

FIXTURES = Path(__file__).resolve().parent / "fixtures"
AGENT_RECIPE_MIN = FIXTURES / "agent_recipe_min.yaml"


def _diagnose(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> Path:
    rows_path = tmp_path / "rows.jsonl"
    rows_path.write_text(pilot_rows_jsonl(), encoding="utf-8")
    diag_dir = tmp_path / "diag"
    assert (
        main(
            [
                "diagnose",
                "--records",
                str(rows_path),
                "--config",
                str(AGENT_RECIPE_MIN),
                "--output-dir",
                str(diag_dir),
            ]
        )
        == 0
    )
    capsys.readouterr()  # drop the diagnose stdout so propose can parse its own
    return diag_dir / "diagnosis.json"


def _propose(
    tmp_path: Path,
    diagnosis_path: Path,
    capsys: pytest.CaptureFixture[str],
    *extra: str,
    base_config: Path | None = None,
) -> dict:
    out_dir = tmp_path / "cands"
    argv = [
        "propose-calibrated",
        "--diagnosis",
        str(diagnosis_path),
        "--base-config",
        str(base_config or AGENT_RECIPE_MIN),
        "--signal",
        "difficulty",
        "--output-dir",
        str(out_dir),
        *extra,
    ]
    assert main(argv) == 0
    return json.loads(capsys.readouterr().out)


def _config_sites(config_path: Path, signal: str = "difficulty") -> list[float]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    return [
        s.value
        for s in rule_threshold.find_threshold_sites(config)
        if s.signal == signal
    ]


def test_quantile_proposal_matches_bug3_golden(tmp_path, capsys):
    diagnosis_path = _diagnose(tmp_path, capsys)
    payload = _propose(
        tmp_path, diagnosis_path, capsys, "--from", "2.5", "--quantiles", "0.75"
    )
    assert payload["candidates"] == ["difficulty gte 2.5 -> 1.75 (p75 of n=344)"]
    assert len(payload["files"]) == 1
    config_path = Path(payload["files"][0])
    assert _config_sites(config_path) == [1.75, 1.75, 1.75, 1.75]
    sidecar = json.loads(
        config_path.with_suffix(".candidate.json").read_text(encoding="utf-8")
    )
    assert sidecar["ref"] == "calibrated:difficulty=1.75"
    assert sidecar["change_summary"] == "difficulty gte 2.5 -> 1.75 (p75 of n=344)"
    provenance = sidecar["provenance"]
    assert provenance["proposer"] == "distribution-calibration"
    # one verified diag ref per moved site, decision-first anchors (rl/04 shape)
    assert len(provenance["insight_refs"]) == 4
    resolved = resolve_insight_refs(provenance["insight_refs"])
    assert len(resolved) == 4
    anchors = [ref.split("#", 1)[1] for ref in resolved]
    assert anchors == [
        "threshold.domain_code_complex.difficulty",
        "threshold.domain_code.difficulty",
        "threshold.complex_general.difficulty",
        "threshold.medium_general.difficulty",
    ]
    for ref in resolved:
        assert ref.startswith(f"diag:{diagnosis_path}@sha256:")


def test_target_share_matches_quantile(tmp_path, capsys):
    diagnosis_path = _diagnose(tmp_path, capsys)
    by_quantile = _propose(
        tmp_path, diagnosis_path, capsys, "--from", "2.5", "--quantiles", "0.75"
    )
    quantile_config = Path(by_quantile["files"][0]).read_bytes()
    share_dir = tmp_path / "cands-share"
    argv = [
        "propose-calibrated",
        "--diagnosis",
        str(diagnosis_path),
        "--base-config",
        str(AGENT_RECIPE_MIN),
        "--signal",
        "difficulty",
        "--from",
        "2.5",
        "--target-share",
        "0.25",
        "--output-dir",
        str(share_dir),
    ]
    assert main(argv) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["candidates"] == [
        "difficulty gte 2.5 -> 1.75 (p75 (target-share 0.25) of n=344)"
    ]
    assert Path(payload["files"][0]).read_bytes() == quantile_config


def test_multiple_quantiles_ordered(tmp_path, capsys):
    diagnosis_path = _diagnose(tmp_path, capsys)
    sample = _p50_sample(tmp_path)
    payload = _propose(tmp_path, diagnosis_path, capsys, "--quantiles", "0.9,0.5")
    p90 = round(diagnose.percentile(sample, 90), 2)
    p50 = round(diagnose.percentile(sample, 50), 2)
    assert payload["candidates"] == [
        f"difficulty gte 2.5 -> {p90:g} (p90 of n=344)",
        f"difficulty gte 2.5 -> {p50:g} (p50 of n=344)",
    ]
    # files follow proposal order (p90 first as requested)
    assert [Path(f).name for f in payload["files"]] == sorted(
        Path(f).name for f in payload["files"]
    )
    first_value = _single_new_value(
        yaml.safe_load(Path(payload["files"][0]).read_text(encoding="utf-8"))
    )
    second_value = _single_new_value(
        yaml.safe_load(Path(payload["files"][1]).read_text(encoding="utf-8"))
    )
    assert (first_value, second_value) == (p90, p50)
    assert first_value > second_value  # p90 above p50


def _single_new_value(config: dict) -> float:
    values = {
        round(s.value, 6)
        for s in rule_threshold.find_threshold_sites(config)
        if s.signal == "difficulty"
    }
    assert len(values) == 1
    return values.pop()


def _p50_sample(tmp_path: Path) -> list[float]:
    diagnosis = diagnose.load_diagnosis(tmp_path / "diag" / "diagnosis.json")
    entry = next(
        s
        for s in diagnosis["signal_distribution"]["signals"]
        if s["key"] == "decision:difficulty"
    )
    return entry["values"]


def test_config_hash_mismatch_rejected(tmp_path, capsys):
    diagnosis_path = _diagnose(tmp_path, capsys)
    # a re-formatted base config is a different base: stale evidence must not apply
    mutated = tmp_path / "mutated.yaml"
    mutated.write_text(
        AGENT_RECIPE_MIN.read_text(encoding="utf-8") + "\n# reformatted\n",
        encoding="utf-8",
    )
    with pytest.raises(SystemExit, match="proposer error"):
        _propose(
            tmp_path, diagnosis_path, capsys, "--quantiles", "0.75", base_config=mutated
        )


def test_diagnosis_without_config_hash_rejected(tmp_path, capsys):
    diagnosis = diagnose.load_diagnosis(_diagnose(tmp_path, capsys))
    diagnosis["inputs"]["config_hash"] = ""
    broken = tmp_path / "nohash.json"
    broken.write_text(json.dumps(diagnosis), encoding="utf-8")
    with pytest.raises(SystemExit, match="proposer error"):
        _propose(tmp_path, broken, capsys, "--quantiles", "0.75")


def test_unknown_signal_rejected(tmp_path, capsys):
    diagnosis_path = _diagnose(tmp_path, capsys)
    with pytest.raises(SystemExit, match="proposer error"):
        argv = [
            "propose-calibrated",
            "--diagnosis",
            str(diagnosis_path),
            "--base-config",
            str(AGENT_RECIPE_MIN),
            "--signal",
            "ghost",
            "--quantiles",
            "0.75",
            "--output-dir",
            str(tmp_path / "cands"),
        ]
        main(argv)


def test_quantile_bounds_enforced(tmp_path, capsys):
    diagnosis_path = _diagnose(tmp_path, capsys)
    with pytest.raises(SystemExit, match="outside"):
        _propose(tmp_path, diagnosis_path, capsys, "--quantiles", "0")


def test_flags_mutually_exclusive(tmp_path, capsys):
    diagnosis_path = _diagnose(tmp_path, capsys)
    with pytest.raises(SystemExit, match="mutually exclusive"):
        _propose(
            tmp_path,
            diagnosis_path,
            capsys,
            "--quantiles",
            "0.75",
            "--target-share",
            "0.25",
        )


def test_missing_intent_rejected(tmp_path, capsys):
    diagnosis_path = _diagnose(tmp_path, capsys)
    with pytest.raises(SystemExit, match="one of --quantiles or --target-share"):
        _propose(tmp_path, diagnosis_path, capsys)


def test_old_value_never_proposed(tmp_path):
    # p50 of the sample is the old threshold itself: no candidate survives
    diagnosis = {
        "kind": "recipe-lab-diagnosis-v1",
        "inputs": {"config_hash": dataset_content_hash(AGENT_RECIPE_MIN)},
        "signal_distribution": {
            "signals": [
                {
                    "key": "decision:difficulty",
                    "signal": "difficulty",
                    "n": 3,
                    "values": [2.4, 2.5, 2.6],
                }
            ]
        },
    }
    path = tmp_path / "diagnosis.json"
    path.write_text(json.dumps(diagnosis), encoding="utf-8")
    config = yaml.safe_load(AGENT_RECIPE_MIN.read_text(encoding="utf-8"))
    proposals = distribution_calibration.propose_calibrated(
        diagnosis,
        config,
        "difficulty",
        diagnosis_ref=f"diag:{path}@{dataset_content_hash(path)}",
        config_hash=dataset_content_hash(AGENT_RECIPE_MIN),
        quantiles=[0.5],
        from_value=2.5,
    )
    assert proposals == []

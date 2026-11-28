"""Offline calibration smoke (sub-issue 7, J2).

Reproduces the BUG-3 recalibration (#5564 / PR #5568) end to end from a
synthetic pilot-shaped distribution: diagnose finds the dead lanes, the
compressed difficulty scale and the four unreachable gte 2.5 sites;
propose-calibrated moves them to the observed p75. No GPU, no router.
Mirrors examples/smoke/calibration-smoke.sh.
"""

from __future__ import annotations

import json
from pathlib import Path

import yaml
from recipe_lab.cli import main
from recipe_lab.proposers import rule_threshold
from tests.fixtures import pilot_rows_jsonl

FIXTURES = Path(__file__).resolve().parent / "fixtures"
AGENT_RECIPE_MIN = FIXTURES / "agent_recipe_min.yaml"


def _run(tmp_path: Path, capsys, *argv: str) -> dict:
    capsys.readouterr()
    assert main(list(argv)) == 0
    return json.loads(capsys.readouterr().out)


def test_calibration_smoke_reproduces_bug3(tmp_path, capsys):
    rows_path = tmp_path / "rows.jsonl"
    rows_path.write_text(pilot_rows_jsonl(), encoding="utf-8")

    diag = _run(
        tmp_path,
        capsys,
        "diagnose",
        "--records",
        str(rows_path),
        "--config",
        str(AGENT_RECIPE_MIN),
        "--output-dir",
        str(tmp_path / "diag"),
        "--source",
        "calibration smoke: synthetic BUG-3 pilot",
    )
    assert diag["dead_decisions"] == ["domain_code_complex", "complex_general"]
    assert len(diag["unreachable_sites"]) == 4
    assert any(
        a["type"] == "compressed_range" and a["signal"] == "difficulty"
        for a in diag["alerts"]
    )
    assert (tmp_path / "diag" / "diagnosis.md").is_file()

    payload = _run(
        tmp_path,
        capsys,
        "propose-calibrated",
        "--diagnosis",
        str(tmp_path / "diag" / "diagnosis.json"),
        "--base-config",
        str(AGENT_RECIPE_MIN),
        "--signal",
        "difficulty",
        "--from",
        "2.5",
        "--quantiles",
        "0.75",
        "--output-dir",
        str(tmp_path / "cands"),
    )
    assert payload["candidates"] == ["difficulty gte 2.5 -> 1.75 (p75 of n=344)"]
    candidate_path = Path(payload["files"][0])
    sidecar = json.loads(
        candidate_path.with_suffix(".candidate.json").read_text(encoding="utf-8")
    )
    assert sidecar["provenance"][
        "insight_refs"
    ], "the candidate must cite its diagnosis"

    # identical to what the rule-level proposer produces for 1.75 (site for site)
    calibrated = yaml.safe_load(candidate_path.read_text(encoding="utf-8"))
    golden = rule_threshold.propose_rule_thresholds(
        yaml.safe_load(AGENT_RECIPE_MIN.read_text(encoding="utf-8")),
        "difficulty",
        values=[1.75],
        from_value=2.5,
    )
    assert len(golden) == 1
    assert calibrated == golden[0].config

    # --target-share 0.25 is the same intent as --quantiles 0.75
    share = _run(
        tmp_path,
        capsys,
        "propose-calibrated",
        "--diagnosis",
        str(tmp_path / "diag" / "diagnosis.json"),
        "--base-config",
        str(AGENT_RECIPE_MIN),
        "--signal",
        "difficulty",
        "--from",
        "2.5",
        "--target-share",
        "0.25",
        "--output-dir",
        str(tmp_path / "cands-share"),
    )
    assert Path(share["files"][0]).read_bytes() == candidate_path.read_bytes()

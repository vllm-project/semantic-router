"""Rule-level threshold proposer tests (C), including the BUG-3 golden case.

The golden fixture (agent_recipe_min.yaml) mirrors config/recipes/agent:
four `difficulty gte 2.5` rule sites across four decisions, two of them under
NOT branches; PR #5568 calibrated exactly those four sites to 1.75.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml
from recipe_lab.cli import main
from recipe_lab.contract import ContractError
from recipe_lab.proposers import rule_threshold

from tests.test_cli import _write_frozen_spec

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "agent_recipe_min.yaml"


def _load_base() -> dict:
    return yaml.safe_load(FIXTURE.read_text(encoding="utf-8"))


# ---- site discovery ----


def test_find_threshold_sites_golden_four():
    sites = rule_threshold.find_threshold_sites(_load_base())
    difficulty = [s for s in sites if s.signal == "difficulty"]
    assert len(difficulty) == 4
    assert all(s.op == "gte" and s.value == 2.5 for s in difficulty)
    assert sorted(s.decision for s in difficulty) == [
        "complex_general",
        "domain_code",
        "domain_code_complex",
        "medium_general",
    ]
    assert sum(1 for s in difficulty if s.negated) == 2
    # domain leaves carry labels, not predicates: the signal default is not a site
    assert [s for s in sites if s.signal == "domain"] == []


def test_group_sites_by_signal_op_value():
    groups = rule_threshold.group_sites(
        rule_threshold.find_threshold_sites(_load_base())
    )
    assert list(groups) == [("difficulty", "gte", 2.5)]
    assert len(groups[("difficulty", "gte", 2.5)]) == 4


def test_site_paths_round_trip_to_the_predicate():
    base = _load_base()
    for site in rule_threshold.find_threshold_sites(base):
        node = base
        for key in site.path:
            node = node[key]
        assert node[site.op] == site.value


# ---- ranges ----


def test_signal_range_score_uses_levels():
    assert rule_threshold.signal_range(_load_base(), "difficulty") == (0.0, 4.0)


def test_signal_range_choice_is_binary():
    assert rule_threshold.signal_range(_load_base(), "domain") == (0.0, 1.0)


def test_signal_range_unknown_signal_requires_explicit_bounds():
    with pytest.raises(ContractError, match="--lo/--hi"):
        rule_threshold.signal_range(_load_base(), "nonexistent")


# ---- proposal ----


def test_propose_values_moves_all_selected_sites():
    base = _load_base()
    candidates = rule_threshold.propose_rule_thresholds(
        base, "difficulty", values=[1.75]
    )
    assert len(candidates) == 1
    candidate = candidates[0]
    assert candidate.new_value == 1.75
    assert candidate.ref == "config-rev:difficulty=1.75"
    assert len(candidate.sites) == 4
    # the base config is untouched
    assert all(s.value == 2.5 for s in rule_threshold.find_threshold_sites(base))
    applied = [
        s
        for s in rule_threshold.find_threshold_sites(candidate.config)
        if s.signal == "difficulty"
    ]
    assert all(s.value == 1.75 for s in applied)
    assert sum(1 for s in applied if s.negated) == 2


def test_propose_deltas_moves_relative_to_old():
    candidates = rule_threshold.propose_rule_thresholds(
        _load_base(), "difficulty", deltas=[-0.75]
    )
    assert len(candidates) == 1
    assert candidates[0].new_value == 1.75


def test_propose_deltas_rejects_mixed_old_values():
    base = _load_base()
    base["routing"]["decisions"][5]["rules"] = {
        "operator": "AND",
        "conditions": [
            {"type": "decision", "name": "difficulty", "predicate": {"gte": 1.0}}
        ],
    }
    with pytest.raises(ContractError, match="mixed values"):
        rule_threshold.propose_rule_thresholds(base, "difficulty", deltas=[-0.5])


def test_propose_from_filters_sites():
    base = _load_base()
    base["routing"]["decisions"][5]["rules"] = {
        "operator": "AND",
        "conditions": [
            {"type": "decision", "name": "difficulty", "predicate": {"gte": 1.0}}
        ],
    }
    candidates = rule_threshold.propose_rule_thresholds(
        base, "difficulty", values=[1.75], from_value=2.5
    )
    applied = sorted(
        s.value
        for s in rule_threshold.find_threshold_sites(candidates[0].config)
        if s.signal == "difficulty"
    )
    assert applied == [1.0, 1.75, 1.75, 1.75, 1.75]


def test_propose_clamps_to_signal_range():
    candidates = rule_threshold.propose_rule_thresholds(
        _load_base(), "difficulty", values=[9.0, 1.75]
    )
    assert [c.new_value for c in candidates] == [4.0, 1.75]


def test_propose_explicit_lo_hi_overrides_range():
    candidates = rule_threshold.propose_rule_thresholds(
        _load_base(), "difficulty", values=[3.0], lo=0.0, hi=1.0
    )
    assert [c.new_value for c in candidates] == [1.0]


def test_propose_never_proposes_the_old_value():
    assert (
        rule_threshold.propose_rule_thresholds(
            _load_base(), "difficulty", values=[2.5, 2.5]
        )
        == []
    )


def test_propose_requires_values_or_deltas():
    with pytest.raises(ContractError, match="values or deltas"):
        rule_threshold.propose_rule_thresholds(_load_base(), "difficulty")


# ---- decision filter ----


def test_propose_decisions_filter_moves_only_selected():
    candidates = rule_threshold.propose_rule_thresholds(
        _load_base(), "difficulty", values=[1.75], decisions=["complex_general"]
    )
    applied = rule_threshold.find_threshold_sites(candidates[0].config)
    moved = [
        s.decision for s in applied if s.signal == "difficulty" and s.value == 1.75
    ]
    assert moved == ["complex_general"]
    assert len([s for s in applied if s.signal == "difficulty" and s.value == 2.5]) == 3


def test_propose_unknown_decision_requires_allow_partial():
    with pytest.raises(ContractError, match="--allow-partial"):
        rule_threshold.propose_rule_thresholds(
            _load_base(),
            "difficulty",
            values=[1.75],
            decisions=["complex_general", "no_such"],
        )


def test_propose_allow_partial_warns_and_moves_covered(capsys):
    candidates = rule_threshold.propose_rule_thresholds(
        _load_base(),
        "difficulty",
        values=[1.75],
        decisions=["complex_general", "no_such"],
        allow_partial=True,
    )
    assert len(candidates) == 1
    assert len(candidates[0].sites) == 1
    assert "no rule threshold site" in capsys.readouterr().err


def test_propose_signal_without_rule_sites_errors():
    with pytest.raises(ContractError, match="no rule threshold sites"):
        rule_threshold.propose_rule_thresholds(_load_base(), "domain", values=[0.9])


# ---- CLI: the BUG-3 golden calibration ----


def test_cli_golden_bug3_calibration(tmp_path, capsys):
    out_dir = tmp_path / "cands"
    assert (
        main(
            [
                "propose-rule-thresholds",
                "--base-config",
                str(FIXTURE),
                "--signal",
                "difficulty",
                "--from",
                "2.5",
                "--values",
                "1.75",
                "--output-dir",
                str(out_dir),
            ]
        )
        == 0
    )
    parsed = json.loads(capsys.readouterr().out)
    assert len(parsed["files"]) == 1
    config_path = Path(parsed["files"][0])
    assert config_path.name == "candidate-000-difficulty-1.75.yaml"
    applied = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    sites = [
        s
        for s in rule_threshold.find_threshold_sites(applied)
        if s.signal == "difficulty"
    ]
    assert len(sites) == 4
    assert all(s.value == 1.75 for s in sites)
    assert sorted(s.decision for s in sites) == [
        "complex_general",
        "domain_code",
        "domain_code_complex",
        "medium_general",
    ]
    assert sum(1 for s in sites if s.negated) == 2
    # the sidecar carries exact provenance for `run` to record in the ledger
    sidecar = json.loads(
        config_path.with_suffix(".candidate.json").read_text(encoding="utf-8")
    )
    assert sidecar["kind"] == "config"
    assert sidecar["ref"] == "config-rev:difficulty=1.75"
    assert sidecar["provenance"]["proposer"] == "rule-threshold"
    assert sidecar["provenance"]["insight_refs"] == []
    assert sidecar["sites"] == [
        {
            "decision": "domain_code_complex",
            "signal": "difficulty",
            "op": "gte",
            "from": 2.5,
            "to": 1.75,
            "negated": False,
        },
        {
            "decision": "domain_code",
            "signal": "difficulty",
            "op": "gte",
            "from": 2.5,
            "to": 1.75,
            "negated": True,
        },
        {
            "decision": "complex_general",
            "signal": "difficulty",
            "op": "gte",
            "from": 2.5,
            "to": 1.75,
            "negated": False,
        },
        {
            "decision": "medium_general",
            "signal": "difficulty",
            "op": "gte",
            "from": 2.5,
            "to": 1.75,
            "negated": True,
        },
    ]


def test_cli_list_sites(tmp_path, capsys):
    assert (
        main(
            [
                "propose-rule-thresholds",
                "--base-config",
                str(FIXTURE),
                "--signal",
                "difficulty",
                "--list-sites",
            ]
        )
        == 0
    )
    parsed = json.loads(capsys.readouterr().out)
    assert len(parsed["sites"]) == 4
    assert all(s["op"] == "gte" and s["value"] == 2.5 for s in parsed["sites"])


def test_cli_requires_values_or_deltas(tmp_path):
    with pytest.raises(SystemExit, match="--values or --deltas"):
        main(
            [
                "propose-rule-thresholds",
                "--base-config",
                str(FIXTURE),
                "--signal",
                "difficulty",
                "--output-dir",
                str(tmp_path / "c"),
            ]
        )


def test_cli_values_and_deltas_are_exclusive(tmp_path):
    with pytest.raises(SystemExit, match="mutually exclusive"):
        main(
            [
                "propose-rule-thresholds",
                "--base-config",
                str(FIXTURE),
                "--signal",
                "difficulty",
                "--values",
                "1.75",
                "--deltas",
                "-0.75",
                "--output-dir",
                str(tmp_path / "c"),
            ]
        )


# ---- run --validate and sidecar provenance ----


def _run_args(tmp_path, spec_path, cases, candidates: Path) -> list[str]:
    return [
        "run",
        "--spec",
        spec_path,
        "--cases",
        str(cases),
        "--bank",
        str(tmp_path / "bank.json"),
        "--ledger",
        str(tmp_path / "ledger.jsonl"),
        "--candidates",
        str(candidates),
        "--router-bin",
        "/bin/true",
        "--mock-backend",
        str(tmp_path / "mock.py"),
        "--output-dir",
        str(tmp_path / "out"),
    ]


def _empty_bank(tmp_path) -> None:
    (tmp_path / "bank.json").write_text("", encoding="utf-8")


def test_run_all_candidates_rejected_exits_2(tmp_path, capsys):
    spec_path, cases = _write_frozen_spec(tmp_path)
    _empty_bank(tmp_path)
    cands = tmp_path / "cands"
    cands.mkdir()
    (cands / "bad.yaml").write_text("routing: {}\n", encoding="utf-8")
    validator = tmp_path / "validator.sh"
    validator.write_text(
        "#!/bin/sh\necho 'signal unknown_signal not found' >&2\nexit 1\n",
        encoding="utf-8",
    )
    validator.chmod(0o755)
    args = [
        *_run_args(tmp_path, spec_path, cases, cands),
        "--validate",
        "--validate-bin",
        str(validator),
    ]
    assert main(args) == 2
    parsed = json.loads(capsys.readouterr().out)
    assert parsed["evaluated"] == 0
    assert parsed["best"] is None
    assert parsed["rejected"][0]["path"].endswith("bad.yaml")
    assert "config validate exit 1" in parsed["rejected"][0]["reason"]
    # nothing reached the ledger
    assert (
        not (tmp_path / "ledger.jsonl").exists()
        or (tmp_path / "ledger.jsonl").read_text(encoding="utf-8") == ""
    )


def test_run_validate_missing_binary_exits_2(tmp_path, capsys):
    spec_path, cases = _write_frozen_spec(tmp_path)
    _empty_bank(tmp_path)
    cands = tmp_path / "cands"
    cands.mkdir()
    (cands / "c.yaml").write_text("routing: {}\n", encoding="utf-8")
    args = [
        *_run_args(tmp_path, spec_path, cases, cands),
        "--validate",
        "--validate-bin",
        str(tmp_path / "no-such-bin"),
    ]
    assert main(args) == 2
    assert "validator binary not found" in capsys.readouterr().err


def test_run_sidecar_provenance_and_partial_rejection(monkeypatch, tmp_path, capsys):
    from recipe_lab import cli

    spec_path, cases = _write_frozen_spec(tmp_path)
    _empty_bank(tmp_path)
    cands = tmp_path / "cands"
    cands.mkdir()
    # candidate A: with sidecar provenance (valid under the stub validator)
    (cands / "candidate-000-difficulty-1.75.yaml").write_text(
        "routing: {}\n", encoding="utf-8"
    )
    (cands / "candidate-000-difficulty-1.75.candidate.json").write_text(
        json.dumps(
            {
                "kind": "config",
                "ref": "config-rev:difficulty=1.75",
                "change_summary": "rule thresholds: signal 'difficulty' gte 2.5 -> 1.75 (4 site(s))",
                "provenance": {"proposer": "rule-threshold", "insight_refs": []},
                "sites": [],
            }
        ),
        encoding="utf-8",
    )
    # candidate B: no sidecar -> falls back to the bare-path entry
    (cands / "candidate-001.yaml").write_text("routing: {}\n", encoding="utf-8")
    # candidate C: rejected by the stub validator
    (cands / "candidate-002.yaml").write_text("routing: {}\n", encoding="utf-8")
    validator = tmp_path / "validator.sh"
    validator.write_text(
        '#!/bin/sh\ncase "$4" in *candidate-002*) exit 1;; esac\nexit 0\n',
        encoding="utf-8",
    )
    validator.chmod(0o755)

    def fake_evaluate(**kwargs):
        return {"summary": {"routed_accuracy": 0.5}, "bank_stats": {}, "statuses": {}}

    monkeypatch.setattr(cli, "evaluate_candidate", fake_evaluate)
    args = [
        *_run_args(tmp_path, spec_path, cases, cands),
        "--validate",
        "--validate-bin",
        str(validator),
    ]
    assert main(args) == 0
    parsed = json.loads(capsys.readouterr().out)
    assert parsed["evaluated"] == 2
    assert len(parsed["rejected"]) == 1
    assert parsed["rejected"][0]["path"].endswith("candidate-002.yaml")
    # the ledger kept the sidecar's provenance for candidate A
    rows = [
        json.loads(line)
        for line in (tmp_path / "ledger.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    by_ref = {r["candidate"]["ref"]: r for r in rows}
    assert set(by_ref) == {
        "config-rev:difficulty=1.75",
        str(cands / "candidate-001.yaml"),
    }
    assert (
        by_ref["config-rev:difficulty=1.75"]["candidate"]["provenance"]["proposer"]
        == "rule-threshold"
    )
    fallback = by_ref[str(cands / "candidate-001.yaml")]
    assert fallback["candidate"]["provenance"]["proposer"] == "threshold-search"
    assert fallback["candidate"]["change_summary"] == "candidate 2/2"

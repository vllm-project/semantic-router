"""Unit tests for the diagnose command (sub-issue 7, item D).

The synthetic fixture reproduces the BUG-3 pilot shape (appendix A of
sub-issue-7): 344 difficulty scores in [0.08, 1.87], nothing at or above the
stale 2.5 thresholds, so the strong lanes are dead and the four gte 2.5
sites are unreachable.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml
from recipe_lab import diagnose
from recipe_lab.contract import ContractError, dataset_content_hash
from tests.fixtures import pilot_rows

FIXTURES = Path(__file__).resolve().parent / "fixtures"
AGENT_RECIPE_MIN = FIXTURES / "agent_recipe_min.yaml"


def _config() -> dict:
    return yaml.safe_load(AGENT_RECIPE_MIN.read_text(encoding="utf-8"))


def _write_rows(tmp_path: Path) -> Path:
    path = tmp_path / "rows.jsonl"
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in pilot_rows()),
        encoding="utf-8",
    )
    return path


def _entry(distribution: dict, key: str) -> dict:
    return next(s for s in distribution["signals"] if s["key"] == key)


def test_coverage_lists_every_decision_and_flags_dead():
    coverage = diagnose.decision_coverage(pilot_rows(), _config())
    assert coverage["total_decisions"] == 6
    assert coverage["dead_decisions"] == ["domain_code_complex", "complex_general"]
    assert sum(entry["hits"] for entry in coverage["decisions"]) == 344
    statuses = {entry["decision"]: entry["status"] for entry in coverage["decisions"]}
    assert statuses["domain_code_complex"] == "dead"
    assert statuses["complex_general"] == "dead"
    assert statuses["medium_general"] == "ok"


def test_expect_rare_reported_but_never_warned():
    coverage = diagnose.decision_coverage(
        pilot_rows(), _config(), expect_rare=("domain_code_complex", "complex_general")
    )
    assert coverage["dead_decisions"] == []
    by_name = {entry["decision"]: entry for entry in coverage["decisions"]}
    assert by_name["domain_code_complex"]["expected_rare"] is True
    assert by_name["complex_general"]["expected_rare"] is True
    # still reported: hits and shares stay visible
    assert by_name["complex_general"]["hits"] == 0


def test_rare_flagged_below_pct():
    # the fixture rows cycle four fired decisions evenly: 25% each
    coverage = diagnose.decision_coverage(pilot_rows(), _config(), rare_pct=30.0)
    assert coverage["rare_decisions"] == [
        {"decision": "simple_math_fast_path", "hit_pct": 25.0},
        {"decision": "domain_code", "hit_pct": 25.0},
        {"decision": "medium_general", "hit_pct": 25.0},
        {"decision": "simple_general", "hit_pct": 25.0},
    ]
    # at the default 1% they are all comfortably ok
    assert diagnose.decision_coverage(pilot_rows(), _config())["rare_decisions"] == []


def test_distribution_quantiles_range_and_utilization():
    distribution = diagnose.signal_distribution(pilot_rows(), _config())
    difficulty = _entry(distribution, "decision:difficulty")
    assert difficulty["signal"] == "difficulty"
    assert difficulty["n"] == 344
    assert difficulty["min"] == 0.08
    assert difficulty["max"] == 1.87
    assert difficulty["p75"] == 1.75
    assert difficulty["range"] == [0.0, 4.0]
    assert difficulty["range_utilization"] < 0.5
    assert difficulty["values"] == sorted(difficulty["values"])
    assert len(difficulty["values"]) == 344
    # per-choice keys are their own entries on the choice signal's [0, 1] range
    choice = _entry(distribution, "decision:domain:code")
    assert choice["signal"] == "domain"
    assert choice["range"] == [0.0, 1.0]
    assert choice["n"] == 344
    # keys are listed in sorted order
    keys = [s["key"] for s in distribution["signals"]]
    assert keys == sorted(keys)


def test_distribution_unknown_signal_falls_back_to_observed_range():
    rows = pilot_rows()[:50]
    for _index, row in enumerate(rows):
        row["signal_values"]["decision:ghost"] = 0.42
    distribution = diagnose.signal_distribution(rows, _config())
    ghost = _entry(distribution, "decision:ghost")
    assert ghost["signal"] == "ghost"
    assert ghost["range"] is None
    assert ghost["range_utilization"] is None
    assert (
        ghost["histogram"]["counts"][9] == 50
    )  # a degenerate range parks everything in the last bucket


def test_reachability_flags_the_four_bug3_sites():
    reachability = diagnose.threshold_reachability(pilot_rows(), _config())
    unreachable = [s for s in reachability["sites"] if s["status"] == "unreachable"]
    assert len(unreachable) == 4
    assert all(
        s["signal"] == "difficulty" and s["op"] == "gte" and s["value"] == 2.5
        for s in unreachable
    )
    assert all(s["satisfied"] == 0 and s["evaluated"] == 344 for s in unreachable)
    # two of the four sites live under NOT branches and keep their written predicate
    assert sorted(s["negated"] for s in unreachable) == [False, False, True, True]
    groups = [g for g in reachability["groups"] if g["status"] == "unreachable"]
    assert len(groups) == 1
    assert len(groups[0]["sites"]) == 4
    # the bare difficulty conditions carry no predicate and are not sites
    assert all(s["value"] == 2.5 for s in reachability["sites"])


def test_unobserved_site_never_makes_a_group_unreachable():
    rows = [
        {"request_id": "r0", "matched_decision": "simple_general", "signal_values": {}}
    ]
    reachability = diagnose.threshold_reachability(rows, _config())
    assert all(site["status"] == "unobserved" for site in reachability["sites"])
    assert all(group["status"] == "unobserved" for group in reachability["groups"])


def test_choice_predicate_evaluated_against_strongest_choice():
    config = {
        "recipes": [
            {
                "name": "default",
                "signals": {
                    "systemone": [{"name": "domain", "question": {"type": "choice"}}]
                },
                "decisions": [
                    {
                        "name": "only",
                        "rules": {
                            "operator": "AND",
                            "conditions": [
                                {
                                    "type": "systemone",
                                    "name": "domain",
                                    "predicate": {"gte": 0.9},
                                }
                            ],
                        },
                        "fallback_model": "m",
                    }
                ],
            }
        ]
    }
    rows = [
        {
            "matched_decision": "only",
            "signal_values": {
                "decision:domain:code": 0.3,
                "decision:domain:other": 0.95,
            },
        },
        {
            "matched_decision": "only",
            "signal_values": {
                "decision:domain:code": 0.8,
                "decision:domain:other": 0.4,
            },
        },
    ]
    reachability = diagnose.threshold_reachability(rows, config)
    site = reachability["sites"][0]
    assert site["satisfied"] == 1 and site["evaluated"] == 2


def test_build_diagnosis_compat_arrays_and_alerts():
    diagnosis = diagnose.build_diagnosis(
        pilot_rows(),
        _config(),
        rows_path="rows.jsonl",
        rows_hash="sha256:" + "a" * 64,
        config_path="config.yaml",
        config_hash="sha256:" + "b" * 64,
        source="pilot replay",
    )
    assert diagnosis["kind"] == "recipe-lab-diagnosis-v1"
    assert diagnosis["inputs"]["samples"] == 344
    assert diagnosis["dead_decisions"] == ["domain_code_complex", "complex_general"]
    assert diagnosis["rare_decisions"] == []
    assert len(diagnosis["unreachable_sites"]) == 4
    for site in diagnosis["unreachable_sites"]:
        assert set(site) == {"decision", "signal", "op", "threshold"}
        assert site["threshold"] == 2.5
    types = [alert["type"] for alert in diagnosis["alerts"]]
    assert types.count("dead_decision") == 2
    assert types.count("compressed_range") >= 1  # difficulty uses under half its scale
    assert types.count("unreachable_threshold") == 1
    assert "small_sample" not in types  # 344 samples


def test_small_sample_warns():
    diagnosis = diagnose.build_diagnosis(pilot_rows()[:10], _config(), min_sample=100)
    assert diagnosis["alerts"][-1]["type"] == "small_sample"
    assert diagnosis["alerts"][-1]["samples"] == 10


def test_byte_stable_output(tmp_path):
    rows = pilot_rows()
    config = _config()
    first = diagnose.build_diagnosis(
        rows, config, rows_path="rows.jsonl", config_path="config.yaml"
    )
    second = diagnose.build_diagnosis(
        rows, config, rows_path="rows.jsonl", config_path="config.yaml"
    )
    first_json, first_md = diagnose.write_diagnosis(first, tmp_path / "one")
    second_json, second_md = diagnose.write_diagnosis(second, tmp_path / "two")
    assert first_json.read_bytes() == second_json.read_bytes()
    assert first_md.read_bytes() == second_md.read_bytes()
    payload = json.loads(first_json.read_text(encoding="utf-8"))
    assert payload["kind"] == "recipe-lab-diagnosis-v1"
    assert "# Recipe diagnosis report" in first_md.read_text(encoding="utf-8")


def test_load_diagnosis_kind_checked(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"kind": "something-else"}), encoding="utf-8")
    with pytest.raises(ContractError, match="recipe-lab-diagnosis-v1"):
        diagnose.load_diagnosis(path)


def test_load_diagnostic_rows_rejects_pre_adapter_snapshots(tmp_path):
    path = tmp_path / "old-rows.jsonl"
    path.write_text(json.dumps({"request_id": "r0"}) + "\n", encoding="utf-8")
    with pytest.raises(ContractError, match="signal_values"):
        diagnose.load_diagnostic_rows(path)


def test_load_diagnostic_rows_rejects_bad_json(tmp_path):
    path = tmp_path / "rows.jsonl"
    path.write_text("{not json}\n", encoding="utf-8")
    with pytest.raises(ContractError, match="line 1"):
        diagnose.load_diagnostic_rows(path)


def test_cli_diagnose_writes_artifacts(tmp_path, capsys):
    from recipe_lab.cli import main

    rows_path = _write_rows(tmp_path)
    diag_dir = tmp_path / "diag"
    code = main(
        [
            "diagnose",
            "--records",
            str(rows_path),
            "--config",
            str(AGENT_RECIPE_MIN),
            "--output-dir",
            str(diag_dir),
            "--source",
            "unit rows",
        ]
    )
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["dead_decisions"] == ["domain_code_complex", "complex_general"]
    diagnosis = diagnose.load_diagnosis(payload["diagnosis"])
    assert diagnosis["inputs"]["rows_hash"] == dataset_content_hash(rows_path)
    assert diagnosis["inputs"]["config_hash"] == dataset_content_hash(AGENT_RECIPE_MIN)
    assert diagnosis["inputs"]["source"] == "unit rows"
    assert diagnosis["inputs"]["recipe"] == "default"
    assert "- recipe: `default`" in (diag_dir / "diagnosis.md").read_text(
        encoding="utf-8"
    )
    assert payload["recipe"] == "default"
    assert (diag_dir / "diagnosis.md").is_file()


def test_cli_diagnose_names_the_recipe_of_a_multi_recipe_document(tmp_path, capsys):
    # one recipe per public model: without a selection the diagnosis is
    # ambiguous, and either selector names the recipe to diagnose
    from recipe_lab.cli import main

    rows_path = _write_rows(tmp_path)
    base = _config()["recipes"][0]
    config = {
        "recipes": [
            {**base, "name": "balanced"},
            {**base, "name": "speed-first", "decisions": base["decisions"][:1]},
        ],
        "entrypoints": [
            {"model_names": ["vllm-sr/mom-blend"], "recipe": "balanced"},
            {"model_names": ["vllm-sr/mom-flash"], "recipe": "speed-first"},
        ],
    }
    config_path = tmp_path / "two.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    common = ["--records", str(rows_path), "--config", str(config_path)]

    def run(*selector: str) -> tuple[int, dict]:
        out = tmp_path / "diag" / ("-".join(selector) or "none")
        code = main(["diagnose", *common, "--output-dir", str(out), *selector])
        captured = capsys.readouterr()
        return code, {"out": captured.out, "err": captured.err, "dir": out}

    code, first = run()
    assert code == 2
    assert "pass --recipe" in first["err"]
    assert not first["dir"].exists()

    code, by_name = run("--recipe", "speed-first")
    assert code == 0
    assert json.loads(by_name["out"])["recipe"] == "speed-first"
    diagnosis = diagnose.load_diagnosis(by_name["dir"] / "diagnosis.json")
    assert diagnosis["inputs"]["recipe"] == "speed-first"
    assert diagnosis["decision_coverage"]["total_decisions"] == 1

    code, by_model = run("--public-model", "vllm-sr/mom-blend")
    assert code == 0
    diagnosis = diagnose.load_diagnosis(by_model["dir"] / "diagnosis.json")
    assert diagnosis["inputs"]["recipe"] == "balanced"
    assert diagnosis["decision_coverage"]["total_decisions"] == 6


def test_cli_diagnose_missing_records_exits_2(tmp_path):
    from recipe_lab.cli import main

    code = main(
        [
            "diagnose",
            "--records",
            str(tmp_path / "missing.jsonl"),
            "--config",
            str(AGENT_RECIPE_MIN),
            "--output-dir",
            str(tmp_path / "diag"),
        ]
    )
    assert code == 2

"""Public CLI discovery and result persistence contracts."""

import json

import pytest


def test_builtin_scenarios_listed():
    from tuning.cli import BUILTIN_SCENARIOS

    assert "privacy" in BUILTIN_SCENARIOS
    assert "calibration" in BUILTIN_SCENARIOS


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("privacy", "privacy_routing_tuning"),
        ("calibration", "calibration_tuning"),
        ("tuning.scenarios.privacy:PrivacyScenario", "privacy_routing_tuning"),
    ],
)
def test_load_scenario(name, expected):
    from tuning.cli import _load_scenario

    assert _load_scenario(name).name == expected


def test_unknown_scenario_exits():
    from tuning.cli import _load_scenario

    with pytest.raises(SystemExit):
        _load_scenario("nonexistent_scenario")


def test_candidate_config_is_the_only_runtime_tuning_output():
    from tuning.cli import build_parser

    args = build_parser().parse_args(
        [
            "privacy",
            "--config",
            "config.yaml",
            "--probes",
            "probes.yaml",
            "--candidate-config",
            "candidate.yaml",
        ]
    )
    assert args.candidate_config == "candidate.yaml"
    assert not hasattr(args, "router_pid")
    assert not hasattr(args, "deploy_config")


def test_public_package_exposes_candidate_tuner_only():
    import tuning

    assert "CandidateTuner" in tuning.__all__
    assert "TuningLoop" not in tuning.__all__
    assert not hasattr(tuning.RouterClient(), "get_config_hash")
    assert not hasattr(tuning.RouterClient(), "hot_reload")


def test_candidate_tuner_rejects_active_manifest_overwrite(tmp_path):
    from tuning.client import RouterClient
    from tuning.scenario import CandidateTuner
    from tuning.scenarios.privacy import PrivacyScenario

    config_path = tmp_path / "config.yaml"
    with pytest.raises(ValueError, match="must not overwrite"):
        CandidateTuner(
            scenario=PrivacyScenario(),
            router=RouterClient(),
            config_path=config_path,
            probes_path=tmp_path / "probes.yaml",
            candidate_path=config_path,
        )


def test_save_results_creates_file(tmp_path):
    from tuning.probes import save_results

    path = save_results({"scenario": "test", "accuracy": 0.95}, "test.json", tmp_path)
    assert path.exists()
    assert json.loads(path.read_text())["scenario"] == "test"


def test_save_results_creates_directory(tmp_path):
    from tuning.probes import save_results

    output_dir = tmp_path / "nested" / "results"
    assert save_results({"ok": True}, "out.json", output_dir).exists()
    assert output_dir.is_dir()

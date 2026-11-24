"""C9 bundle tests: packaging is a final-verify-only, byte-stable operation.

Covers the acceptance flow end to end through the CLI: a final live PASS
record with passing transfer guards bundles into the nine artifacts; byte
stability holds across repeated runs; and every rejection path (not final,
not PASS, transfer guard not passing, unknown record, failing config
validate, private content) exits 2 without writing a bundle.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest
import yaml
from recipe_lab.classification import classification_projection_hash
from recipe_lab.cli import main
from recipe_lab.contract import API_VERSION
from recipe_lab.ledger import Ledger
from recipe_lab.report import render_promotion_draft

EXIT_CONTRACT_ERROR = 2

FROZEN_HASH = "sha256:" + "1" * 64
SMALL = "mock/small"
LARGE = "mock/large"

CONFIG_YAML = """\
providers:
  vllm:
    endpoints: ["http://127.0.0.1:8001/v1"]
routing:
  default_model: mock/small
"""
SKELETON_YAML = """\
providers:
  vllm:
    endpoints: ["http://127.0.0.1:8001/v1"]
"""
PROFILE_JSON = '{"schema": "test-profile", "n": 3}\n'


def _sha256(text: str) -> str:
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()


# Y4: the packaged inputs are bound to the evidence - the ledger's candidate
# names the config's sha256, and the bootstrap sidecar cites the
# skeleton/profile pair the candidate was generated with.
CONFIG_SHA256 = _sha256(CONFIG_YAML)
SKELETON_SHA256 = _sha256(SKELETON_YAML)
PROFILE_SHA256 = _sha256(PROFILE_JSON)
# the Y2 classification source's content identity (a fake hash for fixtures)
CLASSIFICATION_SOURCE_SHA256 = "sha256:" + "c" * 64


def _write_spec(tmp_path: Path) -> str:
    spec = {
        "api_version": API_VERSION,
        "objective": {
            "metric_key": "capability.score",
            "direction": "maximize",
            "min_improvement_pp": 1.0,
            "constraints": [
                {
                    "name": "cost-cap",
                    "metric": "cost_per_question",
                    "op": "lte",
                    "bound": 2e-5,
                }
            ],
        },
        "gates": [
            {"name": name, "metric": name, "floor": 0.95}
            for name in ("privacy", "no_pii")
        ],
        "workloads": [
            {
                "name": "holdout-tune",
                "profile": "quick",
                "splits": {
                    "calib": {
                        "manifest": "calib.jsonl",
                        "content_hash": "sha256:" + "0" * 64,
                    }
                },
            },
            {
                "name": "holdout-transfer",
                "profile": "quick",
                "role": "transfer",
                "splits": {
                    "frozen_test": {
                        "manifest": "frozen_test.jsonl",
                        "content_hash": FROZEN_HASH,
                    }
                },
            },
        ],
        "attribution": {"oracle_policy": {"math": LARGE}},
        "budget": {"rounds": 1, "max_minutes_per_trial": 5},
        "models": [
            {"name": SMALL, "endpoint": "http://127.0.0.1:8001/v1"},
            {"name": LARGE, "endpoint": "http://127.0.0.1:8002/v1"},
        ],
        "cost_basis": {
            "source": "provider_pricing",
            "assumed_pricing": {
                name: {"prompt_per_1m": 0.1, "completion_per_1m": 0.4}
                for name in (SMALL, LARGE)
            },
        },
    }
    spec_path = tmp_path / "spec.yaml"
    spec_path.write_text(yaml.safe_dump(spec), encoding="utf-8")
    return str(spec_path)


def _pass_record(**overrides) -> dict:
    record = {
        "round": 3,
        "ts": "2026-01-01T00:00:00Z",
        "candidate": {
            "kind": "config",
            "ref": "candidate-lambda-0.30.yaml",
            "change_summary": "bootstrap lambda 0.30",
            "provenance": {"proposer": "bootstrap", "insight_refs": []},
            "config_sha256": CONFIG_SHA256,
        },
        "final": True,
        "tier": "live",
        "baseline_ref": "run-base-1",
        "metrics": {"capability.score": 0.83},
        "verdict": "PASS",
        "dataset_hash": FROZEN_HASH,
        "reasons": [
            "paired CI lower bound 1.20pp >= min 1.00pp (200 paired cases, delta 2.00pp)"
        ],
        "gates": {"privacy": "pass", "no_pii": "pass"},
        "verification": {
            "baseline": 0.81,
            "guard_measurements": [
                {"name": name, "candidate": 0.98} for name in ("privacy", "no_pii")
            ],
        },
        "bench_refs": ["run-base-1", "run-cand-1"],
        "constraints": {
            "cost-cap": {
                "name": "cost-cap",
                "metric": "cost_per_question",
                "op": "lte",
                "bound": 2e-5,
                "candidate": 1.2e-05,
                "bound_value": 2.0e-05,
                "status": "pass",
            }
        },
        "paired": {
            "paired_cases": 200,
            "quality_delta": 0.02,
            "quality_delta_ci95": [0.012, 0.03],
            "quality_delta_bootstrap_ci95": [0.011, 0.029],
            "wins": 120,
            "losses": 60,
            "ties": 20,
        },
        "workload": "holdout-transfer",
        "split": "frozen_test",
        "notes": (
            "baseline=0.81; repetitions=3; routing_distribution="
            '{"mock/small": 120, "mock/large": 80}'
        ),
    }
    record.update(overrides)
    return record


def _write_ledger(tmp_path: Path, final_record: dict) -> tuple[str, str]:
    """A replay record outside the campaign hash plus the final PASS record."""
    ledger_path = tmp_path / "ledger.jsonl"
    ledger = Ledger(ledger_path)
    ledger.append(
        {
            "round": 2,
            "ts": "2025-12-31T00:00:00Z",
            "candidate": {
                "kind": "config",
                "ref": "candidate-lambda-0.10.yaml",
                "change_summary": "",
                "provenance": {"proposer": "bootstrap", "insight_refs": []},
            },
            "tier": "replay",
            "baseline_ref": "run-base-0",
            "metrics": {"routed_accuracy": 0.8},
            "verdict": "RANKED",
            "dataset_hash": "sha256:" + "2" * 64,
            "reasons": ["replay ranking only"],
        }
    )
    appended = ledger.append(final_record)
    return str(ledger_path), appended["record_id"]


def _validator(tmp_path: Path, *, success: bool) -> Path:
    name = "fake-vllm-sr" if success else "bad-vllm-sr"
    message = "config ok" if success else "bad config"
    if os.name == "nt":
        path = tmp_path / (name + ".cmd")
        path.write_text(f"@echo off\necho {message}\nexit /b {int(not success)}\n")
    else:
        path = tmp_path / name
        path.write_text(f"#!/bin/sh\necho '{message}'\nexit {int(not success)}\n")
        path.chmod(0o755)
    return path


def _write_inputs(tmp_path: Path) -> dict:
    config = tmp_path / "candidate-lambda-0.30.yaml"
    config.write_bytes(CONFIG_YAML.encode())
    skeleton = tmp_path / "scaffold-lambda-0.30.yaml"
    skeleton.write_bytes(SKELETON_YAML.encode())
    profile = tmp_path / "profile.json"
    profile.write_bytes(PROFILE_JSON.encode())
    # the bootstrap provenance sidecar: cites the skeleton/profile pair plus
    # the Y2 classification identity of the evaluated candidate
    sidecar = tmp_path / "candidate-lambda-0.30.provenance.json"
    sidecar.write_text(
        json.dumps(
            {
                "schema": "recipe-lab-bootstrap-v1",
                "proposer": "bootstrap",
                "scaffold": {
                    "path": str(skeleton),
                    "sha256": SKELETON_SHA256,
                },
                "profile": {"path": str(profile), "sha256": PROFILE_SHA256},
                "classification": {
                    "path": "config/config.yaml",
                    "sha256": CLASSIFICATION_SOURCE_SHA256,
                    "projection_sha256": classification_projection_hash(
                        yaml.safe_load(CONFIG_YAML)
                    ),
                },
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    validate_bin = _validator(tmp_path, success=True)
    return {
        "config": str(config),
        "skeleton": str(skeleton),
        "profile": str(profile),
        "sidecar": str(sidecar),
        "validate_bin": str(validate_bin),
    }


def _bundle_args(
    tmp_path: Path,
    spec_path: str,
    ledger_path: str,
    record_id: str,
    inputs: dict,
    out: Path,
    extra: list[str] | None = None,
) -> list[str]:
    return [
        "bundle",
        "--spec",
        spec_path,
        "--ledger",
        ledger_path,
        "--record",
        record_id,
        "--out",
        str(out),
        "--config",
        inputs["config"],
        "--skeleton",
        inputs["skeleton"],
        "--profile",
        inputs["profile"],
        "--candidate-sidecar",
        inputs["sidecar"],
        "--validate-bin",
        inputs["validate_bin"],
        "--router-commit",
        "abc1234",
        "--model-revision",
        f"{SMALL}=rev-a",
        "--model-eval-script",
        inputs["skeleton"],
    ] + (extra or [])


BUNDLE_FILES = {
    "config.yaml",
    "validate.txt",
    "config.diff",
    "report.md",
    "profile.json",
    "ledger.jsonl",
    "provenance.json",
    "PROMOTION.md",
    "REVIEW.md",
}


@pytest.mark.parametrize("candidate_hash", [CONFIG_SHA256, None])
def test_template_bundle_uses_bindings_without_skeleton(
    tmp_path, monkeypatch, candidate_hash
):
    from recipe_lab import bundle

    spec_path = _write_spec(tmp_path)
    ledger_path, record_id = _write_ledger(tmp_path, _pass_record())
    config = tmp_path / "candidate-lambda-0.yaml"
    config.write_bytes(CONFIG_YAML.encode())
    bindings = tmp_path / "bindings-lambda-0.yaml"
    bindings.write_bytes(b"fast: [{model: mock/small}]\n")
    profile = tmp_path / "profile.json"
    profile.write_bytes(PROFILE_JSON.encode())
    sidecar = tmp_path / "bindings-lambda-0.provenance.json"
    sidecar.write_text(
        json.dumps(
            {
                "schema": "recipe-lab-bindings-v1",
                "candidate": {"sha256": candidate_hash} if candidate_hash else {},
                "bindings": {"sha256": _sha256(bindings.read_text(encoding="utf-8"))},
                "profile": {"sha256": PROFILE_SHA256},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(bundle, "_run_config_validate", lambda *args: "config ok\n")
    out = tmp_path / "bundle"
    assert main(
        [
            "bundle",
            "--spec",
            spec_path,
            "--ledger",
            ledger_path,
            "--record",
            record_id,
            "--out",
            str(out),
            "--config",
            str(config),
            "--bindings",
            str(bindings),
            "--profile",
            str(profile),
            "--candidate-sidecar",
            str(sidecar),
        ]
    ) == (0 if candidate_hash else EXIT_CONTRACT_ERROR)
    if not candidate_hash:
        assert not out.exists()
        return
    assert (out / "config.yaml").read_bytes() == config.read_bytes()
    assert (out / "bindings.yaml").read_bytes() == bindings.read_bytes()


def test_validation_renders_only_the_known_input_path(tmp_path, monkeypatch):
    import subprocess

    from recipe_lab import bundle

    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    path = tmp_path / "experiment" / "candidate.yaml"
    prefix = str(path)[:-7]
    other = str(tmp_path / "private-diagnostic")
    output = (
        f"  Path       {prefix}\n             {str(path)[-7:]}\n  Path       {other}\n"
    )
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *a, **k: subprocess.CompletedProcess([], 0, output, ""),
    )
    text = bundle._run_config_validate(path, "validator")
    assert prefix not in text
    assert str(Path("~") / "experiment") in text
    assert other in text  # the bundle scanner still sees unrelated private diagnostics


def test_bundle_happy_path(tmp_path, capsys):
    spec_path = _write_spec(tmp_path)
    ledger_path, record_id = _write_ledger(tmp_path, _pass_record())
    inputs = _write_inputs(tmp_path)
    out = tmp_path / "bundle"
    rc = main(_bundle_args(tmp_path, spec_path, ledger_path, record_id, inputs, out))
    assert rc == 0
    assert {p.name for p in out.iterdir()} == BUNDLE_FILES

    # config.yaml and profile.json are byte-exact copies
    assert (out / "config.yaml").read_bytes() == Path(inputs["config"]).read_bytes()
    assert (out / "profile.json").read_text(encoding="utf-8") == PROFILE_JSON
    # validate.txt carries the validator output
    assert (out / "validate.txt").read_text(encoding="utf-8") == "config ok\n"
    # config.diff is a unified diff against the skeleton
    diff_text = (out / "config.diff").read_text(encoding="utf-8")
    assert diff_text.startswith("--- skeleton")
    assert "+  default_model: mock/small" in diff_text

    ledger_record = Ledger(ledger_path).read()[-1]
    # PROMOTION.md is the record-draft output for this record
    assert (out / "PROMOTION.md").read_text(encoding="utf-8") == render_promotion_draft(
        ledger_record
    )
    # REVIEW.md is the V5 template prefilled with the record identity
    review = (out / "REVIEW.md").read_text(encoding="utf-8")
    assert f"record: {record_id}" in review
    assert "decision (promote / hold / rework):" in review

    # ledger.jsonl is the campaign excerpt: only records sharing the dataset hash
    excerpt = [
        json.loads(line)
        for line in (out / "ledger.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert [row["record_id"] for row in excerpt] == [record_id]

    provenance = json.loads((out / "provenance.json").read_text(encoding="utf-8"))
    assert provenance["kind"] == "recipe-lab-bundle"
    assert provenance["record_id"] == record_id
    assert provenance["router_commit"] == "abc1234"
    assert provenance["models"] == {SMALL: "rev-a", LARGE: None}
    assert provenance["artifacts"]["config"]["sha256"] == (
        "sha256:" + hashlib.sha256(Path(inputs["config"]).read_bytes()).hexdigest()
    )
    assert provenance["run_ids"]["bench_refs"] == ["run-base-1", "run-cand-1"]
    assert (
        provenance["datasets"]["workloads"]["holdout-transfer"]["frozen_test"]
        == FROZEN_HASH
    )
    # Y2: the provenance carries the classification binding the bundle verified
    assert provenance["classification"]["projection_sha256"] == (
        classification_projection_hash(yaml.safe_load(CONFIG_YAML))
    )
    assert provenance["classification"]["source_sha256"] == CLASSIFICATION_SOURCE_SHA256
    provenance_text = (out / "provenance.json").read_text(encoding="utf-8")
    # provenance cites basenames and hashes only - no absolute paths leak in
    assert str(Path(ledger_path).parent) not in provenance_text

    report = (out / "report.md").read_text(encoding="utf-8")
    assert "# recipe-lab acceptance bundle report" in report
    assert f"`{record_id}`" in report
    assert "paired cases: 200" in report
    assert "quality delta CI95: [0.012, 0.03]" in report
    assert "- mock/small: 120" in report  # observed routing distribution
    assert "## Cache and tools" in report
    assert "assumed pricing" in report
    assert "verdict: **PASS**" in report
    assert "holdout-transfer: 1x PASS" in report


def test_bundle_is_byte_stable(tmp_path):
    spec_path = _write_spec(tmp_path)
    ledger_path, record_id = _write_ledger(tmp_path, _pass_record())
    inputs = _write_inputs(tmp_path)
    out1 = tmp_path / "bundle-1"
    out2 = tmp_path / "bundle-2"
    assert (
        main(_bundle_args(tmp_path, spec_path, ledger_path, record_id, inputs, out1))
        == 0
    )
    assert (
        main(_bundle_args(tmp_path, spec_path, ledger_path, record_id, inputs, out2))
        == 0
    )
    files1 = {p.name: p.read_bytes() for p in out1.iterdir()}
    files2 = {p.name: p.read_bytes() for p in out2.iterdir()}
    assert set(files1) == BUNDLE_FILES
    assert files1 == files2


@pytest.mark.parametrize("gate_value", ["fail", "incomplete"])
def test_bundle_rejects_guard_not_passing(tmp_path, capsys, gate_value):
    spec_path = _write_spec(tmp_path)
    ledger_path, record_id = _write_ledger(
        tmp_path, _pass_record(gates={"privacy": gate_value})
    )
    inputs = _write_inputs(tmp_path)
    out = tmp_path / "bundle"
    rc = main(_bundle_args(tmp_path, spec_path, ledger_path, record_id, inputs, out))
    assert rc == EXIT_CONTRACT_ERROR
    assert "transfer guards not passing" in capsys.readouterr().err
    assert not out.exists()


def test_bundle_rejects_non_final_record(tmp_path, capsys):
    spec_path = _write_spec(tmp_path)
    record = _pass_record()
    record.pop("final")
    ledger_path, record_id = _write_ledger(tmp_path, record)
    inputs = _write_inputs(tmp_path)
    out = tmp_path / "bundle"
    rc = main(_bundle_args(tmp_path, spec_path, ledger_path, record_id, inputs, out))
    assert rc == EXIT_CONTRACT_ERROR
    assert "carries no final flag" in capsys.readouterr().err
    assert not out.exists()


def test_bundle_rejects_unnamed_config(tmp_path, capsys):
    # Y4: the packaged config must be the config the verdict graded - the
    # ledger's candidate names its sha256
    spec_path = _write_spec(tmp_path)
    ledger_path, record_id = _write_ledger(tmp_path, _pass_record())
    inputs = _write_inputs(tmp_path)
    tweaked = tmp_path / "tweaked.yaml"
    tweaked.write_text(
        CONFIG_YAML + "  # tweaked after the verdict\n", encoding="utf-8"
    )
    inputs["config"] = str(tweaked)
    out = tmp_path / "bundle"
    rc = main(_bundle_args(tmp_path, spec_path, ledger_path, record_id, inputs, out))
    assert rc == EXIT_CONTRACT_ERROR
    err = capsys.readouterr().err
    assert "hashes to" in err and "candidate names" in err
    assert not out.exists()


def test_bundle_rejects_record_without_config_hash(tmp_path, capsys):
    # Y4: a record whose candidate names no config cannot be packaged
    spec_path = _write_spec(tmp_path)
    record = _pass_record()
    del record["candidate"]["config_sha256"]
    ledger_path, record_id = _write_ledger(tmp_path, record)
    inputs = _write_inputs(tmp_path)
    out = tmp_path / "bundle"
    rc = main(_bundle_args(tmp_path, spec_path, ledger_path, record_id, inputs, out))
    assert rc == EXIT_CONTRACT_ERROR
    assert "carries no config_sha256" in capsys.readouterr().err
    assert not out.exists()


def test_bundle_rejects_skeleton_not_matching_sidecar(tmp_path, capsys):
    # Y4: the skeleton must be the scaffold the bootstrap sidecar cites
    spec_path = _write_spec(tmp_path)
    ledger_path, record_id = _write_ledger(tmp_path, _pass_record())
    inputs = _write_inputs(tmp_path)
    other_skeleton = tmp_path / "other-scaffold.yaml"
    other_skeleton.write_text(
        SKELETON_YAML + "routing:\n  default_model: mock/large\n", encoding="utf-8"
    )
    inputs["skeleton"] = str(other_skeleton)
    out = tmp_path / "bundle"
    rc = main(_bundle_args(tmp_path, spec_path, ledger_path, record_id, inputs, out))
    assert rc == EXIT_CONTRACT_ERROR
    err = capsys.readouterr().err
    assert "--skeleton hashes to" in err and "sidecar cites" in err
    assert not out.exists()


def test_bundle_rejects_profile_not_matching_sidecar(tmp_path, capsys):
    # Y4: the profile must be the C3 profile the bootstrap sidecar cites
    spec_path = _write_spec(tmp_path)
    ledger_path, record_id = _write_ledger(tmp_path, _pass_record())
    inputs = _write_inputs(tmp_path)
    other_profile = tmp_path / "other-profile.json"
    other_profile.write_text('{"schema": "test-profile", "n": 4}\n', encoding="utf-8")
    inputs["profile"] = str(other_profile)
    out = tmp_path / "bundle"
    rc = main(_bundle_args(tmp_path, spec_path, ledger_path, record_id, inputs, out))
    assert rc == EXIT_CONTRACT_ERROR
    err = capsys.readouterr().err
    assert "--profile hashes to" in err and "sidecar cites" in err
    assert not out.exists()


def test_bundle_rejects_sidecar_without_classification_projection(tmp_path, capsys):
    # Y2: a pre-Y2 sidecar (no classification block) cannot prove the packaged
    # config classifies like the evaluated candidate
    spec_path = _write_spec(tmp_path)
    ledger_path, record_id = _write_ledger(tmp_path, _pass_record())
    inputs = _write_inputs(tmp_path)
    sidecar = json.loads(Path(inputs["sidecar"]).read_text(encoding="utf-8"))
    del sidecar["classification"]
    Path(inputs["sidecar"]).write_text(
        json.dumps(sidecar, sort_keys=True) + "\n", encoding="utf-8"
    )
    out = tmp_path / "bundle"
    rc = main(_bundle_args(tmp_path, spec_path, ledger_path, record_id, inputs, out))
    assert rc == EXIT_CONTRACT_ERROR
    assert "no classification projection_sha256" in capsys.readouterr().err
    assert not out.exists()


def test_bundle_rejects_config_with_different_classification(tmp_path, capsys):
    # Y2: the packaged config must classify like the evaluated candidate -
    # a swapped embeddings section fails even with a matching config hash
    spec_path = _write_spec(tmp_path)
    swapped_text = CONFIG_YAML + (
        "global:\n"
        "  model_catalog:\n"
        "    embeddings:\n"
        "      semantic:\n"
        "        embedding_config:\n"
        "          model_type: swapped-model\n"
    )
    record = _pass_record()
    record["candidate"]["config_sha256"] = _sha256(swapped_text)
    ledger_path, record_id = _write_ledger(tmp_path, record)
    inputs = _write_inputs(tmp_path)
    swapped = tmp_path / "swapped.yaml"
    swapped.write_bytes(swapped_text.encode())
    inputs["config"] = str(swapped)
    out = tmp_path / "bundle"
    rc = main(_bundle_args(tmp_path, spec_path, ledger_path, record_id, inputs, out))
    assert rc == EXIT_CONTRACT_ERROR
    err = capsys.readouterr().err
    assert "classification projection hashes to" in err and "sidecar cites" in err
    assert not out.exists()


def test_bundle_rejects_replay_record(tmp_path, capsys):
    spec_path = _write_spec(tmp_path)
    ledger_path, record_id = _write_ledger(tmp_path, _pass_record())
    replay_id = Ledger(ledger_path).read()[0]["record_id"]
    assert replay_id != record_id
    inputs = _write_inputs(tmp_path)
    out = tmp_path / "bundle"
    rc = main(_bundle_args(tmp_path, spec_path, ledger_path, replay_id, inputs, out))
    assert rc == EXIT_CONTRACT_ERROR
    assert "not a live-tier record" in capsys.readouterr().err
    assert not out.exists()


def test_bundle_rejects_fail_verdict(tmp_path, capsys):
    spec_path = _write_spec(tmp_path)
    ledger_path, _ = _write_ledger(tmp_path, _pass_record(verdict="FAIL"))
    record_id = Ledger(ledger_path).read()[-1]["record_id"]
    inputs = _write_inputs(tmp_path)
    out = tmp_path / "bundle"
    rc = main(_bundle_args(tmp_path, spec_path, ledger_path, record_id, inputs, out))
    assert rc == EXIT_CONTRACT_ERROR
    assert "only PASS records" in capsys.readouterr().err
    assert not out.exists()


def test_bundle_rejects_unknown_record(tmp_path, capsys):
    spec_path = _write_spec(tmp_path)
    ledger_path, _ = _write_ledger(tmp_path, _pass_record())
    inputs = _write_inputs(tmp_path)
    out = tmp_path / "bundle"
    rc = main(
        _bundle_args(tmp_path, spec_path, ledger_path, "rec-000000000000", inputs, out)
    )
    assert rc == EXIT_CONTRACT_ERROR
    assert "not found" in capsys.readouterr().err
    assert not out.exists()


def test_bundle_rejects_failing_validate(tmp_path, capsys):
    spec_path = _write_spec(tmp_path)
    ledger_path, record_id = _write_ledger(tmp_path, _pass_record())
    inputs = _write_inputs(tmp_path)
    bad_bin = _validator(tmp_path, success=False)
    inputs["validate_bin"] = str(bad_bin)
    out = tmp_path / "bundle"
    rc = main(_bundle_args(tmp_path, spec_path, ledger_path, record_id, inputs, out))
    assert rc == EXIT_CONTRACT_ERROR
    assert "config validate" in capsys.readouterr().err
    assert not out.exists()


def test_bundle_rejects_private_path_in_config(tmp_path, capsys):
    spec_path = _write_spec(tmp_path)
    # hash binding only proves the config is the one that was evaluated; the
    # private scan still has to catch a leaked candidate that carries it
    leaked_text = CONFIG_YAML + "  # reviewed against C:\\Users\\alice\\notes.md\n"
    record = _pass_record()
    record["candidate"]["config_sha256"] = _sha256(leaked_text)
    ledger_path, record_id = _write_ledger(tmp_path, record)
    inputs = _write_inputs(tmp_path)
    leaked = tmp_path / "leaked.yaml"
    leaked.write_bytes(leaked_text.encode())
    inputs["config"] = str(leaked)
    out = tmp_path / "bundle"
    rc = main(_bundle_args(tmp_path, spec_path, ledger_path, record_id, inputs, out))
    assert rc == EXIT_CONTRACT_ERROR
    err = capsys.readouterr().err
    assert "private content" in err
    assert "windows-user-path" in err
    assert "alice" not in err  # the matched content is never echoed
    assert not out.exists()


def test_bundle_rejects_custom_private_pattern(tmp_path, capsys):
    spec_path = _write_spec(tmp_path)
    leaked_text = CONFIG_YAML + "  # endpoint vllm-internal.example:8000\n"
    record = _pass_record()
    record["candidate"]["config_sha256"] = _sha256(leaked_text)
    ledger_path, record_id = _write_ledger(tmp_path, record)
    inputs = _write_inputs(tmp_path)
    leaked = tmp_path / "leaked.yaml"
    leaked.write_bytes(leaked_text.encode())
    inputs["config"] = str(leaked)
    out = tmp_path / "bundle"
    rc = main(
        _bundle_args(
            tmp_path,
            spec_path,
            ledger_path,
            record_id,
            inputs,
            out,
            extra=["--private-pattern", "vllm-internal"],
        )
    )
    assert rc == EXIT_CONTRACT_ERROR
    err = capsys.readouterr().err
    assert "private content" in err
    assert "custom[0]" in err
    assert not out.exists()


@pytest.mark.parametrize(
    "defect",
    [
        "wrong_dataset",
        "development_split",
        "missing_guard",
        "failing_guard",
        "missing_constraint",
        "cost_exceeds_limit",
        "wrong_bound",
        "objective",
        "small_sample",
        "tampered_record",
        "private_endpoint",
    ],
)
def test_bundle_replays_promotion_evidence(tmp_path, capsys, defect):
    spec_path = _write_spec(tmp_path)
    record = _pass_record()
    extra = []
    if defect == "wrong_dataset":
        record["dataset_hash"] = "sha256:" + "e" * 64
    elif defect == "development_split":
        record.update(
            workload="holdout-tune", split="calib", dataset_hash="sha256:" + "0" * 64
        )
    elif defect == "missing_guard":
        record["gates"].clear()
    elif defect == "failing_guard":
        record["verification"]["guard_measurements"][0]["candidate"] = 0.2
    elif defect == "missing_constraint":
        record["constraints"].clear()
    elif defect == "cost_exceeds_limit":
        record["constraints"]["cost-cap"]["candidate"] = 0.1
    elif defect == "wrong_bound":
        record["constraints"]["cost-cap"]["bound_value"] = 0.2
    elif defect == "objective":
        record["paired"]["quality_delta_ci95"] = [-0.04, -0.02]
    elif defect == "small_sample":
        record["paired"]["paired_cases"] = 1
    elif defect == "private_endpoint":
        extra = ["--model-revision", f"{LARGE}=endpoint 192.0.2.5"]
    ledger_path, rec_id = _write_ledger(tmp_path, record)
    if defect == "tampered_record":
        path = Path(ledger_path)
        path.write_text(path.read_text().replace("run-cand-1", "run-cand-2"))
    out = tmp_path / "bundle"
    assert (
        main(
            _bundle_args(
                tmp_path,
                spec_path,
                ledger_path,
                rec_id,
                _write_inputs(tmp_path),
                out,
                extra,
            )
        )
        == EXIT_CONTRACT_ERROR
    )
    assert not out.exists()
    if defect == "private_endpoint":
        assert "192.0.2.5" not in capsys.readouterr().err

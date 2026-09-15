"""Installed Recipe discovery and explicit binding need no live Router."""

import json
from pathlib import Path

import pytest
import yaml
from cli.builtin_recipes import list_builtin_recipes
from cli.commands.runtime_kb import _resolve_kb_source_root
from cli.main import main as cli
from cli.model_bundle import model_bundle_digest
from click.testing import CliRunner


def _inputs(tmp_path: Path):
    providers = {
        "version": "v0.3",
        "listeners": [{"name": "http", "address": "0.0.0.0", "port": 8899}],
        "providers": {
            "defaults": {"model": "model-a"},
            "models": [
                {
                    "name": name,
                    "backend_refs": [
                        {
                            "name": f"primary-{name}",
                            "provider": "vllm",
                            "endpoint": f"localhost:{port}",
                            "protocol": "http",
                        }
                    ],
                }
                for name, port in (("model-a", 8000), ("model-b", 8001))
            ],
        },
        "routing": {"modelCards": [{"name": "model-a"}, {"name": "model-b"}]},
    }
    bundle = list_builtin_recipes()["bundles"][0]
    balance = next(item for item in bundle["recipes"] if item["name"] == "balance")
    bindings = {
        item["name"]: [{"model": "model-a"}, {"model": "model-b"}][
            : item["minimum_candidates"]
        ]
        for item in balance["decisions"]
    }
    source, binding_file = tmp_path / "providers.yaml", tmp_path / "bindings.yaml"
    source.write_text(yaml.safe_dump(providers, sort_keys=False))
    binding_file.write_text(yaml.safe_dump(bindings))
    return source, binding_file, bindings


def _init(source, bindings, output, extra=()):
    return CliRunner().invoke(
        cli,
        [
            "recipe",
            "builtin",
            "init",
            "balance",
            "--bundle",
            "mom-v1",
            "--config",
            str(source),
            "--bindings",
            str(bindings),
            "--model-name",
            "my/balance",
            "--output",
            str(output),
            *extra,
        ],
    )


def test_offline_list_and_export_preserve_verified_bundle(tmp_path):
    runner = CliRunner()
    result = runner.invoke(cli, ["recipe", "builtin", "list"])
    assert result.exit_code == 0, result.output
    bundle = next(
        item
        for item in json.loads(result.output)["bundles"]
        if item["name"] == "mom-v1"
    )
    assert any(item["name"] == "balance" for item in bundle["recipes"])
    destination = tmp_path / "bundle"
    result = runner.invoke(
        cli, ["recipe", "builtin", "export", "mom-v1", "--output-dir", str(destination)]
    )
    assert result.exit_code == 0, result.output
    assert model_bundle_digest(destination) == bundle["sha256"]
    assert sorted(path.name for path in destination.iterdir()) == sorted(
        bundle["files"]
    )
    repeated = runner.invoke(
        cli, ["recipe", "builtin", "export", "mom-v1", "--output-dir", str(destination)]
    )
    assert repeated.exit_code != 0
    assert model_bundle_digest(destination) == bundle["sha256"]


def test_init_selects_balance_binds_models_and_validates_without_touching_source(
    tmp_path,
):
    source, bindings, refs = _inputs(tmp_path)
    original = source.read_bytes()
    output = tmp_path / "config.yaml"
    result = _init(source, bindings, output)
    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["valid"] is True
    document = yaml.safe_load(output.read_text())
    assert source.read_bytes() == original
    assert document["providers"] == yaml.safe_load(original)["providers"]
    assert document["routing"] == yaml.safe_load(original)["routing"]
    assert [recipe["name"] for recipe in document["recipes"]] == ["balance"]
    assert document["entrypoints"] == [
        {"model_names": ["my/balance"], "recipe": "balance"}
    ]
    for decision in document["recipes"][0]["routing"]["decisions"]:
        assert decision["modelRefs"] == refs[decision["name"]]
    validated = CliRunner().invoke(cli, ["config", "validate", "--config", str(output)])
    assert validated.exit_code == 0, validated.output
    assert _init(source, bindings, output).exit_code != 0


@pytest.mark.parametrize(
    "problem, expected",
    [
        ("missing", "Bindings must cover every decision"),
        ("insufficient", "requires at least 2 distinct model candidates"),
        ("unknown", "unconfigured provider models"),
    ],
)
def test_init_refuses_incomplete_or_invented_bindings(tmp_path, problem, expected):
    source, bindings, refs = _inputs(tmp_path)
    if problem == "missing":
        refs.pop("complex")
    elif problem == "insufficient":
        refs["complex"] = [{"model": "model-a"}]
    else:
        refs["simple"] = [{"model": "invented"}]
    bindings.write_text(yaml.safe_dump(refs))
    output = tmp_path / "config.yaml"
    result = _init(source, bindings, output)
    assert result.exit_code != 0
    assert expected in result.output
    assert not output.exists()


def test_explicit_text_only_derivative_keeps_remaining_candidate_requirements(tmp_path):
    source, bindings, refs = _inputs(tmp_path)
    refs.pop("omni")
    bindings.write_text(yaml.safe_dump(refs))
    output = tmp_path / "text-only.yaml"
    result = _init(source, bindings, output, ["--exclude-decision", "omni"])
    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["excluded_decisions"] == ["omni"]
    decisions = yaml.safe_load(output.read_text())["recipes"][0]["routing"]["decisions"]
    assert "omni" not in {decision["name"] for decision in decisions}
    complex_lane = next(
        decision for decision in decisions if decision["name"] == "complex"
    )
    assert complex_lane["algorithm"]["minimum_candidates"] == 2
    assert len(complex_lane["modelRefs"]) == 2


def test_unknown_exclusion_is_an_error(tmp_path):
    source, bindings, _ = _inputs(tmp_path)
    output = tmp_path / "config.yaml"
    result = _init(source, bindings, output, ["--exclude-decision", "typo"])
    assert result.exit_code != 0
    assert "Unknown excluded decisions" in result.output
    assert not output.exists()


def test_init_preserves_config_relative_kb_assets(tmp_path):
    source, bindings, _ = _inputs(tmp_path)
    kb_root = tmp_path / "private-kb"
    kb_root.mkdir()
    manifest = kb_root / "labels.json"
    manifest.write_text('{"labels":{"safe":{"exemplars":["hello"]}}}')
    original_manifest = manifest.read_bytes()
    document = yaml.safe_load(source.read_text())
    document["global"] = {
        "model_catalog": {
            "kbs": [
                {
                    "name": "private-kb",
                    "source": {"path": "private-kb", "manifest": "labels.json"},
                }
            ]
        }
    }
    source.write_text(yaml.safe_dump(document))
    original_source = source.read_bytes()

    relocated = tmp_path / "derived" / "config.yaml"
    result = _init(source, bindings, relocated)
    assert result.exit_code != 0
    assert "--output must be in the same directory as --config" in result.output
    assert not relocated.parent.exists()

    sibling = tmp_path / "balance.yaml"
    result = _init(source, bindings, sibling)
    assert result.exit_code == 0, result.output
    assert _resolve_kb_source_root(sibling, "private-kb") == kb_root.resolve()
    assert yaml.safe_load(sibling.read_text())["global"] == document["global"]
    assert source.read_bytes() == original_source
    assert manifest.read_bytes() == original_manifest

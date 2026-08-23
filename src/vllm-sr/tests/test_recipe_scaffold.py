"""Tests for maintained recipe scaffolding."""

from __future__ import annotations

import json
from pathlib import Path

import yaml
from click.testing import CliRunner

from cli.commands.recipe import recipe
from cli.main import main
from cli.recipe_scaffold import RecipeScaffoldError, scaffold_recipe


def test_recipe_scaffold_creates_five_files(tmp_path: Path):
    output = tmp_path / "starter"
    result = scaffold_recipe("starter", output=output)

    assert result["recipe_id"] == "starter"
    assert sorted(result["files"]) == sorted(
        [
            "README.md",
            "config.yaml",
            "metadata.yaml",
            "probes.yaml",
            "recipe.dsl",
        ]
    )
    config = yaml.safe_load((output / "config.yaml").read_text(encoding="utf-8"))
    assert config["version"] == "v0.3"
    assert config["global"]["router"]["auto_model_names"] == ["vllm-sr/auto"]


def test_recipe_scaffold_multi_profile(tmp_path: Path):
    output = tmp_path / "multi"
    scaffold_recipe("multi", output=output, multi_profile=True)
    config = yaml.safe_load((output / "config.yaml").read_text(encoding="utf-8"))
    assert config["entrypoints"]
    assert config["recipes"]
    assert "decisions" not in config.get("routing", {})
    assert config["recipes"][0]["routing"]["decisions"]


def test_recipe_scaffold_from_recipe_redacts_endpoints(tmp_path: Path):
    output = tmp_path / "fork-balance"
    scaffold_recipe("fork-balance", output=output, from_recipe="balance")
    config = yaml.safe_load((output / "config.yaml").read_text(encoding="utf-8"))
    endpoints = [
        backend.get("endpoint")
        for model in config["providers"]["models"]
        for backend in model.get("backend_refs", [])
        if backend.get("endpoint")
    ]
    assert endpoints
    assert all(endpoint == "host.docker.internal:8000" for endpoint in endpoints)


def test_recipe_scaffold_refuses_overwrite(tmp_path: Path):
    output = tmp_path / "dup"
    scaffold_recipe("dup", output=output)
    try:
        scaffold_recipe("dup", output=output)
    except RecipeScaffoldError as error:
        assert "refusing to overwrite" in str(error)
    else:
        raise AssertionError("expected RecipeScaffoldError")


def test_recipe_scaffold_cli_is_registered():
    runner = CliRunner()

    top = runner.invoke(main, ["--help"])
    group = runner.invoke(main, ["recipe", "--help"])
    command = runner.invoke(main, ["recipe", "scaffold", "--help"])

    assert top.exit_code == group.exit_code == command.exit_code == 0
    assert "scaffold" in group.output
    assert "--from-recipe" in command.output


def test_recipe_scaffold_cli_writes_files(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    runner = CliRunner()
    result = runner.invoke(
        recipe,
        [
            "scaffold",
            "--name",
            "cli-starter",
            "--output",
            str(tmp_path / "cli-starter"),
        ],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["recipe_id"] == "cli-starter"
    assert (tmp_path / "cli-starter" / "config.yaml").is_file()

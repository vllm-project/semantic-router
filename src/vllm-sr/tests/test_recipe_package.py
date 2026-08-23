import hashlib
import json
import os
import time
import zipfile
from pathlib import Path

import pytest
import yaml
from cli.main import main
from cli.recipe_package import (
    RECIPE_FILES,
    RecipePackageError,
    pack_recipe,
    recipe_digest,
)
from click.testing import CliRunner


_RECIPE_CONFIG = """version: v0.4
recipes:
  - name: test-recipe
    document:
      decisions:
        - name: default
          rules:
            operator: AND
            conditions: []
"""


def _write_config_document(recipe: Path, document: dict) -> None:
    (recipe / "config.yaml").write_text(
        yaml.safe_dump(document, sort_keys=False), encoding="utf-8"
    )


def _recipe_with_plugin_configuration(configuration: dict) -> dict:
    document = yaml.safe_load(_RECIPE_CONFIG)
    document["recipes"][0]["document"]["decisions"][0]["plugins"] = [
        {"type": "request_params", "configuration": configuration}
    ]
    return document


def _write_recipe(root: Path, *, reverse: bool = False) -> None:
    contents = {
        "metadata.yaml": """schema_version: vllm-sr/recipe-metadata/v1
id: test-recipe
name: Test Recipe
version: 1.2.3
description: Deterministic package fixture.
authors:
  - name: Test Author
license: Apache-2.0
tags:
  - test
links:
  source: https://example.com/source
""",
        "config.yaml": _RECIPE_CONFIG,
        "probes.yaml": "schema_version: v1\ndecisions: []\n",
        "recipe.dsl": "MODEL test\n",
        "README.md": "# Test Recipe\n",
    }
    root.mkdir()
    filenames = list(RECIPE_FILES)
    if reverse:
        filenames.reverse()
    for index, filename in enumerate(filenames):
        path = root / filename
        path.write_text(contents[filename], encoding="utf-8")
        timestamp = 1_600_000_000 + (index * 97)
        os.utime(path, (timestamp, timestamp))


def test_pack_is_byte_identical_across_mtime_and_creation_order(tmp_path: Path):
    first = tmp_path / "first"
    second = tmp_path / "second"
    _write_recipe(first)
    _write_recipe(second, reverse=True)
    for path in second.iterdir():
        os.utime(path, (time.time(), time.time()))

    first_result = pack_recipe(first, tmp_path / "first.zip")
    second_result = pack_recipe(second, tmp_path / "second.zip")

    assert first_result.archive_sha256 == second_result.archive_sha256
    assert first_result.recipe_digest == second_result.recipe_digest
    assert first_result.path.read_bytes() == second_result.path.read_bytes()
    with zipfile.ZipFile(first_result.path) as archive:
        assert archive.namelist() == sorted(RECIPE_FILES)
        for info in archive.infolist():
            assert info.date_time == (1980, 1, 1, 0, 0, 0)
            assert (info.external_attr >> 16) & 0o777 == 0o644


def test_recipe_digest_matches_go_snapshot_algorithm_and_changes_on_tamper(
    tmp_path: Path,
):
    recipe = tmp_path / "recipe"
    _write_recipe(recipe)
    result = pack_recipe(recipe, tmp_path / "before.zip")

    expected = hashlib.sha256()
    for filename in RECIPE_FILES:
        expected.update(filename.encode("utf-8"))
        expected.update(b"\0")
        expected.update((recipe / filename).read_bytes())
        expected.update(b"\0")
    assert result.recipe_digest == f"sha256:{expected.hexdigest()}"

    (recipe / "README.md").write_text("# Tampered\n", encoding="utf-8")
    tampered = pack_recipe(recipe, tmp_path / "after.zip")
    assert tampered.recipe_digest != result.recipe_digest
    assert tampered.archive_sha256 != result.archive_sha256


def test_recipe_digest_cross_language_golden():
    files = {
        "metadata.yaml": b"schema_version: vllm-sr/recipe-metadata/v1\nid: fixture\n",
        "config.yaml": _RECIPE_CONFIG.encode("utf-8"),
        "probes.yaml": b"schema_version: v1\ndecisions: []\n",
        "recipe.dsl": b"MODEL fixture\n",
        "README.md": b"# Fixture\n",
    }

    assert recipe_digest(files) == (
        "sha256:53cc6ed9fa3018b6cdec963c70c34dfb535eb69f81fb950ae502447f3f3c3cb0"
    )


@pytest.mark.parametrize("invalid", ["missing", "extra", "symlink"])
def test_pack_rejects_invalid_directories(tmp_path: Path, invalid: str):
    recipe = tmp_path / "recipe"
    _write_recipe(recipe)
    if invalid == "missing":
        (recipe / "README.md").unlink()
    elif invalid == "extra":
        (recipe / "secret.txt").write_text("nope\n", encoding="utf-8")
    else:
        outside = tmp_path / "outside.md"
        outside.write_text("outside\n", encoding="utf-8")
        (recipe / "README.md").unlink()
        (recipe / "README.md").symlink_to(outside)

    with pytest.raises(RecipePackageError):
        pack_recipe(recipe, tmp_path / "invalid.zip")


def test_pack_ignores_real_cli_runtime_state_but_rejects_symlink(tmp_path: Path):
    recipe = tmp_path / "recipe"
    _write_recipe(recipe)
    runtime_state = recipe / ".vllm-sr"
    runtime_state.mkdir()
    (runtime_state / "compiled-bootstrap.yaml").write_text(
        "private runtime state\n", encoding="utf-8"
    )

    result = pack_recipe(recipe, tmp_path / "valid.zip")
    with zipfile.ZipFile(result.path) as archive:
        assert archive.namelist() == sorted(RECIPE_FILES)

    for path in runtime_state.iterdir():
        path.unlink()
    runtime_state.rmdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    runtime_state.symlink_to(outside, target_is_directory=True)
    with pytest.raises(RecipePackageError, match="real directory"):
        pack_recipe(recipe, tmp_path / "invalid.zip")


@pytest.mark.parametrize(
    "source_url",
    [
        "http://example.com/source",
        "https://contributor:secret@example.com/source",
        "https://example.com/source?token=private",
        "https://example.com/source#private",
        "https://example.com/source?",
        "https://example.com/source#",
    ],
)
def test_pack_rejects_metadata_urls_with_private_or_ambiguous_components(
    tmp_path: Path, source_url: str
):
    recipe = tmp_path / "recipe"
    _write_recipe(recipe)
    metadata = (recipe / "metadata.yaml").read_text(encoding="utf-8")
    (recipe / "metadata.yaml").write_text(
        metadata.replace("https://example.com/source", source_url),
        encoding="utf-8",
    )

    with pytest.raises(
        RecipePackageError, match="without userinfo, query, or fragment"
    ):
        pack_recipe(recipe, tmp_path / "invalid.zip")


def test_pack_rejects_malformed_metadata_url_as_package_error(tmp_path: Path):
    recipe = tmp_path / "recipe"
    _write_recipe(recipe)
    metadata = (recipe / "metadata.yaml").read_text(encoding="utf-8")
    (recipe / "metadata.yaml").write_text(
        metadata.replace("https://example.com/source", "https://[invalid/source"),
        encoding="utf-8",
    )

    with pytest.raises(RecipePackageError, match="valid HTTPS URL"):
        pack_recipe(recipe, tmp_path / "invalid.zip")


def test_pack_rejects_symlinked_output_without_overwriting_target(tmp_path: Path):
    recipe = tmp_path / "recipe"
    _write_recipe(recipe)
    outside = tmp_path / "outside.zip"
    outside.write_bytes(b"outside sentinel")
    output = tmp_path / "output.zip"
    output.symlink_to(outside)

    with pytest.raises(RecipePackageError, match="symbolic link"):
        pack_recipe(recipe, output)

    assert output.is_symlink()
    assert outside.read_bytes() == b"outside sentinel"


def test_pack_defaults_to_metadata_identity_filename_and_prints_json(tmp_path: Path):
    recipe = tmp_path / "source"
    _write_recipe(recipe)

    result = CliRunner().invoke(main, ["recipe", "pack", str(recipe)])

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    archive = tmp_path / "test-recipe-1.2.3.vllm-sr-recipe.zip"
    assert payload["path"] == str(archive)
    assert payload["sha256"].startswith("sha256:")
    assert payload["recipe_digest"].startswith("sha256:")
    assert archive.is_file()


def test_pack_rejects_literal_credential_in_current_model_shape(tmp_path: Path):
    recipe = tmp_path / "recipe"
    _write_recipe(recipe)
    secret_value = "super-sensitive-value"
    _write_config_document(
        recipe,
        {
            "version": "v0.4",
            "models": [
                {
                    "name": "private/model",
                    "card": {"capabilities": ["chat"]},
                    "connections": [
                        {
                            "provider": "openai-compatible",
                            "endpoint": "https://models.example.test/v1",
                            "model": "private/model",
                            "api_key": secret_value,
                        }
                    ],
                }
            ],
        },
    )

    result = CliRunner().invoke(
        main, ["recipe", "pack", str(recipe), "--output", str(tmp_path / "out")]
    )

    assert result.exit_code != 0
    assert "models[0].connections[0].api_key" in result.output
    assert secret_value not in result.output
    assert not (tmp_path / "out" / "test-recipe-1.2.3.vllm-sr-recipe.zip").exists()


@pytest.mark.parametrize(
    ("configuration", "expected_suffix"),
    [
        ({"api-key": "canary"}, "api-key"),
        ({"access_key": "canary"}, "access_key"),
        ({"client-secret": "canary"}, "client-secret"),
        ({"auth_token": "canary"}, "auth_token"),
        ({"bearer_token": "canary"}, "bearer_token"),
        ({"private_key": "canary"}, "private_key"),
        ({"headers": {"Authorization": "canary"}}, "headers.Authorization"),
        ({"extra_headers": {"X-API-Key": "canary"}}, "extra_headers.X-API-Key"),
        ({"extra_headers": {"XApiKey": "canary"}}, "extra_headers.XApiKey"),
        ({"headers": {"Cookie": "session=canary"}}, "headers.Cookie"),
        (
            {"headers": {"Proxy-Authorization": "Bearer canary"}},
            "headers.Proxy-Authorization",
        ),
    ],
)
def test_pack_literal_credential_detector_matches_runtime_fields(
    tmp_path: Path, configuration: dict, expected_suffix: str
):
    recipe = tmp_path / "recipe"
    _write_recipe(recipe)
    secret_value = "credential-value-canary"
    encoded = yaml.safe_dump(configuration).replace("canary", secret_value)
    _write_config_document(
        recipe,
        _recipe_with_plugin_configuration(yaml.safe_load(encoded)),
    )

    result = CliRunner().invoke(
        main, ["recipe", "pack", str(recipe), "--output", str(tmp_path / "out")]
    )

    assert result.exit_code != 0
    assert (
        "recipes[0].document.decisions[0].plugins[0].configuration." + expected_suffix
    ) in result.output
    assert secret_value not in result.output


@pytest.mark.parametrize("field", ["base_url", "url", "endpoint"])
@pytest.mark.parametrize(
    "url_template",
    [
        "https://user:{secret}@example.com/v1",
        "https://example.com/v1?api_key={secret}",
        "https://example.com/v1?access-token={secret}",
    ],
)
def test_pack_rejects_embedded_url_credential_without_exposing_it(
    tmp_path: Path, field: str, url_template: str
):
    recipe = tmp_path / "recipe"
    _write_recipe(recipe)
    secret_value = "url-password-canary"
    _write_config_document(
        recipe,
        {
            "version": "v0.4",
            "models": [
                {
                    "name": "private/model",
                    "card": {"capabilities": ["chat"]},
                    "connections": [
                        {
                            "provider": "openai-compatible",
                            field: url_template.format(secret=secret_value),
                            "model": "private/model",
                        }
                    ],
                }
            ],
        },
    )

    result = CliRunner().invoke(
        main, ["recipe", "pack", str(recipe), "--output", str(tmp_path / "out")]
    )

    assert result.exit_code != 0
    assert f"models[0].connections[0].{field}" in result.output
    assert secret_value not in result.output


def test_pack_rejects_credential_references_in_recipe_logic(tmp_path: Path):
    recipe = tmp_path / "recipe"
    _write_recipe(recipe)
    _write_config_document(
        recipe,
        _recipe_with_plugin_configuration(
            {
                "secret_env": "MODEL_SECRET",
                "password": "${DATABASE_PASSWORD}",
            }
        ),
    )

    result = CliRunner().invoke(
        main, ["recipe", "pack", str(recipe), "--output", str(tmp_path / "out")]
    )

    assert result.exit_code != 0
    assert "credential field" in result.output
    assert "MODEL_SECRET" not in result.output
    assert "DATABASE_PASSWORD" not in result.output


@pytest.mark.parametrize(
    "value", ["$DATABASE_PASSWORD", "${database_password}", "${DATABASE_PASSWORD}"]
)
def test_pack_rejects_every_credential_environment_reference(
    tmp_path: Path, value: str
):
    recipe = tmp_path / "recipe"
    _write_recipe(recipe)
    _write_config_document(
        recipe,
        _recipe_with_plugin_configuration({"password": value}),
    )

    result = CliRunner().invoke(
        main, ["recipe", "pack", str(recipe), "--output", str(tmp_path / "out")]
    )

    assert result.exit_code != 0
    assert "plugins[0].configuration.password" in result.output
    assert value not in result.output


def test_pack_rejects_literal_env_fallback_and_pure_reference(tmp_path: Path):
    recipe = tmp_path / "recipe"
    _write_recipe(recipe)
    _write_config_document(
        recipe,
        _recipe_with_plugin_configuration(
            {
                "api_key": "${PROVIDER_API_KEY}",
                "password": "${DATABASE_PASSWORD:-unsafe-default}",
            }
        ),
    )

    result = CliRunner().invoke(
        main, ["recipe", "pack", str(recipe), "--output", str(tmp_path / "out")]
    )

    assert result.exit_code != 0
    assert "plugins[0].configuration.password" in result.output
    assert "unsafe-default" not in result.output

    _write_config_document(
        recipe,
        _recipe_with_plugin_configuration({"api_key": "${PROVIDER_API_KEY}"}),
    )
    referenced = CliRunner().invoke(
        main,
        ["recipe", "pack", str(recipe), "--output", str(tmp_path / "referenced")],
    )

    assert referenced.exit_code != 0
    assert "plugins[0].configuration.api_key" in referenced.output
    assert "PROVIDER_API_KEY" not in referenced.output


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("models", []),
        ("models", [{"name": "runtime-model"}]),
        ("entrypoints", []),
        ("entrypoints", [{"name": "runtime-entrypoint"}]),
        ("global", {}),
        ("global", {"services": {}}),
        ("listeners", []),
    ],
)
def test_pack_rejects_runtime_owned_top_level_fields(
    tmp_path: Path, field: str, value: object
):
    recipe = tmp_path / "recipe"
    _write_recipe(recipe)
    document = yaml.safe_load(_RECIPE_CONFIG)
    document[field] = value
    _write_config_document(recipe, document)

    result = CliRunner().invoke(
        main, ["recipe", "pack", str(recipe), "--output", str(tmp_path / "out")]
    )

    assert result.exit_code != 0
    assert "Recipe distribution" in result.output
    assert field in result.output


@pytest.mark.parametrize(
    ("field", "value", "expected"),
    [
        ("version", "v0.3", "v0.4"),
        ("recipes", [], "at least 1 item"),
    ],
)
def test_pack_requires_complete_v04_recipe_distribution(
    tmp_path: Path, field: str, value: object, expected: str
):
    recipe = tmp_path / "recipe"
    _write_recipe(recipe)
    document = yaml.safe_load(_RECIPE_CONFIG)
    document[field] = value
    _write_config_document(recipe, document)

    result = CliRunner().invoke(
        main, ["recipe", "pack", str(recipe), "--output", str(tmp_path / "out")]
    )

    assert result.exit_code != 0
    assert "Recipe distribution" in result.output
    assert expected in result.output


@pytest.mark.parametrize(
    "config",
    [
        "version: v0.4\nsecret: &private-value sensitive\ncopy: *private-value\n",
        "version: v0.4\ndefaults: &private-map {token: sensitive}\nrequest:\n  <<: *private-map\n",
        "version: v0.4\nrequest:\n  <<: {token: sensitive}\n",
        "version: v0.4\nprovider:\n  base_url: !!binary aHR0cHM6Ly91c2VyOnNlbnNpdGl2ZUBleGFtcGxlLmNvbS92MQ==\n",
        "version: v0.4\nrequest:\n  headers:\n    !!binary QXV0aG9yaXphdGlvbg==: Bearer sensitive\n",
        "version: v0.4\nvalue: !!str sensitive\n",
    ],
)
def test_pack_rejects_yaml_indirection_without_exposing_values(
    tmp_path: Path, config: str
):
    recipe = tmp_path / "recipe"
    _write_recipe(recipe)
    (recipe / "config.yaml").write_text(config, encoding="utf-8")

    result = CliRunner().invoke(
        main, ["recipe", "pack", str(recipe), "--output", str(tmp_path / "out")]
    )

    assert result.exit_code != 0
    assert "Managed Recipe YAML must not contain" in result.output
    assert "sensitive" not in result.output


@pytest.mark.parametrize(
    "probes",
    [
        "schema_version: v1\ndefaults: &private sensitive\ncopy: *private\n",
        "schema_version: v1\nvalue: !!str sensitive\n",
    ],
)
def test_pack_rejects_probe_yaml_indirection_without_exposing_values(
    tmp_path: Path, probes: str
):
    recipe = tmp_path / "recipe"
    _write_recipe(recipe)
    (recipe / "probes.yaml").write_text(probes, encoding="utf-8")

    result = CliRunner().invoke(
        main, ["recipe", "pack", str(recipe), "--output", str(tmp_path / "out")]
    )

    assert result.exit_code != 0
    assert "Managed Recipe YAML must not contain" in result.output
    assert "sensitive" not in result.output


def test_recipe_pack_help_is_registered():
    runner = CliRunner()

    top = runner.invoke(main, ["--help"])
    group = runner.invoke(main, ["recipe", "--help"])
    command = runner.invoke(main, ["recipe", "pack", "--help"])

    assert top.exit_code == group.exit_code == command.exit_code == 0
    assert "recipe" in top.output
    assert "pack" in group.output
    assert "exact five-file RECIPE_DIR" in command.output

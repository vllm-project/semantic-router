import hashlib
import json
import os
import time
import zipfile
from pathlib import Path

import pytest
from cli.main import main
from cli.recipe_package import (
    RECIPE_FILES,
    RecipePackageError,
    pack_recipe,
    recipe_digest,
)
from click.testing import CliRunner


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
        "config.yaml": "version: v0.3\nlisteners: []\n",
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
        "config.yaml": b"version: v0.3\n",
        "probes.yaml": b"schema_version: v1\ndecisions: []\n",
        "recipe.dsl": b"MODEL fixture\n",
        "README.md": b"# Fixture\n",
    }

    assert recipe_digest(files) == (
        "sha256:b461747f69f276aea124f9d0907e373f9f201245f635d840276765dd6829ef7b"
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
    (runtime_state / "runtime-config.yaml").write_text(
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


def test_pack_rejects_literal_credential_with_key_path_only(tmp_path: Path):
    recipe = tmp_path / "recipe"
    _write_recipe(recipe)
    secret_value = "super-sensitive-value"
    (recipe / "config.yaml").write_text(
        f"version: v0.3\nproviders:\n  password: {secret_value}\n",
        encoding="utf-8",
    )

    result = CliRunner().invoke(
        main, ["recipe", "pack", str(recipe), "--output", str(tmp_path / "out")]
    )

    assert result.exit_code != 0
    assert "providers.password" in result.output
    assert secret_value not in result.output
    assert not (tmp_path / "out" / "test-recipe-1.2.3.vllm-sr-recipe.zip").exists()


@pytest.mark.parametrize(
    ("config", "expected_path"),
    [
        ("providers:\n  api-key: canary\n", "providers.api-key"),
        ("providers:\n  access_key: canary\n", "providers.access_key"),
        ("providers:\n  client-secret: canary\n", "providers.client-secret"),
        ("stores:\n  auth_token: canary\n", "stores.auth_token"),
        ("stores:\n  bearer_token: canary\n", "stores.bearer_token"),
        ("tls:\n  private_key: canary\n", "tls.private_key"),
        (
            "request:\n  headers:\n    Authorization: canary\n",
            "request.headers.Authorization",
        ),
        (
            "request:\n  extra_headers:\n    X-API-Key: canary\n",
            "request.extra_headers.X-API-Key",
        ),
        (
            "request:\n  extra_headers:\n    XApiKey: canary\n",
            "request.extra_headers.XApiKey",
        ),
        (
            "request:\n  headers:\n    Cookie: session=canary\n",
            "request.headers.Cookie",
        ),
        (
            "request:\n  headers:\n    Proxy-Authorization: Bearer canary\n",
            "request.headers.Proxy-Authorization",
        ),
    ],
)
def test_pack_literal_credential_detector_matches_runtime_fields(
    tmp_path: Path, config: str, expected_path: str
):
    recipe = tmp_path / "recipe"
    _write_recipe(recipe)
    secret_value = "credential-value-canary"
    (recipe / "config.yaml").write_text(
        "version: v0.3\n" + config.replace("canary", secret_value),
        encoding="utf-8",
    )

    result = CliRunner().invoke(
        main, ["recipe", "pack", str(recipe), "--output", str(tmp_path / "out")]
    )

    assert result.exit_code != 0
    assert expected_path in result.output
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
    (recipe / "config.yaml").write_text(
        f"version: v0.3\nprovider:\n  {field}: {url_template.format(secret=secret_value)}\n",
        encoding="utf-8",
    )

    result = CliRunner().invoke(
        main, ["recipe", "pack", str(recipe), "--output", str(tmp_path / "out")]
    )

    assert result.exit_code != 0
    assert f"provider.{field}" in result.output
    assert secret_value not in result.output


def test_pack_allows_api_key_env_and_exact_braced_credential_reference(tmp_path: Path):
    recipe = tmp_path / "recipe"
    _write_recipe(recipe)
    (recipe / "config.yaml").write_text(
        """version: v0.3
providers:
  api_key_env: PROVIDER_API_KEY
stores:
  password: ${DATABASE_PASSWORD}
""",
        encoding="utf-8",
    )

    result = CliRunner().invoke(
        main, ["recipe", "pack", str(recipe), "--output", str(tmp_path / "out")]
    )

    assert result.exit_code == 0, result.output


@pytest.mark.parametrize("value", ["$DATABASE_PASSWORD", "${database_password}"])
def test_pack_rejects_noncanonical_credential_environment_reference(
    tmp_path: Path, value: str
):
    recipe = tmp_path / "recipe"
    _write_recipe(recipe)
    (recipe / "config.yaml").write_text(
        f"version: v0.3\nstores:\n  password: {value}\n",
        encoding="utf-8",
    )

    result = CliRunner().invoke(
        main, ["recipe", "pack", str(recipe), "--output", str(tmp_path / "out")]
    )

    assert result.exit_code != 0
    assert "stores.password" in result.output
    assert value not in result.output


def test_pack_rejects_literal_env_fallback_but_not_pure_reference(tmp_path: Path):
    recipe = tmp_path / "recipe"
    _write_recipe(recipe)
    (recipe / "config.yaml").write_text(
        """version: v0.3
providers:
  api_key: ${PROVIDER_API_KEY}
  password: ${DATABASE_PASSWORD:-unsafe-default}
""",
        encoding="utf-8",
    )

    result = CliRunner().invoke(
        main, ["recipe", "pack", str(recipe), "--output", str(tmp_path / "out")]
    )

    assert result.exit_code != 0
    assert "providers.password" in result.output
    assert "providers.api_key" not in result.output
    assert "unsafe-default" not in result.output

    (recipe / "config.yaml").write_text(
        "version: v0.3\nproviders:\n  api_key: ${PROVIDER_API_KEY}\n",
        encoding="utf-8",
    )
    allowed = CliRunner().invoke(
        main,
        ["recipe", "pack", str(recipe), "--output", str(tmp_path / "allowed")],
    )

    assert allowed.exit_code == 0, allowed.output


@pytest.mark.parametrize(
    "mcp_config",
    [
        "",
        "          transport_type: stdio\n          command: private-command\n",
        "          command: private-command\n",
        "          transport_type: stdio\n          env: {PRIVATE_NAME: private-value}\n",
        "          transport_type: http\n          args: [private-argument]\n",
    ],
)
def test_pack_rejects_enabled_local_process_mcp_without_exposing_values(
    tmp_path: Path, mcp_config: str
):
    recipe = tmp_path / "recipe"
    _write_recipe(recipe)
    (recipe / "config.yaml").write_text(
        """version: v0.3
global:
  model_catalog:
    modules:
      classifier:
        mcp:
          enabled: true
"""
        + mcp_config
        + "          tool_name: classify\n",
        encoding="utf-8",
    )

    result = CliRunner().invoke(
        main, ["recipe", "pack", str(recipe), "--output", str(tmp_path / "out")]
    )

    assert result.exit_code != 0
    assert "global.model_catalog.modules.classifier.mcp" in result.output
    for private_value in ("private-command", "private-value", "private-argument"):
        assert private_value not in result.output


@pytest.mark.parametrize(
    "mcp_config",
    [
        "          transport_type: http\n          url: https://example.com/mcp\n",
        "          transport_type: streamable-http\n          url: https://example.com/mcp\n",
        "          transport_type: sse\n          url: https://example.com/mcp\n",
        "          url: https://example.com/mcp\n",
    ],
)
def test_pack_allows_enabled_remote_mcp_transports(tmp_path: Path, mcp_config: str):
    recipe = tmp_path / "recipe"
    _write_recipe(recipe)
    (recipe / "config.yaml").write_text(
        """version: v0.3
global:
  model_catalog:
    modules:
      classifier:
        mcp:
          enabled: true
"""
        + mcp_config
        + "          tool_name: classify\n",
        encoding="utf-8",
    )

    result = CliRunner().invoke(
        main, ["recipe", "pack", str(recipe), "--output", str(tmp_path / "out")]
    )

    assert result.exit_code == 0, result.output


@pytest.mark.parametrize(
    "config",
    [
        "version: v0.3\nsecret: &private-value sensitive\ncopy: *private-value\n",
        "version: v0.3\ndefaults: &private-map {token: sensitive}\nrequest:\n  <<: *private-map\n",
        "version: v0.3\nrequest:\n  <<: {token: sensitive}\n",
        "version: v0.3\nprovider:\n  base_url: !!binary aHR0cHM6Ly91c2VyOnNlbnNpdGl2ZUBleGFtcGxlLmNvbS92MQ==\n",
        "version: v0.3\nrequest:\n  headers:\n    !!binary QXV0aG9yaXphdGlvbg==: Bearer sensitive\n",
        "version: v0.3\nvalue: !!str sensitive\n",
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

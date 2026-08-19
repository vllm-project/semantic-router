from __future__ import annotations

import os
import stat
from pathlib import Path

import pytest
import yaml
from cli.commands import runtime as runtime_commands
from cli.commands.runtime_management_credentials import (
    catalog_management_credential_environment,
    management_credential_env_names,
)
from cli.commands.runtime_support import sensitive_env_names
from cli.main import main
from cli.model_bundle import MODEL_BUNDLE_FILES
from cli.recipe_directory import resolve_active_recipe_directory
from cli.recipe_topology_contract import MANAGEMENT_CREDENTIAL_ENV
from click.testing import CliRunner


def _capture_deployment(monkeypatch):
    captured: list[dict[str, object]] = []

    class _Backend:
        def deploy(self, **kwargs):
            captured.append(kwargs)

    monkeypatch.setattr(
        runtime_commands, "_build_backend", lambda *_args, **_kwargs: _Backend()
    )
    return captured


def _invoke_catalog_serve(monkeypatch, tmp_path: Path, *arguments: str):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv(MANAGEMENT_CREDENTIAL_ENV, raising=False)
    captured = _capture_deployment(monkeypatch)
    result = CliRunner().invoke(
        main,
        ["serve", *arguments, "--image-pull-policy", "never"],
    )
    return result, captured


def _source_document(deployment: dict[str, object]) -> dict[str, object]:
    return yaml.safe_load(Path(str(deployment["source_config_file"])).read_text())


def _entrypoint_models(document: dict[str, object]) -> list[str]:
    return [str(entrypoint["model_names"][0]) for entrypoint in document["entrypoints"]]


def test_serve_one_catalog_virtual_model_uses_private_workspace_source(
    monkeypatch, tmp_path: Path
):
    result, captured = _invoke_catalog_serve(
        monkeypatch, tmp_path, "vllm-sr/mom-v1-blend"
    )

    assert result.exit_code == 0, result.output
    assert len(captured) == 1
    deployment = captured[0]
    source_path = Path(str(deployment["source_config_file"]))
    assert source_path.name == "config.yaml"
    assert source_path.parent.parent == tmp_path / ".vllm-sr" / "catalog-sources"
    assert stat.S_IMODE(source_path.parent.stat().st_mode) == 0o700
    assert stat.S_IMODE(source_path.stat().st_mode) == 0o644
    assert {path.name for path in source_path.parent.iterdir()} == set(
        MODEL_BUNDLE_FILES
    )
    assert all(
        stat.S_IMODE((source_path.parent / name).stat().st_mode) == 0o644
        for name in MODEL_BUNDLE_FILES
    )
    assert resolve_active_recipe_directory(source_path) is not None
    assert not (tmp_path / "config.yaml").exists()

    document = _source_document(deployment)
    assert _entrypoint_models(document) == ["vllm-sr/mom-v1-blend"]
    assert len(document["recipes"]) == 1
    assert len(document["providers"]["models"]) == 7
    metadata = yaml.safe_load((source_path.parent / "metadata.yaml").read_text())
    probes = yaml.safe_load((source_path.parent / "probes.yaml").read_text())
    assert metadata["id"] == probes["name"] == "mom-v1"
    assert {decision["model"] for decision in probes["decisions"]} == {
        "vllm-sr/mom-v1-blend"
    }
    assert sum(len(decision["variants"]) for decision in probes["decisions"]) > 0

    env_vars = deployment["env_vars"]
    token = env_vars[MANAGEMENT_CREDENTIAL_ENV]
    assert len(token) == 64
    int(token, 16)
    assert env_vars["VLLM_SR_STATE_ROOT_DIR"] == str(tmp_path)
    assert token not in source_path.read_text(encoding="utf-8")
    assert MANAGEMENT_CREDENTIAL_ENV not in os.environ

    credential_path = tmp_path / ".vllm-sr" / "catalog-credentials" / "vllm-sr.token"
    assert credential_path.read_text(encoding="ascii").strip() == token
    assert stat.S_IMODE(credential_path.stat().st_mode) == 0o600


def test_serve_multiple_catalog_models_preserves_operand_order(
    monkeypatch, tmp_path: Path
):
    result, captured = _invoke_catalog_serve(
        monkeypatch,
        tmp_path,
        "vllm-sr/mom-v1-flash",
        "--minimal",
        "vllm-sr/mom-v1-lite",
    )

    assert result.exit_code == 0, result.output
    document = _source_document(captured[0])
    assert _entrypoint_models(document) == [
        "vllm-sr/mom-v1-flash",
        "vllm-sr/mom-v1-lite",
    ]
    assert len(document["recipes"]) == 2
    assert len(document["providers"]["models"]) == 7
    assert "vllm-sr/mom-v1-blend" not in _entrypoint_models(document)
    assert captured[0]["enable_observability"] is False
    source_path = Path(str(captured[0]["source_config_file"]))
    probes = yaml.safe_load((source_path.parent / "probes.yaml").read_text())
    assert {decision["model"] for decision in probes["decisions"]} == {
        "vllm-sr/mom-v1-flash",
        "vllm-sr/mom-v1-lite",
    }


def test_serve_catalog_source_and_management_token_are_stable(
    monkeypatch, tmp_path: Path
):
    first, first_capture = _invoke_catalog_serve(
        monkeypatch, tmp_path, "vllm-sr/mom-v1-lite"
    )
    second, second_capture = _invoke_catalog_serve(
        monkeypatch, tmp_path, "vllm-sr/mom-v1-lite"
    )

    assert first.exit_code == second.exit_code == 0
    assert (
        first_capture[0]["source_config_file"]
        == second_capture[0]["source_config_file"]
    )
    assert (
        first_capture[0]["env_vars"][MANAGEMENT_CREDENTIAL_ENV]
        == second_capture[0]["env_vars"][MANAGEMENT_CREDENTIAL_ENV]
    )


@pytest.mark.parametrize(
    ("env_name", "serve_args", "warns"),
    (
        (None, ("--image", "example.invalid/all:custom"), True),
        (None, ("--router-image", "example.invalid/router:custom"), True),
        ("VLLM_SR_IMAGE", (), True),
        ("VLLM_SR_ROUTER_IMAGE", (), True),
        ("VLLM_SR_IMAGE_AMD", ("--platform", "amd"), True),
        ("VLLM_SR_IMAGE_NVIDIA", ("--platform", "nvidia"), True),
        ("VLLM_SR_IMAGE_AMD", ("--platform", "nvidia"), False),
        ("VLLM_SR_IMAGE_NVIDIA", ("--platform", "amd"), False),
    ),
)
def test_serve_catalog_warns_for_effective_router_image_override(
    monkeypatch,
    tmp_path: Path,
    env_name: str | None,
    serve_args: tuple[str, ...],
    warns: bool,
):
    image_env_names = (
        "VLLM_SR_IMAGE",
        "VLLM_SR_ROUTER_IMAGE",
        "VLLM_SR_IMAGE_AMD",
        "VLLM_SR_IMAGE_NVIDIA",
    )
    for name in image_env_names:
        monkeypatch.delenv(name, raising=False)
    if env_name is not None:
        monkeypatch.setenv(env_name, "example.invalid/router:custom")

    result, captured = _invoke_catalog_serve(
        monkeypatch,
        tmp_path,
        "vllm-sr/mom-v1-blend",
        *serve_args,
    )

    assert result.exit_code == 0, result.output
    assert len(captured) == 1
    assert ("custom Router image" in result.stderr) is warns
    if warns:
        assert "operator-managed compatibility override" in result.stderr


def test_serve_catalog_rejects_config_custom_model_and_kubernetes_before_writes(
    monkeypatch, tmp_path: Path
):
    monkeypatch.chdir(tmp_path)
    runner = CliRunner()

    conflict = runner.invoke(
        main,
        ["serve", "vllm-sr/mom-v1-blend", "--config", "custom.yaml"],
    )
    custom = runner.invoke(main, ["serve", "my/qwen"])
    kubernetes = runner.invoke(
        main, ["serve", "vllm-sr/mom-v1-blend", "--target", "k8s"]
    )

    assert conflict.exit_code == custom.exit_code == kubernetes.exit_code == 1
    assert "mutually exclusive" in conflict.stderr
    assert "provider aliases or model checkpoints" in custom.stderr
    assert "local Docker target" in kubernetes.stderr
    assert not (tmp_path / ".vllm-sr").exists()


def test_serve_catalog_rejects_unknown_model_and_orphan_catalog_version(
    monkeypatch, tmp_path: Path
):
    monkeypatch.chdir(tmp_path)
    runner = CliRunner()

    unknown = runner.invoke(main, ["serve", "vllm-sr/not-installed"])
    orphan_version = runner.invoke(main, ["serve", "--catalog-version", "latest"])

    assert unknown.exit_code == orphan_version.exit_code == 1
    assert "unknown built-in models" in unknown.stderr
    assert "requires at least one MODEL" in orphan_version.stderr


def test_serve_catalog_rejects_silent_algorithm_override_before_writes(
    monkeypatch, tmp_path: Path
):
    monkeypatch.chdir(tmp_path)

    result = CliRunner().invoke(
        main,
        ["serve", "vllm-sr/mom-v1-blend", "--algorithm", "static"],
    )

    assert result.exit_code == 1
    assert (
        "Catalog MODEL operands use their verified recipe algorithms" in result.stderr
    )
    assert not (tmp_path / ".vllm-sr").exists()


def test_serve_catalog_rejects_symlinked_workspace_state(monkeypatch, tmp_path: Path):
    outside = tmp_path / "outside"
    outside.mkdir()
    (tmp_path / ".vllm-sr").symlink_to(outside, target_is_directory=True)
    monkeypatch.chdir(tmp_path)

    result = CliRunner().invoke(main, ["serve", "vllm-sr/mom-v1-blend"])

    assert result.exit_code == 1
    assert "must not be a symbolic link" in result.stderr
    assert list(outside.iterdir()) == []


def test_serve_help_explains_virtual_custom_and_multi_model_contracts():
    result = CliRunner().invoke(main, ["serve", "--help"])

    assert result.exit_code == 0
    assert "serve [OPTIONS] [MODEL]..." in result.output
    assert "vllm-sr serve vllm-sr/mom-v1-blend" in result.output
    assert "vllm-sr/mom-v1-lite vllm-sr/mom-v1-flash" in result.output
    assert "does not download or launch" in result.output
    assert "physical LLM engines" in result.output
    assert "vllm-sr serve --config my-models.yaml" in result.output
    assert "\x08" not in result.output


def test_model_list_points_to_catalog_and_custom_serve_flows():
    result = CliRunner().invoke(main, ["model", "list"])

    assert result.exit_code == 0
    assert "Next step" in result.output
    assert "vllm-sr serve <MODEL> [MODEL]..." in result.output
    assert "vllm-sr serve --config <PATH>" in result.output
    assert result.stderr == ""


def test_serve_catalog_fails_closed_for_active_recipe(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(
        runtime_commands, "active_recipe_package_for_stack", lambda **_kwargs: True
    )
    result, captured = _invoke_catalog_serve(
        monkeypatch, tmp_path, "vllm-sr/mom-v1-blend"
    )

    assert result.exit_code == 1
    assert "active managed Recipe" in result.stderr
    assert captured == []


def test_serve_catalog_fails_closed_when_runtime_edits_are_preserved(
    monkeypatch, tmp_path: Path
):
    preserved = tmp_path / ".vllm-sr" / "runtime-config.yaml"
    preserved.parent.mkdir(parents=True)
    preserved.write_text("version: dashboard-edited\n", encoding="utf-8")
    monkeypatch.setattr(
        runtime_commands,
        "materialize_runtime_config",
        lambda *_args, **_kwargs: preserved,
    )
    result, captured = _invoke_catalog_serve(
        monkeypatch, tmp_path, "vllm-sr/mom-v1-blend"
    )

    assert result.exit_code == 1
    assert "Dashboard changes preserved" in result.stderr
    assert captured == []


def test_management_credential_schema_is_sensitive_and_context_is_restored(
    monkeypatch, tmp_path: Path
):
    config = tmp_path / "config.yaml"
    config.write_text(
        """global:
  services:
    management_api:
      auth:
        tokens:
          - env: CUSTOM_MANAGEMENT_TOKEN
""",
        encoding="utf-8",
    )
    monkeypatch.setenv("CUSTOM_MANAGEMENT_TOKEN", "never-print-this-value")

    assert management_credential_env_names(config) == {"CUSTOM_MANAGEMENT_TOKEN"}
    assert "CUSTOM_MANAGEMENT_TOKEN" in sensitive_env_names(config)
    with catalog_management_credential_environment(
        config, state_root=tmp_path, stack_name="test-stack"
    ) as bindings:
        assert bindings == {"CUSTOM_MANAGEMENT_TOKEN": "never-print-this-value"}
    assert os.environ["CUSTOM_MANAGEMENT_TOKEN"] == "never-print-this-value"


@pytest.mark.parametrize(
    "environment_name", ["lowercase_token", "HOME", "PATH", "VLLM_SR_PLATFORM"]
)
def test_management_credential_schema_rejects_noncanonical_or_reserved_env_name(
    tmp_path: Path, environment_name: str
):
    config = tmp_path / "config.yaml"
    config.write_text(
        f"""global:
  services:
    management_api:
      auth:
        tokens:
          - env: {environment_name}
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="env name is invalid"):
        management_credential_env_names(config)


def test_catalog_management_credential_rejects_invalid_supplied_or_stored_token(
    monkeypatch, tmp_path: Path
):
    config = tmp_path / "config.yaml"
    config.write_text(
        """global:
  services:
    management_api:
      auth:
        tokens:
          - env: VLLM_SR_DASHBOARD_RECIPE_TOKEN
""",
        encoding="utf-8",
    )
    monkeypatch.setenv(MANAGEMENT_CREDENTIAL_ENV, "invalid")
    try:
        with catalog_management_credential_environment(
            config, state_root=tmp_path, stack_name="test-stack"
        ):
            raise AssertionError("invalid token was accepted")
    except ValueError as error:
        assert "64 lowercase hexadecimal" in str(error)

    monkeypatch.delenv(MANAGEMENT_CREDENTIAL_ENV)
    stored = tmp_path / ".vllm-sr" / "catalog-credentials" / "test-stack.token"
    stored.parent.mkdir(parents=True)
    stored.write_text("0" * 64, encoding="ascii")
    stored.chmod(0o644)
    try:
        with catalog_management_credential_environment(
            config, state_root=tmp_path, stack_name="test-stack"
        ):
            raise AssertionError("public credential file was accepted")
    except ValueError as error:
        assert "not private" in str(error)

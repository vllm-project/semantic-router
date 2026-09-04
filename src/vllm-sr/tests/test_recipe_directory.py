import subprocess
from types import SimpleNamespace

import pytest
from cli import container_start, core
from cli.recipe_directory import (
    RecipeDirectoryError,
    resolve_active_recipe_directory,
)


def _write_managed_recipe(tmp_path):
    files = {
        "config.yaml": "version: v0.3\nlisteners: []\n",
        "metadata.yaml": "schema_version: vllm-sr/recipe-metadata/v1\n",
        "probes.yaml": "schema_version: v1\n",
        "recipe.dsl": "ROUTE default\n",
        "README.md": "# Test Recipe\n",
    }
    for filename, content in files.items():
        (tmp_path / filename).write_text(content, encoding="utf-8")
    return tmp_path / "config.yaml"


def _capture_run_commands(monkeypatch):
    captured = []

    def fake_run(cmd, capture_output, text, check, env=None):
        captured.append(cmd)
        return SimpleNamespace(stdout="container-id\n", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr(
        container_start, "_render_split_envoy_config", lambda *args, **kwargs: None
    )
    return captured


def _find_container_run_cmd(commands, container_name):
    return next(
        cmd
        for cmd in commands
        if "--name" in cmd and cmd[cmd.index("--name") + 1] == container_name
    )


def test_bare_config_is_not_a_managed_recipe(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text("version: v0.3\n", encoding="utf-8")

    assert resolve_active_recipe_directory(config_path) is None


def test_partial_recipe_requires_metadata(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text("version: v0.3\n", encoding="utf-8")
    (tmp_path / "probes.yaml").write_text("schema_version: v1\n", encoding="utf-8")

    with pytest.raises(RecipeDirectoryError, match=r"metadata.yaml is missing"):
        resolve_active_recipe_directory(config_path)


def test_managed_recipe_requires_all_fixed_assets(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text("version: v0.3\n", encoding="utf-8")
    (tmp_path / "metadata.yaml").write_text(
        "schema_version: vllm-sr/recipe-metadata/v1\n", encoding="utf-8"
    )

    with pytest.raises(RecipeDirectoryError, match=r"probes.yaml"):
        resolve_active_recipe_directory(config_path)


def test_start_rejects_incomplete_recipe_before_runtime_cleanup(tmp_path, monkeypatch):
    config_path = tmp_path / "config.yaml"
    config_path.write_text("version: v0.3\n", encoding="utf-8")
    (tmp_path / "metadata.yaml").write_text(
        "schema_version: vllm-sr/recipe-metadata/v1\n", encoding="utf-8"
    )
    cleaned = []
    monkeypatch.setattr(core, "print_vllm_logo", lambda: None)
    monkeypatch.setattr(core, "ensure_clean_runtime_container", cleaned.append)

    with pytest.raises(RecipeDirectoryError, match="Incomplete managed Recipe"):
        core.start_vllm_sr(str(config_path), env_vars={})

    assert cleaned == []


def test_managed_recipe_uses_fixed_config_name(tmp_path):
    config_path = _write_managed_recipe(tmp_path)
    renamed_config = tmp_path / "custom.yaml"
    renamed_config.write_text(config_path.read_text(encoding="utf-8"), encoding="utf-8")

    with pytest.raises(RecipeDirectoryError, match=r"must be named config.yaml"):
        resolve_active_recipe_directory(renamed_config)


def test_resolves_complete_managed_recipe(tmp_path):
    config_path = _write_managed_recipe(tmp_path)

    recipe = resolve_active_recipe_directory(config_path)

    assert recipe is not None
    assert recipe.root == tmp_path
    assert [name for name, _ in recipe.assets] == [
        "config.yaml",
        "metadata.yaml",
        "probes.yaml",
        "recipe.dsl",
        "README.md",
    ]


@pytest.mark.parametrize("target", ["config.yaml", "outside"])
def test_managed_recipe_rejects_symbolic_link_assets(tmp_path, target):
    config_path = _write_managed_recipe(tmp_path)
    readme_path = tmp_path / "README.md"
    readme_path.unlink()
    if target == "outside":
        outside = tmp_path.parent / f"{tmp_path.name}-outside.md"
        outside.write_text("private material\n", encoding="utf-8")
        readme_path.symlink_to(outside)
    else:
        readme_path.symlink_to(tmp_path / target)

    with pytest.raises(RecipeDirectoryError, match="must not be a symbolic link"):
        resolve_active_recipe_directory(config_path)


def test_managed_recipe_runtime_state_stays_under_vllm_sr(tmp_path):
    config_path = _write_managed_recipe(tmp_path)

    _, runtime_paths, _ = container_start._prepare_runtime_paths(str(config_path))

    assert runtime_paths["models_dir"] == str(tmp_path / ".vllm-sr" / "models")
    assert not (tmp_path / "models").exists()
    visible_entries = {
        path.name for path in tmp_path.iterdir() if path.name != ".vllm-sr"
    }
    assert visible_entries == {
        "config.yaml",
        "metadata.yaml",
        "probes.yaml",
        "recipe.dsl",
        "README.md",
    }


def test_dashboard_receives_only_fixed_recipe_asset_mounts(tmp_path, monkeypatch):
    config_path = _write_managed_recipe(tmp_path)
    monkeypatch.setattr(container_start, "get_container_runtime", lambda: "docker")
    monkeypatch.setattr(
        container_start,
        "get_runtime_images",
        lambda **kwargs: {
            "router": "router-image",
            "envoy": "envoy-image",
            "dashboard": "dashboard-image",
        },
    )
    monkeypatch.setattr(
        container_start,
        "resolve_container_cli_path",
        lambda preferred_path=None: str(tmp_path / "docker"),
    )
    captured = _capture_run_commands(monkeypatch)

    return_code, _, _ = container_start.container_start_vllm_sr(
        str(config_path),
        {"VLLM_SR_MANAGED_STORAGE_BACKENDS": "milvus,postgres,redis"},
        [{"name": "http-8899", "address": "0.0.0.0", "port": 8899}],
        minimal=False,
    )

    assert return_code == 0
    dashboard_cmd = _find_container_run_cmd(captured, "vllm-sr-dashboard-container")
    router_cmd = _find_container_run_cmd(captured, "vllm-sr-router-container")
    envoy_cmd = _find_container_run_cmd(captured, "vllm-sr-envoy-container")
    assert "VLLM_SR_ACTIVE_RECIPE_DIR=/app/recipe" in dashboard_cmd
    assert "DASHBOARD_CONFIG_DIR=/app" in dashboard_cmd
    assert (
        "VLLM_SR_RECIPE_STORE_DIR=/app/.vllm-sr/recipe-store/vllm-sr" in dashboard_cmd
    )
    assert "VLLM_SR_RECIPE_STORE_DIR=/app/.vllm-sr/recipe-store/vllm-sr" in router_cmd
    assert not any(item.startswith("VLLM_SR_RECIPE_STORE_DIR=") for item in envoy_cmd)
    for command in (router_cmd, dashboard_cmd):
        assert "VLLM_SR_MANAGED_STORAGE_BACKENDS=milvus,postgres,redis" in command
    assert not any(
        item.startswith("VLLM_SR_MANAGED_STORAGE_BACKENDS=") for item in envoy_cmd
    )
    assert "VLLM_SR_STATE_ROOT_DIR=/app" in dashboard_cmd
    assert "VLLM_SR_CONFIG_BASE_DIR=/app" in dashboard_cmd
    assert "/app/.vllm-sr/runtime-config.yaml" in dashboard_cmd
    assert f"{tmp_path / 'config.yaml'}:/app/source-config.yaml:ro,z" in dashboard_cmd
    assert (tmp_path / ".vllm-sr" / "runtime-config.yaml").is_file()
    assert (tmp_path / ".vllm-sr" / "recipe-store" / "vllm-sr").is_dir()
    for filename in (
        "config.yaml",
        "metadata.yaml",
        "probes.yaml",
        "recipe.dsl",
        "README.md",
    ):
        mount = f"{tmp_path / filename}:/app/recipe/{filename}:ro,z"
        assert mount in dashboard_cmd
        assert mount not in router_cmd


def test_recipe_env_bindings_use_name_only_container_arguments(tmp_path, monkeypatch):
    config_path = _write_managed_recipe(tmp_path)
    monkeypatch.setenv("CUSTOM_API_KEY", "never-include-this-secret")
    monkeypatch.setattr(container_start, "get_container_runtime", lambda: "docker")
    monkeypatch.setattr(
        container_start,
        "get_runtime_images",
        lambda **kwargs: {
            "router": "router-image",
            "envoy": "envoy-image",
            "dashboard": "dashboard-image",
        },
    )
    monkeypatch.setattr(
        container_start,
        "resolve_container_cli_path",
        lambda preferred_path=None: str(tmp_path / "docker"),
    )
    captured = _capture_run_commands(monkeypatch)

    return_code, _, _ = container_start.container_start_vllm_sr(
        str(config_path),
        {
            "CUSTOM_API_KEY": "never-include-this-secret",
            "VLLM_SR_RECIPE_ENV_ALLOWLIST": "CUSTOM_API_KEY",
        },
        [{"name": "http-8899", "address": "0.0.0.0", "port": 8899}],
        minimal=False,
    )

    assert return_code == 0
    router_cmd = _find_container_run_cmd(captured, "vllm-sr-router-container")
    dashboard_cmd = _find_container_run_cmd(captured, "vllm-sr-dashboard-container")
    envoy_cmd = _find_container_run_cmd(captured, "vllm-sr-envoy-container")
    for command in (router_cmd, dashboard_cmd):
        assert "CUSTOM_API_KEY" in command
        assert "CUSTOM_API_KEY=never-include-this-secret" not in command
        assert "never-include-this-secret" not in command
    assert "CUSTOM_API_KEY" not in envoy_cmd
    assert "never-include-this-secret" not in envoy_cmd


def test_source_discovered_secrets_use_name_only_arguments_and_logs(
    tmp_path, monkeypatch, caplog
):
    config_path = _write_managed_recipe(tmp_path)
    config_path.write_text(
        """version: v0.3
listeners: []
providers:
  models:
    - name: custom
      api_key_env: CUSTOM_PROVIDER_API_KEY
""",
        encoding="utf-8",
    )
    custom_value = "custom-provider-secret-canary"
    static_value = "openai-secret-canary"
    monkeypatch.setenv("CUSTOM_PROVIDER_API_KEY", custom_value)
    monkeypatch.setenv("OPENAI_API_KEY", static_value)
    monkeypatch.setattr(container_start, "get_container_runtime", lambda: "docker")
    monkeypatch.setattr(
        container_start,
        "get_runtime_images",
        lambda **kwargs: {
            "router": "router-image",
            "envoy": "envoy-image",
            "dashboard": "dashboard-image",
        },
    )
    monkeypatch.setattr(
        container_start,
        "resolve_container_cli_path",
        lambda preferred_path=None: str(tmp_path / "docker"),
    )
    captured = _capture_run_commands(monkeypatch)

    with caplog.at_level("DEBUG", logger="cli.container_start"):
        return_code, _, _ = container_start.container_start_vllm_sr(
            str(config_path),
            {
                "CUSTOM_PROVIDER_API_KEY": custom_value,
                "OPENAI_API_KEY": static_value,
            },
            [{"name": "http-8899", "address": "0.0.0.0", "port": 8899}],
            minimal=False,
        )

    assert return_code == 0
    router_cmd = _find_container_run_cmd(captured, "vllm-sr-router-container")
    dashboard_cmd = _find_container_run_cmd(captured, "vllm-sr-dashboard-container")
    envoy_cmd = _find_container_run_cmd(captured, "vllm-sr-envoy-container")
    for command in (router_cmd, dashboard_cmd):
        for name, value in (
            ("CUSTOM_PROVIDER_API_KEY", custom_value),
            ("OPENAI_API_KEY", static_value),
        ):
            assert name in command
            assert f"{name}={value}" not in command
            assert value not in command
    for secret in (custom_value, static_value):
        assert secret not in envoy_cmd
        assert secret not in caplog.text
    assert "CUSTOM_PROVIDER_API_KEY=" not in caplog.text
    assert "OPENAI_API_KEY=" not in caplog.text


def test_bare_config_cannot_inherit_stale_recipe_directory_env(tmp_path, monkeypatch):
    config_path = tmp_path / "config.yaml"
    config_path.write_text("version: v0.3\n", encoding="utf-8")
    monkeypatch.setattr(container_start, "get_container_runtime", lambda: "docker")
    monkeypatch.setattr(
        container_start,
        "get_runtime_images",
        lambda **kwargs: {
            "router": "router-image",
            "envoy": "envoy-image",
            "dashboard": "dashboard-image",
        },
    )
    monkeypatch.setattr(
        container_start,
        "resolve_container_cli_path",
        lambda preferred_path=None: str(tmp_path / "docker"),
    )
    captured = _capture_run_commands(monkeypatch)

    return_code, _, _ = container_start.container_start_vllm_sr(
        str(config_path),
        {"VLLM_SR_ACTIVE_RECIPE_DIR": "/stale/recipe"},
        [{"name": "http-8899", "address": "0.0.0.0", "port": 8899}],
        minimal=False,
    )

    assert return_code == 0
    dashboard_cmd = _find_container_run_cmd(captured, "vllm-sr-dashboard-container")
    assert not any(
        value.startswith("VLLM_SR_ACTIVE_RECIPE_DIR=") for value in dashboard_cmd
    )

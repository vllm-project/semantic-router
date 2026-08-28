import hashlib
import importlib
import json
import re
import sys
from pathlib import Path

import pytest
import yaml
from click.testing import CliRunner

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

BootstrapResult = importlib.import_module("cli.bootstrap").BootstrapResult
runtime_commands = importlib.import_module("cli.commands.runtime")
serve_config = importlib.import_module("cli.commands.runtime_serve_config")
main = importlib.import_module("cli.main").main
recipe_package = importlib.import_module("cli.recipe_package")
runtime_config_lock = importlib.import_module("cli.runtime_config_lock")

_PYPROJECT_VERSION_PATTERN = re.compile(
    r'^version = "(?P<version>[^"]+)"$', re.MULTILINE
)


def _project_version() -> str:
    pyproject_path = Path(__file__).resolve().parents[1] / "pyproject.toml"
    match = _PYPROJECT_VERSION_PATTERN.search(
        pyproject_path.read_text(encoding="utf-8")
    )
    assert match is not None
    return match.group("version")


def test_cli_help_lists_registered_commands():
    runner = CliRunner()

    result = runner.invoke(main, ["--help"])

    assert result.exit_code == 0
    for command_name in (
        "serve",
        "config",
        "validate",
        "status",
        "logs",
        "stop",
        "dashboard",
        "chat",
    ):
        assert command_name in result.output
    assert " init" not in result.output


def test_cli_version_matches_project_metadata():
    runner = CliRunner()

    result = runner.invoke(main, ["--version"])
    expected_version = _project_version()

    assert result.exit_code == 0
    assert result.output.strip() == f"vllm-sr version: {expected_version}"


def test_serve_materializes_active_config_under_custom_host_state_root(
    monkeypatch, tmp_path: Path
):
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    config_path = source_dir / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "version": "v0.3",
                "listeners": [
                    {"name": "http-8899", "address": "0.0.0.0", "port": 8899}
                ],
                "routing": {"decisions": [{"name": "default"}]},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    state_root = tmp_path / "host-state"
    bootstrap = BootstrapResult(
        config_path=config_path,
        output_dir=source_dir / ".vllm-sr",
        setup_mode=False,
    )
    captured: dict[str, object] = {}

    class _StubBackend:
        def deploy(self, **kwargs):
            owned_lock = kwargs["runtime_config_lock"]
            assert owned_lock._closed is False
            with pytest.raises(
                runtime_config_lock.RuntimeConfigLockError,
                match="operation is in progress",
            ):
                runtime_config_lock.acquire_runtime_config_lock(
                    runtime_config_path=kwargs["runtime_config_file"],
                    state_root_dir=state_root,
                    stack_name="vllm-sr",
                    timeout_seconds=0,
                )
            captured.update(kwargs)

    monkeypatch.setenv("VLLM_SR_STATE_ROOT_DIR", str(state_root))
    monkeypatch.setattr(
        runtime_commands, "ensure_bootstrap_workspace", lambda _: bootstrap
    )
    monkeypatch.setattr(
        runtime_commands, "_build_backend", lambda *a, **kw: _StubBackend()
    )

    result = CliRunner().invoke(
        main,
        [
            "serve",
            "--config",
            str(config_path),
            "--image-pull-policy",
            "never",
        ],
    )

    assert result.exit_code == 0, result.output
    expected = state_root / ".vllm-sr" / "runtime-config.yaml"
    assert Path(captured["config_file"]) == expected
    assert Path(captured["runtime_config_file"]) == expected
    assert captured["runtime_config_lock"]._closed is True
    assert expected.is_file()
    assert not (source_dir / ".vllm-sr" / "runtime-config.yaml").exists()
    assert "VLLM_SR_STATE_ROOT_DIR" not in captured["env_vars"]


def test_k8s_serve_keeps_non_persistent_effective_config_flow(
    monkeypatch, tmp_path: Path
):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "version": "v0.3",
                "listeners": [
                    {"name": "http-8899", "address": "0.0.0.0", "port": 8899}
                ],
                "routing": {"decisions": [{"name": "default"}]},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    bootstrap = BootstrapResult(
        config_path=config_path,
        output_dir=tmp_path / ".vllm-sr",
        setup_mode=False,
    )
    captured: dict[str, object] = {}

    class _StubBackend:
        def deploy(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(
        runtime_commands, "ensure_bootstrap_workspace", lambda _: bootstrap
    )
    monkeypatch.setattr(
        runtime_commands, "_build_backend", lambda *a, **kw: _StubBackend()
    )
    monkeypatch.setattr(
        serve_config,
        "materialize_runtime_config",
        lambda *_a, **_kw: (_ for _ in ()).throw(
            AssertionError("Kubernetes must not materialize persistent local state")
        ),
    )

    result = CliRunner().invoke(
        main,
        [
            "serve",
            "--target",
            "k8s",
            "--config",
            str(config_path),
            "--algorithm",
            "multi_factor",
        ],
    )

    assert result.exit_code == 0, result.output
    effective = Path(captured["config_file"])
    assert effective == config_path
    assert (
        captured["config_document"]["routing"]["decisions"][0]["algorithm"]["type"]
        == "multi_factor"
    )
    assert "VLLM_SR_SOURCE_CONFIG_PATH" not in captured["env_vars"]
    assert "VLLM_SR_RUNTIME_CONFIG_PATH" not in captured["env_vars"]
    assert not (tmp_path / ".vllm-sr").exists()


def test_serve_help_describes_docker_only_runtime():
    runner = CliRunner()

    result = runner.invoke(main, ["serve", "--help"])

    assert result.exit_code == 0
    assert "Local Docker deployment" in result.output
    assert "Podman" not in result.output
    assert "--topology" not in result.output
    assert "--log-level" in result.output
    assert "latency_aware" in result.output
    assert "session_aware" not in result.output
    assert "--sim-image" in result.output
    assert "--recipe-env NAME" in result.output
    assert "router_r1" not in result.output
    assert "thompson" not in result.output


def test_source_config_keeps_legacy_env_passthrough_and_explicit_package_allowlist(
    monkeypatch, tmp_path: Path, caplog
):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "version": "v0.3",
                "listeners": [
                    {"name": "http-8899", "address": "0.0.0.0", "port": 8899}
                ],
                "providers": {
                    "models": [{"name": "custom", "api_key_env": "CUSTOM_API_KEY"}]
                },
            },
            sort_keys=False,
        )
    )
    bootstrap = BootstrapResult(
        config_path=config_path,
        output_dir=tmp_path / ".vllm-sr",
        setup_mode=False,
    )
    captured: dict[str, object] = {}

    class _StubBackend:
        def deploy(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(
        runtime_commands, "ensure_bootstrap_workspace", lambda _: bootstrap
    )
    monkeypatch.setattr(
        runtime_commands, "_build_backend", lambda *a, **kw: _StubBackend()
    )
    monkeypatch.setenv("CUSTOM_API_KEY", "never-print-this-value")

    legacy = CliRunner().invoke(main, ["serve", "--config", str(config_path)])
    assert legacy.exit_code == 0, legacy.output
    assert captured["env_vars"]["CUSTOM_API_KEY"] == "never-print-this-value"
    assert captured["env_vars"]["VLLM_SR_RECIPE_ENV_ALLOWLIST"] == ""
    assert "never-print-this-value" not in legacy.output
    captured.clear()

    store_dir = tmp_path / ".vllm-sr" / "recipe-store" / "vllm-sr"
    recipe_files = {
        "metadata.yaml": b"schema_version: vllm-sr/recipe-metadata/v1\nid: test\n",
        "config.yaml": config_path.read_bytes(),
        "probes.yaml": b"schema_version: vllm-sr/recipe-probes/v1\nname: test\n",
        "recipe.dsl": b'recipe "test" {}\n',
        "README.md": b"# Test\n",
    }
    recipe_digest = recipe_package.recipe_digest(recipe_files)
    object_dir = (
        store_dir / "objects" / "sha256" / recipe_digest.removeprefix("sha256:")
    )
    object_dir.mkdir(parents=True)
    for filename in recipe_package.RECIPE_FILES:
        (object_dir / filename).write_bytes(recipe_files[filename])
    runtime_bytes = (tmp_path / ".vllm-sr" / "runtime-config.yaml").read_bytes()
    store_dir.joinpath("active.json").write_text(
        json.dumps(
            {
                "schema_version": "vllm-sr/recipe-active/v1",
                "recipe_digest": recipe_digest,
                "config_digest": "sha256:"
                + hashlib.sha256(recipe_files["config.yaml"]).hexdigest(),
                "realized_config_digest": "sha256:"
                + hashlib.sha256(runtime_bytes).hexdigest(),
                "activated_at": "2026-08-12T04:01:00Z",
            }
        ),
        encoding="utf-8",
    )
    missing = CliRunner().invoke(main, ["serve", "--config", str(config_path)])
    assert missing.exit_code != 0
    assert "--recipe-env CUSTOM_API_KEY" in caplog.text
    assert "never-print-this-value" not in caplog.text
    assert captured == {}

    result = CliRunner().invoke(
        main,
        [
            "serve",
            "--config",
            str(config_path),
            "--recipe-env",
            "CUSTOM_API_KEY",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "never-print-this-value" not in result.output
    assert captured["env_vars"]["CUSTOM_API_KEY"] == "never-print-this-value"
    assert captured["env_vars"]["VLLM_SR_RECIPE_ENV_ALLOWLIST"] == "CUSTOM_API_KEY"


def test_active_package_is_validated_before_source_materialization(
    monkeypatch, tmp_path: Path
):
    initial = yaml.safe_dump(
        {
            "version": "v0.3",
            "listeners": [{"name": "http-8899", "address": "0.0.0.0", "port": 8899}],
        },
        sort_keys=False,
    ).encode()
    source = tmp_path / "config.yaml"
    source.write_bytes(initial)
    active = serve_config.materialize_runtime_config(source, initial)
    recipe_files = {
        "metadata.yaml": b"schema_version: vllm-sr/recipe-metadata/v1\nid: test\n",
        "config.yaml": initial,
        "probes.yaml": b"schema_version: vllm-sr/recipe-probes/v1\nname: test\n",
        "recipe.dsl": b'recipe "test" {}\n',
        "README.md": b"# Test\n",
    }
    recipe_digest = recipe_package.recipe_digest(recipe_files)
    store = tmp_path / ".vllm-sr" / "recipe-store" / "vllm-sr"
    object_dir = store / "objects" / "sha256" / recipe_digest.removeprefix("sha256:")
    object_dir.mkdir(parents=True)
    for filename, data in recipe_files.items():
        (object_dir / filename).write_bytes(data)
    (store / "active.json").write_text(
        json.dumps(
            {
                "schema_version": "vllm-sr/recipe-active/v1",
                "recipe_digest": recipe_digest,
                "config_digest": "sha256:" + hashlib.sha256(initial).hexdigest(),
                "realized_config_digest": "sha256:"
                + hashlib.sha256(initial).hexdigest(),
                "activated_at": "2026-08-12T04:01:00Z",
            }
        ),
        encoding="utf-8",
    )
    source.write_text(
        "version: v0.3\nlisteners:\n  - name: changed\n    port: 9999\n",
        encoding="utf-8",
    )
    bootstrap = BootstrapResult(
        config_path=source,
        output_dir=tmp_path / ".vllm-sr",
        setup_mode=False,
    )
    captured: dict[str, object] = {}

    class _StubBackend:
        def deploy(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(
        runtime_commands, "ensure_bootstrap_workspace", lambda _: bootstrap
    )
    monkeypatch.setattr(
        runtime_commands, "_build_backend", lambda *a, **kw: _StubBackend()
    )
    monkeypatch.setattr(
        serve_config,
        "build_effective_config_bytes",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("active package must not rebuild from source")
        ),
    )

    result = CliRunner().invoke(main, ["serve", "--config", str(source)])

    assert result.exit_code == 0, result.output
    assert active.read_bytes() == initial
    assert Path(captured["runtime_config_file"]) == active


def test_serve_passes_log_level_to_backend_env(monkeypatch, tmp_path: Path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "version": "v0.3",
                "listeners": [
                    {"name": "http-8899", "address": "0.0.0.0", "port": 8899}
                ],
                "routing": {"decisions": [{"name": "default"}]},
            },
            sort_keys=False,
        )
    )
    bootstrap = BootstrapResult(
        config_path=config_path,
        output_dir=tmp_path / ".vllm-sr",
        setup_mode=False,
    )
    captured: dict[str, object] = {}

    class _StubBackend:
        def deploy(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(
        runtime_commands, "ensure_bootstrap_workspace", lambda _: bootstrap
    )
    monkeypatch.setattr(
        runtime_commands, "_build_backend", lambda *a, **kw: _StubBackend()
    )

    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "serve",
            "--config",
            str(config_path),
            "--log-level",
            "debug",
            "--image-pull-policy",
            "never",
        ],
    )

    assert result.exit_code == 0
    assert captured["env_vars"]["SR_LOG_LEVEL"] == "debug"


def test_serve_keeps_observability_enabled_in_setup_mode(monkeypatch, tmp_path: Path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "version": "v0.3",
                "listeners": [
                    {"name": "http-8899", "address": "0.0.0.0", "port": 8899}
                ],
                "routing": {"decisions": [{"name": "default"}]},
                "setup": {"mode": True},
            },
            sort_keys=False,
        )
    )
    bootstrap = BootstrapResult(
        config_path=config_path,
        output_dir=tmp_path / ".vllm-sr",
        setup_mode=True,
    )
    captured: dict[str, object] = {}

    class _StubBackend:
        def deploy(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(
        runtime_commands, "ensure_bootstrap_workspace", lambda _: bootstrap
    )
    monkeypatch.setattr(
        runtime_commands, "_build_backend", lambda *a, **kw: _StubBackend()
    )

    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "serve",
            "--config",
            str(config_path),
            "--image-pull-policy",
            "never",
        ],
    )

    assert result.exit_code == 0
    assert captured["enable_observability"] is True


def test_serve_restart_uses_completed_runtime_instead_of_readonly_setup_source(
    monkeypatch, tmp_path: Path
):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "version": "v0.3",
                "listeners": [
                    {"name": "http-8899", "address": "0.0.0.0", "port": 8899}
                ],
                "setup": {"mode": True},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    active = serve_config.materialize_runtime_config(
        config_path, config_path.read_bytes()
    )
    completed = yaml.safe_dump(
        {
            "version": "v0.3",
            "listeners": [{"name": "http-8899", "address": "0.0.0.0", "port": 8899}],
            "routing": {"decisions": [{"name": "default"}]},
        },
        sort_keys=False,
    )
    active.write_text(completed, encoding="utf-8")
    bootstrap = BootstrapResult(
        config_path=config_path,
        output_dir=tmp_path / ".vllm-sr",
        setup_mode=True,
    )
    captured: dict[str, object] = {}

    class _StubBackend:
        def deploy(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(
        runtime_commands, "ensure_bootstrap_workspace", lambda _: bootstrap
    )
    monkeypatch.setattr(
        runtime_commands, "_build_backend", lambda *a, **kw: _StubBackend()
    )

    result = CliRunner().invoke(
        main,
        [
            "serve",
            "--config",
            str(config_path),
            "--minimal",
            "--image-pull-policy",
            "never",
        ],
    )

    assert result.exit_code == 0, result.output
    assert active.read_text(encoding="utf-8") == completed
    assert "VLLM_SR_SETUP_MODE" not in captured["env_vars"]
    assert "DASHBOARD_SETUP_MODE" not in captured["env_vars"]
    assert captured["env_vars"]["DISABLE_DASHBOARD"] == "true"


def test_serve_recovers_pending_config_before_choosing_setup_mode(
    monkeypatch, tmp_path: Path
):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "version": "v0.3",
                "listeners": [
                    {"name": "http-8899", "address": "0.0.0.0", "port": 8899}
                ],
                "routing": {"decisions": [{"name": "default"}]},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    bootstrap = BootstrapResult(
        config_path=config_path,
        output_dir=tmp_path / ".vllm-sr",
        setup_mode=False,
    )
    captured: dict[str, object] = {}

    class _StubBackend:
        def deploy(self, **kwargs):
            captured.update(kwargs)

    def recover_to_setup(**kwargs):
        Path(kwargs["runtime_config_path"]).write_text(
            yaml.safe_dump(
                {
                    "version": "v0.3",
                    "listeners": [
                        {
                            "name": "http-8899",
                            "address": "0.0.0.0",
                            "port": 8899,
                        }
                    ],
                    "setup": {"mode": True},
                },
                sort_keys=False,
            ),
            encoding="utf-8",
        )
        return True

    monkeypatch.setattr(
        runtime_commands, "ensure_bootstrap_workspace", lambda _: bootstrap
    )
    monkeypatch.setattr(
        serve_config,
        "recover_pending_recipe_activation_for_stack",
        recover_to_setup,
    )
    monkeypatch.setattr(
        runtime_commands, "_build_backend", lambda *a, **kw: _StubBackend()
    )

    result = CliRunner().invoke(
        main,
        [
            "serve",
            "--config",
            str(config_path),
            "--image-pull-policy",
            "never",
        ],
    )

    assert result.exit_code == 0, result.output
    assert captured["env_vars"]["VLLM_SR_SETUP_MODE"] == "true"
    assert captured["env_vars"]["DASHBOARD_SETUP_MODE"] == "true"


def test_inject_algorithm_into_config_updates_all_decisions(tmp_path: Path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "version": "v0.3",
                "routing": {
                    "decisions": [
                        {"name": "fast"},
                        {"name": "slow", "algorithm": {"type": "static"}},
                    ]
                },
            },
            sort_keys=False,
        )
    )

    rewritten_path = runtime_commands.inject_algorithm_into_config(
        config_path, "multi_factor"
    )

    with rewritten_path.open() as handle:
        rewritten = yaml.safe_load(handle)

    assert rewritten_path != config_path
    assert [
        decision["algorithm"]["type"] for decision in rewritten["routing"]["decisions"]
    ] == ["multi_factor", "multi_factor"]


def test_inject_algorithm_replaces_stale_type_specific_blocks(tmp_path: Path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "version": "v0.3",
                "routing": {
                    "decisions": [
                        {
                            "name": "fast",
                            "algorithm": {
                                "type": "hybrid",
                                "hybrid": {"cost_weight": 0.4},
                            },
                        },
                        {
                            "name": "slow",
                            "algorithm": {
                                "type": "rl_driven",
                                "rl_driven": {"storage_path": "/tmp/rl.json"},
                                "gmtrouter": {"storage_path": "/tmp/gmt.json"},
                                "bandit": {"strategy": "thompson"},
                                "personalization": {"per_user": True},
                            },
                        },
                    ]
                },
            },
            sort_keys=False,
        )
    )

    rewritten_path = runtime_commands.inject_algorithm_into_config(
        config_path, "multi_factor"
    )

    with rewritten_path.open() as handle:
        rewritten = yaml.safe_load(handle)

    algorithms = [
        decision["algorithm"] for decision in rewritten["routing"]["decisions"]
    ]
    assert algorithms == [
        {"type": "multi_factor"},
        {"type": "multi_factor"},
    ]


def test_inject_latency_aware_algorithm_keeps_matching_config_block(tmp_path: Path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "version": "v0.3",
                "listeners": [
                    {"name": "http-8899", "address": "0.0.0.0", "port": 8899}
                ],
                "routing": {
                    "decisions": [
                        {
                            "name": "latency",
                            "algorithm": {
                                "type": "hybrid",
                                "hybrid": {"experience_weight": 0.6},
                            },
                        },
                    ]
                },
            },
            sort_keys=False,
        )
    )

    rewritten_path = runtime_commands.inject_algorithm_into_config(
        config_path, "latency_aware"
    )

    with rewritten_path.open() as handle:
        rewritten = yaml.safe_load(handle)

    assert rewritten["routing"]["decisions"][0]["algorithm"] == {
        "type": "latency_aware",
        "latency_aware": {"tpot_percentile": 90, "ttft_percentile": 95},
    }


def test_inject_workflows_algorithm_keeps_matching_config_block(tmp_path: Path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "version": "v0.3",
                "routing": {
                    "decisions": [
                        {
                            "name": "flow",
                            "modelRefs": [{"model": "worker-a"}],
                            "algorithm": {
                                "type": "hybrid",
                                "hybrid": {"experience_weight": 0.6},
                            },
                        },
                    ]
                },
            },
            sort_keys=False,
        )
    )

    rewritten_path = runtime_commands.inject_algorithm_into_config(
        config_path, "workflows"
    )

    with rewritten_path.open() as handle:
        rewritten = yaml.safe_load(handle)

    assert rewritten["routing"]["decisions"][0]["algorithm"] == {
        "type": "workflows",
        "workflows": {
            "template": "micro_agent",
            "mode": "static",
            "roles": [{"name": "worker", "models": ["worker-a"]}],
        },
    }

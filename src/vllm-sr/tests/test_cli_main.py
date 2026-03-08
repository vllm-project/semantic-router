from pathlib import Path

import yaml
from cli.bootstrap import BootstrapResult
from cli.commands import runtime as runtime_commands
from cli.main import main
from click.testing import CliRunner


def test_cli_help_lists_registered_commands():
    runner = CliRunner()

    result = runner.invoke(main, ["--help"])

    assert result.exit_code == 0
    for command_name in (
        "init",
        "serve",
        "config",
        "validate",
        "status",
        "logs",
        "stop",
        "dashboard",
    ):
        assert command_name in result.output


def test_inject_algorithm_into_config_updates_all_decisions(tmp_path: Path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "version": "v0.1",
                "decisions": [
                    {"name": "fast"},
                    {"name": "slow", "algorithm": {"type": "static"}},
                ],
            },
            sort_keys=False,
        )
    )

    rewritten_path = runtime_commands.inject_algorithm_into_config(config_path, "elo")

    with rewritten_path.open() as handle:
        rewritten = yaml.safe_load(handle)

    assert rewritten_path != config_path
    assert [decision["algorithm"]["type"] for decision in rewritten["decisions"]] == [
        "elo",
        "elo",
    ]


def test_serve_uses_algorithm_translated_config(monkeypatch, tmp_path: Path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "version": "v0.1",
                "listeners": [
                    {"name": "http-8899", "address": "0.0.0.0", "port": 8899}
                ],
                "decisions": [{"name": "default"}],
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

    monkeypatch.setattr(
        runtime_commands, "ensure_bootstrap_workspace", lambda _: bootstrap
    )
    monkeypatch.setattr(
        runtime_commands,
        "start_vllm_sr",
        lambda **kwargs: captured.update(kwargs),
    )

    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "serve",
            "--config",
            str(config_path),
            "--algorithm",
            "elo",
            "--image-pull-policy",
            "never",
        ],
    )

    assert result.exit_code == 0
    with Path(captured["config_file"]).open() as handle:
        translated = yaml.safe_load(handle)
    assert translated["decisions"][0]["algorithm"]["type"] == "elo"
    assert captured["pull_policy"] == "never"

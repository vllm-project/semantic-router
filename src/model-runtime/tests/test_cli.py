import pytest
from vllm_sr_runtime.cli import build_parser, config_from_args, main


def test_serve_arguments_map_to_the_config():
    args = build_parser().parse_args(
        [
            "serve",
            "vllm-sr/Decision-2.0-Kai-0.6B",
            "--revision",
            "cd49ea38",
            "--device",
            "rocm:1",
            "--uds",
            "/tmp/x.sock",
            "--profile",
            "shared_context",
            "--accept-licence",
            "cc-by-4.0",
        ]
    )
    config = config_from_args(args)
    assert (
        config.model == "vllm-sr/Decision-2.0-Kai-0.6B"
        and config.revision == "cd49ea38"
    )
    assert (
        config.device == "rocm:1"
        and config.uds == "/tmp/x.sock"
        and config.profile == "shared_context"
    )
    assert config.host == "127.0.0.1" and config.accept_licences == ("cc-by-4.0",)


def test_unknown_profile_is_rejected():
    with pytest.raises(SystemExit):
        build_parser().parse_args(["serve", "m", "--profile", "turbo"])


def test_models_and_fixture_commands(capsys, tmp_path):
    assert main(["models"]) == 0
    out = capsys.readouterr().out
    assert (
        "vllm-sr/Decision-2.0-Vega-27B@7aec49ae11a18741706da549ab626b9052795fe7" in out
    )
    assert main(["fixture", str(tmp_path / "pkg"), "--backbone", "qwen3"]) == 0
    assert (tmp_path / "pkg" / "MODEL_MANIFEST.json").is_file()

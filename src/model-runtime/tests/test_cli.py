import os

import pytest
from vllm_srun import cli
from vllm_srun.cli import build_parser, config_from_args, main
from vllm_srun.plugins import registry


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
    (model,) = config.models
    assert (model.model, model.revision) == (
        "vllm-sr/Decision-2.0-Kai-0.6B",
        "cd49ea38",
    )
    assert model.device == "rocm:1" and model.profile == "shared_context"
    assert config.uds == "/tmp/x.sock" and config.host == "127.0.0.1"
    assert config.accept_licences == ("cc-by-4.0",)


def test_profiles_are_any_registered_profile_plugin(monkeypatch):
    with pytest.raises(SystemExit, match="unknown profile 'turbo'"):
        config_from_args(
            build_parser().parse_args(["serve", "m", "--profile", "turbo"])
        )
    names = registry.names
    monkeypatch.setattr(
        registry,
        "names",
        lambda kind: [*names(kind), "turbo"] if kind == "profiles" else names(kind),
    )
    args = build_parser().parse_args(["serve", "m", "--profile", "turbo"])
    assert config_from_args(args).models[0].profile == "turbo"


def test_models_and_fixture_commands(capsys, tmp_path):
    assert main(["models"]) == 0
    out = capsys.readouterr().out
    assert (
        "vllm-sr/Decision-2.0-Vega-27B@7aec49ae11a18741706da549ab626b9052795fe7" in out
    )
    assert main(["fixture", str(tmp_path / "pkg"), "--variant", "qwen3"]) == 0
    assert (tmp_path / "pkg" / "MODEL_MANIFEST.json").is_file()
    assert (
        main(
            [
                "fixture",
                str(tmp_path / "hybrid"),
                "--family",
                "decision2",
                "--variant",
                "qwen3_5",
            ]
        )
        == 0
    )
    assert (tmp_path / "hybrid" / "MODEL_MANIFEST.json").is_file()
    for family in ("nobody", "fixtures", "omni"):
        with pytest.raises(ValueError, match="no fixture writer"):
            main(["fixture", str(tmp_path / family), "--family", family])


BUILTIN_FAMILIES = (
    "decision1",
    "decision2",
    "multimodal_embedding",
    "task_heads",
    "vela2",
)


@pytest.mark.parametrize("family", BUILTIN_FAMILIES)
def test_every_built_in_family_writes_fixtures_it_detects(family, tmp_path):
    import importlib

    from vllm_srun.plugins.base import PackageRef

    assert set(BUILTIN_FAMILIES) <= set(registry.names("families"))
    module = importlib.import_module(f"vllm_srun.testing.{family}")
    loaded = registry.plugin("families", family).load()()
    for variant in module.VARIANTS:
        output = tmp_path / variant
        args = ["fixture", str(output), "--family", family, "--variant", variant]
        assert main(args) == 0
        assert loaded.detect(PackageRef(output)), variant


def test_several_models_share_the_process_options(tmp_path):
    args = build_parser().parse_args(
        [
            "serve",
            "vllm-sr/Decision-2.0-Kai-0.6B@cd49ea38",
            str(tmp_path),
            "--device",
            "cpu",
        ]
    )
    config = config_from_args(args)
    assert len(config.models) == 2
    kai, local = config.models
    assert (kai.model, kai.revision, kai.device) == (
        "vllm-sr/Decision-2.0-Kai-0.6B",
        "cd49ea38",
        "cpu",
    )
    assert (local.model, local.revision) == (str(tmp_path), None)
    (single,) = config_from_args(
        build_parser().parse_args(["serve", "acme/model@" + "a" * 40])
    ).models
    assert single.model == "acme/model" and single.revision == "a" * 40


def test_a_models_file_lists_each_model(tmp_path):
    path = tmp_path / "models.yaml"
    path.write_text(
        "models:\n"
        "  - {model: vllm-sr/Decision-2.0-Kai-0.6B, name: kai, device: cpu}\n"
        "  - {model: /models/pii, name: pii, profile: exact, options: {head: default}}\n"
    )
    config = config_from_args(
        build_parser().parse_args(["serve", "--models", str(path)])
    )
    assert [model.name for model in config.models] == ["kai", "pii"]
    assert config.models[1].options == {"head": "default"}
    assert config.served_models() == config.models


@pytest.mark.parametrize(
    "content,message",
    [
        ("models: []\n", "nonempty"),
        ("models:\n  - {name: x}\n", "model"),
        ("models:\n  - {model: a, colour: red}\n", "unknown fields"),
        ("models:\n  - {model: a, name: x}\n  - {model: b, name: x}\n", "duplicate"),
    ],
)
def test_malformed_models_files_are_refused(tmp_path, content, message):
    path = tmp_path / "models.yaml"
    path.write_text(content)
    with pytest.raises(ValueError, match=message):
        config_from_args(build_parser().parse_args(["serve", "--models", str(path)]))


@pytest.mark.parametrize(
    "argv",
    [
        ["serve"],
        ["serve", "a", "b", "--revision", "c" * 40],
        ["serve", "a", "--models", "m.yaml"],
        ["serve", "a@" + "b" * 40, "--revision", "c" * 40],
    ],
)
def test_conflicting_model_arguments_are_refused(argv):
    with pytest.raises(SystemExit):
        config_from_args(build_parser().parse_args(argv))


def test_a_uid_without_a_passwd_entry_gets_a_user_home_and_compile_cache(
    monkeypatch, tmp_path
):
    def missing(uid):
        raise KeyError(f"getpwuid(): uid not found: {uid}")

    monkeypatch.setattr(cli.pwd, "getpwuid", missing)
    monkeypatch.setattr(cli.tempfile, "gettempdir", lambda: str(tmp_path))
    monkeypatch.delenv("USER", raising=False)
    monkeypatch.delenv("TORCHINDUCTOR_CACHE_DIR", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path / "missing"))
    cli.default_identity()
    assert os.environ["USER"] == "vllm-srun"
    assert os.environ["HOME"] == str(tmp_path)
    assert os.environ["TORCHINDUCTOR_CACHE_DIR"] == str(tmp_path / "torchinductor")


def test_a_known_uid_keeps_its_environment(monkeypatch):
    monkeypatch.setattr(cli.pwd, "getpwuid", lambda uid: object())
    monkeypatch.delenv("TORCHINDUCTOR_CACHE_DIR", raising=False)
    monkeypatch.setenv("HOME", "/")
    cli.default_identity()
    assert os.environ["HOME"] == "/" and "TORCHINDUCTOR_CACHE_DIR" not in os.environ

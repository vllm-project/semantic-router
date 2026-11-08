import os
import stat
from pathlib import Path

import pytest
import yaml
from cli.commands import runtime_paths
from cli.commands.runtime_materialize import main
from cli.commands.runtime_support import realize_runtime_config


def _write_source(path: Path) -> bytes:
    raw = yaml.safe_dump(
        {
            "version": "v0.5",
            "recipes": [
                {
                    "name": "default",
                    "decisions": [
                        {
                            "name": "default",
                            "priority": 0,
                            "modelRefs": [{"model": "test-model"}],
                        }
                    ],
                }
            ],
            "global": {
                "model_catalog": {
                    "modules": {"classifier": {"domain": {"use_cpu": True}}}
                }
            },
        },
        sort_keys=False,
    ).encode("utf-8")
    path.write_bytes(raw)
    return raw


def _domain_use_cpu(realized: dict) -> bool:
    return realized["global"]["model_catalog"]["modules"]["classifier"]["domain"][
        "use_cpu"
    ]


def test_realize_runtime_config_applies_overrides_without_mutating_source(
    tmp_path: Path,
):
    source = tmp_path / "raw.yaml"
    original = _write_source(source)
    target = tmp_path / "state" / "runtime-config.yaml"

    result = realize_runtime_config(
        source, target, algorithm="multi_factor", platform="rocm"
    )

    assert result == target
    assert source.read_bytes() == original
    realized = yaml.safe_load(target.read_text(encoding="utf-8"))
    assert realized["recipes"][0]["decisions"][0]["algorithm"]["type"] == "multi_factor"
    assert _domain_use_cpu(realized) is False
    assert stat.S_IMODE(target.stat().st_mode) == 0o600


def test_realize_runtime_config_strict_atomic_replace(tmp_path: Path, monkeypatch):
    source = tmp_path / "raw.yaml"
    _write_source(source)
    target = tmp_path / "state" / "runtime-config.yaml"
    target.parent.mkdir()
    target.write_text("old\n", encoding="utf-8")
    observed: dict[str, Path] = {}
    real_replace = os.replace

    def record_replace(source_path, target_path):
        observed["source"] = Path(source_path)
        observed["target"] = Path(target_path)
        real_replace(source_path, target_path)

    monkeypatch.setattr(runtime_paths.os, "replace", record_replace)

    realize_runtime_config(source, target)

    assert observed["source"].parent == target.parent
    assert observed["target"] == target
    assert not list(target.parent.glob(".*.tmp"))


def test_runtime_materialize_module_cli(tmp_path: Path, capsys):
    source = tmp_path / "raw.yaml"
    original = _write_source(source)
    target = tmp_path / "state" / "runtime-config.yaml"

    exit_code = main(
        [
            "--source",
            str(source),
            "--target",
            str(target),
            "--algorithm",
            "multi_factor",
            "--platform",
            "rocm",
        ]
    )

    assert exit_code == 0
    assert capsys.readouterr().out.strip() == str(target)
    assert source.read_bytes() == original
    assert target.is_file()


def test_runtime_materialize_module_uses_current_runtime_env(
    tmp_path: Path, monkeypatch, capsys
):
    source = tmp_path / "raw.yaml"
    _write_source(source)
    target = tmp_path / "state" / "runtime-config.yaml"
    monkeypatch.setenv("VLLM_SR_ALGORITHM_OVERRIDE", "multi_factor")
    monkeypatch.setenv("DASHBOARD_PLATFORM", "rocm")

    assert main(["--source", str(source), "--target", str(target)]) == 0

    assert capsys.readouterr().out.strip() == str(target)
    realized = yaml.safe_load(target.read_text(encoding="utf-8"))
    assert realized["recipes"][0]["decisions"][0]["algorithm"]["type"] == "multi_factor"
    assert _domain_use_cpu(realized) is False


def test_managed_materialization_realizes_empty_imported_listener(tmp_path: Path):
    source = tmp_path / "imported.yaml"
    _write_source(source)
    document = yaml.safe_load(source.read_bytes())
    # Dashboard's typed import emits an empty mapping for an omitted listener.
    document["global"]["services"] = {"management_api": {}}
    source.write_text(yaml.safe_dump(document), encoding="utf-8")
    original = source.read_bytes()
    target = tmp_path / "state" / "runtime-config.yaml"

    assert (
        main(["--source", str(source), "--target", str(target), "--managed-listener"])
        == 0
    )

    realized = yaml.safe_load(target.read_bytes())
    assert realized["global"]["services"]["management_api"] == {
        "bind_address": "0.0.0.0",
        "port": 8080,
    }
    assert source.read_bytes() == original
    assert not (target.parent / ".vllm-sr").exists()


@pytest.mark.parametrize(
    ("management", "error"),
    [
        ({"bind_address": "127.0.0.1"}, "split Docker requires"),
        ({"bind_address": "0.0.0.0", "port": 50051}, "conflicts"),
        (
            {"bind_address": "0.0.0.0", "remote_exposure": True},
            "requires bearer auth tokens",
        ),
    ],
)
def test_managed_materialization_rejects_before_publishing(
    tmp_path: Path, management: dict, error: str
):
    source = tmp_path / "imported.yaml"
    _write_source(source)
    document = yaml.safe_load(source.read_bytes())
    document["global"]["services"] = {"management_api": management}
    source.write_text(yaml.safe_dump(document), encoding="utf-8")
    original = source.read_bytes()
    target = tmp_path / "runtime-config.yaml"
    active = b"version: v0.5\nsetup:\n  mode: true\n"
    target.write_bytes(active)

    with pytest.raises(ValueError, match=error):
        realize_runtime_config(source, target, managed_listener=True)

    assert target.read_bytes() == active
    assert source.read_bytes() == original


def test_runtime_materialize_keeps_authored_gpu_embedding_budget(
    tmp_path: Path, monkeypatch
):
    monkeypatch.setenv("DASHBOARD_PLATFORM", "rocm")
    monkeypatch.delenv("VLLM_SR_ROCM_PRESERVE_CPU", raising=False)
    monkeypatch.delenv("VLLM_SR_ROCM_FORCE_GPU", raising=False)
    source = tmp_path / "raw.yaml"
    _write_source(source)
    document = yaml.safe_load(source.read_text())
    deployments = {
        "gpu-embedding": {
            "provider": "model_runtime",
            "artifact": "vllm-sr/Vela-1.0-Encoder-307M-Embedding",
            "device": "rocm:0",
            "input": {"max_tokens": 128, "overflow": "reject"},
        }
    }
    document["global"]["model_catalog"]["deployments"] = deployments
    document["global"]["model_catalog"]["embeddings"] = {"default": "gpu-embedding"}
    source.write_text(yaml.safe_dump(document), encoding="utf-8")
    original = source.read_bytes()
    target = tmp_path / "state" / "runtime-config.yaml"

    assert main(["--source", str(source), "--target", str(target)]) == 0

    realized = yaml.safe_load(target.read_text())
    catalog = realized["global"]["model_catalog"]
    assert catalog["embeddings"] == {"default": "gpu-embedding"}
    assert catalog["deployments"] == deployments
    assert source.read_bytes() == original

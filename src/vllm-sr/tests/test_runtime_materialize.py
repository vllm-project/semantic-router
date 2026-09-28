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
            "version": "v0.3",
            "routing": {
                "decisions": [
                    {
                        "name": "default",
                        "priority": 0,
                        "modelRefs": [{"model": "test-model"}],
                    }
                ]
            },
            "global": {
                "model_catalog": {
                    "embeddings": {
                        "semantic": {
                            "use_cpu": True,
                            "embedding_config": {"model_type": "mmbert"},
                        }
                    }
                }
            },
        },
        sort_keys=False,
    ).encode("utf-8")
    path.write_bytes(raw)
    return raw


def _write_kb_source(path: Path, source_path: str) -> bytes:
    raw = yaml.safe_dump(
        {
            "version": "v0.3",
            "global": {
                "model_catalog": {
                    "kbs": [
                        {
                            "name": "privacy_kb",
                            "source": {
                                "path": source_path,
                                "manifest": "labels.json",
                            },
                        }
                    ]
                }
            },
        },
        sort_keys=False,
    ).encode("utf-8")
    path.write_bytes(raw)
    return raw


def test_realize_runtime_config_applies_overrides_without_mutating_source(
    tmp_path: Path,
):
    source = tmp_path / "raw.yaml"
    original = _write_source(source)
    target = tmp_path / "state" / "runtime-config.yaml"

    result = realize_runtime_config(
        source, target, algorithm="multi_factor", platform="amd"
    )

    assert result == target
    assert source.read_bytes() == original
    realized = yaml.safe_load(target.read_text(encoding="utf-8"))
    assert realized["routing"]["decisions"][0]["algorithm"]["type"] == "multi_factor"
    assert (
        realized["global"]["model_catalog"]["embeddings"]["semantic"]["use_cpu"] is True
    )
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
            "amd",
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
    monkeypatch.setenv("DASHBOARD_PLATFORM", "amd")

    assert main(["--source", str(source), "--target", str(target)]) == 0

    assert capsys.readouterr().out.strip() == str(target)
    realized = yaml.safe_load(target.read_text(encoding="utf-8"))
    assert realized["routing"]["decisions"][0]["algorithm"]["type"] == "multi_factor"
    assert (
        realized["global"]["model_catalog"]["embeddings"]["semantic"]["use_cpu"] is True
    )


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
    active = b"version: v0.3\nsetup:\n  mode: true\n"
    target.write_bytes(active)

    with pytest.raises(ValueError, match=error):
        realize_runtime_config(source, target, managed_listener=True)

    assert target.read_bytes() == active
    assert source.read_bytes() == original


def test_runtime_materialize_keeps_authored_gpu_embedding_budget(
    tmp_path: Path, monkeypatch
):
    monkeypatch.setenv("DASHBOARD_PLATFORM", "amd")
    monkeypatch.delenv("VLLM_SR_AMD_PRESERVE_CPU", raising=False)
    monkeypatch.delenv("VLLM_SR_AMD_FORCE_GPU", raising=False)
    source = tmp_path / "raw.yaml"
    _write_source(source)
    document = yaml.safe_load(source.read_text())
    deployments = {
        "gpu-embedding": {
            "artifact": "models/mmbert-embedding",
            "provider": "ort",
            "device": "migraphx:0",
            "precision": "native",
            "input": {"max_tokens": 128, "overflow": "reject"},
        }
    }
    bindings = {
        "embedding": {
            "deployment": "gpu-embedding",
            "contract": "embedding.v1",
            "adapter": "mmbert",
        }
    }
    document["global"]["model_catalog"]["deployments"] = deployments
    document["routing"]["model_bindings"] = bindings
    source.write_text(yaml.safe_dump(document), encoding="utf-8")
    original = source.read_bytes()
    target = tmp_path / "state" / "runtime-config.yaml"

    assert main(["--source", str(source), "--target", str(target)]) == 0

    realized = yaml.safe_load(target.read_text())
    catalog = realized["global"]["model_catalog"]
    assert catalog["embeddings"]["semantic"]["use_cpu"] is True
    assert catalog["deployments"] == deployments
    assert realized["routing"]["model_bindings"] == bindings
    assert source.read_bytes() == original


@pytest.mark.parametrize(
    "source_path",
    ["../dashboard-data", "/app/.vllm-sr", "knowledge_bases/../dashboard-data"],
)
def test_package_activation_rejects_unsafe_kb_paths_without_side_effects(
    tmp_path: Path, source_path: str
):
    state = tmp_path / ".vllm-sr"
    state.mkdir()
    raw = state / "raw.yaml"
    _write_kb_source(raw, source_path)
    protected = state / "dashboard-data"
    protected.mkdir()
    marker = protected / "session.db"
    marker.write_bytes(b"do-not-copy")
    before = {path.relative_to(state) for path in state.rglob("*")}
    target = state / "runtime-config.yaml"

    with pytest.raises(ValueError, match=r"Knowledge-base source\.path"):
        realize_runtime_config(raw, target, package_activation=True)

    assert not target.exists()
    assert {path.relative_to(state) for path in state.rglob("*")} == before
    assert marker.read_bytes() == b"do-not-copy"


def test_managed_preflight_preserves_kb_without_bootstrapping(tmp_path: Path):
    source = tmp_path / "imported.yaml"
    _write_kb_source(source, "remote-kb/")
    kb_source = tmp_path / "remote-kb"
    kb_source.mkdir()
    (kb_source / "labels.json").write_text('{"labels": []}')
    original = source.read_bytes()
    before = set(tmp_path.rglob("*"))
    target = tmp_path / "prepared.yaml"

    assert (
        main(
            [
                "--source",
                str(source),
                "--target",
                str(target),
                "--managed-listener",
                "--skip-kb-bootstrap",
            ]
        )
        == 0
    )

    prepared = yaml.safe_load(target.read_bytes())
    assert prepared["global"]["model_catalog"]["kbs"][0]["source"]["path"] == (
        "remote-kb/"
    )
    assert set(tmp_path.rglob("*")) == before | {target}
    assert source.read_bytes() == original

    realize_runtime_config(source, target, managed_listener=True)

    copied = tmp_path / ".vllm-sr" / "knowledge_bases" / "remote-kb" / "labels.json"
    assert copied.read_bytes() == (kb_source / "labels.json").read_bytes()
    realized = yaml.safe_load(target.read_bytes())
    assert realized["global"]["model_catalog"]["kbs"][0]["source"]["path"] == (
        "knowledge_bases/remote-kb/"
    )


def test_managed_listener_rejection_does_not_bootstrap_kb(tmp_path: Path):
    source = tmp_path / "imported.yaml"
    _write_kb_source(source, "remote-kb/")
    document = yaml.safe_load(source.read_bytes())
    document["global"]["services"] = {"management_api": {"bind_address": "127.0.0.1"}}
    source.write_text(yaml.safe_dump(document), encoding="utf-8")
    before = set(tmp_path.rglob("*"))

    with pytest.raises(ValueError, match="split Docker requires"):
        realize_runtime_config(
            source, tmp_path / "runtime-config.yaml", managed_listener=True
        )

    assert set(tmp_path.rglob("*")) == before


def test_package_activation_keeps_canonical_kb_reference_side_effect_free(
    tmp_path: Path,
):
    state = tmp_path / ".vllm-sr"
    state.mkdir()
    raw = state / "raw.yaml"
    _write_kb_source(raw, "knowledge_bases/privacy/")
    target = state / "runtime-config.yaml"

    realize_runtime_config(raw, target, package_activation=True)

    realized = yaml.safe_load(target.read_text(encoding="utf-8"))
    assert (
        realized["global"]["model_catalog"]["kbs"][0]["source"]["path"]
        == "knowledge_bases/privacy/"
    )
    assert not (state / "knowledge_bases").exists()


def test_runtime_materialize_cli_package_activation_is_side_effect_free(
    tmp_path: Path, capsys
):
    state = tmp_path / ".vllm-sr"
    state.mkdir()
    raw = state / "raw.yaml"
    _write_kb_source(raw, "knowledge_bases/privacy/")
    target = state / "runtime-config.yaml"

    assert (
        main(
            [
                "--source",
                str(raw),
                "--target",
                str(target),
                "--package-activation",
            ]
        )
        == 0
    )

    assert capsys.readouterr().out.strip() == str(target)
    assert not (state / "knowledge_bases").exists()

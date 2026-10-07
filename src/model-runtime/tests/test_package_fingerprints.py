"""Fingerprint keys must use POSIX separators on every host."""

import json
import shutil
from pathlib import Path, PureWindowsPath

import pytest
from vllm_srun.families.decision2 import package as pkg
from vllm_srun.registry.artifacts import sha256_file, sha256_json


class WindowsRelativePath(type(Path())):
    """Use host file IO while rendering relative paths as Windows paths."""

    def relative_to(self, *args, **kwargs):
        return PureWindowsPath(super().relative_to(*args, **kwargs).as_posix())


@pytest.mark.parametrize("path_type", [Path, WindowsRelativePath])
@pytest.mark.parametrize("fixture_name", ["qwen3_package", "adapter_package"])
def test_checkpoint_identity_uses_posix_keys(request, path_type, fixture_name):
    root = request.getfixturevalue(fixture_name)
    adapter = fixture_name == "adapter_package"
    names = {
        "decision_config.json",
        "decision_head.safetensors",
        "tokenizer.json",
        "tokenizer_config.json",
    }
    names.update(
        {"adapter/adapter_config.json", "adapter/adapter_model.safetensors"}
        if adapter
        else {"backbone/config.json", "backbone/model.safetensors"}
    )
    # Independent, literal keys avoid generating expectations with the verifier.
    expected = {name: sha256_file(root / name) for name in names}
    assert pkg.checkpoint_files(path_type(root)) == expected

    decision_config = json.loads((root / "decision_config.json").read_text())
    base_root = None
    if adapter:
        base_root = root.parent / f"{root.name}-base"
        source = {
            name: sha256_file(base_root / name)
            for name in ("config.json", "model-00001-of-00001.safetensors")
        }
        assert pkg.source_files(path_type(base_root)) == source
        expected = {
            **{f"checkpoint/{name}": digest for name, digest in expected.items()},
            **{f"source/{name}": digest for name, digest in source.items()},
        }
        base_root = path_type(base_root)
    assert pkg.model_identity(
        path_type(root), decision_config, base_root
    ) == sha256_json(expected)


@pytest.mark.parametrize("path_type", [Path, WindowsRelativePath])
def test_nested_source_fingerprint_uses_posix_keys(
    adapter_package, tmp_path, path_type
):
    base_root = adapter_package.parent / f"{adapter_package.name}-base"
    shutil.copytree(base_root, tmp_path / "backbone")
    expected = {
        f"backbone/{name}": sha256_file(base_root / name)
        for name in ("config.json", "model-00001-of-00001.safetensors")
    }
    assert pkg.source_files(path_type(tmp_path)) == expected

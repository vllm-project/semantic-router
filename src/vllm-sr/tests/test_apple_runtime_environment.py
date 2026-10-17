"""Image extras and environment markers belong to the native target, not Linux."""

import json
import sys
from importlib import metadata
from pathlib import Path

import pytest
from cli import apple_runtime_environment as environment


class ImageDistribution:
    version = "0.2.0"
    _path = Path("/image/vllm_srun.dist-info")
    requires = (
        "base>=1",
        "mac-only>=1; sys_platform == 'darwin'",
        "linux-only>=1; sys_platform == 'linux'",
        "pillow>=1; extra == 'multimodal'",
        "opencv-python-headless>=1; extra == 'multimodal' and platform_system == 'Darwin'",
        "test-only>=1; extra == 'test'",
        "old-python>=1; python_version < '3.10'",
    )

    def locate_file(self, _name):
        return Path("/image/vllm_srun")


def test_image_extras_are_evaluated_for_managed_macos_python(monkeypatch, capsys):
    monkeypatch.setattr(metadata, "distribution", lambda name: ImageDistribution())
    monkeypatch.setattr(metadata, "version", lambda name: "1.0.0")
    monkeypatch.setenv("VLLM_SRUN_EXTRAS", "multimodal")
    target = environment.target_marker_environment()
    monkeypatch.setattr(sys, "argv", ["-c", json.dumps(target)])
    exec(environment.IMAGE_METADATA, {})
    result = json.loads(capsys.readouterr().out)
    assert result["requirements"] == [
        "base==1.0.0",
        "mac-only==1.0.0",
        "opencv-python-headless==1.0.0",
        "pillow==1.0.0",
    ]
    assert result["extras"] == ["multimodal"]
    assert result["target"]["platform_machine"] == "arm64"
    assert result["target"]["python_full_version"] == environment.PYTHON_VERSION


def test_missing_image_extras_manifest_requires_a_matching_rebuild(monkeypatch):
    monkeypatch.setattr(metadata, "distribution", lambda name: ImageDistribution())
    monkeypatch.delenv("VLLM_SRUN_EXTRAS", raising=False)
    monkeypatch.setattr(
        sys, "argv", ["-c", json.dumps(environment.target_marker_environment())]
    )
    with pytest.raises(ValueError, match="rebuild the image"):
        exec(environment.IMAGE_METADATA, {})

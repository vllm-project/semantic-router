"""Model selection and input policies survive CLI runtime materialization."""

from pathlib import Path

import pytest
import yaml
from cli.commands.runtime_support import realize_runtime_config

REPO_ROOT = Path(__file__).resolve().parents[3]


def _without_use_cpu(value):
    if isinstance(value, dict):
        return {
            key: _without_use_cpu(item)
            for key, item in value.items()
            if key != "use_cpu"
        }
    return value


@pytest.mark.parametrize("platform", ["cpu", "rocm", "cuda"])
def test_reference_vela_policy_survives_runtime_materialization(
    tmp_path: Path, monkeypatch, platform: str
):
    for name in ("AMD", "NVIDIA"):
        monkeypatch.delenv(f"VLLM_SR_{name}_FORCE_GPU", raising=False)
        monkeypatch.delenv(f"VLLM_SR_{name}_PRESERVE_CPU", raising=False)
    source = REPO_ROOT / "config" / "config.yaml"
    original = source.read_bytes()
    authored = yaml.safe_load(original)["global"]["model_catalog"]
    target = tmp_path / "runtime-config.yaml"
    realize_runtime_config(source, target, algorithm=None, platform=platform)
    assert source.read_bytes() == original
    catalog = yaml.safe_load(target.read_text())["global"]["model_catalog"]
    assert catalog["system"] == authored["system"]
    for section in ("embeddings", "modules"):
        assert _without_use_cpu(catalog[section]) == _without_use_cpu(authored[section])
    assert catalog["embeddings"] == {"default": {"deployment": "vela-embedding"}}
    # The default judgment binding selects a declared logical deployment;
    # platform materialization must preserve its resource and input policy.
    assert catalog["system"]["decision_model"] == {"deployment": "primary"}
    assert catalog["deployments"] == authored["deployments"]
    assert catalog["deployments"]["primary"]["artifact"] == "vllm-sr/vela3-fast"

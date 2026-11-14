"""Model selection and input policies survive CLI runtime materialization."""

from pathlib import Path

import yaml
from cli.commands.runtime_support import realize_runtime_config

REPO_ROOT = Path(__file__).resolve().parents[3]


def test_reference_vela_policy_survives_runtime_materialization(tmp_path: Path):
    source = REPO_ROOT / "config" / "config.yaml"
    original = source.read_bytes()
    authored = yaml.safe_load(original)["global"]["model_catalog"]
    target = tmp_path / "runtime-config.yaml"
    realize_runtime_config(source, target, algorithm=None)
    assert source.read_bytes() == original
    realized = yaml.safe_load(target.read_text())["global"]
    catalog = realized["model_catalog"]
    assert catalog["system"] == authored["system"]
    assert catalog["embeddings"] == authored["embeddings"]
    assert catalog["embeddings"] == {"default": {"deployment": "vela-embedding"}}
    # The signals' prompt compression is compress.signals of a global plugin.
    assert realized["plugins"] == yaml.safe_load(original)["global"]["plugins"]
    # The default judgment binding selects a declared logical deployment;
    # runtime materialization must preserve its resource and input policy.
    assert catalog["system"]["decision_model"] == {"deployment": "primary"}
    assert catalog["deployments"] == authored["deployments"]
    assert catalog["deployments"]["primary"]["artifact"] == "vllm-sr/vela3-fast"

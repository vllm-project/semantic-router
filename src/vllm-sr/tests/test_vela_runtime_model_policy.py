"""Model selection and input policies survive CLI runtime materialization."""

from pathlib import Path

import pytest
import yaml
from cli.commands.runtime_support import realize_runtime_config

REPO_ROOT = Path(__file__).resolve().parents[3]


@pytest.mark.parametrize("platform", ["cpu", "amd", "nvidia"])
def test_reference_vela_policy_survives_runtime_materialization(
    tmp_path: Path, monkeypatch, platform: str
):
    for name in ("AMD", "NVIDIA"):
        monkeypatch.delenv(f"VLLM_SR_{name}_FORCE_GPU", raising=False)
        monkeypatch.delenv(f"VLLM_SR_{name}_PRESERVE_CPU", raising=False)
    source = REPO_ROOT / "config" / "config.yaml"
    original = source.read_bytes()
    target = tmp_path / "runtime-config.yaml"
    realize_runtime_config(source, target, algorithm=None, platform=platform)
    assert source.read_bytes() == original
    catalog = yaml.safe_load(target.read_text())["global"]["model_catalog"]
    assert catalog["system"]["domain_classifier"] == (
        "models/Vela-1.0-Encoder-307M-Domain"
    )
    assert catalog["system"]["prompt_guard"] == "models/Vela-1.0-Encoder-307M-Guard"
    semantic = catalog["embeddings"]["semantic"]
    assert semantic["mmbert_model_path"] == "models/Vela-1.0-Encoder-307M-Embedding"
    assert semantic["embedding_config"]["full_context"] is False
    modules = catalog["modules"]
    fact_check = modules["hallucination_mitigation"]["fact_check"]
    assert fact_check["threshold"] == 0.95
    for module in (
        modules["classifier"]["domain"],
        modules["classifier"]["pii"],
        fact_check,
        modules["feedback_detector"],
        modules["modality_detector"]["classifier"],
    ):
        assert module["max_sequence_length"] == 0
        assert module["use_cpu"] is (platform == "cpu")


@pytest.mark.parametrize("budget,full_context", [(0, False), (32768, True)])
def test_explicit_old_models_and_long_policy_are_not_rewritten(
    tmp_path: Path, monkeypatch, budget: int, full_context: bool
):
    monkeypatch.delenv("VLLM_SR_AMD_FORCE_GPU", raising=False)
    monkeypatch.delenv("VLLM_SR_AMD_PRESERVE_CPU", raising=False)
    catalog = {
        "system": {"domain_classifier": "models/mmbert32k-intent-classifier-merged"},
        "embeddings": {
            "semantic": {
                "mmbert_model_path": "models/mmbert-embed-32k-2d-matryoshka",
                "embedding_config": {
                    "model_type": "mmbert",
                    "full_context": full_context,
                },
            }
        },
        "modules": {"classifier": {"domain": {"max_sequence_length": budget}}},
    }
    source = tmp_path / "source.yaml"
    source.write_text(
        yaml.safe_dump({"version": "v0.3", "global": {"model_catalog": catalog}})
    )
    target = tmp_path / "runtime.yaml"
    realize_runtime_config(source, target, algorithm=None, platform="amd")
    actual = yaml.safe_load(target.read_text())["global"]["model_catalog"]
    assert actual["system"] == catalog["system"]
    assert actual["modules"]["classifier"]["domain"]["max_sequence_length"] == budget
    semantic = actual["embeddings"]["semantic"]
    assert (
        semantic["mmbert_model_path"]
        == catalog["embeddings"]["semantic"]["mmbert_model_path"]
    )
    assert semantic["embedding_config"]["full_context"] == full_context

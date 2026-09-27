"""Official Qwen3 source support must retain native decision outputs on reload."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from transformers import AutoTokenizer, Qwen3Config, Qwen3ForCausalLM

from training.model.decision_model import DecisionModel, QWEN3_ARCHITECTURE


class _Tokenizer:
    def save_pretrained(self, path: Path) -> None:
        (path / "tokenizer_config.json").write_text("{}\n", encoding="utf-8")


def test_official_qwen3_base_full_checkpoint_roundtrip(
    tmp_path: Path, monkeypatch
) -> None:
    torch.manual_seed(3)
    source = tmp_path / "official-source"
    config = Qwen3Config(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=128,
        head_dim=8,
    )
    Qwen3ForCausalLM(config).save_pretrained(source)
    monkeypatch.setattr(
        AutoTokenizer, "from_pretrained", lambda *args, **kwargs: _Tokenizer()
    )
    model, tokenizer = DecisionModel.from_base(source, "fixed-official-revision")
    assert model.metadata["architecture"] == QWEN3_ARCHITECTURE
    assert model.metadata["backbone_model_type"] == "qwen3"
    model.eval()
    inputs = {
        "input_ids": torch.tensor([[3, 4, 5, 6, 7, 8]]),
        "attention_mask": torch.ones((1, 6), dtype=torch.long),
        "candidate_positions": torch.tensor([[2, 4]]),
        "candidate_mask": torch.tensor([[True, True]]),
        "query_positions": torch.tensor([5]),
    }
    with torch.inference_mode():
        before = model(**inputs)
    output = tmp_path / "decision-checkpoint"
    model.save(output, tokenizer)
    assert (
        json.loads((output / "decision_config.json").read_text())["architecture"]
        == QWEN3_ARCHITECTURE
    )
    restored, _ = DecisionModel.from_checkpoint(output)
    restored.eval()
    with torch.inference_mode():
        after = restored(**inputs)
    torch.testing.assert_close(after, before, rtol=0, atol=1e-6)

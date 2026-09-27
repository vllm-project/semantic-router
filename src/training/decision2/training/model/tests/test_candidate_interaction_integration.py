"""Tiny Qwen3 roundtrip and frozen CLI constraints for the new head."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import torch
from transformers import AutoTokenizer, Qwen3Config, Qwen3ForCausalLM

from training.model import train
from training.model.decision_model import (
    QWEN3_INTERACTION_ARCHITECTURE,
    DecisionModel,
    collate,
)
from training.model.loss import per_example_loss


class _Tokenizer:
    def save_pretrained(self, path: Path) -> None:
        (path / "tokenizer_config.json").write_text("{}\n", encoding="utf-8")


def _batch() -> dict[str, torch.Tensor]:
    examples = []
    for index, (kind, count) in enumerate((("choice", 3), ("noul", 2), ("score", 3))):
        examples.append(
            {
                "id": f"item-{index}",
                "ids": [3 + index, 5, 7, 11, 13, 17, 19],
                "candidate_positions": list(range(1, count + 1)),
                "query_position": 6,
                "label": index % count,
                "keys": [str(j) for j in range(count)],
                "teacher_probs": None,
                "task_type": kind,
            }
        )
    return collate(examples, pad_id=0)


def test_matched_zero_step_one_update_and_native_reload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    torch.manual_seed(59)
    source = tmp_path / "official-source"
    Qwen3ForCausalLM(
        Qwen3Config(
            vocab_size=64,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=128,
            head_dim=8,
        )
    ).save_pretrained(source)
    monkeypatch.setattr(AutoTokenizer, "from_pretrained", lambda *a, **k: _Tokenizer())

    torch.manual_seed(17)
    control, _ = DecisionModel.from_base(source, "fixed-official-revision")
    torch.manual_seed(17)
    treatment, tokenizer = DecisionModel.from_base(
        source, "fixed-official-revision", head_variant="candidate-interaction"
    )
    assert treatment.metadata["architecture"] == QWEN3_INTERACTION_ARCHITECTURE
    assert treatment.metadata["head_variant"] == "candidate-interaction"
    assert treatment.metadata["interaction_dim"] == 64
    for name, value in control.head.state_dict().items():
        torch.testing.assert_close(
            treatment.head.state_dict()[name], value, rtol=0, atol=0
        )
    batch = _batch()
    control.eval()
    treatment.eval()
    with torch.inference_mode():
        torch.testing.assert_close(treatment(**batch), control(**batch), rtol=0, atol=0)

    treatment.train()
    optimizer = torch.optim.AdamW(treatment.parameters(), lr=1e-4)
    logits = treatment(**batch)
    loss = per_example_loss(
        logits,
        batch["labels"],
        batch["candidate_mask"],
        objective="ce_brier",
        brier_weight=0.5,
    )["total"].mean()
    assert torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(treatment.head.interaction_out.weight.grad).all()
    optimizer.step()
    treatment.eval()
    with torch.inference_mode():
        before = treatment(**batch).softmax(-1)
    checkpoint = tmp_path / "candidate-interaction-checkpoint"
    treatment.save(checkpoint, tokenizer)
    restored, _ = DecisionModel.from_checkpoint(checkpoint)
    restored.eval()
    with torch.inference_mode():
        after = restored(**batch).softmax(-1)
    torch.testing.assert_close(after, before, rtol=0, atol=1e-6)
    with pytest.raises(ValueError, match="task_type_ids"):
        restored(
            **{key: value for key, value in batch.items() if key != "task_type_ids"}
        )
    metadata_path = checkpoint / "decision_config.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["head_variant"] = "shared"
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ValueError, match="variant disagree"):
        DecisionModel.from_checkpoint(checkpoint)


def test_cli_rejects_changed_source_data_or_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(train.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(train.torch.cuda, "is_bf16_supported", lambda: True)
    argv = [
        "train",
        "--model-path",
        "pinned-source",
        "--init-kind",
        "base",
        "--base-revision",
        train.TYPED_HEAD_SOURCE_REVISION,
        "--train",
        "train.jsonl",
        "--select",
        "select.jsonl",
        "--cal",
        "cal.jsonl",
        "--output",
        "new-run",
        "--head-variant",
        "candidate-interaction",
        "--objective",
        "ce_brier",
        "--epochs",
        "1",
        "--max-steps",
        "466",
        "--microbatch",
        "1",
        "--accumulation",
        "16",
        "--eval-batch",
        "2",
        "--max-length",
        "8192",
        "--backbone-lr",
        "2e-5",
        "--head-lr",
        "2e-4",
        "--save-every",
        "64",
    ]
    monkeypatch.setattr(sys, "argv", argv)
    train.validate_args(train.parse_args())
    changed = argv.copy()
    changed[changed.index("--max-steps") + 1] = "465"
    monkeypatch.setattr(sys, "argv", changed)
    with pytest.raises(ValueError, match="frozen official source"):
        train.validate_args(train.parse_args())

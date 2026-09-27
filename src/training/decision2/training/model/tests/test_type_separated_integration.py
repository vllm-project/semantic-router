"""The experimental typed head must load natively without changing shared heads."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import torch
from transformers import AutoTokenizer, Qwen3Config, Qwen3ForCausalLM

from training.model import train
from training.model.decision_model import (
    QWEN3_ARCHITECTURE,
    QWEN3_TYPED_ARCHITECTURE,
    DecisionModel,
    collate,
)
from training.model.loss import per_example_loss


class _Tokenizer:
    def save_pretrained(self, path: Path) -> None:
        (path / "tokenizer_config.json").write_text("{}\n", encoding="utf-8")


@pytest.fixture
def source(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    torch.manual_seed(191)
    path = tmp_path / "official-source"
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
    Qwen3ForCausalLM(config).save_pretrained(path)
    monkeypatch.setattr(AutoTokenizer, "from_pretrained", lambda *a, **k: _Tokenizer())
    return path


def _inputs() -> dict[str, torch.Tensor]:
    items = []
    for index, kind in enumerate(("choice", "noul", "score")):
        count = 3 if kind != "noul" else 2
        items.append(
            {
                "id": kind,
                "ids": [3 + index, 5, 7, 11, 13, 17, 19],
                "candidate_positions": list(range(1, count + 1)),
                "query_position": 6,
                "label": index % count,
                "keys": [str(i) for i in range(count)],
                "teacher_probs": None,
                "task_type": kind,
            }
        )
    return collate(items, pad_id=0)


def test_typed_head_one_step_native_save_reload(source: Path, tmp_path: Path) -> None:
    model, tokenizer = DecisionModel.from_base(
        source, "fixed-official-revision", head_variant="type-separated"
    )
    assert model.metadata["architecture"] == QWEN3_TYPED_ARCHITECTURE
    assert model.metadata["head_variant"] == "type-separated"
    assert len(model.head.heads) == 3
    batch = _inputs()
    assert batch["task_type_ids"].tolist() == [0, 1, 2]
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    logits = model(**batch)
    loss = per_example_loss(
        logits,
        batch["labels"],
        batch["candidate_mask"],
        objective="ce_brier",
        brier_weight=0.5,
    )["total"].mean()
    assert torch.isfinite(loss)
    loss.backward()
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in model.head.parameters()
    )
    optimizer.step()
    model.eval()
    with torch.inference_mode():
        before = model(**batch).softmax(-1)
    checkpoint = tmp_path / "typed-checkpoint"
    model.save(checkpoint, tokenizer)
    restored, _ = DecisionModel.from_checkpoint(checkpoint)
    restored.eval()
    with torch.inference_mode():
        after = restored(**batch).softmax(-1)
    torch.testing.assert_close(after, before, rtol=0, atol=1e-6)
    with pytest.raises(ValueError, match="task_type_ids"):
        restored(**{k: v for k, v in batch.items() if k != "task_type_ids"})


def test_shared_checkpoint_remains_shared_and_spoofing_fails(
    source: Path, tmp_path: Path
) -> None:
    model, tokenizer = DecisionModel.from_base(source, "fixed-official-revision")
    assert model.metadata["architecture"] == QWEN3_ARCHITECTURE
    checkpoint = tmp_path / "shared-checkpoint"
    model.save(checkpoint, tokenizer)
    config_path = checkpoint / "decision_config.json"
    old_metadata = json.loads(config_path.read_text(encoding="utf-8"))
    old_metadata.pop("head_variant")
    config_path.write_text(json.dumps(old_metadata), encoding="utf-8")
    restored, _ = DecisionModel.from_checkpoint(checkpoint)
    restored.eval()
    with torch.inference_mode():
        logits = restored(**_inputs())
    assert torch.isfinite(logits[_inputs()["candidate_mask"]]).all()
    old_metadata["head_variant"] = "type-separated"
    config_path.write_text(json.dumps(old_metadata), encoding="utf-8")
    with pytest.raises(ValueError, match="variant disagree"):
        DecisionModel.from_checkpoint(checkpoint)


def test_typed_head_only_supported_on_qwen3(source: Path, monkeypatch) -> None:
    from transformers import AutoConfig

    original = AutoConfig.from_pretrained

    def qwen35(*args, **kwargs):
        config = original(*args, **kwargs)
        config.model_type = "qwen3_5"
        return config

    monkeypatch.setattr(AutoConfig, "from_pretrained", qwen35)
    with pytest.raises(ValueError, match="requires Qwen3"):
        DecisionModel.from_base(
            source, "fixed-official-revision", head_variant="type-separated"
        )


def test_typed_training_cli_rejects_budget_or_weight_origin_changes(
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
        "type-separated",
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
    args = train.parse_args()
    train.validate_args(args)
    for changed in (("--max-steps", "465"), ("--head-lr", "3e-4")):
        mutation = argv.copy()
        mutation[mutation.index(changed[0]) + 1] = changed[1]
        monkeypatch.setattr(sys, "argv", mutation)
        with pytest.raises(ValueError, match="frozen official source"):
            train.validate_args(train.parse_args())

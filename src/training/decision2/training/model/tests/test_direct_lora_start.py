"""Fresh optimizer continuation must preserve an existing LoRA start."""

from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from training.model import train
from training.model.lora import LORA_FORMAT

INITIAL_SHA = "a" * 64


def _args(monkeypatch: pytest.MonkeyPatch, *extra: str):
    monkeypatch.setattr(train.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(train.torch.cuda, "is_bf16_supported", lambda: True)
    monkeypatch.setattr(train, "version", lambda package: "test-peft")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train",
            "--train",
            "train.jsonl",
            "--select",
            "select.jsonl",
            "--cal",
            "cal.jsonl",
            "--output",
            "run",
            "--model-path",
            "selected-lora",
            "--init-kind",
            "decision2-lora",
            "--train-mode",
            "lora",
            "--source-path",
            "immutable-base",
            "--initial-model-sha256",
            INITIAL_SHA,
            "--lora-rank",
            "8",
            "--lora-alpha",
            "16",
            "--lora-dropout",
            "0.05",
            *extra,
        ],
    )
    return train.parse_args()


def test_direct_lora_cli_requires_pinned_original_and_existing_adapter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = _args(monkeypatch)
    train.validate_args(args)
    for name, value in (
        ("source_path", None),
        ("initial_model_sha256", None),
        ("train_mode", "full"),
        ("base_revision", "other"),
    ):
        changed = SimpleNamespace(**vars(args))
        setattr(changed, name, value)
        with pytest.raises(ValueError, match="Direct LoRA continuation"):
            train.validate_args(changed)


def test_legacy_run_contract_has_no_new_initial_identity_field(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = _args(monkeypatch)
    assert train.direct_lora_contract_fields(args) == {
        "initial_model_sha256": INITIAL_SHA
    }
    for kind in ("base", "posttrained", "decision1", "decision2"):
        old = SimpleNamespace(**vars(args))
        old.init_kind = kind
        old.initial_model_sha256 = None
        assert train.direct_lora_contract_fields(old) == {}


class _Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.base = nn.Linear(2, 2, bias=False)
        self.base.requires_grad_(False)
        self.lora_A = nn.Parameter(torch.ones(2, 2))


def test_direct_lora_loader_binds_identity_and_keeps_selected_tensors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = _args(monkeypatch)
    model = SimpleNamespace(
        metadata={
            "checkpoint_format": LORA_FORMAT,
            "head_dim": 256,
            "lora": {
                "rank": 8,
                "alpha": 16,
                "dropout": 0.05,
                "peft_version": train.version("peft"),
                "source_fingerprint": {"files_sha256": {"base": "b" * 64}},
            },
        },
        backbone=_Backbone(),
        head=nn.Linear(2, 1),
    )
    identity = {
        "model_sha256": INITIAL_SHA,
        "files_sha256": {
            "checkpoint/adapter/adapter_model.safetensors": "c" * 64,
            "checkpoint/decision_head.safetensors": "d" * 64,
        },
    }
    called = []
    monkeypatch.setattr(train, "checkpoint_fingerprint", lambda *a: identity)
    monkeypatch.setattr(
        train.DecisionModel,
        "from_checkpoint",
        lambda *a, **kw: (called.append(kw) or model, object()),
    )
    loaded, _, frozen = train.load_direct_lora_start(args)
    assert loaded is model and frozen == identity
    assert called == [{"source_path": "immutable-base", "trainable_adapter": True}]
    assert model.metadata["continuation_origin"] == {
        "initial_model_sha256": INITIAL_SHA,
        "initial_adapter_sha256": "c" * 64,
        "initial_head_sha256": "d" * 64,
    }
    changed = SimpleNamespace(**vars(args))
    changed.initial_model_sha256 = "e" * 64
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        train.load_direct_lora_start(changed)


def test_direct_lora_loader_rejects_topology_or_unfrozen_base(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = _args(monkeypatch)
    identity = {
        "model_sha256": INITIAL_SHA,
        "files_sha256": {
            "checkpoint/adapter/adapter_model.safetensors": "c" * 64,
            "checkpoint/decision_head.safetensors": "d" * 64,
        },
    }
    model = SimpleNamespace(
        metadata={
            "checkpoint_format": LORA_FORMAT,
            "head_dim": 256,
            "lora": {
                "rank": 16,
                "alpha": 16,
                "dropout": 0.05,
                "peft_version": train.version("peft"),
                "source_fingerprint": {"files_sha256": {"base": "b" * 64}},
            },
        },
        backbone=_Backbone(),
        head=nn.Linear(2, 1),
    )
    monkeypatch.setattr(train, "checkpoint_fingerprint", lambda *a: identity)
    monkeypatch.setattr(
        train.DecisionModel, "from_checkpoint", lambda *a, **kw: (model, object())
    )
    with pytest.raises(ValueError, match="topology"):
        train.load_direct_lora_start(args)
    model.metadata["lora"]["rank"] = 8
    model.backbone.base.requires_grad_(True)
    with pytest.raises(ValueError, match="only the existing trainable LoRA"):
        train.load_direct_lora_start(args)

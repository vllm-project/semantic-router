"""Fresh optimizer continuation must preserve an existing LoRA start."""

from __future__ import annotations

import json
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
            "--direct-lora-parity-receipt",
            "parity.json",
            "--direct-lora-parity-sha256",
            "b" * 64,
            "--direct-lora-arm",
            "A",
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
        "initial_model_sha256": INITIAL_SHA,
        "direct_lora_parity_sha256": "b" * 64,
        "direct_lora_arm": "A",
    }
    for kind in ("base", "posttrained", "decision1", "decision2"):
        old = SimpleNamespace(**vars(args))
        old.init_kind = kind
        old.initial_model_sha256 = None
        assert train.direct_lora_contract_fields(old) == {}


def test_direct_lora_optimizer_is_gated_by_both_zero_step_starts(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    args = _args(monkeypatch)
    args.initial_model_sha256 = (
        "d9f4990427156a7712325de16f6105659fc00015d44c3e2f9331f52481d350d2"
    )
    args.train = str(tmp_path / "arm-a.jsonl")
    args.direct_lora_parity_receipt = str(tmp_path / "parity.json")
    args.direct_lora_parity_sha256 = "f" * 64
    arm_shas = {
        "A": "6a6ef7d3f2eac2a63cdd61cd806275e67c0aa772e78b45bf0953200f2a776235",
        "B": "1c705c9a8271ce2e526b6bc91affe18d463a52bb86ce99b6b1bcb5007d467b41",
    }
    receipt = {
        "schema_version": "decision2-score-en-direct-lora-start-parity/1",
        "status": "PASS",
        "source_model_sha256": args.initial_model_sha256,
        "roster_sha256": "193404fb2ed3905cbb9e34379400a2f33a40d86aaae971940454c6fe71163bc5",
        "tolerance": 1e-4,
        "arms": {
            arm: {
                "status": "PASS",
                "train_sha256": digest,
                "same_argmax": 32,
                "max_absolute_option_probability_drift": 0.0,
            }
            for arm, digest in arm_shas.items()
        },
    }
    path = tmp_path / "parity.json"
    path.write_text(json.dumps(receipt), encoding="utf-8")
    monkeypatch.setattr(
        train,
        "file_sha256",
        lambda target: "f" * 64 if str(target) == str(path) else arm_shas["A"],
    )
    train.verify_direct_lora_parity_gate(args)
    receipt["arms"]["B"]["same_argmax"] = 31
    path.write_text(json.dumps(receipt), encoding="utf-8")
    with pytest.raises(ValueError, match="arm B failed"):
        train.verify_direct_lora_parity_gate(args)
    receipt["arms"]["B"]["same_argmax"] = 32
    args.direct_lora_arm = "B"
    with pytest.raises(ValueError, match="frozen arm differs"):
        train.verify_direct_lora_parity_gate(args)


def test_v7p_direct_lora_receipt_binds_all_three_arms_without_rewriting_v6(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    args = _args(monkeypatch)
    args.initial_model_sha256 = (
        "d9f4990427156a7712325de16f6105659fc00015d44c3e2f9331f52481d350d2"
    )
    args.train = str(tmp_path / "arm-a.jsonl")
    args.direct_lora_parity_receipt = str(tmp_path / "v7p-start.json")
    args.direct_lora_parity_sha256 = "f" * 64
    receipt = {
        "schema_version": "decision2-score-v7p-direct-lora-start/2",
        "status": "PASS",
        "source_model_sha256": args.initial_model_sha256,
        "source_files_sha256": {"base": "1" * 64},
        "roster_sha256": "2" * 64,
        "roster_items": 32,
        "tolerance": 1e-4,
        "container_image_id": "sha256:" + "3" * 64,
        "arms": {
            name: {
                "status": "PASS",
                "train_sha256": "a" * 64 if name != "B" else "b" * 64,
                "prediction_sha256": (
                    "4" if name == "A" else "5" if name == "B" else "6"
                )
                * 64,
                "same_argmax": 32,
                "max_absolute_option_probability_drift": 0.0,
            }
            for name in ("A", "B", "C")
        },
    }
    path = tmp_path / "v7p-start.json"
    path.write_text(json.dumps(receipt), encoding="utf-8")
    monkeypatch.setattr(
        train,
        "file_sha256",
        lambda target: "f" * 64 if str(target) == str(path) else "a" * 64,
    )
    train.verify_direct_lora_parity_gate(args)
    args.direct_lora_arm = "C"
    train.verify_direct_lora_parity_gate(args)
    args.direct_lora_arm = "B"
    with pytest.raises(ValueError, match="frozen arm TRAIN differs"):
        train.verify_direct_lora_parity_gate(args)
    args.direct_lora_arm = "A"
    receipt["arms"]["C"]["same_argmax"] = 31
    path.write_text(json.dumps(receipt), encoding="utf-8")
    with pytest.raises(ValueError, match="arm C failed parity"):
        train.verify_direct_lora_parity_gate(args)
    receipt["arms"]["C"]["same_argmax"] = 32
    receipt["arms"]["C"]["train_sha256"] = "c" * 64
    path.write_text(json.dumps(receipt), encoding="utf-8")
    with pytest.raises(ValueError, match="objective arm changed TRAIN"):
        train.verify_direct_lora_parity_gate(args)


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

"""Gold-free CPU checks for the prospective Gemma TRAIN-only one-step gate."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from scripts import preflight_gemma4_one_step as probe
from torch import nn
from training.model.gemma4 import encode_gemma


class TinyTokenizer:
    bos_token_id = 2
    pad_token_id = 0

    def encode(self, value: str, *, add_special_tokens: bool) -> list[int]:
        assert not add_special_tokens
        return [ord(char) + 3 for char in value]


def fake_train_row() -> dict:
    row_id = "unit-test-private-train-row"
    return {
        "id": row_id,
        "group_id": row_id,
        "split": "train",
        "evaluation_role": "train",
        "task_type": "score",
        "family": "stage4_ordinal",
        "language": "zh",
        "label": 1,
        "state": "蓝色凭证已经签名。",
        "instructions": "判断证据强度。",
        "options": [
            {"key": "0", "description": "没有支持"},
            {"key": "1", "description": "部分支持"},
            {"key": "2", "description": "充分支持"},
        ],
    }


class DummyBackbone(nn.Module):
    def forward(self, *, input_ids, attention_mask, use_cache):
        assert not use_cache
        assert input_ids.shape == attention_mask.shape
        hidden = input_ids.float().unsqueeze(-1).repeat(1, 1, 4)
        return SimpleNamespace(last_hidden_state=hidden)


class DummyHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, choices, query):
        return self.scale * choices[:, :, 0] + query[:, :1]


class GemmaOneStepCpuTests(unittest.TestCase):
    def test_exact_train_file_and_row_are_required(self) -> None:
        tokenizer = TinyTokenizer()
        row = fake_train_row()
        encoded = encode_gemma(row, tokenizer, probe.MAX_LENGTH)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "train.jsonl"
            line = json.dumps(row, ensure_ascii=False) + "\n"
            path.write_text(line, encoding="utf-8")
            row_spec = {
                "id": row["id"],
                "group_id": row["group_id"],
                "row_sha256": hashlib.sha256(line.encode()).hexdigest(),
                "prompt_sha256": encoded["prompt_sha256"],
                "token_ids_sha256": encoded["token_ids_sha256"],
                "token_count": len(encoded["ids"]),
                "task_type": "score",
                "family": "stage4_ordinal",
                "language": "zh",
                "label_index": 1,
                "option_count": 3,
            }
            with (
                patch.object(
                    probe, "TRAIN_SHA256", hashlib.sha256(line.encode()).hexdigest()
                ),
            ):
                got, digest = probe.pinned_train_row(path, tokenizer, row_spec)
                self.assertEqual(got["task_type"], "score")
                self.assertEqual(got["label"], 1)
                self.assertEqual(digest, row_spec["row_sha256"])
                row["evaluation_role"] = "final"
                path.write_text(
                    json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8"
                )
                with self.assertRaisesRegex(ValueError, "TRAIN file differs"):
                    probe.pinned_train_row(path, tokenizer, row_spec)

    def test_lock_binds_source_train_input_compute_and_stop_rules(self) -> None:
        source = {
            "config_sha256": "config",
            "tokenizer_json_sha256": "tokenizer",
            "weights": {"shard_sha256": {"part": "shard"}},
        }
        row_spec = {
            "id": "private-train-id",
            "group_id": "private-train-id",
            "row_sha256": "private-row-hash",
            "prompt_sha256": "prompt-hash",
            "token_ids_sha256": "token-hash",
            "token_count": 268,
            "task_type": "score",
            "family": "stage4_ordinal",
            "language": "zh",
            "label_index": 1,
            "option_count": 3,
        }
        encoded = {
            "id": row_spec["id"],
            "prompt_sha256": row_spec["prompt_sha256"],
            "token_ids_sha256": row_spec["token_ids_sha256"],
            "ids": [1] * row_spec["token_count"],
            "task_type": "score",
            "label": 1,
            "keys": ["0", "1", "2"],
        }
        lock = {
            "schema_version": probe.LOCK_VERSION,
            "source_id": probe.SOURCE_ID,
            "source_revision": probe.SOURCE_REVISION,
            "config_sha256": "config",
            "tokenizer_json_sha256": "tokenizer",
            "shard_sha256": {"part": "shard"},
            "code_sha256": probe.code_hashes(),
            "runner_sha256": "runner",
            "train_file_sha256": probe.TRAIN_SHA256,
            "train_row": row_spec,
            "gpu_ordinal": probe.GPU_ORDINAL,
            "max_length": probe.MAX_LENGTH,
            "seed": probe.SEED,
            "rank": 8,
            "alpha": 16,
            "dropout": 0.05,
            "head_dim": 256,
            "lora_lr": probe.LORA_LR,
            "head_lr": probe.HEAD_LR,
            "weight_decay": 0.0,
            "grad_clip": probe.GRAD_CLIP,
            "max_grad_norm": probe.MAX_GRAD_NORM,
            "max_reload_logit_drift": probe.MAX_RELOAD_LOGIT_DRIFT,
            "optimizer_updates": 1,
            "max_wall_seconds": probe.MAX_WALL_SECONDS,
            "select_cal_formal_labels_read": False,
        }
        probe.verify_lock(lock, source, encoded, row_spec["row_sha256"], "runner")
        lock["max_reload_logit_drift"] = 1e-2
        with self.assertRaisesRegex(ValueError, "differs from its frozen"):
            probe.verify_lock(lock, source, encoded, row_spec["row_sha256"], "runner")
        lock["max_reload_logit_drift"] = probe.MAX_RELOAD_LOGIT_DRIFT
        lock["select_cal_formal_labels_read"] = True
        with self.assertRaisesRegex(ValueError, "differs from its frozen"):
            probe.verify_lock(lock, source, encoded, row_spec["row_sha256"], "runner")

    def test_private_lock_file_mode_and_head_shape(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "lock.json"
            path.write_text("{}", encoding="utf-8")
            os.chmod(path, 0o600)
            self.assertEqual(probe.read_private_lock(path), {})
            os.chmod(path, 0o644)
            with self.assertRaisesRegex(ValueError, "owner-held"):
                probe.read_private_lock(path)
        encoded = {
            "ids": [2, 3, 4, 5, 6, 7],
            "candidate_positions": [1, 2, 3],
            "query_position": 5,
        }
        output = probe.decision_logits(DummyBackbone(), DummyHead(), encoded)
        self.assertEqual(tuple(output.shape), (1, 3))
        encoded["candidate_positions"] = [1, 2]
        with self.assertRaisesRegex(RuntimeError, "logits are invalid"):
            probe.decision_logits(DummyBackbone(), DummyHead(), encoded)


if __name__ == "__main__":
    unittest.main()

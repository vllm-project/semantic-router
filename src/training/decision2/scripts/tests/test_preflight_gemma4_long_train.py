"""CPU-only contract tests for the proposed three-type long-TRAIN gate."""

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from scripts import preflight_gemma4_long_train as probe
from torch import nn


class StubBackbone(nn.Module):
    def forward(self, *, input_ids, attention_mask, use_cache):
        assert not use_cache
        assert input_ids.shape == attention_mask.shape
        return SimpleNamespace(
            last_hidden_state=input_ids.float().unsqueeze(-1).repeat(1, 1, 4)
        )


class StubHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, candidates, query):
        return self.scale * candidates[:, :, 0] + query[:, :1]


class GemmaLongTrainCpuTests(unittest.TestCase):
    def test_longest_common_train_row_per_type_is_deterministic(self) -> None:
        rows = [
            {
                "id": "choice-short",
                "group_id": "g1",
                "task_type": "choice",
                "family": "f",
                "language": "en",
                "label": 0,
                "options": [{}, {}],
                "q_len": 7,
                "g_len": 8,
            },
            {
                "id": "choice-long",
                "group_id": "g2",
                "task_type": "choice",
                "family": "f",
                "language": "en",
                "label": 1,
                "options": [{}, {}],
                "q_len": 9,
                "g_len": 9,
            },
            {
                "id": "choice-excluded",
                "group_id": "g3",
                "task_type": "choice",
                "family": "f",
                "language": "en",
                "label": 0,
                "options": [{}, {}],
                "q_len": 11,
                "g_len": 10,
            },
            {
                "id": "noul-long",
                "group_id": "g4",
                "task_type": "noul",
                "family": "f",
                "language": "zh",
                "label": 0,
                "options": [{}, {}],
                "q_len": 8,
                "g_len": 8,
            },
            {
                "id": "score-long",
                "group_id": "g5",
                "task_type": "score",
                "family": "f",
                "language": "en",
                "label": 1,
                "options": [{}, {}, {}],
                "q_len": 7,
                "g_len": 7,
            },
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "train.jsonl"
            path.write_text("".join(json.dumps(row) + "\n" for row in rows))
            digest = hashlib.sha256(path.read_bytes()).hexdigest()

            def tokenized(row, _tokenizer, _max_length, *, kind):
                length = row[f"{kind}_len"]
                return {
                    "ids": [0] * length,
                    "prompt_sha256": f"prompt-{row['id']}",
                    "token_ids_sha256": f"tokens-{row['id']}",
                }

            with (
                patch.object(probe, "TRAIN_SHA256", digest),
                patch.object(probe, "MAX_LENGTH", 10),
                patch.object(
                    probe,
                    "PINNED_LONG_LENGTHS",
                    {"choice": 9, "noul": 8, "score": 7},
                ),
                patch.object(probe, "load_partition", return_value=rows),
                patch.object(
                    probe,
                    "encode",
                    side_effect=lambda row, tok, cap: tokenized(
                        row, tok, cap, kind="q"
                    ),
                ),
                patch.object(
                    probe,
                    "encode_gemma",
                    side_effect=lambda row, tok, cap: tokenized(
                        row, tok, cap, kind="g"
                    ),
                ),
            ):
                chosen = probe.select_long_rows(path, object(), object())
        self.assertEqual(
            [item["id"] for item in chosen],
            ["choice-long", "noul-long", "score-long"],
        )
        self.assertEqual([item["token_count"] for item in chosen], [9, 8, 7])

    def test_lock_fixes_gpu_objective_budget_and_no_gold(self) -> None:
        source = {
            "config_sha256": "config",
            "tokenizer_json_sha256": "gemma-tokenizer",
            "weights": {"shard_sha256": {"part": "shard"}},
        }
        with patch.object(probe, "code_hashes", return_value={"probe": "hash"}):
            lock = probe.expected_lock(source, [{"task_type": "choice"}], "runner")
        self.assertEqual(lock["gpu_ordinal"], "3")
        self.assertEqual(lock["optimizer_updates"], 3)
        self.assertEqual(lock["max_wall_seconds"], 2700)
        self.assertEqual(lock["objective"], "ce_brier")
        self.assertEqual(lock["brier_weight"], 0.5)
        self.assertTrue(lock["gradient_checkpointing"])
        self.assertFalse(lock["select_cal_formal_labels_read"])
        self.assertFalse(lock["model_quality_evaluated"])

    def test_cpu_admission_rejects_mutated_private_lock(self) -> None:
        source = {
            "config_sha256": "config",
            "tokenizer_json_sha256": "gemma-tokenizer",
            "weights": {"shard_sha256": {"part": "shard"}},
        }
        selected = [{"task_type": kind} for kind in probe.TASK_TYPES]
        with patch.object(probe, "code_hashes", return_value={"probe": "hash"}):
            tampered = probe.expected_lock(source, selected, "runner")
            tampered["max_length"] = 4095
            with (
                patch.object(probe, "inspect", return_value=source),
                patch.object(
                    probe,
                    "sha256_file",
                    side_effect=lambda path: (
                        probe.QWEN_TOKENIZER_SHA256
                        if path.name == "tokenizer.json"
                        else "runner"
                    ),
                ),
                patch("transformers.AutoTokenizer.from_pretrained"),
                patch.object(probe, "select_long_rows", return_value=selected),
                patch.object(probe, "read_private_lock", return_value=tampered),
            ):
                with self.assertRaisesRegex(ValueError, "differs from private lock"):
                    probe.checked_inputs(
                        Path("/fake/source"),
                        Path("/fake/train"),
                        Path("/fake/qwen"),
                        Path("/fake/runner"),
                        Path("/fake/lock"),
                    )

    def test_dynamic_logits_accepts_native_two_three_and_six_options(self) -> None:
        for count in (2, 3, 6):
            encoded = {
                "ids": list(range(count + 4)),
                "candidate_positions": list(range(1, count + 1)),
                "query_position": count + 3,
                "keys": [str(i) for i in range(count)],
            }
            logits = probe.dynamic_logits(StubBackbone(), StubHead(), encoded)
            self.assertEqual(tuple(logits.shape), (1, count))
            self.assertTrue(torch.isfinite(logits).all())


if __name__ == "__main__":
    unittest.main()

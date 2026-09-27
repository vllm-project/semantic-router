"""CPU/meta checks for the isolated official Gemma text adapter path."""

from __future__ import annotations

import json
import unittest
from collections import Counter
from types import SimpleNamespace

from scripts import preflight_gemma4_adapter as probe
from torch import nn
from training.model.gemma4 import encode_gemma, lora_plan


class TinyTokenizer:
    bos_token_id = 2
    pad_token_id = 0

    def encode(self, value: str, *, add_special_tokens: bool) -> list[int]:
        assert not add_special_tokens
        return [ord(char) + 3 for char in value]


class FakeAttention(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.q_proj = nn.Linear(2816, width, bias=False, device="meta")
        self.o_proj = nn.Linear(width, 2816, bias=False, device="meta")
        self.k_proj = nn.Linear(2816, width // 2, bias=False, device="meta")


class FakeLayer(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.self_attn = FakeAttention(width)
        self.router = nn.Module()
        self.router.proj = nn.Linear(2816, 128, bias=False, device="meta")


class FakeText(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(
            model_type="gemma4_text", num_hidden_layers=30, hidden_size=2816
        )
        self.layers = nn.ModuleList(
            FakeLayer(8192 if i in (5, 11, 17, 23, 29) else 4096) for i in range(30)
        )


class GemmaAdapterPreflightTests(unittest.TestCase):
    def test_pinned_text_target_topology_and_count(self) -> None:
        text = FakeText()
        plan = lora_plan(text, rank=8, head_dim=256)
        self.assertEqual(len(plan["target_modules"]), 60)
        self.assertEqual(plan["wide_attention_layers"], [5, 11, 17, 23, 29])
        self.assertEqual(plan["adapter_parameters"], 3_645_440)
        self.assertEqual(plan["decision_head_parameters"], 2_895_360)
        self.assertEqual(plan["combined_trainable_parameters"], 6_540_800)
        self.assertFalse(
            any("vision" in name or "router" in name for name in plan["target_modules"])
        )
        text.layers[5].self_attn.o_proj = nn.Identity()
        with self.assertRaisesRegex(ValueError, "lacks linear q/o"):
            lora_plan(text)
        text = FakeText()
        text.config.model_type = "gemma4"
        with self.assertRaisesRegex(ValueError, "Expected pinned Gemma"):
            lora_plan(text)

    def test_gemma_bos_offsets_every_candidate_without_truncation(self) -> None:
        tokenizer = TinyTokenizer()
        rows = probe.synthetic_prompts32()
        self.assertEqual(len(rows), 32)
        self.assertEqual(len({row["id"] for row in rows}), 32)
        self.assertEqual(
            Counter(row["questions"]["q"]["type"] for row in rows),
            {"choice": 11, "noul": 11, "score": 10},
        )
        encoded = probe.encode_roster(tokenizer)
        self.assertEqual(len(encoded), 32)
        for item in encoded:
            self.assertEqual(item["ids"][0], tokenizer.bos_token_id)
            self.assertTrue(
                all(p < item["query_position"] for p in item["candidate_positions"])
            )
        self.assertEqual(probe.roster_sha256(encoded), probe.roster_sha256(encoded))
        schema = probe.question_to_row(rows[0], "q", rows[0]["questions"]["q"])
        with self.assertRaisesRegex(ValueError, "exceeds max_length"):
            encode_gemma(schema, tokenizer, 20)

    def test_source_adapter_compare_rejects_drift_and_mismatch(self) -> None:
        rows = probe.encode_roster(TinyTokenizer())
        selected = [[0.25] * 2816] * 3
        template = {
            "probe_version": probe.PROBE_VERSION,
            "mode": "source-32",
            "source": {
                "source_id": probe.SOURCE_ID,
                "source_revision": probe.SOURCE_REVISION,
            },
            "loaded": {"loaded_text_parameters": 25_233_141_760},
            "prompt_version": probe.GEMMA_PROMPT_VERSION,
            "roster_sha256": probe.roster_sha256(rows),
            "prompt_count": 32,
            "code_sha256": probe.code_hashes(),
            "lock_sha256": "frozen-lock-hash",
            "lora_plan": lora_plan(FakeText()),
            "training_steps": 0,
            "formal_labels_read": False,
            "model_quality_evaluated": False,
            "readings": [
                {
                    "id": item["id"],
                    "task_type": item["task_type"],
                    "input_tokens": len(item["ids"]),
                    "token_ids_sha256": item["token_ids_sha256"],
                    "selected_hidden": selected,
                }
                for item in rows
            ],
        }
        source = json.loads(json.dumps(template))
        adapter = json.loads(json.dumps(template))
        adapter["mode"] = "adapter-32"
        adapter["lora_plan"].update({"alpha": 16, "dropout": 0.05})
        self.assertEqual(
            probe.compare(source, adapter)["status"],
            "source_to_fresh_lora_identity_passed",
        )
        adapter["readings"][0]["selected_hidden"][0][0] += 0.0002
        with self.assertRaisesRegex(ValueError, "changes source hidden"):
            probe.compare(source, adapter)
        adapter["readings"][0]["selected_hidden"][0][0] -= 0.0002
        adapter["readings"][0]["token_ids_sha256"] = "different"
        with self.assertRaisesRegex(ValueError, "prompt identities changed"):
            probe.compare(source, adapter)
        adapter["readings"][0]["token_ids_sha256"] = source["readings"][0][
            "token_ids_sha256"
        ]
        adapter["formal_labels_read"] = True
        with self.assertRaisesRegex(ValueError, "training or quality"):
            probe.compare(source, adapter)

    def test_exact_lock_rejects_changed_roster_or_source(self) -> None:
        rows = probe.encode_roster(TinyTokenizer())
        source = {
            "config_sha256": "config-hash",
            "tokenizer_json_sha256": "tokenizer-hash",
            "weights": {"shard_sha256": {"part": "shard-hash"}},
        }
        lock = {
            "schema_version": probe.LOCK_VERSION,
            "source_id": probe.SOURCE_ID,
            "source_revision": probe.SOURCE_REVISION,
            "config_sha256": "config-hash",
            "tokenizer_json_sha256": "tokenizer-hash",
            "shard_sha256": {"part": "shard-hash"},
            "code_sha256": probe.code_hashes(),
            "runner_sha256": "runner-hash",
            "roster_sha256": probe.roster_sha256(rows),
            "cells": ["source-32", "adapter-32"],
            "gpu_ordinal": "3",
            "max_length": 8192,
            "prompt_count": 32,
            "optimizer_updates": 0,
            "formal_labels_read": False,
            "adapter_plan": {
                "rank": 8,
                "alpha": 16,
                "dropout": 0.05,
                "adapter_parameters": 3_645_440,
                "decision_head_parameters": 2_895_360,
            },
            "max_wall_seconds": 1200,
        }
        probe.verify_lock(lock, source, rows, "runner-hash")
        source["weights"]["shard_sha256"]["part"] = "other"
        with self.assertRaisesRegex(ValueError, "differs from its frozen lock"):
            probe.verify_lock(lock, source, rows, "runner-hash")
        source["weights"]["shard_sha256"]["part"] = "shard-hash"
        rows[0]["token_ids_sha256"] = "other"
        with self.assertRaisesRegex(ValueError, "differs from its frozen lock"):
            probe.verify_lock(lock, source, rows, "runner-hash")


if __name__ == "__main__":
    unittest.main()

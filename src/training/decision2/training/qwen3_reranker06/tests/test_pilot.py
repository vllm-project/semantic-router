"""CPU contract tests for the frozen long-context reranker pilot."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from training.qwen3_reranker06 import pilot, train


class FakeTokenizer:
    def encode(self, prompt: str, add_special_tokens: bool = False) -> list[int]:
        assert not add_special_tokens
        return list(range(len(prompt.split())))


def row(kind: str, index: int, family: str = "stage4_scope") -> dict:
    keys = ["false", "true"] if kind == "noul" else ["0", "1", "2"]
    return {
        "id": f"{kind}-{family}-{index}",
        "task_type": kind,
        "family": family,
        "state": "Evidence for candidate two",
        "instructions": "Choose the supported answer",
        "options": [{"key": key, "description": f"option {key}"} for key in keys],
        "label": 1,
    }


class PilotContracts(unittest.TestCase):
    def test_candidate_text_has_all_options_without_gold(self) -> None:
        item = row("choice", 1)
        question = pilot.row_question(item)
        prompts, keys = pilot.candidate_texts(item["state"], question)
        self.assertEqual(keys, ["0", "1", "2"])
        self.assertEqual(len(prompts), 3)
        for prompt in prompts:
            self.assertIn("All candidate answers:", prompt)
            self.assertIn("option 0", prompt)
            self.assertIn("option 2", prompt)
            self.assertNotIn("label", prompt)
        self.assertEqual(pilot.target_key(row("noul", 2)), "yes")

    def test_context_refuses_truncation(self) -> None:
        item = row("choice", 1)
        question = pilot.row_question(item)
        sequences, _ = pilot.encode(FakeTokenizer(), item["state"], question, 999)
        self.assertTrue(sequences)
        with self.assertRaisesRegex(ValueError, "context_overflow"):
            pilot.encode(FakeTokenizer(), item["state"], question, 1)

    def test_quotas_and_long_strata_are_deterministic(self) -> None:
        records = []
        for (kind, source_class), quota in pilot.QUOTAS.items():
            family = (
                "stage4_scope"
                if source_class == "structured"
                else "human_goemotions_choice"
            )
            for index in range(quota + 20):
                length = 700 if index < 35 else 250
                records.append((row(kind, index, family), length))
        first = pilot.choose_rows(records)
        self.assertEqual(
            [r["id"] for r in first], [r["id"] for r in pilot.choose_rows(records)]
        )
        self.assertEqual(len(first), 512)
        self.assertEqual(len({r["id"] for r in first}), 512)
        self.assertEqual(sum(r["task_type"] == "score" for r in first), 128)

    def test_projection_preserves_boolean_and_ordinal_semantics(self) -> None:
        question = pilot.row_question(row("noul", 1))
        answer = pilot.project(question, ["no", "yes"], [-2, 2])
        self.assertGreater(answer["noul"], 0.5)
        question = pilot.row_question(row("score", 1))
        answer = pilot.project(question, ["0", "1", "2"], [-2, -1, 3])
        self.assertEqual(answer["native_level"], 2)
        self.assertGreater(answer["score"], 1.5)

    def test_tampered_adapter_is_rejected_before_reload(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            directory = Path(name)
            weights = directory / "adapter_model.safetensors"
            weights.write_bytes(b"first")
            source = {"model_revision": pilot.MODEL_REVISION}
            receipt = {
                "contract": pilot.ADAPTER,
                "source": source,
                "source_train_sha256": "32a1226931967fd7a0f53cb2a4fd51189d5b643c56eeb1b88cea94ebf4312398",
                "optimizer_updates": train.UPDATES,
                "adapter_sha256": pilot.file_sha256(weights),
            }
            (directory / "decision2_pilot_receipt.json").write_text(
                json.dumps(receipt), encoding="utf-8"
            )
            self.assertEqual(train.verify_adapter(directory, source), receipt)
            weights.write_bytes(b"second")
            with self.assertRaisesRegex(ValueError, "receipt mismatch"):
                train.verify_adapter(directory, source)


if __name__ == "__main__":
    unittest.main()

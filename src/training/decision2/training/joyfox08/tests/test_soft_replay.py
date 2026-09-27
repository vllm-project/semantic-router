"""Gold-free contracts for the fixed Joyfox soft-replay arm."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from training.joyfox08 import soft_replay
from training.model.data import file_sha256


class SoftReplayTest(unittest.TestCase):
    def test_smoke_selection_is_deterministic_and_includes_longest(self) -> None:
        prepared = [
            {
                "row": {"id": f"{kind}-{index}", "task_type": kind},
                "effective_length": 1024 if index == 29 else index + 1,
            }
            for kind in soft_replay.SMOKE_QUOTAS
            for index in range(30)
        ]
        selected = soft_replay.smoke_ids(prepared)
        self.assertEqual(len(selected), 32)
        self.assertEqual(selected, soft_replay.smoke_ids(list(reversed(prepared))))
        for kind in soft_replay.SMOKE_QUOTAS:
            self.assertIn(f"{kind}-29", selected)
        with self.assertRaisesRegex(ValueError, "Insufficient score"):
            soft_replay.smoke_ids(prepared[:60])

    def test_two_process_source_numeric_gate_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            first_smoke = root / "first.json"
            second_smoke = root / "second.json"
            first_receipt = root / "first.receipt.json"
            second_receipt = root / "second.receipt.json"
            records = [
                {
                    "id": f"row-{index}",
                    "input_sha256": "same",
                    "keys": ["a", "b"],
                    "token_count": 10,
                    "effective_length": 10,
                    "logits": [1.0, 0.0],
                }
                for index in range(32)
            ]
            first_smoke.write_text(json.dumps(records))
            second_smoke.write_text(json.dumps(records))
            common = {
                "contract": soft_replay.CONTRACT,
                "source": {"model_revision": "pinned"},
                "runtime_image_id": soft_replay.IMAGE_ID,
                "torch": "fixed",
                "hip": "fixed",
                "transformers": "fixed",
                "peft": "fixed",
                "sample_sha256": soft_replay.SAMPLE_SHA256,
                "sample_manifest_sha256": soft_replay.MANIFEST_SHA256,
                "token_count": 320,
                "smoke_ids": [record["id"] for record in records],
                "smoke_ids_sha256": soft_replay._hash_json(
                    [record["id"] for record in records]
                ),
                "cache_sha256": "cached",
            }
            first_receipt.write_text(
                json.dumps(
                    {
                        **common,
                        "mode": "cache",
                        "smoke_sha256": file_sha256(first_smoke),
                    }
                )
            )
            second_receipt.write_text(
                json.dumps(
                    {
                        **common,
                        "mode": "repeat",
                        "smoke_sha256": file_sha256(second_smoke),
                    }
                )
            )
            result = soft_replay.compare_source_passes(
                first_smoke=first_smoke,
                second_smoke=second_smoke,
                first_receipt=first_receipt,
                second_receipt=second_receipt,
                output=root / "compare.json",
            )
            self.assertTrue(result["numeric_gate_passed"])
            records[0]["logits"] = [0.0, 1.0]
            changed = root / "changed.json"
            changed.write_text(json.dumps(records))
            second_receipt.write_text(
                json.dumps(
                    {
                        **common,
                        "mode": "repeat",
                        "smoke_sha256": file_sha256(changed),
                    }
                )
            )
            result = soft_replay.compare_source_passes(
                first_smoke=first_smoke,
                second_smoke=changed,
                first_receipt=first_receipt,
                second_receipt=second_receipt,
                output=root / "changed.compare.json",
            )
            self.assertFalse(result["numeric_gate_passed"])
            self.assertEqual(result["category_changes"], 1)

    def test_softmax_rejects_nonfinite_source(self) -> None:
        with self.assertRaisesRegex(ValueError, "Nonfinite"):
            soft_replay._softmax([0.0, float("nan")])


if __name__ == "__main__":
    unittest.main()

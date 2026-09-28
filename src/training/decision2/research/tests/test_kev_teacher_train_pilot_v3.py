"""Gold-free contracts for the third disjoint Kev TRAIN signal screen."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from research.eikos_teacher_train_pilot import roster as v1_roster
from research.kev_teacher_train_pilot_v2 import roster as v2_roster
from research.kev_teacher_train_pilot_v3 import preflight, roster


def row(kind: str, index: int) -> dict:
    return {
        "id": f"{kind}-{index}",
        "input_sha256": f"sha-{kind}-{index}",
        "group_id": f"{kind}-group-{index}",
        "task_type": kind,
    }


class KevTeacherV3Contracts(unittest.TestCase):
    def test_roster_excludes_both_prior_groups_and_is_deterministic(self) -> None:
        rows = [
            row(kind, index)
            for kind in ("choice", "noul", "score")
            for index in range(128)
        ]
        first, second, third = v1_roster(rows), v2_roster(rows), roster(rows)
        old = {item["group_id"] for item in first + second}
        self.assertEqual(len(third), 96)
        self.assertEqual(len({item["group_id"] for item in third}), 96)
        self.assertFalse(old & {item["group_id"] for item in third})
        self.assertEqual(third, roster(list(reversed(rows))))
        self.assertEqual(
            {
                kind: sum(item["task_type"] == kind for item in third)
                for kind in ("choice", "noul", "score")
            },
            {"choice": 32, "noul": 32, "score": 32},
        )

    def test_cpu_preflight_rejects_missing_base_shard(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            snapshot = Path(directory)
            (snapshot / "tokenizer.json").write_text("{}")
            (snapshot / "model.safetensors.index.json").write_text(
                json.dumps({"weight_map": {"weight": "missing.safetensors"}})
            )
            with (
                patch(
                    "research.kev_teacher_train_pilot_v3.local_revision",
                    return_value=True,
                ),
                patch(
                    "research.kev_teacher_train_pilot_v3.verify_provenance",
                    return_value={"source_hashes": {"kev/api.py": "hash"}},
                ),
                patch("huggingface_hub.snapshot_download", return_value=str(snapshot)),
            ):
                with self.assertRaisesRegex(ValueError, "weight cache is incomplete"):
                    preflight(
                        Path(directory),
                        Path(directory),
                        "139fdd94f1b6a6ad80cc15e08fcb99cac885a101",
                    )


if __name__ == "__main__":
    unittest.main()

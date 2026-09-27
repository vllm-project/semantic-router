"""Synthetic ANLI development-screen tests; no source examples or gold files."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pyarrow as pa
import pyarrow.parquet as pq

from training.data import audit_anli_score_diagnostic as anli
from training.data.plan_goldfree_inventory import (
    CORE_ROLES,
    NATIVE_ROLE_COUNTS,
    PARTITION_ROLE_COUNTS,
)
from training.model.data import file_sha256
from training.model.decision_model import segments


class CharacterTokenizer:
    def encode(self, value: str, *, add_special_tokens: bool) -> list[int]:
        return [ord(char) for char in value]


def pair(round_id: int, premise: str, hypothesis: str, label: int = 0) -> anli.Pair:
    return anli.Pair(round_id, premise, hypothesis, label, False)


class AnliScoreDiagnosticTests(unittest.TestCase):
    def test_source_rounds_labels_groups_and_reasons(self) -> None:
        shared = "A synthetic premise about a unique blue lantern shipment."
        rows = [
            anli.Pair(1, shared, "The shipment is blue.", 0, True),
            anli.Pair(2, shared, "The shipment is red.", 2, False),
            anli.Pair(
                3,
                "An unrelated purple bicycle was repaired.",
                "A bicycle was repaired.",
                0,
                True,
            ),
        ]
        result = anli.source_profile(rows)
        self.assertEqual(result["total_rows"], 3)
        self.assertEqual(result["independent_normalized_premises_all_rounds"], 2)
        self.assertEqual(result["premise_groups_crossing_rounds"], 1)
        self.assertEqual(result["rounds"]["r1"]["labels"]["entailment"], 1)
        self.assertEqual(result["rounds"]["r2"]["labels"]["contradiction"], 1)
        self.assertEqual(result["rounds"]["r2"]["reason_present"], 0)

    def test_native_score_prompt_ignores_source_label_and_reason(self) -> None:
        first = anli.Pair(1, "Evidence text", "Claim text", 0, False)
        second = anli.Pair(1, "Evidence text", "Claim text", 2, True)
        left = anli.score_row(first, 5)
        right = anli.score_row(second, 5)
        self.assertEqual(left, right)
        self.assertEqual(left["task_type"], "score")
        self.assertEqual([item["key"] for item in left["options"]], ["0", "1", "2"])
        rendered = "".join([segments(left)[0], *segments(left)[1], segments(left)[2]])
        self.assertIn("Evidence text", rendered)
        self.assertIn("Claim text", rendered)
        self.assertNotIn("reason", rendered.lower())
        profile = anli.prompt_profile(
            [
                first,
                pair(2, "Other evidence", "Other claim"),
                pair(3, "Third evidence", "Third claim"),
            ],
            CharacterTokenizer(),
        )
        self.assertEqual(
            profile["source_to_score_level"],
            {"entailment": 2, "neutral": 1, "contradiction": 0},
        )
        self.assertEqual(profile["rounds"]["r1"]["requests"], 1)
        self.assertFalse(profile["reason_in_prompt"])

    def test_pinned_dev_schema_and_cross_round_uid_identity(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            frozen = {}
            for round_id in (1, 2, 3):
                filename = f"dev_r{round_id}.parquet"
                table = pa.table(
                    {
                        "uid": [f"synthetic-{round_id}"],
                        "premise": [
                            f"A synthetic source premise for round {round_id}."
                        ],
                        "hypothesis": ["A synthetic test hypothesis."],
                        "label": [round_id - 1],
                        "reason": ["synthetic explanation"],
                    }
                )
                pq.write_table(table, root / filename)
                frozen[round_id] = (filename, 1, file_sha256(root / filename))
            with patch.dict(anli.ROUND_FILES, frozen, clear=True):
                rows, hashes = anli.read_dev(root)
                self.assertEqual(len(rows), 3)
                self.assertEqual(len(hashes), 3)
                anli.ROUND_FILES[3] = (frozen[3][0], 1, "0" * 64)
                with self.assertRaisesRegex(ValueError, "changed"):
                    anli.read_dev(root)

    def test_missing_reference_roles_hold_without_opening_role_files(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manifest = Path(directory) / "candidate.json"
            manifest.write_text(
                json.dumps(
                    [
                        {"role": role, "path": "/nonexistent", "sha256": "0" * 64}
                        for role in NATIVE_ROLE_COUNTS
                    ]
                ),
                encoding="utf-8",
            )
            roles = anli.load_projected_roles(manifest)
            result = anli.overlap_profile([pair(1, "premise", "hypothesis")], roles)
            self.assertEqual(result["status"], "HOLD_MISSING_PROTECTED_ROLES")
            self.assertEqual(set(result["missing_roles"]), set(PARTITION_ROLE_COUNTS))
            self.assertEqual(
                anli.overlap_profile([], None)["status"],
                "HOLD_MISSING_PROTECTED_INVENTORY",
            )

    def test_complete_synthetic_reference_flags_source_overlap(self) -> None:
        source = pair(
            1,
            "A distinctive synthetic premise about the blue lantern's owner.",
            "The lantern belongs to the applicant.",
        )
        roles = {
            role: [{"id": role, "state": "Unrelated text unique to " + role * 7}]
            for role in CORE_ROLES
        }
        roles["rights_clean_train"] = [{"id": "train-control", "state": source.premise}]
        with patch.dict(NATIVE_ROLE_COUNTS, dict.fromkeys(NATIVE_ROLE_COUNTS, 1)):
            with patch.dict(
                PARTITION_ROLE_COUNTS,
                {
                    name: (split, 1)
                    for name, (split, _) in PARTITION_ROLE_COUNTS.items()
                },
            ):
                result = anli.overlap_profile([source], roles)
                self.assertEqual(result["status"], "HOLD_OBSERVABLE_OVERLAP")
                self.assertGreater(
                    result["by_role"]["rights_clean_train"]["counts"][
                        "exact_normalized"
                    ],
                    0,
                )
                self.assertNotIn(source.premise, json.dumps(result))
                self.assertEqual(
                    anli.overlap_profile([source], {**roles, "unvetted": []})["status"],
                    "HOLD_UNATTESTED_OPTIONAL_ROLES",
                )


if __name__ == "__main__":
    unittest.main()

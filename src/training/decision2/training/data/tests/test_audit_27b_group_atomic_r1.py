"""Synthetic contracts for locked r1 full-input and provenance screens."""

from __future__ import annotations

import unittest

from training.data.audit_27b_group_atomic_r1 import (
    near_pair_profile,
    source_group_pairs,
    surface_pair_ids,
)


class NativeInputAdmissionContracts(unittest.TestCase):
    def test_complete_input_tracks_option_change_and_state_exact(self) -> None:
        left = [
            {
                "id": "train-1",
                "state": "The reviewer approved invoice A after the final audit.",
                "instructions": "Pick the supported decision.",
                "options": '["approve","reject"]',
            }
        ]
        right = [
            {
                **left[0],
                "id": "protected-1",
                "options": '["reject","approve"]',
            }
        ]
        complete = surface_pair_ids(left, right, surface="complete_input")
        state = surface_pair_ids(left, right, surface="state_evidence")
        self.assertEqual(complete["exact_raw"], [])
        self.assertEqual(state["exact_raw"], [("train-1", "protected-1")])

    def test_shared_scaffold_near_is_separate_from_distinct_state(self) -> None:
        common = " ".join(
            f"Check distinct clause {index:03d} before choosing the valid option."
            for index in range(100)
        )
        left = [
            {
                "id": "train",
                "state": "Parcel A arrived at noon.",
                "instructions": common,
            }
        ]
        right = [
            {
                "id": "protected",
                "state": "Contract B expired yesterday.",
                "instructions": common,
            }
        ]
        complete = surface_pair_ids(left, right, surface="complete_input")
        state = surface_pair_ids(left, right, surface="state_evidence")
        self.assertEqual(complete["near"], [("train", "protected")])
        self.assertEqual(state["near"], [])

    def test_source_group_collision_uses_source_and_group(self) -> None:
        selected = [
            {"id": "t1", "source": "s1", "group_id": "g"},
            {"id": "t2", "source": "s2", "group_id": "g"},
        ]
        partitions = {
            "select": [
                {"id": "p1", "source": "s1", "group_id": "g"},
                {"id": "p2", "source": "s3", "group_id": "g"},
            ]
        }
        self.assertEqual(
            source_group_pairs(selected, partitions), {"select": [("t1", "p1")]}
        )
        self.assertEqual(
            source_group_pairs(selected, partitions, match_source=False),
            {
                "select": [
                    ("t1", "p1"),
                    ("t1", "p2"),
                    ("t2", "p1"),
                    ("t2", "p2"),
                ]
            },
        )

    def test_near_pair_profile_uses_inputs_and_family_only(self) -> None:
        selected = {
            "t": {
                "state": "Package A arrived on Monday after the audit.",
                "family": "f",
                "source": "train-source",
                "group_id": "g1",
            }
        }
        metadata = {
            "p": {
                "family": "f",
                "source": "protected-source",
                "group_id": "g2",
                "label": object(),
            }
        }
        protected = {"p": {"state": "Package A arrived on Tuesday after the audit."}}
        result = near_pair_profile([("t", "p")], selected, metadata, protected)
        self.assertEqual(result["pairs"], 1)
        self.assertEqual(result["same_family"], 1)
        self.assertEqual(result["same_source_group"], 0)


if __name__ == "__main__":
    unittest.main()

"""Prospective v2 oracle, orthogonality and native-contract checks."""

from __future__ import annotations

import collections
import json
import tempfile
import unittest
from pathlib import Path

from training.data.audit_sol2b_policy_conflicts import verify_blind_packet
from training.data.audit_sol2b_policy_conflicts_v2 import preliminary_overlap
from training.data.build_sol2b_policy_conflicts import (
    Candidate,
    label_counts,
    reference_level,
)
from training.data.build_sol2b_policy_conflicts import (
    build_group as build_v1_group,
)
from training.data.build_sol2b_policy_conflicts_v2 import (
    DIAGNOSTIC_SEED,
    SEED,
    build_all,
    build_group,
    write_private_packet,
)
from training.model.data import validate_row


class Sol2BPolicyConflictsV2Test(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.rows, cls.oracles = build_all()

    def test_512_complete_native_groups_and_joint_noul_balance(self) -> None:
        self.assertEqual((len(self.rows), len(self.oracles)), (1536, 512))
        by_group = collections.defaultdict(set)
        for row in self.rows:
            validate_row(row, "train")
            by_group[row["group_id"]].add(row["task_type"])
        self.assertEqual(len(by_group), 512)
        self.assertTrue(
            all(types == {"choice", "noul", "score"} for types in by_group.values())
        )
        self.assertEqual(
            label_counts(self.rows),
            {
                "choice": {"A": 256, "B": 256},
                "noul": {"false": 256, "true": 256},
                "score": {"0": 172, "1": 172, "2": 168},
            },
        )
        joint = collections.Counter(
            (
                oracle["facts"]["family"],
                oracle["noul_query_mode"],
                oracle["answers"]["noul"],
                oracle["noul_subject_side"],
            )
            for oracle in self.oracles
        )
        self.assertEqual(len(joint), 4 * 2 * 2 * 2)
        self.assertEqual(set(joint.values()), {16})
        self.assertEqual(len({oracle["fact_digest"] for oracle in self.oracles}), 512)

    def test_oracle_reexecution_and_position_logic(self) -> None:
        for index, oracle in enumerate(self.oracles):
            choice, noul, score = self.rows[index * 3 : index * 3 + 3]
            candidates = {
                facts["name"]: Candidate(
                    name=facts["name"],
                    credential=facts["credential"],
                    opening=facts["opening"],
                    required=facts["required"],
                    events=tuple(tuple(event) for event in facts["events"]),
                    hold=facts["hold"],
                    release=facts["release"],
                    signatures=facts["signatures"],
                    pending=facts["pending"],
                )
                for facts in (oracle["facts"]["left"], oracle["facts"]["right"])
            }
            levels = {
                name: reference_level(candidate)
                for name, candidate in candidates.items()
            }
            self.assertEqual(levels, oracle["levels"])
            choice_answer = choice["options"][choice["label"]]["description"]
            self.assertEqual(choice_answer, oracle["answers"]["choice"])
            self.assertEqual(levels[choice_answer], max(levels.values()))
            self.assertEqual(
                choice["options"][1 - choice["label"]]["description"],
                next(name for name in levels if name != choice_answer),
            )
            noul_level = levels[oracle["noul_subject"]]
            noul_answer = (
                noul_level == 2
                if oracle["noul_query_mode"] == "full"
                else noul_level >= 1
            )
            self.assertEqual(noul_answer, oracle["answers"]["noul"])
            self.assertEqual(
                noul["options"][noul["label"]]["key"],
                str(noul_answer).lower(),
            )
            self.assertEqual(levels[oracle["score_subject"]], score["label"])

    def test_frozen_diagnostic_is_group_disjoint(self) -> None:
        diagnostic, _ = build_all(groups_per_family=32, seed=DIAGNOSTIC_SEED)
        self.assertEqual(len({row["group_id"] for row in diagnostic}), 128)
        self.assertFalse(
            {row["input_sha256"] for row in diagnostic}
            & {row["input_sha256"] for row in self.rows}
        )

    def test_preliminary_overlap_counts_and_v1_independence(self) -> None:
        v1_rows, _ = build_v1_group("eligibility", 0)
        structured_state = {**v1_rows[0], "state": {"field": "structured"}}
        report = preliminary_overlap(
            self.rows[:3],
            {"self": self.rows[:3], "v1": v1_rows, "structured": [structured_state]},
        )
        self.assertEqual(report["self"]["shared_input_sha256"], 3)
        self.assertEqual(report["self"]["shared_normalized_state"], 1)
        self.assertEqual(report["self"]["shared_fact_digest"], 1)
        self.assertEqual(report["v1"]["shared_input_sha256"], 0)
        self.assertEqual(report["v1"]["shared_fact_digest"], 0)
        self.assertEqual(report["structured"]["shared_normalized_state"], 0)

    def test_blind_packet_structure_and_determinism(self) -> None:
        first, first_oracle = build_group("eligibility", 7, seed=SEED)
        second, second_oracle = build_group("eligibility", 7, seed=SEED)
        self.assertEqual((first, first_oracle), (second, second_oracle))
        with tempfile.TemporaryDirectory() as directory:
            packet = Path(directory) / "packet"
            write_private_packet(packet)
            blind = [
                json.loads(line)
                for line in (packet / "blind-review-48.jsonl").read_text().splitlines()
            ]
            self.assertEqual(verify_blind_packet(blind)["blind_groups"], 48)
            blind[0]["questions"][0]["label"] = 0
            with self.assertRaisesRegex(ValueError, "exposes a label"):
                verify_blind_packet(blind)


if __name__ == "__main__":
    unittest.main()

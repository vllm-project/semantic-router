"""Executable-oracle and data-contract tests for the Sol 2B source candidate."""

from __future__ import annotations

import collections
import itertools
import json
import tempfile
import unittest
from pathlib import Path

from training.data.audit_sol2b_policy_conflicts import (
    shortcut_probe,
    verify_blind_packet,
)
from training.data.build_sol2b_policy_conflicts import (
    FAMILIES,
    SEED,
    Candidate,
    build_all,
    build_group,
    label_counts,
    oracle_level,
    reference_level,
    write_private_packet,
)
from training.model.data import validate_row


class Sol2BPolicyConflictsTest(unittest.TestCase):
    def test_ordered_reference_oracle_matches_primary_on_state_grid(self) -> None:
        for (
            credential,
            hold,
            release,
            pending,
            signatures,
            opening,
        ) in itertools.product(
            (False, True),
            (False, True),
            (False, True),
            (False, True),
            range(4),
            range(4),
        ):
            item = Candidate(
                name="test",
                credential=credential,
                opening=opening,
                required=2,
                events=((1, True), (-2, False), (0, True)),
                hold=hold,
                release=release,
                signatures=signatures,
                pending=pending,
            )
            self.assertEqual(oracle_level(item), reference_level(item))

    def test_complete_native_groups_and_balanced_labels(self) -> None:
        rows, oracles = build_all()
        self.assertEqual((len(rows), len(oracles)), (1536, 512))
        self.assertEqual(len({item["fact_digest"] for item in oracles}), 512)
        self.assertEqual(
            label_counts(rows),
            {
                "choice": {"A": 256, "B": 256},
                "noul": {"false": 256, "true": 256},
                "score": {"0": 172, "1": 172, "2": 168},
            },
        )
        types = collections.defaultdict(set)
        for row in rows:
            validate_row(row, "train")
            types[row["group_id"]].add(row["task_type"])
        self.assertTrue(
            all(value == {"choice", "noul", "score"} for value in types.values())
        )
        self.assertEqual(
            collections.Counter(item["facts"]["family"] for item in oracles),
            dict.fromkeys(FAMILIES, 128),
        )

    def test_choice_key_position_and_oracle_answers(self) -> None:
        for family in FAMILIES:
            for index in range(128):
                rows, oracle = build_group(family, index)
                choice, noul, score = rows
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
                answer = choice["options"][choice["label"]]["description"]
                self.assertEqual(answer, oracle["answers"]["choice"])
                self.assertEqual(levels[answer], max(levels.values()))
                self.assertEqual(
                    noul["options"][noul["label"]]["key"],
                    str(oracle["answers"]["noul"]).lower(),
                )
                noul_level = levels[oracle["noul_subject"]]
                self.assertEqual(
                    oracle["answers"]["noul"],
                    (
                        noul_level == 2
                        if oracle["noul_query_mode"] == "full"
                        else noul_level >= 1
                    ),
                )
                self.assertEqual(score["label"], oracle["answers"]["score"])
                self.assertEqual(levels[oracle["score_subject"]], score["label"])
                reverse = list(reversed(choice["options"]))
                reverse_label = next(
                    i
                    for i, option in enumerate(reverse)
                    if option["description"] == oracle["answers"]["choice"]
                )
                self.assertEqual(reverse_label, 1 - choice["label"])
                self.assertEqual(
                    oracle["levels"][oracle["answers"]["choice"]],
                    max(oracle["levels"].values()),
                )

    def test_generation_is_deterministic(self) -> None:
        first, first_oracle = build_group("eligibility", 7)
        second, second_oracle = build_group("eligibility", 7)
        self.assertEqual((first, first_oracle), (second, second_oracle))

    def test_blind_packet_structure_and_answer_exclusion(self) -> None:
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

    def test_fixed_state_removed_shortcut_screen_fails_noul(self) -> None:
        diagnostic, _ = build_all(groups_per_family=32, seed=SEED + "-diagnostic")
        result = shortcut_probe(diagnostic)
        self.assertEqual(
            {kind: values["n"] for kind, values in result.items()},
            {"choice": 128, "noul": 128, "score": 128},
        )
        self.assertEqual(result["noul"]["majority_accuracy"], 0.5)
        self.assertEqual(result["noul"]["state_removed_accuracy"], 0.671875)
        self.assertFalse(result["noul"]["gate_pass"])


if __name__ == "__main__":
    unittest.main()

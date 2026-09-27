"""Synthetic-only tests for preregistered ANLI TRAIN source screening."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pyarrow as pa
import pyarrow.parquet as pq

from training.data import audit_anli_score_train_source as screen
from training.model.data import file_sha256


class CharacterTokenizer:
    def encode(self, value: str, *, add_special_tokens: bool) -> list[int]:
        return [ord(character) for character in value]


def row(
    round_id: int, position: int, premise: str, hypothesis: str, label: int = 0
) -> screen.TrainRow:
    return screen.TrainRow(round_id, position, premise, hypothesis, label, False)


class AnliTrainScreenTests(unittest.TestCase):
    def test_pinned_train_schema_hash_and_uid_uniqueness(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            frozen = {}
            for round_id in (1, 2, 3):
                filename = f"synthetic-r{round_id}.parquet"
                pq.write_table(
                    pa.table(
                        {
                            "uid": [f"synthetic-{round_id}"],
                            "premise": [f"Synthetic source premise {round_id}"],
                            "hypothesis": ["An invented claim"],
                            "label": [round_id - 1],
                            "reason": [""],
                        }
                    ),
                    root / filename,
                )
                frozen[round_id] = (filename, 1, file_sha256(root / filename))
            with patch.dict(screen.TRAIN_FILES, frozen, clear=True):
                rows, hashes = screen.read_train(root)
                self.assertEqual(len(rows), 3)
                self.assertEqual(len(hashes), 3)
                screen.TRAIN_FILES[3] = (frozen[3][0], 1, "0" * 64)
                with self.assertRaisesRegex(ValueError, "changed"):
                    screen.read_train(root)

    def test_group_sampling_is_label_blind_whole_and_budgeted(self) -> None:
        shared = "A shared premise in two rounds must be excluded."
        first = "A long, unique synthetic premise for the first round."
        second = "A long, unique synthetic premise for the second round."
        third = "A long, unique synthetic premise for the third round."
        dev = "A premise already present in open development."
        rows = [
            row(1, 0, first, "The item is blue.", 0),
            row(1, 1, first, "The item is red.", 2),
            row(1, 2, shared, "A shared example.", 1),
            row(2, 0, shared, "A second shared example.", 0),
            row(2, 1, second, "Another source example.", 0),
            row(2, 2, dev, "A development example.", 1),
            row(3, 0, third, "A third source example.", 2),
        ]
        with patch.object(screen, "MAX_ROWS_PER_ROUND", 2):
            selected, profile, lengths = screen.sample_whole_groups(
                rows, {screen.normalize(dev)}, set(), CharacterTokenizer()
            )
        self.assertEqual(len(selected), 4)
        self.assertEqual(len(lengths), 4)
        self.assertEqual(profile["selected_rows_by_round"], {"r1": 2, "r2": 1, "r3": 1})
        self.assertEqual(profile["excluded_cross_round_groups"], 1)
        self.assertEqual(profile["excluded_dev_exact_groups"], 1)
        self.assertEqual(
            {item.group for item in selected},
            {first.casefold(), second.casefold(), third.casefold()},
        )
        changed_labels = [
            row(
                item.round,
                item.position,
                item.premise,
                item.hypothesis,
                (item.label + 1) % 3,
            )
            for item in rows
        ]
        label_blind, _, _ = screen.sample_whole_groups(
            changed_labels, {screen.normalize(dev)}, set(), CharacterTokenizer()
        )
        self.assertEqual(
            [(item.round, item.position) for item in selected],
            [(item.round, item.position) for item in label_blind],
        )

    def test_protected_exact_quarantines_whole_premise_group(self) -> None:
        premise = "A distinctive blue lantern was shipped to the applicant."
        rows = [
            row(1, 0, premise, "The lantern was blue."),
            row(1, 1, premise, "The lantern was red."),
        ]
        excluded, counts = screen.protected_exact_groups(
            rows,
            {
                "synthetic_protected": [
                    {
                        "id": "protected",
                        "state": premise,
                    }
                ]
            },
        )
        self.assertEqual(excluded, {screen.normalize(premise)})
        self.assertEqual(counts["synthetic_protected"]["exact_raw"], 1)
        self.assertEqual(counts["synthetic_protected"]["exact_normalized"], 1)

    def test_shortcut_cues_are_fixed_and_aggregate(self) -> None:
        self.assertIn("not", screen._cue_names("The object is not visible."))
        self.assertIn("number_or_date", screen._cue_names("The event was in 1998."))
        self.assertIn("length_0_80", screen._cue_names("A brief example"))
        self.assertNotIn("premise", screen._cue_names("A brief example"))
        rows = [
            row(
                round_id,
                index,
                f"Synthetic premise {round_id}-{index}",
                "The item is not blue.",
                index % 3,
            )
            for round_id in (1, 2, 3)
            for index in range(60)
        ]
        result = screen.shortcut_profile(rows)
        self.assertEqual(set(result), {"r1", "r2", "r3"})
        self.assertEqual(result["r1"]["rows"], 60)
        self.assertIn("not", result["r1"]["hypothesis_only_cue_associations"])
        self.assertNotIn("Synthetic premise", str(result))


if __name__ == "__main__":
    unittest.main()

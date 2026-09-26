"""Train-only TweetEval conversion and gold-free CSS exclusion checks."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from transfer import build as transfer

from training.data import build_tweeteval_human as human


class TweetEvalHumanTests(unittest.TestCase):
    def test_balanced_sample_is_deterministic_and_uses_only_source_rows(self) -> None:
        rows = [
            {
                "source_id": f"hate/{index}",
                "gold_key": "not-hate" if index < 9 else "hate",
            }
            for index in range(12)
        ]
        first = human.balanced_sample(rows, 6, "seed", "hate")
        second = human.balanced_sample(rows, 6, "seed", "hate")
        self.assertEqual(first, second)
        self.assertEqual(
            {row["source_id"] for row in first}, {row["source_id"] for row in second}
        )
        self.assertEqual(sum(row["gold_key"] == "hate" for row in first), 3)

    def test_choice_order_changes_by_source_but_gold_stays_bound(self) -> None:
        rows = [
            human.make_train_row(
                {
                    "source_id": f"stance/hillary/{index}",
                    "state": f"A distinct opinion {index}",
                    "gold_key": "none",
                    "bucket": "stance/hillary",
                    "context_sha256": f"{index:064x}",
                },
                "fixed",
            )
            for index in range(20)
        ]
        self.assertGreater(
            len({tuple(option["key"] for option in row["options"]) for row in rows}), 1
        )
        self.assertTrue(
            all(row["options"][row["label"]]["key"] == "none" for row in rows)
        )

    def test_css_exclusions_read_prompt_files_without_opening_gold(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            pilot = root / "css-pilot.prompts.jsonl"
            evaluation = root / "css-evaluation.prompts.jsonl"
            dev = root / "dev.prompts.jsonl"
            pilot.write_text(
                json.dumps(
                    {
                        "id": "css/semeval_stance/one",
                        "state": "pilot context",
                        "questions": {"label": {"type": "choice"}},
                    }
                )
                + "\n"
            )
            evaluation.write_text(
                json.dumps(
                    {
                        "id": "css/emotion/two",
                        "state": "final context",
                        "questions": {"label": {"type": "choice"}},
                    }
                )
                + "\n"
            )
            dev.write_text(
                json.dumps({"id": "dev/three", "state": "development context"}) + "\n"
            )
            from training.data import build_pilot as data

            manifest = {
                "data_revision": transfer.DATA_REVISION,
                "replication_revision": transfer.REPLICATION_REVISION,
                "outputs": {
                    "pilot_prompts": {
                        "file": pilot.name,
                        "sha256": data.sha_file(pilot),
                        "n": 1,
                    },
                    "evaluation_prompts": {
                        "file": evaluation.name,
                        "sha256": data.sha_file(evaluation),
                        "n": 1,
                    },
                },
            }
            (root / "css-manifest.json").write_text(json.dumps(manifest))
            # If an implementation starts depending on final gold, no such
            # file exists in this fixture and this test fails.
            rows, receipts = human.read_gold_free_prompts(root, dev)
            self.assertEqual(
                [row["id"] for row in rows],
                ["css/semeval_stance/one", "css/emotion/two", "dev/three"],
            )
            self.assertEqual(receipts["evaluation_prompts"]["rows"], 1)

    def test_source_reader_never_opens_test_or_validation_labels(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for bucket in human.BUCKETS:
                directory = root / "datasets" / bucket
                directory.mkdir(parents=True)
                task = human.bucket_task(bucket)
                mapping = (
                    root / "datasets" / "stance" / "mapping.txt"
                    if task == "stance"
                    else directory / "mapping.txt"
                )
                mapping.write_text(
                    "".join(
                        f"{index}\t{label}\n"
                        for index, label in enumerate(human.OPTIONS[task])
                    )
                )
                (directory / "train_text.txt").write_text("Only this train sentence.\n")
                (directory / "train_labels.txt").write_text("0\n")
                (directory / "test_labels.txt").write_text("should never be read\n")
                if bucket == "hate":
                    (directory / "train_text.txt").write_text(
                        "Only this train sentence.\n\n"
                    )
                    (directory / "train_labels.txt").write_text("0\n1\n")
            with mock.patch.object(
                transfer, "git_head", return_value=human.TWEETEVAL_REVISION
            ), mock.patch.object(transfer, "require_clean_tracked_files"):
                source, receipts = human.source_rows(root)
            self.assertEqual(
                sum(len(rows) for rows in source.values()), len(human.BUCKETS)
            )
            self.assertEqual(receipts["hate"]["empty_text_rows"], 1)
            self.assertTrue(
                all(
                    set(info["files"])
                    == {"mapping.txt", "train_text.txt", "train_labels.txt"}
                    for info in receipts.values()
                )
            )


if __name__ == "__main__":
    unittest.main()

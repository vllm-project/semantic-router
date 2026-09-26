"""Frozen multilingual pilot quotas, lineage and gold-blind packet contract."""

from __future__ import annotations

import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path

from training.data import build_massive_locale_review as locale_review


class LocaleReviewTests(unittest.TestCase):
    def make_candidate(self, root: Path) -> tuple[Path, str, str]:
        candidate = root / "candidate"
        candidate.mkdir()
        targets = [
            intent for names in locale_review.INTENT_STRATA.values() for intent in names
        ]
        rows = []
        options = [{"key": key, "description": f"Action {key}"} for key in "ABCDEF"]
        for index in range(481):
            source_id = str(1000 + index)
            intent = targets[index] if index < len(targets) else "weather_query"
            for source_locale in ("en-US", *locale_review.LOCALES):
                rows.append(
                    {
                        "id": f"massive-1.1:{source_id}:{source_locale}",
                        "group_id": f"massive-1.1:{source_id}",
                        "state": f"{source_locale} request {source_id}",
                        "instructions": f"{source_locale} choose one",
                        "options": options,
                        "label": 2,
                        "audit_metadata": {
                            "source_id": source_id,
                            "source_locale": source_locale,
                            "intent": intent,
                        },
                    }
                )
        locale_review.write_jsonl(candidate / "train.private.jsonl", rows)
        train_sha = locale_review.sha(candidate / "train.private.jsonl")
        (candidate / "LICENSE").write_text("CC BY 4.0")
        (candidate / "NOTICE.md").write_text("MASSIVE and SLURP")
        manifest = {
            "training_approved": False,
            "cross_locale_semantic_review_pending": True,
            "source_groups": {"kept_train": 481},
            "rows": {"train": 3367},
            "train_intent_coverage": 59,
            "outputs": {"train.private.jsonl": train_sha},
            "rights": {
                "original_license_sha256": locale_review.sha(candidate / "LICENSE"),
                "original_notice_sha256": locale_review.sha(candidate / "NOTICE.md"),
            },
        }
        (candidate / "manifest.json").write_text(json.dumps(manifest))
        return candidate, locale_review.sha(candidate / "manifest.json"), train_sha

    def test_fixed_18_group_six_locale_packet_hides_gold(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            candidate, manifest_sha, train_sha = self.make_candidate(root)
            first = root / "first"
            with self.assertRaisesRegex(ValueError, "input changed"):
                locale_review.build(candidate, first, expected_manifest_sha="0" * 64)
            receipt = locale_review.build(
                candidate,
                first,
                expected_manifest_sha=manifest_sha,
                expected_train_sha=train_sha,
            )
            packet = locale_review.read_jsonl(
                first / "locale-pilot.blind.private.jsonl"
            )
            key = locale_review.read_jsonl(first / "locale-pilot.key.private.jsonl")
            self.assertEqual(len(packet), 108)
            self.assertEqual(len(key), 108)
            self.assertEqual(len({row["parallel_group"] for row in packet}), 18)
            self.assertEqual(
                Counter(row["locale"] for row in packet),
                dict.fromkeys(locale_review.LOCALES, 18),
            )
            self.assertEqual(
                Counter(row["selection_stratum"] for row in key),
                dict.fromkeys(locale_review.INTENT_STRATA, 36),
            )
            self.assertEqual(len({row["source_id"] for row in key}), 18)
            self.assertEqual(
                {row["review_id"] for row in packet}, {row["review_id"] for row in key}
            )
            self.assertTrue(
                all(
                    "source_id" not in row
                    and "source_intent" not in row
                    and "gold_option_key" not in row
                    and "label" not in row
                    and "selection_stratum" not in row
                    for row in packet
                )
            )
            self.assertTrue(
                all(
                    [option["key"] for option in row["options"]] == list("ABCDEF")
                    for row in packet
                )
            )
            self.assertTrue(all(row["gold_option_key"] == "C" for row in key))
            self.assertFalse(receipt["training_approved"])
            self.assertEqual(first.stat().st_mode & 0o777, 0o700)
            self.assertTrue(
                all(path.stat().st_mode & 0o777 == 0o600 for path in first.iterdir())
            )
            second = root / "second"
            locale_review.build(
                candidate,
                second,
                expected_manifest_sha=manifest_sha,
                expected_train_sha=train_sha,
            )
            self.assertEqual(
                locale_review.sha(first / "locale-pilot.blind.private.jsonl"),
                locale_review.sha(second / "locale-pilot.blind.private.jsonl"),
            )
            self.assertEqual(
                locale_review.sha(first / "locale-pilot.key.private.jsonl"),
                locale_review.sha(second / "locale-pilot.key.private.jsonl"),
            )

    def test_parallel_option_drift_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            candidate, _, _ = self.make_candidate(root)
            rows = locale_review.read_jsonl(candidate / "train.private.jsonl")
            for row in rows:
                if (
                    row["audit_metadata"]["source_id"] == "1000"
                    and row["audit_metadata"]["source_locale"] == "ar-SA"
                ):
                    row["options"] = [*row["options"]]
                    row["options"][0] = {"key": "A", "description": "Different action"}
                    break
            (candidate / "train.private.jsonl").unlink()
            locale_review.write_jsonl(candidate / "train.private.jsonl", rows)
            manifest = json.loads((candidate / "manifest.json").read_text())
            manifest["outputs"]["train.private.jsonl"] = locale_review.sha(
                candidate / "train.private.jsonl"
            )
            (candidate / "manifest.json").write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "option, label or intent drift"):
                locale_review.build(
                    candidate,
                    root / "review",
                    expected_manifest_sha=locale_review.sha(
                        candidate / "manifest.json"
                    ),
                    expected_train_sha=manifest["outputs"]["train.private.jsonl"],
                )


if __name__ == "__main__":
    unittest.main()

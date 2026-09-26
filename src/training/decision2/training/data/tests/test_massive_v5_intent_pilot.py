"""Focused contracts for the prospective MASSIVE v5 intent-ID pilot."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from training.data import build_massive_multilingual as massive
from training.data import build_massive_v5_intent_pilot as v5
from training.data import build_pilot as pilot


def fixture_source(groups_per_intent: int = 4):
    source = {locale: {} for locale in massive.LOCALES}
    phrases = (
        "the copper lantern at dawn",
        "a willow basket near midnight",
        "the southern bridge after lunch",
        "a paper crane before sunrise",
    )
    domains = {
        "alarm_set": "ring a bedside alarm clock for",
        "weather_query": "forecast the rain and wind around",
        "play_music": "start a jazz playlist featuring",
        "general_joke": "tell a silly joke about",
    }
    for intent in v5.SOURCE_INTENT_ORDER:
        for number in range(groups_per_intent):
            identifier = f"{intent}-{number}"
            for locale in massive.LOCALES:
                source[locale][identifier] = {
                    "id": identifier,
                    "locale": locale,
                    "partition": "train",
                    "intent": intent,
                    "scenario": intent.split("_")[0],
                    "utt": f"{locale} {domains[intent]} {phrases[number]}",
                    "quality_pass": True,
                    "passing_votes": 3,
                }
    return source


class MassiveV5Contracts(unittest.TestCase):
    def test_selection_is_train_only_and_excludes_legacy_groups(self):
        source = fixture_source()
        source["en-US"]["alarm_set-0"]["partition"] = "dev"
        for locale in massive.LOCALES[1:]:
            source[locale]["alarm_set-0"]["partition"] = "dev"
        qualified = [
            identifier
            for identifier, row in source["en-US"].items()
            if row["partition"] == "train"
        ]
        with mock.patch.object(massive, "qualify", return_value={"train": qualified}):
            result = v5.candidates(source, {"weather_query-0"})
        self.assertNotIn("alarm_set-0", result["alarm_set"])
        self.assertNotIn("weather_query-0", result["weather_query"])
        self.assertEqual(len(result["alarm_set"]), 3)
        self.assertEqual(len(result["weather_query"]), 3)

    def test_one_locale_overlap_quarantines_whole_source_group(self):
        source = fixture_source()
        by_intent = {
            intent: [f"{intent}-{number}" for number in range(4)]
            for intent in v5.SOURCE_INTENT_ORDER
        }
        collision = source["zh-CN"]["alarm_set-0"]["utt"]
        protected = v5._context_rows([collision], "protected")
        chosen, audit = v5.overlap_filter(source, by_intent, protected)
        self.assertEqual(len(chosen), 12)
        self.assertNotIn("alarm_set-0", chosen)
        self.assertEqual(audit["quarantined_source_groups"], 1)

    def test_rows_and_two_blind_stages_have_no_visible_gold(self):
        source = fixture_source(groups_per_intent=3)
        identifiers = [
            f"{intent}-{number}"
            for intent in v5.SOURCE_INTENT_ORDER
            for number in range(3)
        ]
        rows = v5.make_rows(source, identifiers)
        self.assertEqual(len(rows), 84)
        self.assertEqual({row["task_type"] for row in rows}, {"choice"})
        self.assertEqual({row["split"] for row in rows}, {"train"})
        self.assertEqual({len(row["options"]) for row in rows}, {4})
        local, parallel, key = v5.make_review_packets(rows, b"test-private-salt")
        self.assertEqual((len(local), len(parallel), len(key)), (84, 12, 84))
        self.assertEqual(len({row["review_id"] for row in local}), 84)
        self.assertTrue(all("group_token" not in row for row in local))
        self.assertEqual(len({row["group_token"] for row in parallel}), 12)
        self.assertFalse(any("intent" in row or "source_id" in row for row in local))
        self.assertTrue(all("gold_option_key" in row for row in key))
        self.assertTrue(all(len(row["localized"]) == 6 for row in parallel))

    def test_legacy_roster_requires_all_seven_locales(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "train.private.jsonl"
            source = fixture_source(groups_per_intent=1)
            with path.open("w", encoding="utf-8") as stream:
                for intent in v5.SOURCE_INTENT_ORDER:
                    for locale in massive.LOCALES:
                        stream.write(
                            json.dumps(
                                {
                                    "split": "train",
                                    "audit_metadata": {
                                        "source_id": intent,
                                        "source_locale": locale,
                                        "source_partition": "train",
                                    },
                                }
                            )
                            + "\n"
                        )
            with mock.patch.object(v5, "LEGACY_TRAIN_SHA", pilot.sha_file(path)):
                with self.assertRaisesRegex(ValueError, "group inventory"):
                    v5.legacy_ids(path)
            self.assertTrue(source)


if __name__ == "__main__":
    unittest.main()

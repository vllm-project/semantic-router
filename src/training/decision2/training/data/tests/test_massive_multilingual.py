"""MASSIVE data contracts: human votes, lineage, options and quarantine."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from training.data import build_massive_multilingual as massive


def judgment(*, intent=1, grammar=3, language="target"):
    return {
        "intent_score": intent,
        "grammar_score": grammar,
        "language_identification": language,
    }


def fixture_source():
    source = {locale: {} for locale in massive.LOCALES}
    intents = [
        "alarm_set",
        "alarm_remove",
        "alarm_query",
        *(
            name
            for name in sorted(massive.INTENT_DESCRIPTIONS)
            if name not in {"alarm_set", "alarm_remove", "alarm_query"}
        ),
    ]
    for identifier, intent in enumerate(intents):
        scenario = intent.split("_")[0]
        for locale in massive.LOCALES:
            utterance = (
                "please wake me at dawn"
                if identifier == 0
                else (
                    "cancel the morning alarm"
                    if identifier == 1
                    else (
                        "what alarms are active"
                        if identifier == 2
                        else f"request {identifier} for {intent}"
                    )
                )
            )
            source[locale][str(identifier)] = {
                "id": str(identifier),
                "locale": locale,
                "partition": "train",
                "intent": intent,
                "scenario": scenario,
                "utt": f"{locale}: {utterance}",
                "quality_pass": True,
                "passing_votes": 3,
            }
    return source


class MassiveContractTests(unittest.TestCase):
    def test_two_of_three_joint_intent_grammar_and_language_votes(self):
        self.assertTrue(
            massive.quality_pass(
                [
                    judgment(intent=1),
                    judgment(intent=2, language="target|english"),
                    judgment(intent=0),
                ]
            )
        )
        self.assertFalse(
            massive.quality_pass(
                [judgment(grammar=2), judgment(language="english"), judgment(intent=2)]
            )
        )
        self.assertFalse(massive.quality_pass([judgment(), judgment()]))
        self.assertFalse(massive.quality_pass([judgment()]))
        with self.assertRaises(ValueError):
            massive.quality_pass([judgment()] * 4)

    def test_dynamic_options_include_hard_neighbors_and_stable_permutation(self):
        scenarios = {
            "alarm_set": "alarm",
            "alarm_remove": "alarm",
            "alarm_query": "alarm",
            "weather_query": "weather",
            "news_query": "news",
            "play_game": "play",
            "qa_stock": "qa",
            "social_post": "social",
        }
        intents = sorted(scenarios)
        first = massive.option_intents("alarm_set", "alarm", intents, scenarios, "42")
        self.assertEqual(
            first,
            massive.option_intents("alarm_set", "alarm", intents, scenarios, "42"),
        )
        self.assertEqual(len(set(first)), 6)
        self.assertEqual(
            set(first) & {"alarm_query", "alarm_remove"},
            {"alarm_query", "alarm_remove"},
        )
        self.assertIn("alarm_set", first)

    def test_supply_aware_balancing_preserves_all_intents(self):
        source = fixture_source()
        for identifier, donor in (("60", "0"), ("61", "1")):
            for locale in massive.LOCALES:
                source[locale][identifier] = {
                    **source[locale][donor],
                    "id": identifier,
                    "utt": f"{locale}: distinct request {identifier}",
                }
        eligible = [str(i) for i in range(62)]
        quotas = massive.balanced_quotas(source, eligible, 62, "train")
        self.assertEqual(sum(quotas.values()), 62)
        self.assertEqual(len(quotas), 60)
        self.assertTrue(all(value >= 1 for value in quotas.values()))
        selected = massive.select_ids(source, eligible, 60)
        self.assertEqual(len(selected), 60)
        self.assertEqual(len({source["en-US"][i]["intent"] for i in selected}), 60)
        missing_dev = [
            identifier
            for identifier in eligible
            if source["en-US"][identifier]["intent"] != "alarm_set"
        ]
        self.assertEqual(len(massive.select_ids(source, missing_dev, 59)), 59)
        with self.assertRaisesRegex(ValueError, "lacks a required intent"):
            massive.select_ids(source, missing_dev, 59, require_all_intents=True)

    def test_source_id_is_one_seven_locale_group(self):
        source = fixture_source()
        rows = massive.make_rows(source, ["0"], "train")
        self.assertEqual(len(rows), 7)
        self.assertEqual({row["group_id"] for row in rows}, {"massive-1.1:0"})
        self.assertEqual({row["label"] for row in rows}, {rows[0]["label"]})
        self.assertEqual(
            {tuple(option["description"] for option in row["options"]) for row in rows},
            {tuple(option["description"] for option in rows[0]["options"])},
        )
        self.assertEqual(
            {row["language"] for row in rows},
            {locale.split("-")[0] for locale in massive.LOCALES},
        )
        self.assertEqual(len({row["instructions"] for row in rows}), 7)

    def test_semantic_review_packet_hides_gold_from_reviewer(self):
        source = fixture_source()
        packet, key = massive.make_blind_review(source, [str(i) for i in range(60)])
        self.assertEqual(len(packet), 420)
        self.assertEqual(len(key), 420)
        self.assertEqual(
            {r["review_id"] for r in packet}, {r["review_id"] for r in key}
        )
        self.assertTrue(
            all(
                "intent" not in row
                and "gold_option_key" not in row
                and "source_id" not in row
                for row in packet
            )
        )
        self.assertTrue(all(row["gold_option_key"] in "ABCDEF" for row in key))

    def test_exact_protected_hit_quarantines_every_locale(self):
        source = fixture_source()
        protected = [
            {
                "id": "unrelated",
                "group_id": "protected",
                "state": source["ja-JP"]["1"]["utt"],
            }
        ]
        eligible, audit = massive.quarantine(
            source, {"train": ["0", "1"], "dev": ["2"]}, protected
        )
        self.assertNotIn("1", eligible["train"])
        self.assertEqual(eligible["train"], ["0"])
        self.assertEqual(audit["quarantined_groups"], 1)

    def test_cross_split_near_context_is_rejected(self):
        source = fixture_source()
        train = massive.make_rows(source, ["0"], "train")
        dev = massive.make_rows(source, ["1"], "select")
        dev[0]["state"] = train[0]["state"]
        with self.assertRaisesRegex(ValueError, "context collision"):
            massive.cross_split_audit(train, dev)

    def test_cross_split_quarantine_removes_entire_parallel_group(self):
        source = fixture_source()
        source["ja-JP"]["1"]["utt"] = source["ja-JP"]["0"]["utt"]
        kept, receipt = massive.quarantine_train_neighbors(source, ["0"], ["1", "2"])
        self.assertEqual(kept, ["2"])
        self.assertEqual(receipt["candidate_groups"], 2)
        self.assertEqual(receipt["quarantined_groups"], 1)
        self.assertGreaterEqual(receipt["exact_context_rows"], 1)

    def test_frozen_reference_hash_required_and_identical_files_loaded_once(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "reference.jsonl"
            path.write_text(json.dumps({"id": "x", "state": "A request"}) + "\n")
            receipt = {
                "one": ("reference.jsonl", massive.sha(path)),
                "two": ("reference.jsonl", massive.sha(path)),
            }
            with mock.patch.object(massive, "REFERENCE_FILES", receipt):
                rows, metadata = massive.load_references(root)
                self.assertEqual(len(rows), 1)
                self.assertEqual(set(metadata), {"one", "two"})
                path.write_text(json.dumps({"id": "x", "state": "Changed"}) + "\n")
                with self.assertRaisesRegex(ValueError, "Frozen protected"):
                    massive.load_references(root)

    def test_no_final_gold_reference_contract(self):
        self.assertFalse(
            any(
                "final" in role.lower() or "target" in relative.lower()
                for role, (relative, _) in massive.REFERENCE_FILES.items()
            )
        )
        self.assertTrue(
            all(
                len(digest) == 64 and all(c in "0123456789abcdef" for c in digest)
                for _, digest in massive.REFERENCE_FILES.values()
            )
        )
        for stem in ("human", "structured", "clean"):
            self.assertTrue(
                {f"{stem}_train", f"{stem}_select", f"{stem}_cal"}
                <= set(massive.REFERENCE_FILES)
            )


if __name__ == "__main__":
    unittest.main()

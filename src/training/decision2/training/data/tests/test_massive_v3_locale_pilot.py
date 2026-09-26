"""Contract tests for the quarantined, gold-blind localized-option pilot."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from training.data import build_massive_v3_locale_pilot as pilot


def prior_review_fixture() -> tuple[list[dict], list[dict], list[dict], list[dict]]:
    packet, key, verdict, postkey = [], [], [], []
    options = [
        {"key": letter, "description": description}
        for letter, description in zip(
            "ABCDEF",
            (
                "Set an alarm",
                "Cancel an alarm",
                "Check existing alarms",
                "Ask a math question",
                "Ask for event recommendations",
                "Turn the lights off",
            ),
        )
    ]
    for group in range(18):
        for locale in pilot.LOCALES:
            review_id = f"r{group:02d}-{locale}"
            parallel_group = f"g{group:02d}"
            shared = {
                "review_id": review_id,
                "parallel_group": parallel_group,
                "locale": locale,
            }
            packet.append(
                {
                    **shared,
                    "english_utterance": f"set alarm {group}",
                    "localized_utterance": f"localized {group}",
                    "options": options,
                }
            )
            key.append(
                {
                    **shared,
                    "source_id": str(group),
                    "source_intent": "alarm_set",
                    "gold_option_key": "A",
                }
            )
            strict = group < 10
            verdict.append(
                {
                    **shared,
                    "strict_parallel_pass": strict,
                    "label_validity": strict,
                    "source_intent_preserved": True,
                    "semantic_tier": "exact" if strict else "material",
                    "localized_utterance_tier": "target_language",
                }
            )
            postkey.append(
                {
                    **shared,
                    "source_id": str(group),
                    "source_intent": "alarm_set",
                    "key_option": "A",
                    "blind_label_validity": strict,
                    "blind_strict_parallel_pass": strict,
                }
            )
    return packet, key, verdict, postkey


class MassiveV3LocalePilotTests(unittest.TestCase):
    def test_only_complete_strict_groups_survive(self) -> None:
        selected, quarantine = pilot.choose_groups(*prior_review_fixture())
        self.assertEqual(len(selected), 60)
        self.assertEqual(
            {k["source_id"] for _, k, _ in selected}, {str(i) for i in range(10)}
        )
        self.assertEqual(quarantine, {"semantic_or_label_failure": 8})
        self.assertEqual({p["locale"] for p, _, _ in selected}, set(pilot.LOCALES))

    def test_one_invalid_locale_quarantines_entire_group(self) -> None:
        rows = prior_review_fixture()
        row = next(v for v in rows[2] if v["review_id"] == "r00-ar-SA")
        row["label_validity"] = False
        row["strict_parallel_pass"] = False
        post = next(v for v in rows[3] if v["review_id"] == "r00-ar-SA")
        post["blind_label_validity"] = False
        post["blind_strict_parallel_pass"] = False
        with self.assertRaisesRegex(ValueError, "ten all-six"):
            pilot.choose_groups(*rows)

    def test_sealed_key_join_must_match_blind_verdict(self) -> None:
        rows = prior_review_fixture()
        rows[3][0]["blind_label_validity"] = False
        with self.assertRaisesRegex(ValueError, "lineage"):
            pilot.choose_groups(*rows)

    def test_localized_options_preserve_keys_and_distinct_meanings(self) -> None:
        path = Path(pilot.__file__).with_name("massive_v3_option_descriptions.json")
        translations = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(len(translations), 41)
        self.assertTrue(
            all(len(value) == len(pilot.LOCALES) for value in translations.values())
        )
        options = [
            {"key": letter, "description": description}
            for letter, description in zip(
                "ABCDEF",
                (
                    "Lower audio volume",
                    "Raise audio volume",
                    "Mute audio",
                    "Change audio volume without a specified direction",
                    "Check social media",
                    "Ask a factual question",
                ),
            )
        ]
        for locale in pilot.LOCALES:
            translated = pilot.localize_options(options, locale, translations)
            self.assertEqual([row["key"] for row in translated], list("ABCDEF"))
            self.assertEqual(len({row["description"] for row in translated}), 6)
            self.assertTrue(
                all(
                    a["description"] != b["description"]
                    for a, b in zip(options, translated)
                )
            )


if __name__ == "__main__":
    unittest.main()

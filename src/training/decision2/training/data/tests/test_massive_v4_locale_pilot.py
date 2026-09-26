"""Contracts for the preregistered MASSIVE v4 blind locale packet."""

from __future__ import annotations

import hashlib
import json
import unittest
from pathlib import Path

from training.data import build_massive_v4_locale_pilot as pilot


def translations() -> dict:
    path = Path(pilot.__file__).with_name("massive_v3_option_descriptions.json")
    return json.loads(path.read_text(encoding="utf-8"))


def options(*, eligible: bool = True) -> list[dict]:
    descriptions = (
        "Set an alarm",
        "Cancel an alarm",
        "Check existing alarms",
        "Ask a math question",
        "Turn the lights off",
        "Ask about weather" if eligible else "Untranslated fixture action",
    )
    return [
        {"key": key, "description": description}
        for key, description in zip("ABCDEF", descriptions)
    ]


def group(source_id: str, intent: str, *, eligible: bool = True) -> list[dict]:
    result = []
    for locale in ("en-US", *pilot.LOCALES):
        result.append(
            {
                "audit_metadata": {
                    "source_id": source_id,
                    "source_locale": locale,
                    "intent": intent,
                },
                "language": locale.split("-")[0],
                "group_id": f"group-{source_id}",
                "state": f"utterance-{source_id}-{locale}",
                "instructions": f"instruction-{locale}",
                "options": options(eligible=eligible),
                "label": 0,
            }
        )
    return result


def source_inventory() -> tuple[list[dict], set[str]]:
    rows = []
    excluded = {f"prior-{n:02d}" for n in range(10)}
    for source_id in sorted(excluded):
        rows.extend(group(source_id, pilot.INTENTS[0]))
    # Eighty-three non-v3 groups span all twelve preregistered intents.
    for index in range(83):
        intent = pilot.INTENTS[index % 12]
        rows.extend(group(f"eligible-{index:03d}", intent))
    for index in range(388):
        rows.extend(group(f"outside-{index:03d}", "general_greet", eligible=False))
    return rows, excluded


class MassiveV4LocalePilotTests(unittest.TestCase):
    def test_hash_sample_is_fixed_and_excludes_every_v3_group(self) -> None:
        source, excluded = source_inventory()
        first, counts = pilot.select_groups(source, translations(), excluded)
        reversed_source, reversed_counts = pilot.select_groups(
            list(reversed(source)), translations(), excluded
        )
        self.assertEqual(
            counts, {"translatable": 93, "remaining_after_v3_exclusion": 83}
        )
        self.assertEqual(counts, reversed_counts)
        self.assertEqual(
            [(intent, source_id) for intent, source_id, _ in first],
            [(intent, source_id) for intent, source_id, _ in reversed_source],
        )
        self.assertEqual([intent for intent, _, _ in first], list(pilot.INTENTS))
        self.assertTrue(all(source_id not in excluded for _, source_id, _ in first))
        for intent, source_id, _ in first:
            candidates = [
                f"eligible-{index:03d}"
                for index in range(83)
                if pilot.INTENTS[index % 12] == intent
            ]
            expected = min(
                candidates,
                key=lambda item: (
                    hashlib.sha256(
                        "\0".join((pilot.SEED, intent, item)).encode()
                    ).hexdigest(),
                    item,
                ),
            )
            self.assertEqual(source_id, expected)

    def test_packet_never_leaks_source_id_intent_or_gold(self) -> None:
        selected = [
            (intent, f"source-{index:02d}", group(f"source-{index:02d}", intent))
            for index, intent in enumerate(pilot.INTENTS)
        ]
        packet, key = pilot.make_rows(selected, translations())
        self.assertEqual(len(packet), len(key), 72)
        self.assertEqual(len({r["parallel_group"] for r in packet}), 12)
        self.assertEqual({r["locale"] for r in packet}, set(pilot.LOCALES))
        self.assertTrue(
            all(
                not {"source_id", "source_intent", "gold_option_key", "label"}
                & set(row)
                for row in packet
            )
        )
        self.assertEqual({r["gold_option_key"] for r in key}, {"A"})
        self.assertEqual(
            {r["review_id"] for r in packet}, {r["review_id"] for r in key}
        )

    def test_any_parallel_label_drift_aborts_packet(self) -> None:
        selected = [
            (intent, f"source-{index:02d}", group(f"source-{index:02d}", intent))
            for index, intent in enumerate(pilot.INTENTS)
        ]
        selected[0][2][1]["label"] = 1
        with self.assertRaisesRegex(ValueError, "parallel source group"):
            pilot.make_rows(selected, translations())

    def test_collapsed_localized_choices_abort_packet(self) -> None:
        table = translations()
        table["Cancel an alarm"] = list(table["Set an alarm"])
        with self.assertRaisesRegex(ValueError, "collapse"):
            pilot.localize_options(options(), "fr-FR", table)


if __name__ == "__main__":
    unittest.main()

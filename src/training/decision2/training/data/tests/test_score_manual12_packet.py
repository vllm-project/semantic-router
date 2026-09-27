"""Contract checks for a sealed, answer-free manual Score review packet."""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

from training.data import score_manual12_packet as packet


def _cases() -> list[dict]:
    rows = []
    for index, mechanism in enumerate(sorted(packet.MECHANISMS)):
        languages = (
            (("short", "en"), ("long", "zh"))
            if index < 3
            else (("short", "zh"), ("long", "en"))
        )
        for length, language in languages:
            rows.append(
                {
                    "id": f"{mechanism}-{length}",
                    "mechanism": mechanism,
                    "length": length,
                    "language": language,
                    "question": "Pick a level from the decisive source.",
                    "before": f"The source for {mechanism} follows.",
                    "after": "No other result is recorded.",
                    "variants": [
                        {
                            "evidence": f"{mechanism} {length} source says negative.",
                            "fatal": True,
                            "pending": False,
                            "reason": "negative",
                        },
                        {
                            "evidence": f"{mechanism} {length} source says undecided.",
                            "fatal": False,
                            "pending": True,
                            "reason": "undecided",
                        },
                        {
                            "evidence": f"{mechanism} {length} source says positive.",
                            "fatal": False,
                            "pending": False,
                            "reason": "positive",
                        },
                    ],
                }
            )
    return rows


class ManualScorePacketTest(unittest.TestCase):
    def test_seal_hides_key_and_group_distribution(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "cases.json"
            seed = root / "seed.bin"
            output = root / "sealed"
            source.write_text(json.dumps(_cases()), encoding="utf-8")
            seed.write_bytes(bytes(range(32)))
            os.chmod(seed, 0o600)
            result = packet.seal(source, seed, output)
            blind = (output / "blind-questions.jsonl").read_text(encoding="utf-8")
            manifest = json.loads((output / "blind-manifest.json").read_text())
            keys = [
                json.loads(line)
                for line in (output / "sealed-key.jsonl").read_text().splitlines()
            ]
            self.assertEqual(result["manifest"]["items"], 36)
            self.assertEqual(
                set(manifest), {"schema", "items", "questions_sha256", "sealed_at_utc"}
            )
            self.assertNotIn('"label"', blind)
            self.assertNotIn('"group_id"', blind)
            self.assertEqual(sorted({row["label"] for row in keys}), [0, 1, 2])
            self.assertEqual(len({row["review_id"] for row in keys}), 36)
            self.assertEqual(
                packet.sha_file(output / "blind-questions.jsonl"),
                manifest["questions_sha256"],
            )

    def test_rejects_broken_counterfactual_group(self) -> None:
        cases = _cases()
        cases[0]["variants"][1]["fatal"] = True
        cases[0]["variants"][1]["pending"] = False
        with self.assertRaises(ValueError):
            packet._inspect(cases)


if __name__ == "__main__":
    unittest.main()

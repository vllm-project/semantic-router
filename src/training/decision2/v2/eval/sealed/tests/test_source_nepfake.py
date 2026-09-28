from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from v2.eval.sealed import candidates as runner
from v2.eval.sealed.schema import validate
from v2.eval.sealed.sources import nepfake

NFC = "https://nepalfactcheck.org/{}/example-slug/"


def row(example_id, verdict, headline, date, url=None, source="nepalfactcheck"):
    return {
        "example_id": example_id,
        "claim_text": headline,
        "verdict_label": {"REAL": 0, "FALSE_MISLEADING": 1, "UNVERIFIED": 2}[verdict],
        "verdict_label_text": verdict,
        "evidence_text": "निष्कर्ष: लेखको मुख्य भाग।",
        "source_name": source,
        "source_url": url or NFC.format(date[:7].replace("-", "/")),
        "date_published": date,
        "is_native_nepali": True,
        "topic_category": "other",
        "annotator_notes": "",
    }


ROWS = [
    row("NF2_1", "FALSE_MISLEADING", "काठमाडौंमा हिउँ परेको भिडियो भाइरल", "2026-07-14"),
    row("NF2_2", "REAL", "के सरकारले नयाँ बिदा घोषणा गरेको हो?", "2026-06-05"),
    row("NF2_3", "UNVERIFIED", "नदीमा गोही देखिएको दाबी", "2026-06-06"),
    row("NF2_4", "FALSE_MISLEADING", "पुलको तस्बिरबारे भ्रामक दाबी", "2026-06-07"),
    row(
        "NF2_5",
        "REAL",
        "बजेटबारे दाबी",
        "2026-06-08",
        url="https://techpana.com/2026/150000/example",
        source="techpana",
    ),
    row("NF2_6", "FALSE_MISLEADING", "पुरानो तस्बिर फेरि भाइरल", "2026-05-20"),
    row(
        "NF2_7",
        "REAL",
        "मौसमबारे सूचना",
        "2026-07-02",
        url=NFC.format("2026/06"),
    ),
    row("NF2_8", "FALSE_MISLEADING", "Viral video of a flood", "2026-08-01"),
    row("NF2_9", "FALSE_MISLEADING", "प्रधानमन्त्रीको भ्रमणको तस्बिर", "2026-09-01"),
]


def snapshot(tmp: str) -> Path:
    root = Path(tmp) / "snap"
    (root / "data").mkdir(parents=True)
    (root / "data" / "nepfakev2.json").write_text(
        json.dumps(ROWS, ensure_ascii=False), encoding="utf-8"
    )
    return root


class NepFakeTest(unittest.TestCase):
    def test_verdict_wording(self):
        for text in ("यो दाबी मिथ्या हो", "भनाइ सही हो", "झुटो खबर", "Fake news"):
            self.assertTrue(nepfake.states_verdict(text), text)
        for text in ("प्रधानमन्त्रीको भ्रमण", "दाबीको सत्यता के हो?", "सहीछाप"):
            self.assertFalse(nepfake.states_verdict(text), text)

    def test_filters_and_mapping(self):
        with tempfile.TemporaryDirectory() as tmp:
            items = list(nepfake.candidates(snapshot(tmp)))
        self.assertEqual([c.source_item_id for c in items], ["NF2_1", "NF2_2", "NF2_9"])
        self.assertEqual([c.gold for c in items], [True, False, True])
        urls = {r["example_id"]: r["source_url"] for r in ROWS}
        for item in items:
            self.assertEqual(validate(item, nepfake.SPEC), [])
            self.assertEqual(set(item.state), {"headline", "published"})
            self.assertEqual(item.date, item.state["published"])
            self.assertEqual(item.group_id, urls[item.source_item_id])
            self.assertEqual(item.balance_label, str(item.gold))

    def test_runner_counts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = snapshot(tmp)
            receipt = runner.run("nepfake", root, Path(tmp) / "out.jsonl")
        self.assertEqual(receipt["invalid"], {})
        task = receipt["tasks"]["nepfake/false_or_misleading"]
        self.assertEqual((task["label:True"], task["label:False"]), (2, 1))


if __name__ == "__main__":
    unittest.main()

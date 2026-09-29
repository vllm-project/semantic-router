import json
import tempfile
import unittest
from pathlib import Path

from v2.data.m4 import pn1_scan


def row(ident, state, split="train"):
    return {"id": ident, "group_id": f"g:{ident}", "split": split, "state": state}


class ScanTest(unittest.TestCase):
    def run_scan(self, rows, protected):
        with tempfile.TemporaryDirectory() as tmp:
            panel = Path(tmp) / "panel.jsonl"
            panel.write_text(
                "".join(json.dumps(p, ensure_ascii=False) + "\n" for p in protected)
            )
            candidates = [dict(r, sentences=pn1_scan.sentences_of(r)) for r in rows]
            return pn1_scan.scan(candidates, {"panel": [panel]})

    def test_layouts_are_stripped(self):
        self.assertEqual(
            pn1_scan.sentences_of(row("a", "First sentence: x y\nSecond sentence: z")),
            ["x y", "z"],
        )
        with self.assertRaises(ValueError):
            pn1_scan.sentences_of(row("b", "Sentence 1: x\nSentence 2: y"))

    def test_exact_containment_and_near_hits(self):
        rows = [
            row("exact", "A: Tom is very busy today.\nB: unrelated words here"),
            row("contain", "(1) the quick brown fox jumps\n(2) zzzzzz qqqqqq"),
            row("near", "Text 1: トムは毎朝七時に起きます\nText 2: ぜんぜん違う文です"),
            row("clean", "A: nothing matches this one\nB: nor this other one"),
        ]
        protected = [
            {"prompt": {"context": "Tom is very busy today. Other text follows."}},
            {"text": "Yesterday the quick brown fox jumps over the dog"},
            {"q": ["トムは毎朝七時に起きますか"]},
        ]
        hits, counts = self.run_scan(rows, protected)
        found = {(h["id"], h["method"]) for h in hits}
        self.assertIn(("exact", "E"), found)
        self.assertIn(("contain", "C"), found)
        self.assertIn(("near", "N"), found)
        self.assertNotIn("clean", {h["id"] for h in hits})
        self.assertEqual(counts["panel"]["leaves"], 3)


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from v2.eval.sealed import containment


def write_rows(path: Path, rows: list) -> str:
    data = "".join(json.dumps(r) + "\n" for r in rows).encode()
    path.write_bytes(data)
    return hashlib.sha256(data).hexdigest()


class ContainmentTest(unittest.TestCase):
    def test_counts_identical_contained_no_text_and_novel(self):
        long_a = "the first passage has more than five tokens in it"
        long_b = "a second passage that also has enough tokens"
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            base = {"id": "r1", "state": long_a, "options": [long_b, "no"]}
            pool = [base, {"id": "r2", "state": long_b}]
            digest = write_rows(tmp / "pool.jsonl", pool)
            (tmp / "manifest.json").write_text(
                json.dumps({"labels": {"x": {"files": [{"sha256": digest}]}}})
            )
            mixture = tmp / "mix"
            mixture.mkdir()
            write_rows(
                mixture / "train.jsonl",
                [
                    base,
                    dict(base, split="train", label=1),
                    {"id": "r2", "state": long_a},
                    {"id": "r9", "state": long_a},
                    {"id": "r3", "label": 2},
                ],
            )
            (mixture / "notes.txt").write_text(long_a)
            code = containment.main(
                [
                    "--screened",
                    f"event2={tmp / 'manifest.json'}",
                    "--reference",
                    str(tmp / "pool.jsonl"),
                    "--mixture",
                    f"m={mixture}",
                    "--output",
                    str(tmp / "receipt.json"),
                    "--novel",
                    str(tmp / "novel.jsonl"),
                ]
            )
            self.assertEqual(code, 0)
            receipt = json.loads((tmp / "receipt.json").read_text())
            lines = (tmp / "novel.jsonl").read_text().splitlines()
            novel = [json.loads(x) for x in lines]
        self.assertEqual(
            receipt["totals"],
            {"rows": 5, "identical": 1, "contained": 1, "no_text": 1, "novel": 2},
        )
        self.assertEqual(receipt["references"][0]["screened_by"], "event2")
        self.assertEqual(len(receipt["mixtures"]["m"]), 1)
        self.assertEqual(sorted(r["id"] for r in novel), ["r2", "r9"])

    def test_unscreened_reference_stops(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            write_rows(tmp / "pool.jsonl", [{"id": "a"}])
            (tmp / "manifest.json").write_text(json.dumps({"labels": {}}))
            with self.assertRaises(SystemExit):
                containment.main(
                    [
                        "--screened",
                        f"v1={tmp / 'manifest.json'}",
                        "--reference",
                        str(tmp / "pool.jsonl"),
                        "--mixture",
                        f"m={tmp / 'pool.jsonl'}",
                        "--output",
                        str(tmp / "r.json"),
                        "--novel",
                        str(tmp / "n.jsonl"),
                    ]
                )


if __name__ == "__main__":
    unittest.main()

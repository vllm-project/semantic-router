from __future__ import annotations

import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

from v2.eval.sealed import independence


class IndependenceTest(unittest.TestCase):
    def test_names_reads_provenance_not_row_text(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "data"
            root.mkdir()
            rows = [
                {"source": "tydiqa_train", "state": "mentions DeliChess in text only"},
                {
                    "source": "x",
                    "audit_metadata": {"dataset": "SpaceHunterInf/DeliChess"},
                },
            ]
            (root / "train.jsonl").write_text(
                "".join(json.dumps(r) + "\n" for r in rows)
            )
            (root / "registry.json").write_text(
                json.dumps({"arms": ["innoduel sample"]})
            )
            terms = Path(tmp) / "terms.json"
            terms.write_text(
                json.dumps(
                    {
                        "delichess": ["DeliChess"],
                        "innoduel": ["innoduel"],
                        "gapa": ["gapa-x"],
                    }
                )
            )
            out = Path(tmp) / "names.json"
            independence.names(Namespace(terms=terms, root=[root], output=out))
            result = json.loads(out.read_text())
        self.assertEqual(result["sources_found"], ["delichess", "innoduel"])
        ((path, words),) = result["hits"]["delichess"].items()
        self.assertTrue(path.endswith("train.jsonl"))
        self.assertEqual(words, {"DeliChess": 1})

    def test_rows_covers_every_split(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "sources" / "src"
            (root / "data").mkdir(parents=True)
            (root / "data" / "train.csv").write_text(
                "text,label\nthis is a long enough text row,1\nshort,0\n"
            )
            (root / "data" / "test.jsonl").write_text(
                json.dumps({"q": "another sufficiently long question"}) + "\n"
            )
            out = Path(tmp) / "protected.jsonl"
            independence.rows(
                Namespace(sources_dir=Path(tmp) / "sources", source=["src"], output=out)
            )
            lines = [json.loads(x) for x in out.read_text().splitlines()]
        self.assertEqual(len(lines), 2)
        self.assertEqual(
            {l["task"] for l in lines}, {"src/data/train.csv", "src/data/test.jsonl"}
        )


if __name__ == "__main__":
    unittest.main()

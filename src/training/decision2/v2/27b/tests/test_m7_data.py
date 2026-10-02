import importlib
import json
import tempfile
import unittest
from pathlib import Path

m7_data = importlib.import_module("v2.27b.m7.m7_data")


def row(i, group, language, text):
    return {
        "id": f"r{i}",
        "group_id": group,
        "language": language,
        "instructions": "",
        "state": text,
        "options": [],
    }


class MLUpsampleTest(unittest.TestCase):
    def setUp(self):
        self.dir = Path(tempfile.mkdtemp())
        base = [row(i, f"e{i}", "en", "x" * 100) for i in range(60)]
        base += [row(100 + i, f"z{i // 2}", "zh", "y" * 50) for i in range(40)]
        base += [row(200 + i, f"k{i}", "ko", "w" * 50) for i in range(20)]
        base += [row(300, "mixed", "en", "m" * 10), row(301, "mixed", "ja", "m" * 10)]
        ib = [row(400 + i, f"ib{i}", "en", "i" * 100) for i in range(30)]
        self.base = self.dir / "a20.jsonl"
        self.base.write_text("".join(json.dumps(r) + "\n" for r in base))
        self.mix = self.dir / "mix.jsonl"
        self.mix.write_text(
            self.base.read_text() + "".join(json.dumps(r) + "\n" for r in ib)
        )

    def test_restores_the_a20_share_with_whole_non_english_groups(self):
        out = self.dir / "out.jsonl"
        rec = m7_data.ml_upsample(self.base, self.mix, "seed", out)
        self.assertLess(rec["input_ml_share"], rec["a20_ml_share"])
        self.assertAlmostEqual(
            rec["output_ml_share"], rec["a20_ml_share"], delta=m7_data.TOL
        )
        data = out.read_bytes()
        self.assertTrue(data.startswith(self.mix.read_bytes()))
        copies = [
            json.loads(line)
            for line in data.splitlines()[len(self.mix.read_text().splitlines()) :]
        ]
        self.assertTrue(
            copies
            and all(c["id"].endswith("~m2") and c["language"] != "en" for c in copies)
        )
        self.assertNotIn("mixed", {c["group_id"] for c in copies})
        by_group = {}
        for c in copies:
            by_group.setdefault(c["group_id"], 0)
            by_group[c["group_id"]] += 1
        self.assertTrue(all(n == 2 for g, n in by_group.items() if g.startswith("z")))
        again = self.dir / "again.jsonl"
        self.assertEqual(
            m7_data.ml_upsample(self.base, self.mix, "seed", again)["output_sha256"],
            rec["output_sha256"],
        )

    def test_refuses_missing_a20_rows_and_existing_output(self):
        short = self.dir / "short.jsonl"
        short.write_text("".join(self.mix.read_text().splitlines(keepends=True)[1:]))
        with self.assertRaises(ValueError):
            m7_data.ml_upsample(self.base, short, "seed", self.dir / "x.jsonl")
        out = self.dir / "out.jsonl"
        out.write_text("")
        with self.assertRaises(FileExistsError):
            m7_data.ml_upsample(self.base, self.mix, "seed", out)


if __name__ == "__main__":
    unittest.main()

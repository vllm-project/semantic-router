import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE.parents[3]))

from lux9b import m7_data  # noqa: E402


def row(i, group, lang="ja"):
    return {
        "id": f"m4pn1-{i}",
        "group_id": group,
        "input_sha256": f"h{i}",
        "language": lang,
        "task_type": "noul",
    }


class Pn1BlockTest(unittest.TestCase):
    def test_repeat_suffixes_later_copies(self):
        rows = [row(1, "g1"), row(2, "g1"), row(3, "g2")]
        block = m7_data.pn1_block(rows, 2)
        self.assertEqual(len(block), 6)
        self.assertEqual([r["id"] for r in block[:3]], [r["id"] for r in rows])
        self.assertEqual([r["id"] for r in block[3:]], [f"{r['id']}~r2" for r in rows])
        self.assertEqual({r["group_id"] for r in block[3:]}, {"g1", "g2"})
        self.assertEqual(block[3]["input_sha256"], rows[0]["input_sha256"])
        self.assertIs(m7_data.pn1_block(rows, 1)[0], rows[0])

    def test_refuses_bad_inputs(self):
        with self.assertRaises(ValueError):
            m7_data.pn1_block([row(1, "g"), row(1, "g")], 2)
        with self.assertRaises(ValueError):
            m7_data.pn1_block([dict(row(1, "g"), id="x~r2")], 2)
        with self.assertRaises(ValueError):
            m7_data.pn1_block([row(1, "g")], 0)

    def test_type_shares(self):
        rows = [
            dict(row(1, "g"), task_type="noul"),
            dict(row(2, "g"), task_type="choice"),
        ]
        shares = m7_data.type_shares(rows, {"m4pn1-1": 30, "m4pn1-2": 10})
        self.assertEqual(shares, {"choice": 0.25, "noul": 0.75})


class StateSegmentsTest(unittest.TestCase):
    def test_only_state_lines_are_screened(self):
        a = {
            "state": "Sentence one is long enough here.\nshort",
            "instructions": "Same template text for all rows.",
        }
        b = {
            "state": "Another state line that is long.",
            "instructions": "Same template text for all rows.",
        }
        c = {"state": "  SENTENCE one is   long enough here. ", "instructions": "x"}
        self.assertFalse(m7_data.state_segments(a) & m7_data.state_segments(b))
        self.assertEqual(m7_data.state_segments(a), m7_data.state_segments(c))
        self.assertEqual(len(m7_data.state_segments(a)), 1)


if __name__ == "__main__":
    unittest.main()

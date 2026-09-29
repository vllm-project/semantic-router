"""CPU tests for ops/m7/m7_lock.py (a synthetic M7 data tree)."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "ops" / "m7" / "m7_lock.py"
_spec = importlib.util.spec_from_file_location("m7_lock", SCRIPT)
ml = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ml)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def row(i, source="xl", family="fam", group=None):
    return {
        "id": f"r{i}",
        "group_id": group or f"g{i}",
        "source": source,
        "family": family,
        "state": "s",
    }


class LockTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.hs1 = row(9, source="decision2_hardskills_hs1", family="hs1_policy_packet")
        arms = {
            "H": [row(1), row(2), self.hs1],
            "C": [row(1), row(2), row(3)],
            "P": [row(1), row(2), row(4)],
        }
        mix = self.root / "data" / "4b" / "mix"
        files = {}
        for arm, rows in arms.items():
            name = f"m7-4b-{arm}"
            d = mix / name
            d.mkdir(parents=True)
            (d / "train.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
            digest = sha(d / "train.jsonl")
            files[arm] = {"train_sha256": digest, "pools": {"base": 2, "x": 1}}
            e = self.root / "exposure" / name
            e.mkdir(parents=True)
            (e / f"exposure-{name}.json").write_text(
                json.dumps({"groups": [], "files": [{"sha256": digest}]})
            )
            t = self.root / "teacher" / name
            t.mkdir(parents=True)
            (t / "teacher.jsonl.manifest.json").write_text(
                json.dumps(
                    {
                        "train_sha256": digest,
                        "rows": 3,
                        "covered": 2,
                        "missing_by_pool": {"hs1": 1} if arm == "H" else {"pn1": 1},
                        "output_sha256": "ab" * 32,
                    }
                )
            )
        report = {
            "tokens": {"H": 1000, "C": 1001, "P": 999},
            "match": {"C_minus_H": 1, "P_minus_H": -1, "max_relative": 0.001},
            "added_tokens_H": 400,
            "filler": {"P_nested_in_C": True},
            **{k: {} for k in ("base", "hs1", "lp", "pn1", "quarantine_rows_dropped")},
        }
        (mix / "compose.json").write_text(
            json.dumps(
                {
                    "report": report,
                    "files": files,
                    "args": {"hs1_drop_substring": ["DEFECT"]},
                    "inputs_sha256": {},
                }
            )
        )
        self.cleared = self.root / "hs1.train.jsonl"
        self.cleared.write_text(
            json.dumps(self.hs1)
            + "\n"
            + json.dumps(row(10, source="decision2_hardskills_hs1"))
            + "\n"
        )

    def tearDown(self):
        self.tmp.cleanup()

    def run_lock(self, **kw):
        out = self.root / "lock.json"
        args = ["--tier", "4b", "--root", str(self.root), "--out", str(out)]
        if kw.get("cleared", True):
            args += [
                "--hs1-cleared",
                str(self.cleared),
                "--hs1-cleared-sha",
                sha(self.cleared),
            ]
        rc = ml.main(args)
        return rc, json.loads(out.read_text())

    def test_pass_counts_hs1_rows_in_the_cleared_revision(self):
        rc, doc = self.run_lock()
        self.assertEqual((rc, doc["status"]), (0, "PASS"), doc["fails"])
        self.assertEqual(doc["arms"]["H"]["hs1_rows"], 1)
        self.assertEqual(doc["arms"]["H"]["hs1_rows_in_cleared_revision"], 1)
        self.assertEqual(doc["arms"]["C"]["hs1_rows"], 0)

    def test_hs1_row_changed_in_the_cleared_revision_fails(self):
        changed = dict(self.hs1, state="fixed wording")
        self.cleared.write_text(json.dumps(changed) + "\n")
        rc, doc = self.run_lock()
        self.assertEqual(rc, 1)
        self.assertTrue(any("cleared HS1 revision" in f for f in doc["fails"]))

    def test_token_mismatch_and_exposure_fail(self):
        comp = self.root / "data" / "4b" / "mix" / "compose.json"
        doc = json.loads(comp.read_text())
        doc["report"]["match"]["max_relative"] = 0.02
        comp.write_text(json.dumps(doc))
        e = self.root / "exposure" / "m7-4b-C" / "exposure-m7-4b-C.json"
        e.write_text(json.dumps({"groups": ["g1"], "files": [{"sha256": "x"}]}))
        rc, out = self.run_lock(cleared=False)
        self.assertEqual(rc, 1)
        self.assertTrue(any("token mismatch" in f for f in out["fails"]))
        self.assertTrue(any("exposure" in f for f in out["fails"]))


if __name__ == "__main__":
    unittest.main()

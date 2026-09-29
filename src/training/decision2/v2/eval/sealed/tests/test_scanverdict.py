from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from pathlib import Path

from v2.eval.sealed import scanverdict

SCRIPT = Path(__file__).resolve().parents[1] / "event3-recheck-scan.sh"
BASE = {
    "a": ("CLEAN", 0.01),
    "b": ("REVIEW", 0.3),
    "c": ("REVIEW", 0.21),
    "d": ("CLEAN", 0.0),
}


def hit(i: str, verdict: str, containment: float) -> dict:
    return {
        "id": i,
        "source": "s",
        "verdict": verdict,
        "containment": containment,
        "label": "x",
    }


class JudgeTest(unittest.TestCase):
    def run_case(self, changes: dict) -> dict:
        old = {k: hit(k, *v) for k, v in BASE.items()}
        new = {
            k: hit(k, *changes.get(k, v)) for k, v in BASE.items() if changes.get(k, v)
        }
        return scanverdict.judge(new, old)

    def test_same_hits_pass(self) -> None:
        result = self.run_case({})
        self.assertEqual(result["verdict"], "PASS")
        self.assertEqual(result["non_clean_ids"], ["b", "c"])
        self.assertEqual(result["counts"], {"CLEAN": 2, "REVIEW": 2})

    def test_lower_or_cleared_recurring_hits_pass(self) -> None:
        self.assertEqual(
            self.run_case({"b": ("REVIEW", 0.25), "c": ("CLEAN", 0.1)})["verdict"],
            "PASS",
        )

    def test_failures(self) -> None:
        cases = {
            "overlap": {"b": ("OVERLAP", 0.6)},
            "new non-clean id": {"a": ("REVIEW", 0.2)},
            "higher containment": {"c": ("REVIEW", 0.22)},
            "missing candidate": {"d": None},
        }
        for name, change in cases.items():
            with self.subTest(name):
                result = self.run_case(change)
                self.assertEqual(result["verdict"], "FAIL")
                self.assertTrue(result["problems"])


class CommandTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp())
        self.write("old.jsonl", {k: hit(k, *v) for k, v in BASE.items()})
        for name in ("receipt.json", "manifest.json"):
            (self.tmp / name).write_text("{}")

    def write(self, name: str, rows: dict) -> Path:
        path = self.tmp / name
        path.write_text("".join(json.dumps(r) + "\n" for r in rows.values()))
        return path

    def compare(self, rows: dict, out: str) -> int:
        self.write("new.jsonl", rows)
        t = self.tmp
        return scanverdict.main(
            [
                "compare",
                "--hits",
                str(t / "new.jsonl"),
                "--baseline",
                str(t / "old.jsonl"),
                "--receipt",
                str(t / "receipt.json"),
                "--manifest",
                str(t / "manifest.json"),
                "--protected-sha",
                "p" * 64,
                "--output",
                str(t / out),
            ]
        )

    def check(self, out: str, manifest: str | None = None) -> int:
        manifest = manifest or scanverdict.sha_file(self.tmp / "manifest.json")
        return scanverdict.main(
            [
                "check",
                "--verdict",
                str(self.tmp / out),
                "--manifest-sha",
                manifest,
                "--protected-sha",
                "p" * 64,
            ]
        )

    def test_pass_and_interlock(self) -> None:
        self.assertEqual(
            self.compare({k: hit(k, *v) for k, v in BASE.items()}, "ok.json"), 0
        )
        verdict = json.loads((self.tmp / "ok.json").read_text())
        self.assertEqual(verdict["non_clean_ids"], ["b", "c"])
        self.assertNotIn("text", json.dumps(verdict))
        self.assertEqual(self.check("ok.json"), 0)
        self.assertEqual(self.check("ok.json", manifest="0" * 64), 1)
        self.assertEqual(self.check("missing.json"), 1)

    def test_fail_blocks_the_interlock(self) -> None:
        rows = {k: hit(k, *v) for k, v in BASE.items()}
        rows["a"] = hit("a", "OVERLAP", 0.9)
        self.assertEqual(self.compare(rows, "bad.json"), 1)
        self.assertEqual(
            json.loads((self.tmp / "bad.json").read_text())["overlap_ids"], ["a"]
        )
        self.assertEqual(self.check("bad.json"), 1)

    def test_script_syntax(self) -> None:
        subprocess.run(["bash", "-n", str(SCRIPT)], check=True)
        text = SCRIPT.read_text()
        self.assertIn("--network none", text)
        self.assertNotIn("/dev/kfd", text)
        self.assertIn("--workers 48 --exact-min-tokens 8", text)
        self.assertLess(
            text.index('[ "$(sha "$PROT")" = "$PROT_SHA" ]'), text.index("docker run")
        )
        self.assertLess(text.index("ACCESS.log"), text.index("docker run"))


if __name__ == "__main__":
    unittest.main()

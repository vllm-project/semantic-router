from __future__ import annotations

import contextlib
import hashlib
import io
import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path

from v2.eval.sealed import overlap, scanverdict

SCRIPT = Path(__file__).resolve().parents[1] / "event3-recheck-scan.sh"
BASE = {
    "a": ("CLEAN", 0.01),
    "b": ("REVIEW", 0.3),
    "c": ("REVIEW", 0.21),
    "d": ("CLEAN", 0.0),
}
BASE_V2 = {**BASE, "e": ("CLEAN", 0.05)}
PROTECTED = "p" * 64


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


class NodesTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp())
        self.written = 0
        self.base = self.tmp / "base.jsonl"
        self.base.write_text(
            "".join(json.dumps(hit(k, *v)) + "\n" for k, v in BASE_V2.items())
        )
        self.retired = self.retired_list("retired.json", ["a", "d"])

    def retired_list(
        self, name: str, rows: list[str], schema: str = "dev2-c1-retired/1"
    ) -> Path:
        path = self.tmp / name
        value = {"schema": schema, "version": "v1.2", "candidates": ["t/x|1"]}
        path.write_text(json.dumps({**value, "protected_rows": rows}))
        return path

    def scan(
        self, node: str, changes: dict, protected: str = PROTECTED
    ) -> tuple[Path, Path]:
        """One node's hits (reverse id order, CLEAN unless changed, None drops a row)."""
        self.written += 1
        rows = [
            hit(k, *changes.get(k, ("CLEAN", 0.0)))
            for k in sorted(BASE_V2, reverse=True)
            if changes.get(k, ()) is not None
        ]
        data = "".join(json.dumps(r, sort_keys=True) + "\n" for r in rows).encode()
        hits = self.tmp / f"hits-{self.written}.jsonl"
        hits.write_bytes(data)
        receipt = self.tmp / f"receipt-{self.written}.json"
        receipt.write_text(
            json.dumps(
                {
                    "hits_sha256": hashlib.sha256(data).hexdigest(),
                    "manifest_sha256": hashlib.sha256(node.encode()).hexdigest(),
                    "protected": {"sha256": protected, "candidates": len(rows)},
                }
            )
        )
        return hits, receipt

    def extract(
        self,
        node: str,
        hits: Path,
        receipt: Path,
        output: Path,
        baseline: Path | None = None,
    ) -> int:
        argv = ["extract", "--hits", str(hits), "--receipt", str(receipt)]
        argv += ["--baseline", str(baseline or self.base)]
        with contextlib.redirect_stdout(io.StringIO()):
            return scanverdict.main([*argv, "--node", node, "--output", str(output)])

    def node(self, node: str, changes: dict, protected: str = PROTECTED) -> Path:
        hits, receipt = self.scan(node, changes, protected)
        output = self.tmp / f"extract-{self.written}.json"
        self.assertEqual(self.extract(node, hits, receipt, output), 0)
        return output

    def judge(
        self,
        extracts: list[Path],
        extra: tuple[str, ...] = (),
        retired: Path | None = None,
        retired_sha: str | None = None,
        output: Path | None = None,
    ) -> tuple[dict, Path]:
        self.written += 1
        output = output or self.tmp / f"verdict-{self.written}.json"
        retired = retired or self.retired
        argv = ["judge", "--baseline", str(self.base), "--retired", str(retired)]
        argv += ["--retired-sha", retired_sha or scanverdict.sha_file(retired)]
        argv += ["--protected-sha", PROTECTED, "--output", str(output), *extra]
        for path in extracts:
            argv += ["--extract", str(path)]
        with contextlib.redirect_stdout(io.StringIO()) as out:
            code = scanverdict.main(argv)
        verdict = json.loads(output.read_text())
        summary = json.loads(out.getvalue())
        self.assertEqual(summary["verdict"], verdict["verdict"])
        self.assertEqual(summary["verdict_sha256"], scanverdict.sha_file(output))
        self.assertEqual(code, 0 if verdict["verdict"] == "PASS" else 1)
        return verdict, output

    def check_v2(
        self,
        verdict: Path,
        sha: str | None = None,
        retired_sha: str | None = None,
        protected: str = PROTECTED,
    ) -> tuple[int, str, str]:
        argv = ["check-v2", "--verdict", str(verdict)]
        argv += ["--verdict-sha", sha or scanverdict.sha_file(verdict)]
        argv += ["--retired-sha", retired_sha or scanverdict.sha_file(self.retired)]
        argv += ["--protected-sha", protected]
        with contextlib.redirect_stdout(io.StringIO()) as out:
            with contextlib.redirect_stderr(io.StringIO()) as err:
                code = scanverdict.main(argv)
        return code, out.getvalue(), err.getvalue()

    def test_extract_round_trip(self) -> None:
        hits, receipt = self.scan("A", {"c": ("REVIEW", 0.3), "b": ("OVERLAP", 0.7)})
        output = self.tmp / "extract.json"
        self.assertEqual(self.extract("A", hits, receipt, output), 0)
        pinned = json.loads(receipt.read_text())
        self.assertEqual(
            json.loads(output.read_text()),
            {
                "schema": "dev2-c1-scan-extract/1",
                "node": "A",
                "hits_sha256": pinned["hits_sha256"],
                "receipt_sha256": scanverdict.sha_file(receipt),
                "manifest_sha256": pinned["manifest_sha256"],
                "protected_sha256": PROTECTED,
                "baseline_hits_sha256": scanverdict.sha_file(self.base),
                "candidates": 5,
                "counts": {"CLEAN": 3, "OVERLAP": 1, "REVIEW": 1},
                "non_clean": [hit("b", "OVERLAP", 0.7), hit("c", "REVIEW", 0.3)],
                "baseline_rows": [hit("b", "OVERLAP", 0.7), hit("c", "REVIEW", 0.3)],
            },
        )
        self.assertEqual(os.stat(output).st_mode & 0o777, 0o600)
        with self.assertRaises(FileExistsError):
            self.extract("A", hits, receipt, output)

    def test_extract_refuses_other_hits(self) -> None:
        hits, receipt = self.scan("A", {})
        hits.write_bytes(hits.read_bytes() + b"\n")
        output = self.tmp / "extract.json"
        with contextlib.redirect_stderr(io.StringIO()) as err:
            self.assertEqual(self.extract("A", hits, receipt, output), 1)
        self.assertIn("extract refused", err.getvalue())
        self.assertFalse(output.exists())

    def test_extract_reads_an_overlap_scan(self) -> None:
        tokens = " ".join(f"tok{i}" for i in range(30))
        protected = self.tmp / "protected.jsonl"
        protected.write_text(
            "".join(
                json.dumps({"task": "t/x", "source_item_id": str(i), "state": [text]})
                + "\n"
                for i, text in enumerate([tokens, "other words " * 6, "tiny"])
            )
        )
        corpus = self.tmp / "corpus.jsonl"
        corpus.write_text(json.dumps({"t": tokens}) + "\n")
        receipt, hits = self.tmp / "receipt.json", self.tmp / "hits.jsonl"
        argv = ["scan", "--protected", str(protected), "--corpus", f"train={corpus}"]
        with contextlib.redirect_stdout(io.StringIO()):
            with contextlib.redirect_stderr(io.StringIO()):
                code = overlap.main(
                    [*argv, "--output", str(receipt), "--hits", str(hits)]
                )
        self.assertEqual(code, 0)
        output = self.tmp / "extract.json"
        self.assertEqual(self.extract("A", hits, receipt, output), 0)
        value = json.loads(output.read_text())
        self.assertEqual(value["protected_sha256"], scanverdict.sha_file(protected))
        self.assertEqual(value["hits_sha256"], scanverdict.sha_file(hits))
        self.assertIsNone(value["manifest_sha256"])
        self.assertEqual(value["candidates"], 3)
        self.assertEqual(value["counts"], {"CLEAN": 2, "OVERLAP": 1})
        self.assertEqual([h["id"] for h in value["non_clean"]], ["t/x|0"])

    def test_judge_passes_retired_non_clean_rows(self) -> None:
        a = self.node("A", {"a": ("OVERLAP", 0.8)})
        b = self.node("B", {"d": ("REVIEW", 0.3)})
        coverage = self.tmp / "coverage-A.json"
        coverage.write_text("{}")
        verdict, output = self.judge([b, a], extra=("--coverage", f"A={coverage}"))
        self.assertEqual(verdict["verdict"], "PASS")
        self.assertEqual(verdict["problems"], [])
        self.assertEqual(verdict["schema"], "dev2-c1-scan-verdict/2")
        self.assertEqual(verdict["item_set"], "v1.2")
        self.assertEqual(verdict["retired_sha256"], scanverdict.sha_file(self.retired))
        self.assertEqual(verdict["protected_sha256"], PROTECTED)
        self.assertEqual(
            verdict["baseline_hits_sha256"], scanverdict.sha_file(self.base)
        )
        self.assertEqual(verdict["candidates"], 5)
        self.assertEqual(verdict["counts"], {"CLEAN": 3, "OVERLAP": 1, "REVIEW": 1})
        self.assertEqual(verdict["retired_non_clean"], 2)
        self.assertEqual(
            verdict["non_clean_ids"],
            [
                {"id": "a", "verdict": "OVERLAP", "containment": 0.8, "nodes": ["A"]},
                {"id": "d", "verdict": "REVIEW", "containment": 0.3, "nodes": ["B"]},
            ],
        )
        for key in ("overlap_ids", "new_non_clean_ids", "higher_containment_ids"):
            self.assertEqual(verdict[key], [])
        extract_a = json.loads(a.read_text())
        self.assertEqual(
            verdict["nodes"][0],
            {
                "node": "A",
                "manifest_sha256": extract_a["manifest_sha256"],
                "hits_sha256": extract_a["hits_sha256"],
                "receipt_sha256": extract_a["receipt_sha256"],
                "extract_sha256": scanverdict.sha_file(a),
                "coverage_receipt_sha256": scanverdict.sha_file(coverage),
            },
        )
        self.assertEqual(verdict["nodes"][1]["node"], "B")
        self.assertIsNone(verdict["nodes"][1]["coverage_receipt_sha256"])
        self.assertEqual(os.stat(output).st_mode & 0o777, 0o600)
        with self.assertRaises(FileExistsError):
            self.judge([a, b], output=output)

    def test_judge_merges_nodes(self) -> None:
        retired = self.retired_list("retired-b.json", ["b"])
        a = self.node("A", {"b": ("OVERLAP", 0.1), "c": ("REVIEW", 0.2)})
        b = self.node("B", {"b": ("REVIEW", 0.3), "c": ("REVIEW", 0.21)})
        verdict, _ = self.judge([a, b], retired=retired)
        self.assertEqual(verdict["verdict"], "PASS")
        self.assertEqual(
            verdict["non_clean_ids"],
            [
                {
                    "id": "b",
                    "verdict": "OVERLAP",
                    "containment": 0.3,
                    "nodes": ["A", "B"],
                },
                {
                    "id": "c",
                    "verdict": "REVIEW",
                    "containment": 0.21,
                    "nodes": ["A", "B"],
                },
            ],
        )
        self.assertEqual(verdict["counts"], {"CLEAN": 3, "OVERLAP": 1, "REVIEW": 1})
        self.assertEqual(verdict["retired_non_clean"], 1)
        higher, _ = self.judge([self.node("C", {"c": ("REVIEW", 0.22)}), b])
        self.assertEqual(higher["verdict"], "FAIL")
        self.assertEqual(higher["higher_containment_ids"], ["c"])
        self.assertEqual(higher["non_clean_ids"][1]["nodes"], ["B", "C"])

    def test_containment_counts_nodes_where_the_id_is_clean(self) -> None:
        merged = scanverdict.merge(
            [
                {
                    "node": "A",
                    "non_clean": [hit("c", "REVIEW", 0.0)],
                    "baseline_rows": [hit("c", "REVIEW", 0.0)],
                },
                {
                    "node": "B",
                    "non_clean": [],
                    "baseline_rows": [hit("c", "CLEAN", 0.15), hit("b", "CLEAN", 0.1)],
                },
            ]
        )
        self.assertEqual(
            merged, {"c": {"verdict": "REVIEW", "containment": 0.15, "nodes": ["A"]}}
        )
        old = {"c": hit("c", "REVIEW", 0.1)}
        result = scanverdict.judge_merged(merged, old, set())
        self.assertEqual(result["higher_containment_ids"], ["c"])

    def test_extract_against_another_baseline_fails(self) -> None:
        other = self.tmp / "other-base.jsonl"
        other.write_bytes(self.base.read_bytes() + b"\n")
        hits, receipt = self.scan("A", {})
        output = self.tmp / "extract-other.json"
        self.assertEqual(self.extract("A", hits, receipt, output, baseline=other), 0)
        verdict, _ = self.judge([output])
        self.assertEqual(verdict["verdict"], "FAIL")
        self.assertIn(
            "node A was extracted against another baseline", verdict["problems"]
        )

    def test_absent_from_the_baseline_is_new(self) -> None:
        merged = scanverdict.merge(
            [{"node": "A", "non_clean": [hit("z", "REVIEW", 0)]}]
        )
        old = {k: hit(k, *v) for k, v in BASE_V2.items()}
        self.assertEqual(
            scanverdict.judge_merged(merged, old, set())["new_non_clean_ids"], ["z"]
        )
        self.assertEqual(scanverdict.judge_merged(merged, old, {"z"})["problems"], [])

    def test_judge_failures(self) -> None:
        clean = self.node("B", {})
        coverage = self.tmp / "coverage.json"
        coverage.write_text("{}")
        other = self.retired_list("other.json", ["a", "d"], "dev2-c1-retired/0")
        receipt = self.scan("A", {})[1]
        cases = {
            "non-retired OVERLAP": (
                [self.node("A", {"b": ("OVERLAP", 0.6)}), clean],
                {},
                "1 OVERLAP outside the retired rows",
            ),
            "newly non-CLEAN": (
                [self.node("A", {"e": ("REVIEW", 0.2)}), clean],
                {},
                "1 non-CLEAN ids that were CLEAN or absent in the baseline",
            ),
            "higher containment": (
                [self.node("A", {"c": ("REVIEW", 0.22)}), clean],
                {},
                "1 recurring ids with higher containment",
            ),
            "other protected rows": (
                [self.node("A", {}, protected="q" * 64), clean],
                {},
                "node A scanned other protected rows",
            ),
            "node candidate counts": (
                [self.node("A", {"e": None}), clean],
                {},
                "the nodes scanned different numbers of candidates",
            ),
            "baseline candidate count": (
                [self.node("A", {"e": None}), self.node("C", {"e": None})],
                {},
                "the scans and the baseline cover different numbers of candidates",
            ),
            "repeated node": (
                [self.node("B", {}), clean],
                {},
                "repeated node names: B",
            ),
            "not an extract": (
                [receipt, clean],
                {},
                f"{receipt.name} is not a scan extract",
            ),
            "retired sha": (
                [self.node("A", {}), clean],
                {"retired_sha": "0" * 64},
                "the retired list differs from --retired-sha",
            ),
            "retired schema": (
                [self.node("A", {}), clean],
                {"retired": other},
                "the retired list is not dev2-c1-retired/1",
            ),
            "coverage of another node": (
                [self.node("A", {}), clean],
                {"extra": ("--coverage", f"C={coverage}")},
                "coverage receipts must name distinct scanned nodes",
            ),
            "coverage twice": (
                [self.node("A", {}), clean],
                {"extra": ("--coverage", f"A={coverage}", "--coverage", f"A={other}")},
                "coverage receipts must name distinct scanned nodes",
            ),
        }
        ids = {
            "non-retired OVERLAP": ("overlap_ids", ["b"]),
            "newly non-CLEAN": ("new_non_clean_ids", ["e"]),
            "higher containment": ("higher_containment_ids", ["c"]),
        }
        for name, (extracts, options, problem) in cases.items():
            with self.subTest(name):
                verdict, _ = self.judge(extracts, **options)
                self.assertEqual(verdict["verdict"], "FAIL")
                self.assertIn(problem, verdict["problems"])
                if name in ids:
                    key, value = ids[name]
                    self.assertEqual(verdict[key], value)

    def test_check_v2(self) -> None:
        clean = self.node("B", {})
        verdict, passed = self.judge([self.node("A", {"a": ("OVERLAP", 0.8)}), clean])
        code, out, err = self.check_v2(passed)
        self.assertEqual((code, err), (0, ""))
        hits = " ".join(node["hits_sha256"][:12] for node in verdict["nodes"])
        self.assertEqual(
            out,
            f"scan verdict PASS (item set v1.2, 2 nodes, hits {hits}, {verdict['utc']})\n",
        )
        _, failed = self.judge([self.node("A", {"b": ("OVERLAP", 0.6)}), clean])
        cases = {
            "wrong verdict sha": (
                passed,
                {"sha": "0" * 64},
                "not the pinned scan verdict",
            ),
            "FAIL verdict": (failed, {}, "verdict 'FAIL'"),
            "wrong retired sha": (
                passed,
                {"retired_sha": "0" * 64},
                "scan judged another retired list",
            ),
            "wrong protected sha": (
                passed,
                {"protected": "q" * 64},
                "scan used other protected rows",
            ),
            "missing verdict": (
                self.tmp / "missing.json",
                {"sha": "0" * 64},
                "no readable scan verdict",
            ),
        }
        for name, (path, options, reason) in cases.items():
            with self.subTest(name):
                code, out, err = self.check_v2(path, **options)
                self.assertEqual((code, out), (1, ""))
                self.assertIn(reason, err)
        self.assertTrue(
            self.check_v2(failed)[2].startswith("scan verdict refused: verdict 'FAIL'")
        )


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import contextlib
import hashlib
import io
import json
import os
import re
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


SEALED = Path(__file__).resolve().parents[1]
CLASS_MAP = SEALED / "c1-p2-classes.json"
A_SHA = "a" * 64
A_PATH = "/data/dev2/runs/dec/m3/data/m3-v2m-ret/train.jsonl"
A_LABEL = f"/data/dev2/runs/dec/m3/data::{A_PATH}"
POOL = "/data/dev2/private/data/arms-v2"
POOL_LABEL = f"{POOL}::{POOL}/h5/h5.train.jsonl"
RAW = "/data/dev2/private/sources/tatoeba-2026-09-26"


def cell(verdict: str, containment: float, exact: int | None = None) -> dict:
    value = {"verdict": verdict, "containment": containment}
    if exact is not None:
        value["exact_tokens"] = exact
    return value


class ClassesTest(unittest.TestCase):
    """judge-classes on one node: r1 is retired, r2-r5 stay in the item set."""

    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp())
        self.written = 0
        real = json.loads(CLASS_MAP.read_text())
        a = [
            {
                "model": "M",
                "sha256": [A_SHA],
                "paths": [r"^/data/dev2/runs/dec/m3/data/m3-v2m-ret/"],
            }
        ]
        self.spec = {**real, "a": a}
        self.retired = self.tmp / "retired.json"
        self.retired.write_text(
            json.dumps(
                {
                    "schema": "dev2-c1-retired/1",
                    "version": "v1.2",
                    "candidates": ["t|1"],
                    "protected_rows": ["r1"],
                }
            )
        )
        self.flagged = {
            "r1": ("REVIEW", 0.3),
            "r2": ("OVERLAP", 1.0),
            "r3": ("REVIEW", 0.3),
            "r4": ("REVIEW", 0.2),
            "r5": ("OVERLAP", 1.0),
        }
        self.grouped = {
            "r1": {"/data/dev2/runs/dec/m3/data": cell("REVIEW", 0.3)},
            "r2": {RAW: cell("OVERLAP", 1.0, 9)},
            "r3": {POOL: cell("REVIEW", 0.3)},
            "r4": {
                "/data/decision20-20260926/runs/css-transfer-v1": cell("REVIEW", 0.2)
            },
            "r5": {
                "/data/dev2/private/tmp/samples-m2": cell("OVERLAP", 1.0, 4),
                RAW: cell("OVERLAP", 1.0),
            },
            "r9": {},
        }
        self.perfile = {
            "r1": {A_LABEL: cell("REVIEW", 0.3)},
            "r3": {POOL_LABEL: cell("REVIEW", 0.3)},
        }
        self.manifest = {A_LABEL: A_SHA, POOL_LABEL: "b" * 64}

    def write(self, name: str, text: str) -> Path:
        self.written += 1
        path = self.tmp / f"{self.written}-{name}"
        path.write_text(text)
        return path

    def rows(self, cells: dict) -> str:
        return "".join(
            json.dumps({"id": k, "task": "t", "shingles": 9, "labels": v}) + "\n"
            for k, v in cells.items()
        )

    def run_judge(
        self,
        nodes: dict[str, str] | None = None,
        protected: str = PROTECTED,
        retired_sha: str | None = None,
    ) -> tuple[dict, Path, int]:
        nodes = nodes or {}
        extract = {
            "schema": "dev2-c1-scan-extract/1",
            "node": "A",
            "hits_sha256": "1" * 64,
            "receipt_sha256": "2" * 64,
            "manifest_sha256": "3" * 64,
            "protected_sha256": protected,
            "baseline_hits_sha256": "4" * 64,
            "candidates": 9,
            "non_clean": [
                {"id": k, "verdict": v, "containment": c}
                for k, (v, c) in self.flagged.items()
            ],
            "baseline_rows": [],
        }
        manifest = {
            label: {
                "kind": "training",
                "files": [{"path": label.split("::")[1], "bytes": 1, "sha256": sha}],
            }
            for label, sha in self.manifest.items()
        }
        argv = [
            "judge-classes",
            "--extract",
            str(self.write("extract.json", json.dumps(extract))),
        ]
        argv += [
            "--grouped",
            f"{nodes.get('grouped', 'A')}={self.write('g.jsonl', self.rows(self.grouped))}",
        ]
        argv += [
            "--perfile",
            f"{nodes.get('perfile', 'A')}={self.write('p.jsonl', self.rows(self.perfile))}",
        ]
        argv += [
            "--perfile-manifest",
            f"A={self.write('m.json', json.dumps({'schema': 'c1-corpora/1', 'labels': manifest}))}",
        ]
        argv += ["--classes", str(self.write("classes.json", json.dumps(self.spec)))]
        argv += ["--retired", str(self.retired)]
        argv += ["--retired-sha", retired_sha or scanverdict.sha_file(self.retired)]
        output = self.tmp / f"verdict-{self.written}.json"
        argv += ["--protected-sha", PROTECTED, "--output", str(output)]
        with contextlib.redirect_stdout(io.StringIO()) as out:
            code = scanverdict.main(argv)
        verdict = json.loads(output.read_text())
        self.assertEqual(
            json.loads(out.getvalue())["verdict_sha256"], scanverdict.sha_file(output)
        )
        self.assertEqual(code, 0 if verdict["verdict"] == "PASS" else 1)
        return verdict, output, code

    def test_base_passes_and_counts_every_class(self) -> None:
        verdict, _, _ = self.run_judge()
        self.assertEqual(verdict["verdict"], "PASS", verdict["problems"])
        self.assertEqual(verdict["schema"], "dev2-c1-class-verdict/1")
        self.assertEqual(verdict["rule"], self.spec["rule"])
        self.assertEqual(verdict["item_set"], "v1.2")
        self.assertEqual(
            verdict["kept"],
            {
                "rows": 4,
                "by_class": {"a": 0, "b": 1, "c": 2, "d": 1, "e": 1},
                "by_verdict": {"OVERLAP": 2, "REVIEW": 2},
            },
        )
        self.assertEqual(verdict["retired"]["by_class"]["a"], 1)
        self.assertEqual(
            verdict["class_a_sets"], [{"model": "M", "files": [["A", A_PATH, A_SHA]]}]
        )
        self.assertEqual(
            (verdict["class_a_ids"], verdict["class_b_near_exact_ids"]), ([], [])
        )
        dump = {d["name"]: d for d in verdict["disclosed"]}["raw-source sample dump"]
        self.assertEqual(
            (dump["kept_rows"], dump["kept_near_exact"], dump["kept_max_containment"]),
            (1, 1, 1.0),
        )
        self.assertEqual(dump["kept_by_verdict"], {"OVERLAP": 1})

    def test_class_a_hits_fail_in_kept_rows_only(self) -> None:
        self.grouped["r1"]["/data/dev2/runs/dec/m3/data"] = cell("OVERLAP", 0.95)
        self.assertEqual(self.run_judge()[0]["verdict"], "PASS")
        cases = {
            "path": (A_LABEL, A_SHA),
            "copy by hash": ("hf-blobs::/data/dev2/hf-cache/blobs/0f", A_SHA),
        }
        for name, (label, sha) in cases.items():
            with self.subTest(name):
                self.setUp()
                self.manifest[label] = sha
                self.perfile["r3"][label] = cell("REVIEW", 0.0, 6)
                verdict, _, _ = self.run_judge()
                self.assertEqual(verdict["verdict"], "FAIL")
                self.assertEqual(verdict["class_a_ids"], ["r3"])
                self.assertIn(
                    "1 rows still in the item set hit class-(a) training files",
                    verdict["problems"],
                )

    def test_near_exact_class_b(self) -> None:
        cases = {
            "containment 0.85": (cell("REVIEW", 0.85), "FAIL"),
            "exact 8 tokens": (cell("OVERLAP", 0.1, 8), "FAIL"),
            "containment 0.79, exact 7": (cell("OVERLAP", 0.79, 7), "PASS"),
        }
        for name, (value, expected) in cases.items():
            with self.subTest(name):
                self.setUp()
                self.perfile["r3"][POOL_LABEL] = value
                verdict, _, _ = self.run_judge()
                self.assertEqual(verdict["verdict"], expected, verdict["problems"])
                if expected == "FAIL":
                    self.assertEqual(verdict["class_b_near_exact_ids"], ["r3"])
        self.setUp()
        self.grouped["r2"]["training-data:v2/a7/arms/A7m"] = cell("REVIEW", 0.9)
        self.perfile["r2"] = {POOL_LABEL: cell("REVIEW", 0.2)}
        verdict, _, _ = self.run_judge()
        self.assertEqual(verdict["class_b_near_exact_ids"], ["r2"])
        self.assertIn(
            "1 rows still in the item set have a near-exact class-(b) hit",
            verdict["problems"],
        )

    def test_coverage_and_binding_failures(self) -> None:
        cases = {
            "grouped misses a row": (
                lambda: self.grouped.pop("r4"),
                {},
                "the grouped scan of A misses 1 flagged rows",
            ),
            "per-file misses a rooted row": (
                lambda: self.perfile.pop("r3"),
                {},
                "the per-file scan of A misses 1 rows with a hit under the training roots",
            ),
            "pinned class-(a) file absent": (
                lambda: self.manifest.pop(A_LABEL),
                {},
                "1 pinned class-(a) files of M are in no per-file manifest",
            ),
            "another node's grouped scan": (
                lambda: None,
                {"nodes": {"grouped": "B"}},
                "--grouped must name each extract's node once",
            ),
            "other protected rows": (
                lambda: None,
                {"protected": "q" * 64},
                "node A scanned other protected rows",
            ),
            "another retired list": (
                lambda: None,
                {"retired_sha": "0" * 64},
                "the retired list differs from --retired-sha",
            ),
        }
        for name, (change, options, problem) in cases.items():
            with self.subTest(name):
                self.setUp()
                change()
                verdict, _, _ = self.run_judge(**options)
                self.assertEqual(verdict["verdict"], "FAIL")
                self.assertIn(problem, verdict["problems"])

    def test_check_v2_takes_the_class_schema(self) -> None:
        verdict, path, _ = self.run_judge()
        argv = [
            "check-v2",
            "--verdict",
            str(path),
            "--verdict-sha",
            scanverdict.sha_file(path),
        ]
        argv += [
            "--retired-sha",
            scanverdict.sha_file(self.retired),
            "--protected-sha",
            PROTECTED,
        ]
        with contextlib.redirect_stdout(io.StringIO()) as out:
            code = scanverdict.main([*argv, "--schema", "dev2-c1-class-verdict/1"])
        self.assertEqual(code, 0)
        self.assertEqual(
            out.getvalue(),
            f"scan verdict PASS (item set v1.2, rule {self.spec['rule']}, 1 nodes,"
            f" hits {'1' * 12}, {verdict['utc']})\n",
        )
        with contextlib.redirect_stderr(io.StringIO()) as err:
            self.assertEqual(scanverdict.main(argv), 1)
        self.assertIn("not a dev2-c1-scan-verdict/2 scan verdict", err.getvalue())


class ClassMapTest(unittest.TestCase):
    spec = json.loads(CLASS_MAP.read_text())
    classes = scanverdict.compile_classes(spec)

    def test_class_a_is_the_released_training_files_and_the_successor(self) -> None:
        record = (
            SEALED.parents[1]
            / "eval/records/jevbench-value/contamination/contamination.json"
        )
        released = json.loads(record.read_text())["A"]["fresh_screen"]
        m8 = (SEALED.parents[1] / "06b/m8_scorebias.py").read_text()
        successor = dict(
            re.findall(
                r'"(/data/dev2/runs/06b/m6/data/m6-[a-z]+-s\d\.train\.jsonl)": "([0-9a-f]{64})"',
                m8,
            )
        )
        self.assertEqual(len(successor), 6)
        pinned = {s for entry in self.spec["a"] for s in entry["sha256"]}
        self.assertEqual(
            pinned, {v["sha256"] for v in released.values()} | set(successor.values())
        )
        self.assertEqual(len(self.spec["a"]), 7)
        for path in successor:
            self.assertTrue(
                any(p.match(path) for p, _ in self.classes["a_paths"]), path
            )
        self.assertEqual(
            self.spec["near_exact"], {"containment": 0.8, "exact_tokens": 8}
        )

    def test_group_classes(self) -> None:
        cases = {
            "training-data:v2/a7/arms/A7m": "b",
            "hf-blobs": "b",
            "/data/dev2/private/data/arms-v2": "b",
            "/data/dev2/private/data/pi-v2": "b",
            "/data/dev2/runs/data/m3b": "b",
            "/data/dev2/private/27b/m3-data/mixtures-m3-1": "b",
            "/data/dev2/tmp/dec-m5-impl": "b",
            "/data/dev2/private/a7/sources/dec10": "c",
            "/data/dev2/private/sources/m3b/natural-instructions": "c",
            "/data/decision20-20260926/external/convokit": "c",
            "/data/decision20-20260926/external/kev/evals": "d",
            "/data/decision20-20260926/runs/css-transfer-v1": "d",
            "/data/dev2/private/tmp/samples-m2": "e",
            "/data/dev2/private/tmp": "e",
        }
        for group, expected in cases.items():
            with self.subTest(group):
                self.assertEqual(scanverdict.group_class(group, self.classes), expected)


if __name__ == "__main__":
    unittest.main()

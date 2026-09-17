"""Fail-closed receipts for real Go -count runs and published OpenVINO inference."""

import copy
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import openvino_evidence as ov
from ci_results import collection_errors

PACKAGE = "example.org/owned"


def stream(names, repeats=3, subtests=()):
    result = []
    for _ in range(repeats):
        for name in names:
            result.append({"Action": "run", "Package": PACKAGE, "Test": name})
            for subtest in subtests:
                for action in ("run", "pass"):
                    result.append(
                        {
                            "Action": action,
                            "Package": PACKAGE,
                            "Test": name + "/" + subtest,
                        }
                    )
            result.append({"Action": "pass", "Package": PACKAGE, "Test": name})
    result.append({"Action": "pass", "Package": PACKAGE})
    return result


class RepeatedOpenVINOTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def write(self, name, rows):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
        return path

    def collect(self, rows, required=None):
        return ov.run_cases(
            self.write("tests.jsonl", rows),
            PACKAGE,
            required or {"TestOwnedOne"},
            3,
            "owned",
        )

    def test_each_repeat_and_subtest_is_preserved(self):
        cases, expected = self.collect(stream(["TestOwnedOne"], subtests=["lease"]))
        self.assertEqual(len(cases), 6)
        self.assertEqual(len(expected), 3)
        self.assertEqual(len({case["id"] for case in cases}), 6)
        for iteration in range(1, 4):
            self.assertIn(f"owned/run-{iteration}/{PACKAGE}/TestOwnedOne", expected)
        self.assertEqual(
            collection_errors({"cases": cases, "expected_cases": expected}, "test"),
            [],
        )

    def test_real_go_json_preserves_same_process_count_and_parallel_subtests(self):
        go = shutil.which("go")
        self.assertIsNotNone(go, "Go is required to validate its actual JSON contract")
        (self.root / "go.mod").write_text(f"module {PACKAGE}\n\ngo 1.23\n")
        (self.root / "owned_test.go").write_text(
            'package owned\nimport "testing"\nvar iterations int\n'
            "func TestOwnedOne(t *testing.T) {\n"
            " iterations++; if iterations > 3 { t.Fatal(iterations) };\n"
            ' t.Run("parallel", func(t *testing.T) { t.Parallel() })\n}\n'
        )
        for filename, flags in (
            ("discovery.jsonl", ["-list", "^TestOwned"]),
            ("tests.jsonl", ["-count=3", "-run", "^TestOwned"]),
        ):
            process = subprocess.run(
                [go, "test", "-json", *flags, "."],
                cwd=self.root,
                capture_output=True,
                text=True,
                timeout=60,
                check=False,
            )
            self.assertEqual(process.returncode, 0, process.stdout + process.stderr)
            (self.root / filename).write_text(process.stdout)
        package = ov.discovered_package(self.root / "discovery.jsonl", {"TestOwnedOne"})
        cases, expected = ov.run_cases(
            self.root / "tests.jsonl", package, {"TestOwnedOne"}, 3, "owned"
        )
        self.assertEqual(len(cases), 6)
        self.assertEqual(len(expected), 3)

    def test_missing_excess_and_empty_runs_are_rejected(self):
        for repeats in (0, 1, 2, 4):
            with self.subTest(repeats=repeats), self.assertRaises(ValueError):
                self.collect(stream(["TestOwnedOne"], repeats=repeats))
        with self.assertRaises(ValueError):
            self.collect(stream(["TestOwnedOne"], repeats=1) * 3)

    def test_duplicate_run_or_terminal_is_not_a_new_iteration(self):
        good = stream(["TestOwnedOne"], subtests=["lease"])
        for index in (0, 1, 2, 3):
            broken = copy.deepcopy(good)
            broken.insert(index, copy.deepcopy(broken[index]))
            with self.subTest(index=index), self.assertRaises(ValueError):
                self.collect(broken)

    def test_skip_failure_crash_and_unfinished_subtest_are_rejected(self):
        good = stream(["TestOwnedOne"], subtests=["lease"])
        variants = [good[:-1], good[:-2], good[:2] + good[3:]]
        for action in ("skip", "fail"):
            for index in (2, 3, len(good) - 1):
                broken = copy.deepcopy(good)
                broken[index]["Action"] = action
                variants.append(broken)
        for index, broken in enumerate(variants):
            with self.subTest(index=index), self.assertRaises(ValueError):
                self.collect(broken)

    def test_extra_root_or_wrong_package_cannot_supply_required_execution(self):
        for names in (["TestOther"], ["TestOwnedOne", "TestOther"]):
            with self.subTest(names=names), self.assertRaises(ValueError):
                self.collect(stream(names))
        broken = stream(["TestOwnedOne"])
        broken[0]["Package"] = "other/package"
        with self.assertRaises(ValueError):
            self.collect(broken)

    def test_discovery_requires_exact_source_inventory_and_process_success(self):
        rows = [
            {"Action": "output", "Package": PACKAGE, "Output": "TestOwnedOne\n"},
            {"Action": "pass", "Package": PACKAGE},
        ]
        path = self.write("discovery.jsonl", rows)
        self.assertEqual(ov.discovered_package(path, {"TestOwnedOne"}), PACKAGE)
        variants = (rows[:-1], rows + rows[:1], rows[1:], [])
        for broken in variants:
            with self.subTest(rows=broken), self.assertRaises(ValueError):
                ov.discovered_package(
                    self.write("discovery.jsonl", broken), {"TestOwnedOne"}
                )
        with self.assertRaises(ValueError):
            ov.discovered_package(path, {"TestOwnedOne", "TestOwnedMissing"})

    def test_complete_evidence_requires_published_heads_and_both_owned_suites(self):
        (self.root / "inference.json").write_text(
            json.dumps(
                {
                    "passed": True,
                    "provider": "openvino",
                    "device": "CPU",
                    "platform": ov.actual_platform(),
                    "models": [],
                }
            )
        )
        package = (ov.ROOT / "openvino-binding/go.mod").read_text().split()[1]
        published = stream(["TestPublishedOpenVINO"], 1, ["domain", "embedding"])
        for row in published:
            row["Package"] = package
        self.write("tests.jsonl", published)
        bindings = {
            "TestOwnedHandlesRemainIndependent",
            "TestOwnedBudgetsAndPadding",
            "TestOwnedCountsBeyondDeclaredModelLimit",
            "TestOwnedRejectsNullText",
            "TestOwnedInitializationCanRetry",
            "TestOwnedReopenOnSameThread",
            "TestOwnedConcurrentInferAndClose",
        }
        for group, names in (
            ("owned-bindings", bindings),
            ("owned-runtime", ov.RUNTIME_TESTS),
        ):
            self.write(
                group + "/discovery.jsonl",
                [
                    {"Action": "output", "Package": PACKAGE, "Output": name + "\n"}
                    for name in names
                ]
                + [{"Action": "pass", "Package": PACKAGE}],
            )
            self.write(group + "/tests.jsonl", stream(sorted(names)))
        result = ov.evidence(self.root)
        self.assertEqual(len(result["expected_cases"]), 30)
        self.assertEqual(len(result["cases"]), 30)
        self.assertEqual(collection_errors(result, "test"), [])
        self.write("owned-runtime/tests.jsonl", [])
        with self.assertRaises(ValueError):
            ov.evidence(self.root)


if __name__ == "__main__":
    unittest.main()

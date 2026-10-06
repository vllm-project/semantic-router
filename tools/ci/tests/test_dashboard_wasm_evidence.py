"""WASM receipts require discovered Go cases and source-owned Node assertions."""

import json
import os
import shlex
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dashboard_wasm_evidence import evidence, go_evidence, node_evidence


class DashboardWasmExecTests(unittest.TestCase):
    def test_exec_bounds_environment_and_preserves_arguments_paths_and_exit(self):
        wrapper = Path(__file__).resolve().parents[3] / "dashboard/wasm/go_test_exec.sh"
        with tempfile.TemporaryDirectory(prefix="wasm exec ") as directory:
            root = Path(directory)
            binary = root / "bin"
            binary.mkdir()
            runtime = root / "go/lib/wasm"
            runtime.mkdir(parents=True)
            fake_go = binary / "go"
            fake_go.write_text(
                "#!/bin/sh\n"
                'test "$1 $2" = "env GOROOT" || exit 90\n'
                'test -n "$WASM_CI_TEST_PADDING" || exit 91\n'
                f"printf '%s\\n' {shlex.quote(str(root / 'go'))}\n"
            )
            runner = runtime / "go_js_wasm_exec"
            runner.write_text('#!/bin/sh\nexec node "$@"\n')
            node = binary / "node"
            node.write_text(
                f"#!{sys.executable}\n"
                "import json, os, sys\n"
                "print(json.dumps({'argv': sys.argv[1:], 'env': dict(os.environ)}))\n"
                "sys.exit(23 if '--fail' in sys.argv else 0)\n"
            )
            for path in (fake_go, runner, node):
                path.chmod(0o755)
            env = {
                **os.environ,
                "PATH": str(binary) + os.pathsep + os.environ["PATH"],
                "TMPDIR": str(root / "temporary files"),
                "WASM_CI_TEST_PADDING": "x" * 16384,
            }
            for failure in (False, True):
                with self.subTest(failure=failure):
                    arguments = ["test binary.wasm", "-test.run=TestCompile"]
                    if failure:
                        arguments.append("--fail")
                    result = subprocess.run(
                        [str(wrapper), *arguments],
                        env=env,
                        capture_output=True,
                        text=True,
                        check=False,
                    )
                    self.assertEqual(result.returncode, 23 if failure else 0)
                    actual = json.loads(result.stdout)
                    self.assertEqual(actual["argv"], arguments)
                    self.assertEqual(actual["env"]["PATH"], env["PATH"])
                    self.assertEqual(actual["env"]["TMPDIR"], env["TMPDIR"])
                    self.assertNotIn("WASM_CI_TEST_PADDING", actual["env"])
                    self.assertLess(
                        sum(
                            len(key) + len(value)
                            for key, value in actual["env"].items()
                        ),
                        8192,
                    )


class DashboardWasmEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.source = self.root / "test.js"
        self.source.write_text('assert("compile", true, "compiled");\n')
        self.inventory = [
            {"Package": "command-line-arguments", "Output": "TestCompile\n"},
        ]
        self.execution = [
            {
                "Package": "command-line-arguments",
                "Action": "run",
                "Test": "TestCompile",
            },
            {
                "Package": "command-line-arguments",
                "Action": "pass",
                "Test": "TestCompile",
            },
            {"Package": "command-line-arguments", "Action": "pass"},
        ]
        self.write("inventory.jsonl", self.inventory)
        self.write("go-tests.jsonl", self.execution)
        self.write("node-tests.jsonl", [{"id": "compile", "status": "passed"}])

    def write(self, filename, values):
        (self.root / filename).write_text(
            "".join(json.dumps(row) + "\n" for row in values)
        )

    def go(self):
        return go_evidence(self.root / "inventory.jsonl", self.root / "go-tests.jsonl")

    def test_combines_real_framework_ids_and_declared_node_assertions(self):
        result = evidence(self.root, self.source)
        self.assertEqual(
            result["expected_cases"],
            ["go-js-wasm:command-line-arguments/TestCompile", "node-wasm:compile"],
        )
        self.assertEqual(len(result["cases"]), 2)
        self.assertTrue(all(row["status"] == "passed" for row in result["cases"]))

    def test_go_requires_discovery_and_complete_nonduplicate_lifecycles(self):
        for records in (
            [],
            self.execution[:-1],
            self.execution[1:],
            [*self.execution, self.execution[1]],
            [*self.execution, self.execution[0]],
            [self.execution[0], self.execution[2]],
            [{**self.execution[0], "Test": "TestUnexpected"}, *self.execution[1:]],
            [{**self.execution[0], "Action": "skip"}],
            [*self.execution, {"Package": "command-line-arguments", "Action": "fail"}],
            [{**self.execution[0], "Action": "pass"}, *self.execution],
            [
                *self.execution,
                {**self.execution[0], "Test": "TestUnknown/variant"},
                {**self.execution[1], "Test": "TestUnknown/variant"},
            ],
        ):
            with self.subTest(records=records):
                self.write("go-tests.jsonl", records)
                with self.assertRaises(ValueError):
                    self.go()

    def test_empty_or_duplicate_go_discovery_is_rejected(self):
        for records in ([], self.inventory * 2):
            with self.subTest(records=records):
                self.write("inventory.jsonl", records)
                with self.assertRaisesRegex(ValueError, "discovery"):
                    self.go()

    def test_skipped_go_subtest_cannot_hide_under_passing_parent(self):
        self.write(
            "go-tests.jsonl",
            [
                *self.execution,
                {
                    "Package": "command-line-arguments",
                    "Test": "TestCompile/variant",
                    "Action": "skip",
                },
            ],
        )
        with self.assertRaisesRegex(ValueError, "did not pass"):
            self.go()

    def test_node_rejects_missing_extra_duplicate_failed_or_skipped_assertions(self):
        passed = {"id": "compile", "status": "passed"}
        for records in (
            [],
            [passed, passed],
            [passed, {"id": "unknown", "status": "passed"}],
            [{**passed, "status": "failed"}],
            [{**passed, "status": "skipped"}],
            [{**passed, "command_exit": 0}],
        ):
            with self.subTest(records=records):
                self.write("node-tests.jsonl", records)
                with self.assertRaises(ValueError):
                    node_evidence(self.source, self.root / "node-tests.jsonl")

    def test_node_inventory_rejects_empty_duplicate_and_nonliteral_ids(self):
        for source in (
            "",
            'assert("compile", true, "ok");\n' * 2,
            'assert(identity, true, "ok");\n',
        ):
            with self.subTest(source=source):
                self.source.write_text(source)
                with self.assertRaisesRegex(ValueError, "literal source IDs"):
                    node_evidence(self.source, self.root / "node-tests.jsonl")


if __name__ == "__main__":
    unittest.main()

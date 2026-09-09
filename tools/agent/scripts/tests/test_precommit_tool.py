import importlib
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

SCRIPT_DIR = Path(__file__).resolve().parents[1]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
precommit_tool = importlib.import_module("precommit_tool")


class ReadOnlyFormattingTests(unittest.TestCase):
    def test_whitespace_check_does_not_modify_source_and_format_does(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            path = root / "sample.py"
            original = b"value = 1  \n\n"
            path.write_bytes(original)
            with mock.patch.object(precommit_tool, "REPO_ROOT", root):
                self.assertEqual(precommit_tool.run_whitespace(["sample.py"]), 1)
                self.assertEqual(path.read_bytes(), original)
                self.assertEqual(
                    precommit_tool.run_whitespace(["sample.py"], fix=True), 0
                )
                self.assertEqual(path.read_bytes(), b"value = 1\n")

    @unittest.skipUnless(shutil.which("gofmt"), "gofmt is required")
    def test_gofmt_rejects_unformatted_code_without_rewriting_it(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.go"
            original = b"package sample\nfunc f( ){println(1)}\n"
            path.write_bytes(original)
            self.assertEqual(precommit_tool.run_go([str(path)]), 1)
            self.assertEqual(path.read_bytes(), original)
            self.assertEqual(precommit_tool.run_go([str(path)], fix=True), 0)
            self.assertEqual(precommit_tool.run_go([str(path)]), 0)

    @unittest.skipUnless(importlib.util.find_spec("black"), "black is required")
    def test_black_checks_then_explicitly_formats(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.py"
            original = b"value=1\n"
            path.write_bytes(original)
            self.assertEqual(precommit_tool.run_python([str(path)]), 1)
            self.assertEqual(path.read_bytes(), original)
            self.assertEqual(precommit_tool.run_python([str(path)], fix=True), 0)
            self.assertEqual(precommit_tool.run_python([str(path)]), 0)

    @unittest.skipUnless(shutil.which("cargo"), "cargo is required")
    def test_rust_check_uses_owning_crate_and_keeps_both_sources_intact(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            for name in ("first", "second"):
                crate = root / name
                (crate / "src").mkdir(parents=True)
                (crate / "Cargo.toml").write_text(
                    f'[package]\nname="{name}"\nversion="0.1.0"\nedition="2021"\n'
                )
                (crate / "src/lib.rs").write_text("pub fn value()->u8{1}\n")
            before = {p: p.read_bytes() for p in root.rglob("*.rs")}
            with mock.patch.object(precommit_tool, "REPO_ROOT", root):
                self.assertNotEqual(precommit_tool.run_rust(["second/src/lib.rs"]), 0)
                self.assertEqual({p: p.read_bytes() for p in before}, before)
                self.assertEqual(
                    precommit_tool.run_rust(["second/src/lib.rs"], fix=True), 0
                )
                self.assertEqual(precommit_tool.run_rust(["second/src/lib.rs"]), 0)
                self.assertEqual(
                    (root / "first/src/lib.rs").read_bytes(),
                    before[root / "first/src/lib.rs"],
                )


class FileRoutingTests(unittest.TestCase):
    def test_javascript_files_run_in_their_own_packages(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            log = root / "calls.jsonl"
            for name in ("website", "dashboard"):
                package = root / name
                (package / "node_modules/.bin").mkdir(parents=True)
                (package / "package.json").write_text('{"scripts":{"lint":"eslint ."}}')
                (package / "sample.ts").write_text("const value = 1\n")
                eslint = package / "node_modules/.bin/eslint"
                eslint.write_text(
                    f'#!{sys.executable}\nimport json, os, sys\nwith open({str(log)!r}, "a") as output:\n output.write(json.dumps([os.getcwd(), sys.argv[1:]]) + "\\n")\n'
                )
                eslint.chmod(0o755)
            with mock.patch.object(
                precommit_tool, "REPO_ROOT", root
            ), mock.patch.object(
                precommit_tool, "ensure_package_dependencies"
            ), mock.patch.object(
                precommit_tool, "ensure_node_runtime"
            ):
                self.assertEqual(
                    precommit_tool.run_javascript(
                        ["dashboard/sample.ts", "website/sample.ts"]
                    ),
                    0,
                )
            calls = [json.loads(line) for line in log.read_text().splitlines()]
            self.assertEqual(
                calls,
                [
                    [str(root / "dashboard"), ["sample.ts"]],
                    [str(root / "website"), ["sample.ts"]],
                ],
            )

    def test_shell_and_spelling_only_receive_requested_files(self):
        with mock.patch.object(precommit_tool, "run", return_value=0) as run:
            precommit_tool.run_shell(["tools/one.sh"])
            self.assertEqual(run.call_args.args[0][-1], "tools/one.sh")
            self.assertNotIn("find", run.call_args.args[0])
            precommit_tool.run_codespell(["README.md"])
            self.assertEqual(run.call_args.args[0][-1], "README.md")
            self.assertNotIn(".", run.call_args.args[0])

    def test_command_failure_is_preserved(self):
        with mock.patch.object(precommit_tool, "run", return_value=7):
            self.assertEqual(precommit_tool.run_python(["sample.py"]), 7)


if __name__ == "__main__":
    unittest.main()

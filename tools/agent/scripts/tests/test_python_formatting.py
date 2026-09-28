from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import yaml

ROOT: Path = Path(__file__).resolve().parents[4]


class PythonFormattingTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        subprocess.run(
            args=["git", "init", "--quiet"],
            cwd=self.root,
            check=True,
            capture_output=True,
        )
        config = yaml.safe_load(
            (ROOT / ".pre-commit-config.yaml").read_text(encoding="utf-8")
        )
        config["repos"] = [
            repo
            for repo in config["repos"]
            if repo["repo"]
            in {
                "https://github.com/pre-commit/pre-commit-hooks",
                "https://github.com/psf/black",
            }
        ]
        (self.root / ".pre-commit-config.yaml").write_text(
            data=yaml.safe_dump(config), encoding="utf-8"
        )
        helper = Path("tools/agent/scripts/check_python_whitespace.py")
        if (ROOT / helper).is_file():
            (self.root / helper).parent.mkdir(parents=True)
            shutil.copyfile(src=ROOT / helper, dst=self.root / helper)

    def write_file(self, name: str, content: bytes) -> Path:
        source = self.root / name
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_bytes(content)
        subprocess.run(
            args=["git", "add", "--", name],
            cwd=self.root,
            check=True,
            capture_output=True,
        )
        return source

    def run_hook(
        self,
        hook: str | None,
        *filenames: str,
        stage: str = "pre-commit",
    ) -> subprocess.CompletedProcess[str]:
        environment: dict[str, str] = os.environ.copy()
        environment.pop("SKIP", None)
        environment["BLACK_CACHE_DIR"] = str(self.root / ".black-cache")
        command: list[str] = [sys.executable, "-m", "pre_commit", "run"]
        if hook is not None:
            command.append(hook)
        return subprocess.run(
            args=[
                *command,
                "--hook-stage",
                stage,
                "--color=never",
                "--files",
                *filenames,
            ],
            cwd=self.root,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            check=False,
            timeout=180,
        )

    def test_dirty_python_checks_leave_original_bytes(self) -> None:
        original = b"value= [1,2] \t\r\n\r\n"
        source = self.write_file(name="sample.py", content=original)

        result = self.run_hook(None, "sample.py")

        self.assertEqual(source.read_bytes(), original, result.stdout)
        self.assertEqual(result.returncode, 1, result.stdout)
        self.assertIn("would reformat sample.py", result.stdout)
        self.assertNotIn("files were modified by this hook", result.stdout)

    def test_clean_python_checks_pass_without_writes(self) -> None:
        for original in (b'value = "keep  spaces"\n', b"value = 1\r\n", b""):
            with self.subTest(original=original):
                source = self.write_file(name="sample.py", content=original)
                modified = source.stat().st_mtime_ns

                result = self.run_hook(None, "sample.py")

                self.assertEqual(result.returncode, 0, result.stdout)
                self.assertEqual(source.read_bytes(), original, result.stdout)
                self.assertEqual(source.stat().st_mtime_ns, modified, result.stdout)

    def test_trailing_whitespace_coverage_without_writes(self) -> None:
        cases = (
            (b"value = 1 \t\n", 1),
            (b"value = 1 \t\r\n", 1),
            (b"value = 1\v\f\n", 1),
            (b'value = """keep \t\nspaces"""\n', 1),
            (b"# coding: latin-1\nvalue = '\xe9' \t\n", 1),
            (b" \t\r\n", 1),
            (b"value = 1\r", 1),
            (b"value = 1", 0),
            (b"value = 1\r\n", 0),
            (b"\n\n", 0),
            (b"", 0),
        )
        for original, expected in cases:
            with self.subTest(original=original):
                source = self.write_file(name="sample.py", content=original)

                result = self.run_hook("trailing-whitespace", "sample.py")

                self.assertEqual(source.read_bytes(), original, result.stdout)
                self.assertEqual(result.returncode, expected, result.stdout)
                if expected:
                    self.assertIn("sample.py", result.stdout)

    def test_end_of_file_coverage_without_writes(self) -> None:
        cases = (
            (b"value = 1", 1),
            (b"value = 1\n\n", 1),
            (b"value = 1\r\n\r\n", 1),
            (b"value = 1\r\r", 1),
            (b"\n", 1),
            (b"\n\n", 1),
            (b"\r\n\r\n", 1),
            (b"value = 1\n", 0),
            (b"value = 1\r\n", 0),
            (b"value = 1\r", 0),
            (b"", 0),
        )
        for original, expected in cases:
            with self.subTest(original=original):
                source = self.write_file(name="sample.py", content=original)

                result = self.run_hook("end-of-file-fixer", "sample.py")

                self.assertEqual(source.read_bytes(), original, result.stdout)
                self.assertEqual(result.returncode, expected, result.stdout)
                if expected:
                    self.assertIn("sample.py", result.stdout)

    def test_explicit_python_format_applies_native_fixers(self) -> None:
        cases = (
            (b"value= [1,2] \t\r\n\r\n", b"value = [1, 2]\r\n"),
            (b"value= [1,2]", b"value = [1, 2]\n"),
            (b"\n\n", b""),
            (b'value = """keep \t\nspaces"""\n', b'value = """keep\nspaces"""\n'),
        )
        for original, expected in cases:
            with self.subTest(original=original):
                source = self.write_file(name="sample.py", content=original)

                result = self.run_hook("python-format", "sample.py", stage="manual")

                self.assertEqual(source.read_bytes(), expected, result.stdout)
                self.assertEqual(result.returncode, 1, result.stdout)
                self.assertIn("files were modified by this hook", result.stdout)
                checked = self.run_hook(None, "sample.py")
                self.assertEqual(checked.returncode, 0, checked.stdout)
                self.assertEqual(source.read_bytes(), expected, checked.stdout)

    def test_black_errors_fail_without_writes(self) -> None:
        original = b"def broken(:\n    pass\n"
        source = self.write_file(name="sample.py", content=original)

        result = self.run_hook("black", "sample.py")

        self.assertEqual(result.returncode, 1, result.stdout)
        self.assertIn("exit code: 123", result.stdout)
        self.assertIn("cannot format sample.py", result.stdout)
        self.assertEqual(source.read_bytes(), original, result.stdout)

    def test_whitespace_read_errors_fail_without_writes(self) -> None:
        config_path = self.root / ".pre-commit-config.yaml"
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        for repo in config["repos"]:
            for hook_config in repo["hooks"]:
                if hook_config["id"] in {"trailing-whitespace", "end-of-file-fixer"}:
                    hook_config["args"] = ["missing.py"]
        config_path.write_text(data=yaml.safe_dump(config), encoding="utf-8")
        original = b"value = 1\n"
        source = self.write_file(name="sample.py", content=original)

        for hook in ("trailing-whitespace", "end-of-file-fixer"):
            with self.subTest(hook=hook):
                result = self.run_hook(hook, "sample.py")
                self.assertEqual(result.returncode, 1, result.stdout)
                self.assertIn("FileNotFoundError", result.stdout)
                self.assertIn("missing.py", result.stdout)
                self.assertEqual(source.read_bytes(), original, result.stdout)

    def test_python_hook_selection_keeps_existing_exclusions(self) -> None:
        for directory in (".venv", "venv", "env", "__pycache__", "site-packages"):
            with self.subTest(directory=directory):
                filename = f"{directory}/sample.py"
                original = b"value= [1,2]\n"
                source = self.write_file(name=filename, content=original)
                for hook, stage in (
                    ("black", "pre-commit"),
                    ("python-format", "manual"),
                ):
                    result = self.run_hook(hook, filename, stage=stage)
                    self.assertEqual(result.returncode, 0, result.stdout)
                    self.assertEqual(source.read_bytes(), original, result.stdout)

        filename = "website/build/sample.py"
        original = b"value = 1 \t\n\n"
        source = self.write_file(name=filename, content=original)
        for stage in ("pre-commit", "manual"):
            result = self.run_hook("end-of-file-fixer", filename, stage=stage)
            self.assertEqual(result.returncode, 0, result.stdout)
            self.assertEqual(source.read_bytes(), original, result.stdout)
        result = self.run_hook("trailing-whitespace", filename)
        self.assertEqual(source.read_bytes(), original, result.stdout)
        self.assertEqual(result.returncode, 1, result.stdout)

        filename = "nested/website/build/sample.py"
        source = self.write_file(name=filename, content=original)
        result = self.run_hook("end-of-file-fixer", filename)
        self.assertEqual(source.read_bytes(), original, result.stdout)
        self.assertEqual(result.returncode, 1, result.stdout)

    def test_non_python_whitespace_hooks_keep_native_behavior(self) -> None:
        for suffix in ("go", "rs", "js"):
            with self.subTest(suffix=suffix):
                filename = f"sample.{suffix}"
                source = self.write_file(name=filename, content=b"value \t\r\n\r\n")

                trailing = self.run_hook("trailing-whitespace", filename)
                self.assertEqual(trailing.returncode, 1, trailing.stdout)
                self.assertEqual(source.read_bytes(), b"value\r\n\r\n", trailing.stdout)
                ending = self.run_hook("end-of-file-fixer", filename)
                self.assertEqual(ending.returncode, 1, ending.stdout)
                self.assertEqual(source.read_bytes(), b"value\r\n", ending.stdout)

    def test_unselected_files_remain_untouched(self) -> None:
        for filename in ("sample.txt", "sample.PY"):
            with self.subTest(filename=filename):
                original = b"value= [1,2] \t\n\n"
                source = self.write_file(name=filename, content=original)
                result = self.run_hook(None, filename)
                self.assertEqual(result.returncode, 0, result.stdout)
                self.assertEqual(source.read_bytes(), original, result.stdout)


if __name__ == "__main__":
    unittest.main()

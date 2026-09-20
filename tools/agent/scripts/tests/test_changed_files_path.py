import importlib
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

SCRIPT_DIR = Path(__file__).resolve().parents[1]
CI_DIR = SCRIPT_DIR.parents[1] / "ci"
for path in (SCRIPT_DIR, CI_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

changed_files = importlib.import_module("changed_files")
harness = importlib.import_module("harness")


class ChangedFilesTests(unittest.TestCase):
    def test_split_changed_files_accepts_common_separators(self) -> None:
        result = changed_files.split_changed_files(
            "tools/agent/scripts/harness.py tools/make/agent.mk,"
            "\nsrc/semantic-router/pkg/apiserver/server.go"
        )

        self.assertEqual(
            result,
            [
                "src/semantic-router/pkg/apiserver/server.go",
                "tools/agent/scripts/harness.py",
                "tools/make/agent.mk",
            ],
        )

    def test_get_changed_files_reads_path_without_git_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "changed-files.txt"
            path.write_text(
                "./tools/agent/scripts/harness.py\n"
                "tools/make/agent.mk\n"
                "tools/agent/scripts/harness.py\n",
                encoding="utf-8",
            )
            with mock.patch.object(changed_files, "git_changed_files") as git_diff:
                result = changed_files.get_changed_files("", None, str(path))

        self.assertEqual(
            result,
            ["tools/agent/scripts/harness.py", "tools/make/agent.mk"],
        )
        git_diff.assert_not_called()

    def test_missing_changed_files_path_has_clear_error(self) -> None:
        with self.assertRaisesRegex(ValueError, "unable to read changed files"):
            changed_files.load_changed_files("does-not-exist")

    def test_git_changed_files_includes_branch_worktree_and_untracked_paths(
        self,
    ) -> None:
        outputs = {
            (
                "rev-parse",
                "--verify",
                "--end-of-options",
                "origin/main^{commit}",
            ): (0, "base\n"),
            ("merge-base", "HEAD", "origin/main"): (0, "base\n"),
            ("diff", "--name-only", "-z", "base...HEAD"): (
                0,
                "committed.py\0shared.py\0",
            ),
            ("diff", "--name-only", "-z", "HEAD"): (
                0,
                "working tree.py\0shared.py\0",
            ),
            ("ls-files", "--others", "--exclude-standard", "-z"): (
                0,
                "untracked.py\0",
            ),
        }

        def fake_run(
            command: list[str], **_: object
        ) -> subprocess.CompletedProcess[str]:
            returncode, stdout = outputs[tuple(command[1:])]
            return subprocess.CompletedProcess(command, returncode, stdout, "")

        with mock.patch.object(changed_files.subprocess, "run", side_effect=fake_run):
            result = changed_files.git_changed_files("origin/main")

        self.assertEqual(
            result,
            ["committed.py", "shared.py", "untracked.py", "working tree.py"],
        )


class ImpactTests(unittest.TestCase):
    def test_impact_contains_facts_without_skill_or_completion_policy(self) -> None:
        result = harness.build_impact(["tools/make/agent.mk"], "cpu")

        self.assertEqual(
            result["domains"], [{"name": "harness", "owner": "maintainers"}]
        )
        self.assertEqual(result["checks"], ["make harness-check"])
        self.assertNotIn("primary_skill", result)
        self.assertNotIn("completion_boundary", result)
        self.assertNotIn("loop_mode", result)

    def test_verify_requires_an_explicit_selection(self) -> None:
        self.assertEqual(harness.run_verify((), ()), 2)


class BaseReferenceTests(unittest.TestCase):
    def setUp(self) -> None:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.git("init", "--quiet")
        (self.root / "file.txt").write_text("first\n", encoding="utf-8")
        self.git("add", "file.txt")
        self.git("commit", "--quiet", "--signoff", "-m", "First")
        (self.root / "file.txt").write_text("second\n", encoding="utf-8")
        self.git("commit", "--quiet", "--signoff", "-am", "Second")
        patch = mock.patch.object(changed_files, "REPO_ROOT", self.root)
        patch.start()
        self.addCleanup(patch.stop)

    def git(self, *args: str) -> str:
        return subprocess.run(
            args=[
                "git",
                "-c",
                "user.name=Test Author",
                "-c",
                "user.email=test@example.org",
                *args,
            ],
            cwd=self.root,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()

    def test_missing_explicit_reference_does_not_fall_back(self) -> None:
        with self.assertRaisesRegex(ValueError, "invalid base ref 'missing-base'"):
            changed_files.resolve_base_ref("missing-base")

    def test_missing_environment_reference_does_not_fall_back(self) -> None:
        with (
            mock.patch.dict(os.environ, {"BASE_REF": "missing-base"}),
            self.assertRaisesRegex(ValueError, "invalid base ref 'missing-base'"),
        ):
            changed_files.resolve_base_ref(None)

    def test_explicit_reference_takes_precedence(self) -> None:
        with mock.patch.dict(os.environ, {"BASE_REF": "missing-base"}):
            self.assertEqual(changed_files.resolve_base_ref("HEAD"), "HEAD")

    def test_non_commit_reference_is_rejected(self) -> None:
        for reference in ("HEAD:file.txt", "HEAD^{tree}"):
            with (
                self.subTest(reference=reference),
                self.assertRaisesRegex(ValueError, "invalid base ref"),
            ):
                changed_files.resolve_base_ref(reference)

    def test_default_reference_keeps_its_fallback(self) -> None:
        with mock.patch.dict(os.environ, {"BASE_REF": ""}):
            self.assertEqual(changed_files.resolve_base_ref(None), "HEAD^")
            self.git("update-ref", "refs/remotes/origin/main", "HEAD")
            self.assertEqual(changed_files.resolve_base_ref(None), "origin/main")

    def test_cli_reports_invalid_base_without_a_traceback(self) -> None:
        result = subprocess.run(
            args=[
                sys.executable,
                str(SCRIPT_DIR / "harness.py"),
                "changed-files",
                "--base-ref",
                "refs/heads/missing-base-reference-test",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 2)
        self.assertIn("invalid base ref", result.stderr)
        self.assertNotIn("Traceback", result.stderr)


if __name__ == "__main__":
    unittest.main()

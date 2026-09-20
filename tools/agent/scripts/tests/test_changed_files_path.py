import importlib
import json
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
            ("rev-parse", "--verify", "origin/main"): (0, "base\n"),
            ("merge-base", "HEAD", "origin/main"): (0, "base\n"),
            ("diff", "--name-only", "--no-renames", "-z", "base...HEAD"): (
                0,
                "committed.py\0shared.py\0",
            ),
            ("diff", "--name-only", "--no-renames", "-z", "HEAD"): (
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


class RenamedFilesTests(unittest.TestCase):
    old_path = "dashboard/frontend/src/components/Moved.tsx"
    new_path = "website/src/components/Moved.tsx"

    def setUp(self) -> None:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.git("init", "--quiet")
        self.git("config", "diff.renames", "true")
        source = self.root / self.old_path
        source.parent.mkdir(parents=True)
        source.write_text("export const moved = true;\n", encoding="utf-8")
        self.git("add", self.old_path)
        self.git("commit", "--quiet", "--signoff", "-m", "Add component")
        (self.root / self.new_path).parent.mkdir(parents=True)
        self.git("mv", self.old_path, self.new_path)

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

    def test_local_staged_rename_keeps_both_paths(self) -> None:
        with mock.patch.object(changed_files, "REPO_ROOT", self.root):
            paths = changed_files.git_changed_files("HEAD")
        self.assertEqual(paths, [self.old_path, self.new_path])

    def test_local_committed_rename_keeps_both_paths(self) -> None:
        self.git("commit", "--quiet", "--signoff", "-m", "Move component")
        with mock.patch.object(changed_files, "REPO_ROOT", self.root):
            paths = changed_files.git_changed_files("HEAD^")
        self.assertEqual(paths, [self.old_path, self.new_path])

    def test_ci_plan_keeps_the_source_domain_after_a_rename(self) -> None:
        self.git("commit", "--quiet", "--signoff", "-m", "Move component")
        output = self.root / "plan.json"
        subprocess.run(
            args=[
                sys.executable,
                str(CI_DIR / "ci_plan.py"),
                "--base",
                self.git("rev-parse", "HEAD^"),
                "--head",
                self.git("rev-parse", "HEAD"),
                "--output",
                str(output),
            ],
            cwd=self.root,
            capture_output=True,
            text=True,
            check=True,
        )
        plan = json.loads(output.read_text(encoding="utf-8"))
        self.assertEqual(plan["paths"], [self.old_path, self.new_path])
        self.assertIn("dashboard", plan["expected_verification_ids"])
        self.assertFalse(plan["quality_context"]["docs_only"])


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


if __name__ == "__main__":
    unittest.main()

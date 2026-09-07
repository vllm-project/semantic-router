from __future__ import annotations

import subprocess
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT = REPO_ROOT / "tools" / "ci" / "source-tree-revision.sh"


def run(*args: str, cwd: Path) -> str:
    return subprocess.run(
        args,
        cwd=cwd,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


class SourceTreeRevisionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.repo = Path(self.temp_dir.name)
        run("git", "init", "--quiet", cwd=self.repo)
        run("git", "config", "user.email", "test@vllm-sr.ai", cwd=self.repo)
        run("git", "config", "user.name", "vLLM-SR Test", cwd=self.repo)
        (self.repo / ".gitignore").write_text("ignored.txt\n", encoding="utf-8")
        (self.repo / "tracked.txt").write_text("initial\n", encoding="utf-8")
        run("git", "add", ".", cwd=self.repo)
        run("git", "commit", "--quiet", "-m", "initial", cwd=self.repo)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def revision(self) -> str:
        return run("bash", str(SCRIPT), str(self.repo), cwd=self.repo)

    def test_clean_checkout_uses_the_full_git_commit(self) -> None:
        self.assertEqual(
            self.revision(), run("git", "rev-parse", "HEAD", cwd=self.repo)
        )

    def test_dirty_checkout_uses_a_deterministic_sha256_tree(self) -> None:
        (self.repo / "tracked.txt").write_text("changed\n", encoding="utf-8")
        first = self.revision()
        second = self.revision()
        self.assertRegex(first, r"^sha256:[0-9a-f]{64}$")
        self.assertEqual(first, second)

    def test_untracked_source_changes_the_tree_revision(self) -> None:
        (self.repo / "tracked.txt").write_text("changed\n", encoding="utf-8")
        before = self.revision()
        (self.repo / "new-source.txt").write_text("new\n", encoding="utf-8")
        self.assertNotEqual(self.revision(), before)

    def test_gitignored_files_do_not_change_the_tree_revision(self) -> None:
        (self.repo / "tracked.txt").write_text("changed\n", encoding="utf-8")
        before = self.revision()
        (self.repo / "ignored.txt").write_text("local cache\n", encoding="utf-8")
        self.assertEqual(self.revision(), before)


if __name__ == "__main__":
    unittest.main()
